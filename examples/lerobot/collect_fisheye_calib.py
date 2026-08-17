#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
鱼眼双目相机内参标定 —— 采集标定图像对。

原理
----
从 ROS 2 话题订阅左右两路鱼眼图像（跟 record_cameras.py 订阅相机的方式一样，
rgb8 编码 + BEST_EFFORT QoS），每次按键截取当前"最新一帧"左右图像对，落盘成
PNG（无损，不用 JPEG 有损压缩，标定对像素精度敏感）。

因为这台机器的 OpenCV 是无 GUI 后端编译的（cv2.imshow 会直接崩），没法弹窗
预览取景，所以每次截取后脚本会立刻在本地跑一次 cv2.findChessboardCorners，
用文字告诉你这一帧有没有找到棋盘格角点 —— 用这个代替"看预览判断要不要留"。

操作步骤
--------
1. 确认棋盘格参数（--rows/--cols/--square-size），跟你实际打印的板子对上，
   这里只用来做检测反馈，不影响标定本身用的是原始检测结果。
2. 打印棋盘格贴在硬板上，跑起来后把板子举到相机各个位置：
   - 近、中、远都要有（尤其近距离让棋盘格填满大半画面）
   - 四个角落 + 画面中心都要覆盖到（鱼眼边缘畸变大，角落必须覆盖）
   - 各种倾斜角度都要有（不要总是正对镜头）
3. 每次摆好之后按 's' 截取一对，看提示的检测结果：
   - 两路都 "found" 才是有效的一对，留着
   - 有一路 "NOT FOUND" 就重新摆一下再截，脚本不会自动删，需要你自己事后清理
4. 目标 25~35 对有效图像，按 'q' 结束

输出
----
  <out>/left/frame_0000.png, frame_0001.png, ...
  <out>/right/frame_0000.png, frame_0001.png, ...
  左右路用相同文件名对应同一帧，供 cv2.fisheye.calibrate / stereoCalibrate 使用。
  注意：每次启动都会先清空 <out>/left、<out>/right 里上次留下的图，不会跨次累积。

用法
----
    python3 collect_fisheye_calib.py \\
        --left-topic /fisheye/left/image_raw \\
        --right-topic /fisheye/right/image_raw \\
        --rows 10 --cols 7 --square-size 0.025

    python3 collect_fisheye_calib.py --out calib_data/fisheye_intrinsics
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

DEFAULT_OUT = Path(__file__).parent / "calib_data" / "fisheye_intrinsics"


# --------------------------------------------------------------------------- #
# 键盘：跟 record_cameras.py / collect_pivot_calib.py 一致的单键读取
# --------------------------------------------------------------------------- #
def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --------------------------------------------------------------------------- #
# ROS 节点：订阅左右两路图像，始终保留最新一帧
# --------------------------------------------------------------------------- #
class StereoBuffer(Node):
    def __init__(self, left_topic: str, right_topic: str):
        super().__init__("fisheye_calib_collector")
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] = {}
        self._counts = {"left": 0, "right": 0}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, left_topic,
                                 lambda m: self._cb(m, "left"), qos)
        self.create_subscription(Image, right_topic,
                                 lambda m: self._cb(m, "right"), qos)
        self.get_logger().info(f"Subscribed: left={left_topic} right={right_topic}")

    def _cb(self, msg: Image, name: str) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"{name}: expected rgb8, got {msg.encoding}", once=True)
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        row = msg.width * 3
        if msg.step != row:
            buf = buf.reshape(msg.height, msg.step)[:, :row].reshape(-1)
        img = buf.reshape(msg.height, msg.width, 3)
        with self._lock:
            self._latest[name] = img.copy()
            self._counts[name] += 1

    def snapshot(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self._latest)

    def wait_for_both(self, timeout: float = 30.0) -> bool:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "left" in self.snapshot() and "right" in self.snapshot():
                return True
            time.sleep(0.1)
        return False


# --------------------------------------------------------------------------- #
def check_corners(img_rgb: np.ndarray, pattern: tuple[int, int]) -> bool:
    """跟标定代码用同一个检测函数，只用于当场反馈，不影响后续真正标定的检测。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    found, _ = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-topic", default="/fisheye/left/image_raw")
    ap.add_argument("--right-topic", default="/fisheye/right/image_raw")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rows", type=int, default=10,
                    help="棋盘格行方向格子数（不是内角点数）")
    ap.add_argument("--cols", type=int, default=7,
                    help="棋盘格列方向格子数（不是内角点数）")
    ap.add_argument("--square-size", type=float, default=0.025,
                    help="格子实测边长，单位米，仅用于此处打印提示，不写入图像")
    args = ap.parse_args()

    # 内角点数 = 格子数 - 1
    pattern = (args.cols - 1, args.rows - 1)

    left_dir = args.out / "left"
    right_dir = args.out / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # 每次开新的采集session都是一次独立标定，跟上一次(可能是别的分辨率/别的时间)
    # 的图像混在一起会让人分不清哪些帧属于这次，还可能把旧分辨率的图喂进标定。
    # 所以启动时先清空，不留给用户手动清理的步骤。
    stale = sorted(left_dir.glob("frame_*.png")) + sorted(right_dir.glob("frame_*.png"))
    if stale:
        print(f"清空上次采集留下的 {len(stale)} 张旧图像...")
        for f in stale:
            f.unlink()

    rclpy.init()
    node = StereoBuffer(args.left_topic, args.right_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    def teardown() -> None:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()

    print(f"Waiting for frames on {args.left_topic} / {args.right_topic} ...")
    if not node.wait_for_both(timeout=30.0):
        print("ERROR: 没收到两路图像，检查话题名和相机是否在发布。", file=sys.stderr)
        teardown()
        return 1

    snap = node.snapshot()
    for name, img in sorted(snap.items()):
        print(f"  {name}: {img.shape[1]}x{img.shape[0]}")
    print(f"棋盘格内角点: {pattern[0]} x {pattern[1]}  (格子边长 {args.square_size*1000:.1f}mm)")
    print(f"输出目录: {args.out}")
    print("\n's' = 截取一对   'q' = 结束\n")

    frame_i = 0
    saved_pairs = 0
    try:
        while True:
            print(f"[{saved_pairs} 对已存] 按 's' 截取，'q' 退出...", end="\r", flush=True)
            ch = getch()
            if ch == "q":
                break
            if ch != "s":
                continue

            snap = node.snapshot()
            if "left" not in snap or "right" not in snap:
                print("\n  跳过：还没收到两路图像")
                continue

            left_img, right_img = snap["left"], snap["right"]
            left_ok = check_corners(left_img, pattern)
            right_ok = check_corners(right_img, pattern)

            fname = f"frame_{frame_i:04d}.png"
            cv2.imwrite(str(left_dir / fname),
                       cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(right_dir / fname),
                       cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

            status = f"left {'FOUND' if left_ok else 'NOT FOUND'} / right {'FOUND' if right_ok else 'NOT FOUND'}"
            print(f"\n  已存 {fname}   {status}")
            if left_ok and right_ok:
                saved_pairs += 1

            frame_i += 1

    except KeyboardInterrupt:
        print("\n\nCtrl+C — 退出")

    print(f"\n共截取 {frame_i} 对，其中两路都检测到角点的 {saved_pairs} 对。")
    print(f"图像保存在: {args.out}")
    teardown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
