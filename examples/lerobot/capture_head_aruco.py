#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
从 ROS 2 截取 head_left / head_right 两路鱼眼图像，供
solve_pico_to_head_extrinsics.py 用来求 Pico 相机到这两个鱼眼相机的外参。

前提：Pico 头显和这两个鱼眼相机在整个标定过程中刚性固定、彼此不动。此脚本
只负责截鱼眼这两路；Pico 那一帧用 rcvpic.py 单独截（它已经会自动把 Pico 帧和
内参存到 calib_data/pico_camera/ 下）。

跟 collect_fisheye_calib.py 一样，这台机器的 OpenCV 没有 GUI 支持，没法弹窗
预览，所以每次按键截取后会立刻做一次检测，用文字告诉你两路有没有都拍到。

--board charuco 时用的是 ChArUco 标定板而不是单张 ArUco marker：单 marker 只有
4 个共面角点，PnP 解出来的深度（沿 marker 法线方向）噪声很大，两次标定换个位置
摆 marker，结果能差出 1-2 cm；ChArUco 板一次给几十个跨越整块板子的角点，
深度约束好得多。板子用 generate_charuco_board.py 生成/打印，见 charuco_common.py
顶部注释。

用法
----
    # 单 marker（旧方式，噪声较大，见上面的说明）
    python3 capture_head_aruco.py \\
        --left-topic /head/left/image_raw --right-topic /head/right/image_raw \\
        --board marker --dict 5X5_100 --marker-id 0

    # ChArUco 板（推荐）：参数要跟 generate_charuco_board.py 生成板子时一致
    python3 capture_head_aruco.py --board charuco \\
        --squares-x 7 --squares-y 5 --square-size 0.035 --marker-size 0.026
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from charuco_common import (
    ARUCO_DICTS,
    MIN_CHARUCO_CORNERS,
    add_charuco_args,
    build_board,
    detect_charuco,
)

DEFAULT_OUT = Path(__file__).parent / "calib_data" / "pico_to_head"


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class StereoBuffer(Node):
    def __init__(self, left_topic: str, right_topic: str):
        super().__init__("head_aruco_collector")
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] = {}
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, left_topic, lambda m: self._cb(m, "left"), qos)
        self.create_subscription(
            Image, right_topic, lambda m: self._cb(m, "right"), qos
        )
        self.get_logger().info(f"Subscribed: left={left_topic} right={right_topic}")

    def _cb(self, msg: Image, name: str) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"{name}: expected rgb8, got {msg.encoding}", once=True
            )
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        row = msg.width * 3
        if msg.step != row:
            buf = buf.reshape(msg.height, msg.step)[:, :row].reshape(-1)
        img = buf.reshape(msg.height, msg.width, 3)
        with self._lock:
            self._latest[name] = img.copy()

    def snapshot(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self._latest)

    def wait_for_both(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "left" in self.snapshot() and "right" in self.snapshot():
                return True
            time.sleep(0.1)
        return False


def check_marker(img_rgb: np.ndarray, aruco_dict, marker_id: int) -> bool:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    return ids is not None and marker_id in ids.flatten()


def check_charuco(img_rgb: np.ndarray, board) -> tuple[bool, int]:
    """Returns (good_enough, n_corners). Live feedback only -- the actual pose
    solve (with sub-pixel refinement against real K/D) happens in
    solve_pico_to_head_extrinsics.py, not here."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    charuco_corners, _ids = detect_charuco(gray, board)
    n = 0 if charuco_corners is None else len(charuco_corners)
    return n >= MIN_CHARUCO_CORNERS, n


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--left-topic", default="/head/left/image_raw")
    ap.add_argument("--right-topic", default="/head/right/image_raw")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--board",
        choices=("marker", "charuco"),
        default="marker",
        help="'marker': single ArUco marker (legacy, noisy depth). "
        "'charuco': ChArUco board (recommended, see module docstring)",
    )
    ap.add_argument(
        "--marker-id", type=int, default=0, help="only used with --board marker"
    )
    add_charuco_args(
        ap
    )  # --dict is shared; --squares-x/y, --square/marker-size are charuco-only
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="不清空 --out 目录里已有的旧截图（默认每次运行会先清空，"
        "避免这次没截够的编号残留上次的旧图，被 "
        "solve_pico_to_head_extrinsics.py 误当成新数据用）",
    )
    args = ap.parse_args()

    if args.board == "marker":
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
        board = None
    else:
        board, aruco_dict = build_board(args)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.keep_existing:
        stale = sorted(args.out.glob("head_left_*.png")) + sorted(
            args.out.glob("head_right_*.png")
        )
        if stale:
            for f in stale:
                f.unlink()
            print(
                f"已清空 {args.out} 里 {len(stale)} 张旧截图，避免跟这次新采集的编号混用"
                f"（用 --keep-existing 保留旧图）"
            )

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
    if args.board == "marker":
        print(f"ArUco 字典: DICT_{args.dict}  目标 ID: {args.marker_id}")
    else:
        print(
            f"ChArUco 板: {args.squares_x}x{args.squares_y} 格 @ "
            f"{args.square_size * 1000:.1f}mm，marker {args.marker_size * 1000:.1f}mm，"
            f"DICT_{args.dict}（至少要检测到 {MIN_CHARUCO_CORNERS} 个角点才算 FOUND）"
        )
    print(f"输出目录: {args.out}")
    print("\n's' = 截取一对（覆盖上一次）   'q' = 结束（只需要成功的一对就够）\n")

    # 固定编号 0000，每次按 s 直接覆盖：这个脚本的用法本来就是"没找到就重新摆好再
    # 截"，不是攒一叠候选帧。之前按编号累加会在同一次运行里留下 frame_0000（可能是
    # 摆位置摆歪时截的）、frame_0001... solve_pico_to_head_extrinsics.py 默认只读
    # frame_0000，于是每次实际用的是这次运行里第一次按的那对，不是你最后确认
    # FOUND/FOUND 的那对 —— 这正是标定结果每次跑都不一样的原因。固定覆盖后，
    # frame_0000 永远是你最后一次按 s 截到的画面，不用记编号、也不会有旧文件残留。
    frame_i = 0
    n_captured = 0
    try:
        while True:
            print("按 's' 截取（覆盖），'q' 退出...", end="\r", flush=True)
            ch = getch()
            if ch == "q":
                break
            if ch != "s":
                continue

            snap = node.snapshot()
            left_img, right_img = snap["left"], snap["right"]
            if args.board == "marker":
                left_ok = check_marker(left_img, aruco_dict, args.marker_id)
                right_ok = check_marker(right_img, aruco_dict, args.marker_id)
                status = (
                    f"left {'FOUND' if left_ok else 'NOT FOUND'} / "
                    f"right {'FOUND' if right_ok else 'NOT FOUND'}"
                )
            else:
                left_ok, left_n = check_charuco(left_img, board)
                right_ok, right_n = check_charuco(right_img, board)
                status = (
                    f"left {'FOUND' if left_ok else 'NOT FOUND'} ({left_n} corners) / "
                    f"right {'FOUND' if right_ok else 'NOT FOUND'} ({right_n} corners)"
                )

            cv2.imwrite(
                str(args.out / f"head_left_{frame_i:04d}.png"),
                cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(args.out / f"head_right_{frame_i:04d}.png"),
                cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR),
            )

            print(
                f"\n  已存 head_{{left,right}}_{frame_i:04d}.png（覆盖上一次）   {status}"
            )
            n_captured += 1

    except KeyboardInterrupt:
        print("\n\nCtrl+C — 退出")

    print(f"\n共截取 {n_captured} 次（每次覆盖同一对文件），保存在: {args.out}")
    teardown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
