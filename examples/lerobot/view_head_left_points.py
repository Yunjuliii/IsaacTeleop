#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
订阅 head_left 鱼眼相机，把给定的几个像素点画上去，再发布到一个新话题，
供另一台机器上的 rqt_image_view 实时查看。

这台机器上 OpenCV 没编译 GUI 支持（cv2.imshow 用不了），也没有可用的 X11
display，所以走 ROS 话题这条路：本机只管订阅 + 画点 + 重新发布，显示交给
另一台能跑 rqt 的机器。

发布的是 sensor_msgs/CompressedImage（JPEG），不是裸 Image：源话题
/head/left/image_raw 是 640x480 rgb8 未压缩，稳定占 ~28MB/s (~224Mbps)，这个
带宽跨网络传给另一台机器很容易把链路打满，DDS best-effort 下一丢分片就整帧
丢弃，表现出来就是画面卡顿。JPEG 压缩后同样画面通常只要几十 KB/帧，带宽降
一个数量级以上，rqt_image_view 能自动识别 CompressedImage 并显示。

用法
----
    # 本机（Jetson）：
    python3 view_head_left_points.py

    # 另一台机器（同一个 ROS domain / DDS 能互通）：
    ros2 run rqt_image_view rqt_image_view
    # 在 rqt_image_view 的话题下拉框里选 /head/left/image_points/compressed
    # （rqt_image_view 认 image_transport 的 .../compressed 命名约定，直接
    # 显示解码后的画面，不用你自己解压）

默认画的就是你给的三个点（x=481,y=207 / x=134,y=205 / x=88,y=402），也可以用
--points 自己指定，格式 "x,y x,y ..."：

    python3 view_head_left_points.py --points "481,207 134,205 88,402 200,300"
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

DEFAULT_POINTS = [(481, 207), (134, 205), (88, 402)]

# BGR，循环使用；点数超过颜色数就从头循环
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def parse_points(spec: str) -> list[tuple[int, int]]:
    pts = []
    for tok in spec.split():
        x_str, y_str = tok.split(",")
        pts.append((int(x_str), int(y_str)))
    return pts


class PointOverlay(Node):
    def __init__(
        self,
        in_topic: str,
        out_topic: str,
        points: list[tuple[int, int]],
        jpeg_quality: int,
    ):
        super().__init__("head_left_point_overlay")
        self._points = points
        self._jpeg_quality = jpeg_quality
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # CompressedImage，不是 Image：源话题是 640x480 rgb8 未压缩，跨网络传给
        # rqt_image_view 会占满带宽导致丢帧卡顿，JPEG 压缩后能小一个数量级以上。
        self._pub = self.create_publisher(CompressedImage, f"{out_topic}/compressed", qos)
        self.create_subscription(Image, in_topic, self._cb, qos)
        self._n_frames = 0
        self.get_logger().info(
            f"订阅 {in_topic}，画 {len(points)} 个点，"
            f"发布 JPEG(q={jpeg_quality}) 到 {out_topic}/compressed"
        )

    def _cb(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"期望 rgb8，收到 {msg.encoding}，跳过这一帧", once=True
            )
            return

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        row = msg.width * 3
        if msg.step != row:
            buf = buf.reshape(msg.height, msg.step)[:, :row].reshape(-1)
        img = buf.reshape(msg.height, msg.width, 3).copy()

        for i, (x, y) in enumerate(self._points):
            color = COLORS[i % len(COLORS)]
            cv2.circle(img, (x, y), 3, color, thickness=-1)

        # 画的是 rgb8，cv2.imencode 按 BGR 理解通道顺序，编码前转一下，
        # 否则颜色会红蓝互换（画面内容不受影响，只是点的颜色会错）。
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(
            ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            self.get_logger().warn("JPEG 编码失败，跳过这一帧", once=True)
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        out.data = jpg.tobytes()
        self._pub.publish(out)

        self._n_frames += 1
        if self._n_frames % 60 == 0:
            self.get_logger().info(f"已发布 {self._n_frames} 帧", throttle_duration_sec=5)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-topic", default="/head/left/image_raw")
    ap.add_argument("--out-topic", default="/head/left/image_points")
    ap.add_argument(
        "--points",
        type=str,
        default=None,
        help='"x,y x,y ..." 格式，默认用脚本里写死的三个点',
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="0-100，越高画质越好、带宽越大（默认 80）",
    )
    args = ap.parse_args()

    points = parse_points(args.points) if args.points else DEFAULT_POINTS

    rclpy.init()
    node = PointOverlay(args.in_topic, args.out_topic, points, args.jpeg_quality)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
