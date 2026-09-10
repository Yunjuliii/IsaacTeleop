#!/usr/bin/env python3
"""Overlay fixed reference points on the head/right camera and republish it.

Subscribes to /head/right/image_raw (sensor_msgs/Image, rgb8), draws the given
points (in image pixel coordinates) on every frame, and republishes the result
as a sensor_msgs/CompressedImage topic so it can be viewed remotely with rqt
(rqt_image_view) on another machine -- no local display/GUI needed on this box.

Usage::

    python3 view_head_left_with_points.py
    python3 view_head_left_with_points.py --topic /head/right/image_raw \
        --out-topic /head/right/points/compressed \
        --points "148,404" "202,226" "522,225"

On the viewing machine (same ROS_DOMAIN_ID / network):

    rqt_image_view  # then pick /head/right/points/compressed
"""

import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

# 用户提供的三个标注点（图像坐标系：原点左上角，x 向右，y 向下）。
DEFAULT_POINTS = [(148, 404), (202, 226), (522, 225)]


def parse_points(raw: list[str]) -> list[tuple[int, int]]:
    points = []
    for item in raw:
        x_str, y_str = item.split(",")
        points.append((int(x_str), int(y_str)))
    return points


class PointOverlayRepublisher(Node):
    def __init__(
        self,
        topic: str,
        out_topic: str,
        points: list[tuple[int, int]],
        jpeg_quality: int,
    ):
        super().__init__("head_right_point_republisher")
        self._points = points
        self._jpeg_quality = jpeg_quality

        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(CompressedImage, out_topic, qos_out)
        self.create_subscription(Image, topic, self._on_image, qos_in)
        self.get_logger().info(f"Subscribed to {topic}, publishing to {out_topic}")

    def _on_image(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(f"expected rgb8, got {msg.encoding}", once=True)
            return

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        row = msg.width * 3
        if msg.step != row:
            buf = buf.reshape(msg.height, msg.step)[:, :row].reshape(-1)
        img = buf.reshape(msg.height, msg.width, 3)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for i, (x, y) in enumerate(self._points):
            cv2.drawMarker(
                frame, (x, y), (0, 0, 255),
                markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
            )
            cv2.putText(
                frame, f"{i}:({x},{y})", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            self.get_logger().warn("jpeg encode failed", once=True)
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        out.data = encoded.tobytes()
        self._pub.publish(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/head/right/image_raw", help="Input image topic")
    parser.add_argument(
        "--out-topic", default="/head/right/points/compressed",
        help="Output sensor_msgs/CompressedImage topic (view with rqt_image_view)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (1-100)")
    parser.add_argument(
        "--points", nargs="+", metavar="X,Y", default=None,
        help='Points to overlay as "x,y" pairs, e.g. --points 201,225 146,404 520,225. '
             "Defaults to the three points given for this task.",
    )
    args = parser.parse_args()

    points = parse_points(args.points) if args.points else DEFAULT_POINTS

    rclpy.init()
    node = PointOverlayRepublisher(args.topic, args.out_topic, points, args.jpeg_quality)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
