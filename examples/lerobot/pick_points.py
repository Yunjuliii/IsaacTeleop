#!/usr/bin/env python3
"""
点击图片，记录点击点在图像坐标系（像素坐标，原点在左上角，x向右，y向下）下的坐标。

用法:
    python pick_points.py <image_path> [--out points.json]

交互:
    左键点击   : 记录一个点，并在图上画出编号
    按 'u'    : 撤销上一个点
    按 's'    : 保存当前所有点到 json 文件
    按 'q' 或 ESC : 退出（退出前会自动保存）
"""

import argparse
import json
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="Click points on an image and get their pixel coordinates.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--out", default="points.json", help="Path to save clicked points as JSON")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image '{args.image}'", file=sys.stderr)
        sys.exit(1)

    points = []
    window_name = "click points (u=undo, s=save, q/ESC=quit)"

    def redraw():
        disp = img.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"point {len(points) - 1}: (x={x}, y={y})")
            redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('u'):
            if points:
                removed = points.pop()
                print(f"undo point: {removed}")
                redraw()
        elif key == ord('s'):
            with open(args.out, "w") as f:
                json.dump(points, f, indent=2)
            print(f"saved {len(points)} points to {args.out}")

    with open(args.out, "w") as f:
        json.dump(points, f, indent=2)
    print(f"saved {len(points)} points to {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
