#!/usr/bin/env python3
"""
点击图片，记录点击点在图像坐标系（像素坐标，原点在左上角，x向右，y向下）下的坐标。

需要图形界面：远程使用时请先用 `ssh -Y` 登录（X11 转发）再运行。

用法:
    python pick_points.py <image_path> [--out points.json] [--num-points 3] [--scale 2]

    --num-points N : 最多记录 N 个点（默认 0 = 不限制），点够后会立刻打印结果
    --scale S      : 放大 S 倍显示以便精确点击，输出坐标仍是原图像素坐标

交互:
    移动鼠标   : 左上角实时显示鼠标所在位置的原图坐标
    左键点击   : 记录一个点，并在图上画出编号
    按 'u'    : 撤销上一个点
    按 's'    : 保存当前所有点到 json 文件
    按 'q' 或 ESC : 退出（退出前会自动保存）

退出时会打印可直接粘贴到 view_head_left_with_points.py 的
`--points x,y x,y ...` 参数以及 DEFAULT_POINTS 列表。
"""

import argparse
import json
import sys

import cv2


def save(points, path):
    with open(path, "w") as f:
        json.dump(points, f, indent=2)
    print(f"saved {len(points)} points to {path}")


def print_summary(points):
    if not points:
        return
    print("--points " + " ".join(f"{x},{y}" for x, y in points))
    print("DEFAULT_POINTS = " + str([(x, y) for x, y in points]))


def main():
    parser = argparse.ArgumentParser(description="Click points on an image and get their pixel coordinates.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--out", default="points.json", help="Path to save clicked points as JSON")
    parser.add_argument("--num-points", type=int, default=0,
                        help="Maximum number of points to record (0 = unlimited)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Display magnification; output stays in original image pixels")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image '{args.image}'", file=sys.stderr)
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"image {args.image}: {w}x{h}  (left-click=add, u=undo, s=save, q/ESC=quit)")

    scale = args.scale
    if scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        disp_base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    else:
        disp_base = img

    points = []
    hover = [None]
    window_name = "click points (u=undo, s=save, q/ESC=quit)"

    def to_img(x, y):
        """Displayed-window pixel -> original image pixel (clamped to image bounds)."""
        return (min(max(int(round(x / scale)), 0), w - 1),
                min(max(int(round(y / scale)), 0), h - 1))

    def to_disp(x, y):
        return (int(round(x * scale)), int(round(y * scale)))

    def redraw():
        disp = disp_base.copy()
        for i, (x, y) in enumerate(points):
            px, py = to_disp(x, y)
            cv2.drawMarker(disp, (px, py), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            cv2.putText(disp, f"{i}:({x},{y})", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if hover[0] is not None:
            cv2.putText(disp, f"({hover[0][0]},{hover[0][1]})", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            hover[0] = to_img(x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            if args.num_points and len(points) >= args.num_points:
                print(f"already have {args.num_points} points (u=undo, q=quit)")
                return
            points.append(to_img(x, y))
            print(f"point {len(points) - 1}: (x={points[-1][0]}, y={points[-1][1]})")
            redraw()
            if args.num_points and len(points) == args.num_points:
                print_summary(points)
                print("press q to quit, u to undo the last point")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
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
            save(points, args.out)
        # Window closed via the title-bar button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    save(points, args.out)
    print_summary(points)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
