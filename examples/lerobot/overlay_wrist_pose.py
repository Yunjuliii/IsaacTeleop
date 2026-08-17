#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
把 add_wrist_camera_pose.py 算出来的手腕在相机坐标系下的位置，投影回对应的
head_left / head_right 鱼眼视频上画出来，生成带标注的新视频。

投影用的是跟标定时同一个鱼眼畸变模型（cv2.fisheye.projectPoints，用
calibrate_fisheye.py 存的 K, D），不是先去畸变再画：视频本身是原始畸变画面，
直接在畸变模型下投影才能落在正确的像素位置上。

一帧里手腕点满足以下任一条件就不画（只是跳过这一个点，另一只手/另一帧照常画）：
  - 该帧 observation.wrist_{side}_in_head_{cam} 的 valid=0（手柄或头显本身无效）
  - z <= 0（手腕在相机后方，几何上不可能投影到画面里）
  - 投影出的像素落在画面外

用法
----
    python3 overlay_wrist_pose.py --dataset-root ~/datasets/my_task
    python3 overlay_wrist_pose.py --dataset-root ~/datasets/my_task --episode 0 --cam left
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).parent / "calib_data"

# BGR
WRIST_COLORS = {"left": (0, 220, 255), "right": (255, 80, 220)}  # 黄=左手, 品红=右手
POINT_RADIUS = 6
LABEL_OFFSET = (8, -8)


def load_intrinsics(side: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(CALIB_DIR / "fisheye_intrinsics" / f"{side}_intrinsics.npz")
    K = data["K"].astype(np.float64)
    D = data["D"].astype(np.float64)
    w, h = data["image_size"].tolist()
    return K, D, (w, h)


def scale_K(K: np.ndarray, calib_size: tuple[int, int], actual_size: tuple[int, int]) -> np.ndarray:
    """calib_size / actual_size 都是 (w, h)。跟 solve_pico_to_head_extrinsics.py 里
    同名函数一个道理：nvvidconv 是等比单独缩放（非裁剪），所以标定分辨率跟视频实际
    分辨率不一致时，fx/cx 按宽度比例、fy/cy 按高度比例线性缩放即可，不用重新标定。
    D 是无量纲的，原样复用。"""
    cw, ch = calib_size
    aw, ah = actual_size
    if (cw, ch) == (aw, ah):
        return K
    sx, sy = aw / cw, ah / ch
    K = K.copy()
    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K


def project_point(p_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray | None:
    """p_cam: (3,) 相机坐标系下的点。z<=0（在相机后方）返回 None。"""
    if p_cam[2] <= 0:
        return None
    obj = p_cam.reshape(1, 1, 3).astype(np.float64)
    img_pts, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
    return img_pts[0, 0]


def draw_wrist(frame: np.ndarray, pose8: np.ndarray, K: np.ndarray, D: np.ndarray,
               color: tuple[int, int, int], label: str) -> None:
    """pose8: (8,) [x,y,z,qx,qy,qz,qw,valid]。不合法/画面外就什么都不画。"""
    if pose8[7] < 0.5:
        return
    pt = project_point(pose8[0:3], K, D)
    if pt is None:
        return
    h, w = frame.shape[:2]
    x, y = pt
    if not (0 <= x < w and 0 <= y < h):
        return
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, POINT_RADIUS, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, POINT_RADIUS + 2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (center[0] + LABEL_OFFSET[0], center[1] + LABEL_OFFSET[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def process_episode(repo_id: str, root: Path, episode_index: int, cam_sides: list[str],
                    out_dir: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=str(root), episodes=[episode_index],
                        download_videos=False)
    n = len(ds)
    if n == 0:
        print(f"  episode {episode_index}: 没有帧，跳过")
        return

    for cam in cam_sides:
        K, D, (calib_w, calib_h) = load_intrinsics(cam)
        img_key = f"observation.images.head_{cam}"
        left_key = f"observation.wrist_left_in_head_{cam}"
        right_key = f"observation.wrist_right_in_head_{cam}"

        out_path = out_dir / f"head_{cam}_wrist_overlay_ep{episode_index:06d}.mp4"
        # cv2.VideoWriter 在这台机器上只能编 mpeg4（fourcc mp4v），大多数播放器/浏览器
        # 现在只认 h264，所以先写到临时文件，最后用 ffmpeg 转成 h264 再替换掉。
        tmp_path = out_path.with_suffix(".raw.mp4")
        writer = None
        drawn_left = drawn_right = 0

        for i in range(n):
            item = ds[i]
            img = (item[img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]

            if writer is None:
                if (w, h) != (calib_w, calib_h):
                    print(f"  {img_key} 视频分辨率 {w}x{h} 跟标定用的 {calib_w}x{calib_h} "
                          f"不一致（多半是相机分辨率换过之后没有重新标定），已按比例缩放 K 再投影",
                          file=sys.stderr)
                    K = scale_K(K, (calib_w, calib_h), (w, h))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_path), fourcc, ds.fps, (w, h))

            pose_l = item[left_key].numpy()
            pose_r = item[right_key].numpy()
            before = (pose_l[7] > 0.5, pose_r[7] > 0.5)
            draw_wrist(frame, pose_l, K, D, WRIST_COLORS["left"], "L")
            draw_wrist(frame, pose_r, K, D, WRIST_COLORS["right"], "R")
            drawn_left += int(before[0])
            drawn_right += int(before[1])

            writer.write(frame)

        writer.release()

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)],
            check=True,
        )
        tmp_path.unlink()

        print(f"  episode {episode_index} / head_{cam}: {n} 帧 -> {out_path}  "
              f"(L 有效 {drawn_left}, R 有效 {drawn_right} — 这里的\"有效\"只看 valid 位，"
              f"不代表实际画出来了，z<=0 或出画面的没算进去)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", default="teleop/sensing_gmsl2_rig",
                    help="随便填，LeRobotDataset 只用它做缓存标识，不影响读取本地数据"
                         "（默认跟 record_cameras.py 的默认值一致）")
    ap.add_argument("--episode", type=int, default=None,
                    help="只处理这一个 episode（默认：处理数据集里全部 episode）")
    ap.add_argument("--cam", choices=("left", "right", "both"), default="both")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="默认: <dataset-root>/wrist_overlay/")
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: {args.dataset_root} 不是 LeRobot 数据集根目录（缺 meta/info.json）",
              file=sys.stderr)
        return 1

    import json
    info = json.load(open(args.dataset_root / "meta" / "info.json"))
    needed = [f"observation.wrist_left_in_head_left", f"observation.wrist_right_in_head_left"]
    if not all(k in info["features"] for k in needed):
        print("ERROR: 数据集里没有 observation.wrist_*_in_head_* 列，"
              "先跑 add_wrist_camera_pose.py。", file=sys.stderr)
        return 1

    cam_sides = ["left", "right"] if args.cam == "both" else [args.cam]
    out_dir = args.out_dir or (args.dataset_root / "wrist_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [args.episode] if args.episode is not None else list(range(info["total_episodes"]))
    print(f"处理 {len(episodes)} 个 episode，相机: {cam_sides}，输出到: {out_dir}\n")

    for ep in episodes:
        process_episode(args.repo_id, args.dataset_root, ep, cam_sides, out_dir)

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
