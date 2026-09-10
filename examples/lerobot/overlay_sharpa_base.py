#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
把 add_sharpa_retargeted_episode.py 写的 observation.sharpa_base_left/right
（已经是 head_left 鱼眼相机系下的位姿，见该脚本文件头的坐标转换链）投影到
对应 episode 的 head_left 视频上，画成三轴坐标系，生成一份可播放的带标注
mp4。

跟 overlay_wrist_pose.py 是同一套投影/画图代码（同一个鱼眼畸变模型、同样
"原点+三轴"画法），只是这里的 sharpa_base 列本来就已经在相机系下，不需要
再叠一次 observation.wrist_*_in_head_* 或 sharpa_base_*_in_head_* 的相机
选择逻辑——因为 add_sharpa_retargeted_episode.py 只算了 head_left 这一路。

用法
----
    python3 overlay_sharpa_base.py --dataset-root /home/nvidia/IsaacTeleop/dataset --episode 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from overlay_wrist_pose import draw_frame_axes, load_intrinsics, scale_K

CAM = "left"  # sharpa_base_left/right 目前只算了 head_left 这一路


def process_episode(
    repo_id: str,
    root: Path,
    episode_index: int,
    out_path: Path,
    max_seconds: float | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=repo_id, root=str(root), episodes=[episode_index], download_videos=False
    )
    n = len(ds)
    if n == 0:
        print(f"  episode {episode_index}: 没有帧，跳过")
        return
    if max_seconds is not None:
        n = min(n, max(1, round(max_seconds * ds.fps)))

    K, D, (calib_w, calib_h) = load_intrinsics(CAM)
    img_key = f"observation.images.head_{CAM}"
    base_keys = {"left": "observation.sharpa_base_left",
                 "right": "observation.sharpa_base_right"}
    labels = {"left": "SB-L", "right": "SB-R"}

    tmp_path = out_path.with_suffix(".raw.mp4")
    writer = None
    drawn = {"left": 0, "right": 0}

    for i in range(n):
        item = ds[i]
        img = (item[img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        if writer is None:
            if (w, h) != (calib_w, calib_h):
                print(
                    f"  {img_key} 视频分辨率 {w}x{h} 跟标定用的 {calib_w}x{calib_h} "
                    f"不一致，已按比例缩放 K 再投影",
                    file=sys.stderr,
                )
                K = scale_K(K, (calib_w, calib_h), (w, h))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_path), fourcc, ds.fps, (w, h))

        for side, key in base_keys.items():
            pose = item[key].numpy()
            drawn[side] += int(pose[7] > 0.5)
            draw_frame_axes(frame, pose, K, D, labels[side], thickness=2)

        writer.write(frame)

    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)],
        check=True,
    )
    tmp_path.unlink()

    print(f"  episode {episode_index} / head_{CAM}: {n} 帧 -> {out_path}  "
          f'(左手底座有效 {drawn["left"]}, 右手底座有效 {drawn["right"]} — '
          f'这里的"有效"只看 valid 位，不代表实际画出来了，z<=0 或出画面的没算进去)')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", default="teleop/sensing_gmsl2_rig",
                     help="随便填，LeRobotDataset 只用它做缓存标识")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-path", type=Path, default=None,
                     help="默认: <dataset-root>/sharpa_base_overlay/head_left_sharpa_base_overlay_ep<N>.mp4")
    ap.add_argument("--max-seconds", type=float, default=None,
                     help="只叠加开头这么多秒（默认：整段）")
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: {args.dataset_root} 不是 LeRobot 数据集根目录（缺 meta/info.json）",
              file=sys.stderr)
        return 1

    import json
    info = json.load(open(args.dataset_root / "meta" / "info.json"))
    needed = ["observation.sharpa_base_left", "observation.sharpa_base_right"]
    missing = [k for k in needed if k not in info["features"]]
    if missing:
        print(f"ERROR: 数据集缺 {missing}，先跑 add_sharpa_retargeted_episode.py",
              file=sys.stderr)
        return 1

    out_path = args.out_path or (
        args.dataset_root / "sharpa_base_overlay"
        / f"head_left_sharpa_base_overlay_ep{args.episode:06d}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"处理 episode {args.episode}，输出到: {out_path}\n")
    process_episode(args.repo_id, args.dataset_root, args.episode, out_path,
                     max_seconds=args.max_seconds)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
