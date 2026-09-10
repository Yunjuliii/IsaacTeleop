#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
把 add_wrist_pose.py 算出来的手腕在相机坐标系下的位置，投影回对应的
head_left / head_right 鱼眼视频上画出来，生成带标注的新视频。数据集里
如果还有 add_sharpa_joints.py 写的 observation.sharpa_base_*_in_head_*
（stage 帧数据集才有），两个 Sharpa 底座也一并画上。每个位姿画完整
三轴而不是单点：修好姿态目标之后底座原点应该贴着腕点（差 ~0.5cm，
投影后原点基本重合），而两者轴向差一个固定约定旋转 + 逐帧 IK 补偿——
"原点咬合、轴向岔开"就是预期图景，肉眼即可验收。

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
from scipy.spatial.transform import Rotation

CALIB_DIR = Path(__file__).parent / "calib_data"

# 每个位姿画完整三轴（X红 Y绿 Z蓝，BGR），原点画一个小实心点+标签区分
# 四个坐标系：腕用细线，Sharpa 底座用粗线。腕和底座原点只差 ~0.5cm，
# 投影后基本重合——重合的原点 + 岔开的轴向正是预期图景（轴向差一个
# 固定约定旋转 + 逐帧 IK 补偿）。
AXIS_LEN = 0.05          # 轴长 [m]
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))   # X, Y, Z
LABEL_COLORS = {"L": (0, 220, 255), "R": (255, 80, 220),        # 黄=左腕, 品红=右腕
                "SB-L": (80, 255, 80), "SB-R": (255, 220, 0)}   # 绿/青=左右底座
LABEL_OFFSET = (8, -8)


def load_intrinsics(side: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(CALIB_DIR / "fisheye_intrinsics" / f"{side}_intrinsics.npz")
    K = data["K"].astype(np.float64)
    D = data["D"].astype(np.float64)
    w, h = data["image_size"].tolist()
    return K, D, (w, h)


def scale_K(
    K: np.ndarray, calib_size: tuple[int, int], actual_size: tuple[int, int]
) -> np.ndarray:
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


def draw_frame_axes(
    frame: np.ndarray,
    pose8: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    label: str,
    thickness: int = 1,
) -> None:
    """把 pose8 [x,y,z,qx,qy,qz,qw,valid] 画成三轴坐标系。

    原点 valid=0 / 在相机后方 / 出画面就整个不画；单根轴的端点在相机
    后方只跳过那根轴。5cm 的轴长下鱼眼弧度可忽略，两端点投影后直接
    连直线即可。"""
    if pose8[7] < 0.5:
        return
    p = pose8[0:3].astype(np.float64)
    if p[2] <= 0:
        return
    R = Rotation.from_quat(pose8[3:7].astype(np.float64)).as_matrix()
    ends = p[None, :] + AXIS_LEN * R.T          # 行 i = 原点 + 轴长*第 i 列
    obj = np.vstack([p[None, :], ends]).reshape(-1, 1, 3)
    img_pts, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
    pix = img_pts.reshape(-1, 2)

    h, w = frame.shape[:2]
    if not (0 <= pix[0, 0] < w and 0 <= pix[0, 1] < h):
        return
    origin = (int(round(pix[0, 0])), int(round(pix[0, 1])))
    for i in range(3):
        if ends[i, 2] <= 0:
            continue
        e = pix[1 + i]
        # 出画面交给 cv2.line 裁剪，但要挡住鱼眼模型在视场边缘外的发散值。
        if not np.all(np.isfinite(e)) or abs(e[0]) > 4 * w or abs(e[1]) > 4 * h:
            continue
        cv2.line(frame, origin, (int(round(e[0])), int(round(e[1]))),
                 AXIS_COLORS[i], thickness, cv2.LINE_AA)
    color = LABEL_COLORS[label]
    cv2.circle(frame, origin, 3, color, thickness=-1, lineType=cv2.LINE_AA)
    # 腕和底座的原点几乎重合，标签一个放右上一个放右下，避免互相覆盖。
    dy = -LABEL_OFFSET[1] + 8 if label.startswith("SB") else LABEL_OFFSET[1]
    cv2.putText(
        frame,
        label,
        (origin[0] + LABEL_OFFSET[0], origin[1] + dy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def process_episode(
    repo_id: str,
    root: Path,
    episode_index: int,
    cam_sides: list[str],
    out_dir: Path,
    max_seconds: float | None = None,
    has_sharpa: bool = False,
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

    for cam in cam_sides:
        K, D, (calib_w, calib_h) = load_intrinsics(cam)
        img_key = f"observation.images.head_{cam}"
        left_key = f"observation.wrist_left_in_head_{cam}"
        right_key = f"observation.wrist_right_in_head_{cam}"
        base_keys = {
            side: f"observation.sharpa_base_{side}_in_head_{cam}"
            for side in ("left", "right")
        } if has_sharpa else {}

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
                    print(
                        f"  {img_key} 视频分辨率 {w}x{h} 跟标定用的 {calib_w}x{calib_h} "
                        f"不一致（多半是相机分辨率换过之后没有重新标定），已按比例缩放 K 再投影",
                        file=sys.stderr,
                    )
                    K = scale_K(K, (calib_w, calib_h), (w, h))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_path), fourcc, ds.fps, (w, h))

            pose_l = item[left_key].numpy()
            pose_r = item[right_key].numpy()
            before = (pose_l[7] > 0.5, pose_r[7] > 0.5)
            draw_frame_axes(frame, pose_l, K, D, "L", thickness=1)
            draw_frame_axes(frame, pose_r, K, D, "R", thickness=1)
            drawn_left += int(before[0])
            drawn_right += int(before[1])
            for side, key in base_keys.items():
                draw_frame_axes(frame, item[key].numpy(), K, D,
                                f"SB-{side[0].upper()}", thickness=2)

            writer.write(frame)

        writer.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(out_path),
            ],
            check=True,
        )
        tmp_path.unlink()

        print(
            f"  episode {episode_index} / head_{cam}: {n} 帧 -> {out_path}  "
            f'(L 有效 {drawn_left}, R 有效 {drawn_right} — 这里的"有效"只看 valid 位，'
            f"不代表实际画出来了，z<=0 或出画面的没算进去)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--repo-id",
        default="teleop/sensing_gmsl2_rig",
        help="随便填，LeRobotDataset 只用它做缓存标识，不影响读取本地数据"
        "（默认跟 record_cameras.py 的默认值一致）",
    )
    ap.add_argument(
        "--episode",
        type=int,
        default=None,
        help="只处理这一个 episode（默认：处理数据集里全部 episode）",
    )
    ap.add_argument("--cam", choices=("left", "right", "both"), default="both")
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="默认: <dataset-root>/wrist_overlay/"
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="每个 episode 只叠加开头这么多秒（默认：整段都叠加）",
    )
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(
            f"ERROR: {args.dataset_root} 不是 LeRobot 数据集根目录（缺 meta/info.json）",
            file=sys.stderr,
        )
        return 1

    import json

    info = json.load(open(args.dataset_root / "meta" / "info.json"))
    needed = [
        "observation.wrist_left_in_head_left",
        "observation.wrist_right_in_head_left",
    ]
    if not all(k in info["features"] for k in needed):
        print(
            "ERROR: 数据集里没有 observation.wrist_*_in_head_* 列，"
            "先跑 add_wrist_pose.py。",
            file=sys.stderr,
        )
        return 1

    sharpa_needed = [
        f"observation.sharpa_base_{s}_in_head_{c}"
        for s in ("left", "right") for c in ("left", "right")
    ]
    has_sharpa = all(k in info["features"] for k in sharpa_needed)
    if not has_sharpa:
        print("提示: 数据集里没有 observation.sharpa_base_*_in_head_* 列"
              "（local 帧数据集没有这四列，stage 帧的先跑 add_sharpa_joints.py），"
              "本次只画腕点。")

    cam_sides = ["left", "right"] if args.cam == "both" else [args.cam]
    out_dir = args.out_dir or (args.dataset_root / "wrist_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = (
        [args.episode]
        if args.episode is not None
        else list(range(info["total_episodes"]))
    )
    print(f"处理 {len(episodes)} 个 episode，相机: {cam_sides}，输出到: {out_dir}\n")

    for ep in episodes:
        process_episode(
            args.repo_id,
            args.dataset_root,
            ep,
            cam_sides,
            out_dir,
            max_seconds=args.max_seconds,
            has_sharpa=has_sharpa,
        )

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
