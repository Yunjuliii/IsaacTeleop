#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
把 Manus 腕（manus_hand_tracking_plugin.cpp 锚手用的那个 wrist，跟
add_wrist_pose.py 算的"controller 标定腕"是两个不同的点）投影到
head_left/head_right 鱼眼视频上画出来，验收 grip_to_aim 标定链。

跟 observation.wrist_*_in_world 是两个不同的 wrist（细节见
calibrate_grip_to_aim.py 的 docstring 和 add_wrist_pose.py 顶部注释）：

    wrist_in_world(t)      = T_grip->stage(t) . T_wrist->ctrl        (pivot+rotation 标定)
    manusWrist_world(t)    = T_grip->stage(t) . T_grip->aim . kHandOffset
                                                (manus_hand_tracking_plugin.cpp 锚手方式)

两条链只差一个常数 C：

    C = inv(T_wrist->ctrl) . T_grip->aim . kHandOffset
    manusWrist_world(t) = wrist_in_world(t) . C

C 由本脚本现场合成（见 load_latest_grip_to_aim_C 的 docstring），不要用
grip_to_aim/*.npz 里预合成的 C_left/C_right——那是旧 vendor kHandOffset 的
快照，对 2026-08-25 换过插件常数之后录的数据是错的。当前常数下 C 是纯轴
约定旋转（平移≈0），Manus 腕原点应与标定腕原点重合。

所以本脚本直接读 add_wrist_pose.py 已经写进 parquet 的
observation.wrist_{left,right}_in_world，右乘 C，得到 manusWrist_world(t)，
再套 add_wrist_pose.py 同一条 stage->cam 投影链（T_headsetLocal->head_x 取自
solve_pico_to_head_extrinsics.py 的标定），跟 overlay_wrist_pose.py 一样用鱼眼
畸变模型 (cv2.fisheye.projectPoints) 画三轴。

前置条件：先跑过 add_wrist_pose.py（需要 observation.wrist_*_in_world 列）。

用法
----
    python3 overlay_manus_wrist.py --dataset-root ~/datasets/my_task --episode 0
    python3 overlay_manus_wrist.py --dataset-root ~/datasets/my_task --episode 0 --cam right
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import overlay_wrist_pose
from add_wrist_pose import load_headset_to_head_cams, pose_to_matrix
from overlay_wrist_pose import (
    draw_frame_axes,
    load_intrinsics,
    scale_K,
)

CALIB_DIR = Path(__file__).parent / "calib_data"

# draw_frame_axes 按 label 去 overlay_wrist_pose.LABEL_COLORS 里找颜色，"ML"/"MR"
# 不在原表里，运行时会 KeyError——这里直接往那张表里插两个新 label，而不是自己
# 另开一张影子表（否则改了也不会被 draw_frame_axes 用到）。
# 橙=Manus 左腕, 紫=Manus 右腕，跟 overlay_wrist_pose.py 的黄(L)/品红(R) 区分开。
overlay_wrist_pose.LABEL_COLORS["ML"] = (0, 165, 255)
overlay_wrist_pose.LABEL_COLORS["MR"] = (255, 0, 165)


def load_latest_grip_to_aim_C(
    grip_to_aim_dir: Path, controller_dir: Path
) -> dict[str, np.ndarray]:
    """现场合成 C = inv(T_wrist->ctrl) . T_grip->aim . kHandOffset_current。

    不用 npz 里预合成的 C_left/C_right：那是 calibrate_grip_to_aim.py 运行当时
    的 kHandOffset 合成的快照。2026-08-25 插件常数从 vendor nominal 换成了
    rig 标定值（kLeftHandOffset 注释），npz 里的 C 还带着 vendor 偏移的 ~20cm
    平移，对换过常数之后录的数据是错的。T_grip_to_aim_* 字段是 controller
    固件常数，跟 kHandOffset 无关，不过期，可以放心用。

    kHandOffset 取 record_cameras.DEFAULT_AIM_TO_WRIST——它是插件当前
    kLeft/RightHandOffset 的镜像拷贝。当前常数下 C 应当是纯旋转（轴约定
    重映射 C0，平移为零），即 Manus 腕原点与标定腕原点重合。
    """
    from add_wrist_pose import load_wrist_to_ctrl  # noqa: PLC0415
    from record_cameras import DEFAULT_AIM_TO_WRIST  # noqa: PLC0415

    files = sorted(grip_to_aim_dir.glob("grip_to_aim_*.npz"))
    if not files:
        sys.exit(
            f"ERROR: 找不到 grip_to_aim 标定文件（{grip_to_aim_dir}）。"
            "先跑 calibrate_grip_to_aim.py。"
        )
    npz = np.load(files[-1])
    print(f"  grip_to_aim 标定: {files[-1].name}（只用 T_grip_to_aim_*，C 现场合成）")

    out: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        T_ga = npz[f"T_grip_to_aim_{side}"].astype(np.float64)
        p_wc, R_wc = load_wrist_to_ctrl(controller_dir, side)
        T_wrist_ctrl = np.eye(4)
        T_wrist_ctrl[:3, :3] = R_wc
        T_wrist_ctrl[:3, 3] = p_wc
        spec = DEFAULT_AIM_TO_WRIST[side]
        kHandOffset = pose_to_matrix(
            np.asarray(spec["position"], dtype=np.float64),
            np.asarray(spec["quaternion"], dtype=np.float64),
        )
        C = np.linalg.inv(T_wrist_ctrl) @ T_ga @ kHandOffset
        t_norm = np.linalg.norm(C[:3, 3]) * 1000
        print(f"  [{side}] C 平移 {t_norm:.2f} mm（当前插件常数下应接近 0）")
        out[side] = C
    return out


def manus_wrist_in_cam(
    wrist_in_world_8: np.ndarray,
    head_pose_7: np.ndarray,
    T_headset_to_cam: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """单帧：wrist_in_world (8,) + head_pose (7,) -> manus 腕在 cam 系下的 pose8。

    跟 add_wrist_pose.py 的 batch_stage_to_cam + compose_batch 是同一套代数，
    这里逐帧算（overlay 走 LeRobotDataset 逐帧迭代，帧数不多，没必要再批量化）。
    """
    out = np.zeros(8, dtype=np.float32)

    ctrl_valid = wrist_in_world_8[7] > 0.5
    head_valid = np.linalg.norm(head_pose_7[3:7]) > 0.5
    if not (ctrl_valid and head_valid):
        return out

    T_wrist_world = pose_to_matrix(
        wrist_in_world_8[0:3].astype(np.float64), wrist_in_world_8[3:7].astype(np.float64)
    )
    T_manus_world = T_wrist_world @ C

    T_head_world = pose_to_matrix(
        head_pose_7[0:3].astype(np.float64), head_pose_7[3:7].astype(np.float64)
    )
    T_world_to_head = np.linalg.inv(T_head_world)
    T_world_to_cam = T_headset_to_cam @ T_world_to_head

    T_manus_cam = T_world_to_cam @ T_manus_world

    out[0:3] = T_manus_cam[:3, 3]
    out[3:7] = Rotation.from_matrix(T_manus_cam[:3, :3]).as_quat()
    out[7] = 1.0
    return out


def process_episode(
    repo_id: str,
    root: Path,
    episode_index: int,
    cam_sides: list[str],
    out_dir: Path,
    C: dict[str, np.ndarray],
    T_headset_to_head: dict[str, np.ndarray],
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=repo_id, root=str(root), episodes=[episode_index], download_videos=False
    )
    n = len(ds)
    if n == 0:
        print(f"  episode {episode_index}: 没有帧，跳过")
        return

    for cam in cam_sides:
        K, D, (calib_w, calib_h) = load_intrinsics(cam)
        img_key = f"observation.images.head_{cam}"

        out_path = out_dir / f"head_{cam}_manus_wrist_overlay_ep{episode_index:06d}.mp4"
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
                        f"不一致，已按比例缩放 K 再投影",
                        file=sys.stderr,
                    )
                    K = scale_K(K, (calib_w, calib_h), (w, h))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_path), fourcc, ds.fps, (w, h))

            wrist_left_world = item["observation.wrist_left_in_world"].numpy()
            wrist_right_world = item["observation.wrist_right_in_world"].numpy()
            head_pose = item["observation.head_pose"].numpy()
            T_headset_to_cam = T_headset_to_head[cam]

            pose_ml = manus_wrist_in_cam(wrist_left_world, head_pose, T_headset_to_cam, C["left"])
            pose_mr = manus_wrist_in_cam(wrist_right_world, head_pose, T_headset_to_cam, C["right"])
            drawn_left += int(pose_ml[7] > 0.5)
            drawn_right += int(pose_mr[7] > 0.5)

            draw_frame_axes(frame, pose_ml, K, D, "ML", thickness=1)
            draw_frame_axes(frame, pose_mr, K, D, "MR", thickness=1)

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
            f"(ML 有效 {drawn_left}, MR 有效 {drawn_right} — 只看 valid 位，"
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
        help="随便填，LeRobotDataset 只用它做缓存标识，不影响读取本地数据",
    )
    ap.add_argument("--episode", type=int, default=0, help="处理哪个 episode（默认 0）")
    ap.add_argument("--cam", choices=("left", "right", "both"), default="left")
    ap.add_argument(
        "--grip-to-aim-dir", type=Path, default=CALIB_DIR / "grip_to_aim",
        help="calibrate_grip_to_aim.py 输出目录（取最新一份 *.npz）",
    )
    ap.add_argument(
        "--pico-intrinsics", type=Path,
        default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz",
    )
    ap.add_argument(
        "--pico-to-head", type=Path,
        default=CALIB_DIR / "pico_to_head" / "extrinsics.npz",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="默认: <dataset-root>/wrist_overlay/"
    )
    args = ap.parse_args()

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: {args.dataset_root} 不是 LeRobot 数据集根目录（缺 meta/info.json）",
              file=sys.stderr)
        return 1

    import json

    info = json.load(open(info_path))
    needed = ["observation.wrist_left_in_world", "observation.wrist_right_in_world"]
    if not all(k in info["features"] for k in needed):
        print(
            "ERROR: 数据集里没有 observation.wrist_*_in_world 列，先跑 add_wrist_pose.py。",
            file=sys.stderr,
        )
        return 1

    print("加载标定:")
    C = load_latest_grip_to_aim_C(args.grip_to_aim_dir, CALIB_DIR / "controller")
    T_headset_to_head_left, T_headset_to_head_right = load_headset_to_head_cams(
        args.pico_intrinsics, args.pico_to_head
    )
    T_headset_to_head = {"left": T_headset_to_head_left, "right": T_headset_to_head_right}

    cam_sides = ["left", "right"] if args.cam == "both" else [args.cam]
    out_dir = args.out_dir or (args.dataset_root / "wrist_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n处理 episode {args.episode}，相机: {cam_sides}，输出到: {out_dir}\n")
    process_episode(
        args.repo_id, args.dataset_root, args.episode, cam_sides, out_dir, C, T_headset_to_head
    )

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
