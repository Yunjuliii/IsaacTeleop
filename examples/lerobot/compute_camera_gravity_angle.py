#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
算数据集第一帧里，head 相机光轴（相机系 +z，OpenCV 惯例）跟世界系重力方向
（world/stage 系的 y 轴，重力指向 -y）之间的夹角。

变换链跟 add_wrist_pose.py 一致：
    T_headsetLocal->head_x = T_picoCam->head_x . T_headsetLocal->picoCam
    T_headsetLocal->stage(t)  由 observation.head_pose 第一帧给出
                               （position + quaternion_xyzw，headsetLocal 在 stage 系下的 pose）

    T_head_x->stage(t) = T_headsetLocal->stage(t) . inv(T_headsetLocal->head_x)

相机光轴方向（世界系下）:
    axis_world = R_head_x->stage(t) @ [0, 0, 1]

跟重力方向 [0, -1, 0]（-y，因为 world 系是 y-up）算夹角。

用法
----
    python3 compute_camera_gravity_angle.py --dataset-root ~/datasets/my_task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from add_wrist_pose import CALIB_DIR, load_headset_to_head_cams, pose_to_matrix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="record_cameras.py 录的 LeRobot 数据集根目录")
    ap.add_argument("--pico-intrinsics", type=Path,
                    default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz")
    ap.add_argument("--pico-to-head", type=Path,
                    default=CALIB_DIR / "pico_to_head" / "extrinsics.npz")
    ap.add_argument("--frame-index", type=int, default=0,
                    help="用数据集里第几帧的 head_pose，默认第一帧")
    args = ap.parse_args()

    parquet_files = sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        print(f"ERROR: {args.dataset_root / 'data'} 下没找到 parquet 文件", file=sys.stderr)
        return 1

    table = pq.read_table(parquet_files[0])
    if args.frame_index >= table.num_rows:
        print(f"ERROR: frame-index {args.frame_index} 超出行数 {table.num_rows}", file=sys.stderr)
        return 1

    head_pose = np.array(table.column("observation.head_pose")[args.frame_index].as_py(),
                         dtype=np.float64)
    position = head_pose[0:3]
    quat_xyzw = head_pose[3:7]
    if np.linalg.norm(quat_xyzw) < 0.5:
        print(f"ERROR: 第 {args.frame_index} 帧 head_pose 无效（四元数全零）", file=sys.stderr)
        return 1

    T_headset_to_stage = pose_to_matrix(position, quat_xyzw)

    T_headset_to_head_left, T_headset_to_head_right = load_headset_to_head_cams(
        args.pico_intrinsics, args.pico_to_head)
    T_headset_to_head = {"left": T_headset_to_head_left, "right": T_headset_to_head_right}

    gravity_world = np.array([0.0, -1.0, 0.0])  # world/stage 是 y-up，重力指向 -y

    print(f"数据集: {args.dataset_root}")
    print(f"用帧: {parquet_files[0].name} 第 {args.frame_index} 行")
    print(f"head_pose: position={position}, quat_xyzw={quat_xyzw}\n")

    for cam_side, T_headset_to_cam in T_headset_to_head.items():
        T_cam_to_headset = np.linalg.inv(T_headset_to_cam)
        T_cam_to_stage = T_headset_to_stage @ T_cam_to_headset

        axis_cam = np.array([0.0, 0.0, 1.0])  # OpenCV 惯例，相机光轴 = +z
        axis_world = T_cam_to_stage[:3, :3] @ axis_cam
        axis_world /= np.linalg.norm(axis_world)

        cos_gravity = float(np.clip(np.dot(axis_world, gravity_world), -1.0, 1.0))
        angle_gravity = np.degrees(np.arccos(cos_gravity))

        cos_y = float(np.clip(np.dot(axis_world, [0.0, 1.0, 0.0]), -1.0, 1.0))
        angle_y = np.degrees(np.arccos(cos_y))

        print(f"[head_{cam_side}]")
        print(f"  光轴（世界系）      : {axis_world}")
        print(f"  与 +y 轴夹角        : {angle_y:.2f} 度")
        print(f"  与重力方向(-y)夹角  : {angle_gravity:.2f} 度")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
