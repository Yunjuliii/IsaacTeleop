#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
给 record_cameras.py 录的 LeRobot 数据集，按帧算出双手手腕的 pose，作为新列
写回数据集的 parquet ——世界系（stage）一份，head_left / head_right 两个鱼眼
相机坐标系下各一份。

变换链
------
    T_wrist->stage(t)       = T_ctrl->stage(t) . T_wrist->ctrl
                               (T_wrist->ctrl 来自 calib_data/controller/ 的
                               pivot + rotation 标定，跟 wrist_pose_viz_3d.py
                               用的是同一套公式；stage 就是 observation.controller_left/
                               right 本身所在的世界坐标系)

    T_headsetLocal->head_x  = T_picoCam->head_x . T_headsetLocal->picoCam
                               (T_picoCam->head_x 来自 solve_pico_to_head_extrinsics.py，
                               T_headsetLocal->picoCam 取自 Pico 标定包里的
                               position/quaternion_xyzw 字段取逆 —— 那两个字段本身是
                               T_picoCam->headsetLocal，不是 extrinsic_matrix 字段，
                               也不需要额外的 Rx180/Z flip)

    T_wrist->head_x(t)      = T_headsetLocal->head_x . inv(T_headsetLocal->stage(t))
                               . T_wrist->stage(t)

新增列（每行 8 个 float32：x,y,z,qx,qy,qz,qw,valid）：
    observation.wrist_left_in_world
    observation.wrist_right_in_world
    observation.wrist_left_in_head_left
    observation.wrist_left_in_head_right
    observation.wrist_right_in_head_left
    observation.wrist_right_in_head_right

世界系那两列 valid=0 只代表 controller.valid=0；相机系那四列 valid=0 还会
额外因为 head_pose 全零（头显数据本身无效）而置零。两种情况下 pose 都整行
清零，跟 CONTROLLER_POSE_NAMES 的约定一致，别把零值当成"手腕在原点"来用。

用法
----
    python3 add_wrist_pose.py --dataset-root ~/datasets/my_task
    python3 add_wrist_pose.py --dataset-root ~/datasets/my_task --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

CALIB_DIR = Path(__file__).parent / "calib_data"

WRIST_POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "valid"]


# --------------------------------------------------------------------------- #
# 标定加载
# --------------------------------------------------------------------------- #
def load_wrist_to_ctrl(
    controller_dir: Path, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """跟 wrist_pose_viz_3d.py::load_calib 一致：取最新一份 pivot + rotation 标定。"""
    pivot_files = sorted(controller_dir.glob(f"calib_pivot_{side}_*.npz"))
    rot_files = sorted(controller_dir.glob(f"calib_rotation_{side}_*.npz"))
    if not pivot_files:
        sys.exit(f"ERROR: 找不到 {side} 手 pivot 标定文件（{controller_dir}）")
    if not rot_files:
        sys.exit(f"ERROR: 找不到 {side} 手 rotation 标定文件（{controller_dir}）")
    p = np.load(pivot_files[-1])[f"p_wrist_ctrl_{side}"].astype(np.float64)
    R = np.load(rot_files[-1])[f"R_wrist_ctrl_{side}"].astype(np.float64)
    print(f"  [{side}] pivot    : {pivot_files[-1].name}")
    print(f"  [{side}] rotation : {rot_files[-1].name}")
    return p, R


def pose_to_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def load_headset_to_head_cams(
    pico_intrinsics: Path, pico_to_head: Path
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (T_headsetLocal->head_left, T_headsetLocal->head_right)，都是 4x4。"""
    pico = np.load(pico_intrinsics)
    # PICO legacy API 返回的 position/quaternion_xyzw 是 T_picoCam->headsetLocal：
    #   p_headset = R_lrot @ p_picoCam + l_pos
    # 不能直接当成 T_headsetLocal->picoCam（也不要用 extrinsic_matrix 字段）。
    l_pos = pico["position"].astype(np.float64)
    l_rot = pico["quaternion_xyzw"].astype(np.float64)
    T_picocam_to_headset = pose_to_matrix(l_pos, l_rot)
    T_headset_to_picocam = np.linalg.inv(T_picocam_to_headset)

    ext = np.load(pico_to_head)
    T_picocam_to_head_left = ext["T_pico_to_head_left"].astype(np.float64)
    T_picocam_to_head_right = ext["T_pico_to_head_right"].astype(np.float64)

    T_headset_to_head_left = T_picocam_to_head_left @ T_headset_to_picocam
    T_headset_to_head_right = T_picocam_to_head_right @ T_headset_to_picocam
    return T_headset_to_head_left, T_headset_to_head_right


# --------------------------------------------------------------------------- #
# 批量位姿代数（N 帧一起算，避免逐行 for 循环拖慢大数据集）
# --------------------------------------------------------------------------- #
def batch_wrist_to_stage(
    ctrl: np.ndarray, p_wrist_ctrl: np.ndarray, R_wrist_ctrl: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ctrl: (N, 8) = [x,y,z,qx,qy,qz,qw,valid]。返回 (R (N,3,3), t (N,3), valid (N,))。"""
    pc = ctrl[:, 0:3].astype(np.float64)
    qc = ctrl[:, 3:7].astype(np.float64)
    valid = ctrl[:, 7] > 0.5
    # 无效帧（手柄追踪丢失）整行是零，包含零范数四元数——SciPy 对此直接抛异常
    # 而不是返回垃圾值，跟 batch_stage_to_cam 里 head_pose 的处理是同一个坑，
    # 同样先替换成单位四元数再算，结果本来就会被下面的 valid 过滤掉。
    qc_safe = qc.copy()
    qc_safe[~valid] = [0.0, 0.0, 0.0, 1.0]
    Rc = Rotation.from_quat(
        qc_safe
    ).as_matrix()  # (N, 3, 3)，无效帧是占位值，valid 会滤掉
    Rw = Rc @ R_wrist_ctrl  # 手腕姿态，stage 系
    pw = np.matmul(Rc, p_wrist_ctrl) + pc  # 手腕原点，stage 系
    return Rw, pw, valid


def batch_stage_to_cam(
    head_pose: np.ndarray, T_headset_to_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """head_pose: (N, 7) = [x,y,z,qx,qy,qz,qw]。返回 T_stage->cam 逐帧的 (R, t, valid)。"""
    ph = head_pose[:, 0:3].astype(np.float64)
    qh = head_pose[:, 3:7].astype(np.float64)
    # 头显 pose 没有单独的 valid 标志位，全零四元数（范数接近 0）是唯一能识别出的
    # "无效帧"信号 —— record_cameras.py 里 head 无效时就整行写零。
    valid = np.linalg.norm(qh, axis=1) > 0.5

    # 无效帧的四元数是全零（范数为 0），SciPy 对零范数四元数会直接抛异常而不是
    # 返回垃圾值，所以先把这些行替换成单位四元数再算，结果本来就会被 valid 滤掉。
    qh_safe = qh.copy()
    qh_safe[~valid] = [0.0, 0.0, 0.0, 1.0]
    Rh = Rotation.from_quat(
        qh_safe
    ).as_matrix()  # (N, 3, 3)，无效帧这里是占位值，valid 会滤掉
    Rh_inv = np.transpose(Rh, (0, 2, 1))  # R^T
    ph_inv = -np.einsum("nij,nj->ni", Rh_inv, ph)  # -R^T . p

    R_hc = T_headset_to_cam[:3, :3]
    t_hc = T_headset_to_cam[:3, 3]

    R_result = R_hc @ Rh_inv  # (N, 3, 3)
    t_result = np.einsum("ij,nj->ni", R_hc, ph_inv) + t_hc
    return R_result, t_result, valid


def compose_batch(
    R_a: np.ndarray, t_a: np.ndarray, R_b: np.ndarray, t_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T_result = T_a . T_b，逐帧。"""
    R = R_a @ R_b
    t = np.einsum("nij,nj->ni", R_a, t_b) + t_a
    return R, t


def pack_pose_valid(R: np.ndarray, t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(R (N,3,3), t (N,3), valid (N,)) -> (N, 8) float32 [x,y,z,qx,qy,qz,qw,valid]，
    无效帧整行清零（沿用 CONTROLLER_POSE_NAMES 的约定）。"""
    n = R.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    if valid.any():
        q = Rotation.from_matrix(R[valid]).as_quat()
        out[valid, 0:3] = t[valid]
        out[valid, 3:7] = q
        out[valid, 7] = 1.0
    return out


# --------------------------------------------------------------------------- #
# parquet 读写
# --------------------------------------------------------------------------- #
def add_fixed_size_list_column(
    table: pa.Table, name: str, values: np.ndarray
) -> pa.Table:
    """values: (N, 8) float32 -> fixed_size_list<float>[8] 列，追加到 table。"""
    flat = pa.array(values.reshape(-1), type=pa.float32())
    col = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    if name in table.column_names:
        table = table.drop([name])
    return table.append_column(name, col)


def process_file(
    path: Path,
    wrist_to_ctrl: dict[str, tuple[np.ndarray, np.ndarray]],
    T_headset_to_head: dict[str, np.ndarray],
    dry_run: bool,
) -> dict[str, int]:
    table = pq.read_table(path)
    n = table.num_rows

    head_pose = np.stack(
        table.column("observation.head_pose").to_numpy(zero_copy_only=False)
    )
    ctrl_left = np.stack(
        table.column("observation.controller_left").to_numpy(zero_copy_only=False)
    )
    ctrl_right = np.stack(
        table.column("observation.controller_right").to_numpy(zero_copy_only=False)
    )

    counts: dict[str, int] = {}
    for wrist_side, ctrl in (("left", ctrl_left), ("right", ctrl_right)):
        p_wc, R_wc = wrist_to_ctrl[wrist_side]
        # 世界（stage）系下的手腕 pose 只算这一次，相机系的四列都是在它上面接着乘。
        Rw, pw, ctrl_valid = batch_wrist_to_stage(ctrl, p_wc, R_wc)

        world_col = f"observation.wrist_{wrist_side}_in_world"
        world_packed = pack_pose_valid(Rw, pw, ctrl_valid)
        counts[world_col] = int(ctrl_valid.sum())
        if not dry_run:
            table = add_fixed_size_list_column(table, world_col, world_packed)

        for cam_side in ("left", "right"):
            R_sc, t_sc, head_valid = batch_stage_to_cam(
                head_pose, T_headset_to_head[cam_side]
            )
            R_res, t_res = compose_batch(R_sc, t_sc, Rw, pw)
            valid = ctrl_valid & head_valid

            cam_col = f"observation.wrist_{wrist_side}_in_head_{cam_side}"
            cam_packed = pack_pose_valid(R_res, t_res, valid)
            counts[cam_col] = int(valid.sum())
            if not dry_run:
                table = add_fixed_size_list_column(table, cam_col, cam_packed)

    if not dry_run:
        pq.write_table(table, path)

    counts["_rows"] = n
    return counts


def update_info_json(dataset_root: Path, dry_run: bool) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for wrist_side in ("left", "right"):
        names = [f"observation.wrist_{wrist_side}_in_world"] + [
            f"observation.wrist_{wrist_side}_in_head_{cam_side}"
            for cam_side in ("left", "right")
        ]
        for name in names:
            info["features"][name] = {
                "dtype": "float32",
                "shape": [8],
                "names": WRIST_POSE_NAMES,
            }

    if dry_run:
        print(f"  (dry-run，不写 {info_path})")
        return
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  已更新: {info_path}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="record_cameras.py 录的 LeRobot 数据集根目录",
    )
    ap.add_argument("--controller-dir", type=Path, default=CALIB_DIR / "controller")
    ap.add_argument(
        "--pico-intrinsics",
        type=Path,
        default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz",
    )
    ap.add_argument(
        "--pico-to-head",
        type=Path,
        default=CALIB_DIR / "pico_to_head" / "extrinsics.npz",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计每列有多少有效帧，不实际写回 parquet / info.json",
    )
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(
            f"ERROR: {args.dataset_root} 看起来不是 LeRobot 数据集根目录（缺 meta/info.json）",
            file=sys.stderr,
        )
        return 1

    print("加载标定:")
    wrist_to_ctrl = {
        side: load_wrist_to_ctrl(args.controller_dir, side)
        for side in ("left", "right")
    }
    T_headset_to_head_left, T_headset_to_head_right = load_headset_to_head_cams(
        args.pico_intrinsics, args.pico_to_head
    )
    T_headset_to_head = {
        "left": T_headset_to_head_left,
        "right": T_headset_to_head_right,
    }

    parquet_files = sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        print(
            f"ERROR: {args.dataset_root / 'data'} 下没找到 parquet 文件",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n共 {len(parquet_files)} 个 parquet 文件{'（dry-run，不会实际写回）' if args.dry_run else ''}"
    )

    totals: dict[str, int] = {}
    total_rows = 0
    for path in parquet_files:
        counts = process_file(path, wrist_to_ctrl, T_headset_to_head, args.dry_run)
        rows = counts.pop("_rows")
        total_rows += rows
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        print(f"  {path.relative_to(args.dataset_root)}: {rows} 行")

    print(f"\n共 {total_rows} 行，各列有效帧数:")
    for name, cnt in totals.items():
        pct = 100.0 * cnt / total_rows if total_rows else 0.0
        print(f"  {name}: {cnt}/{total_rows} ({pct:.1f}%)")

    print("\n更新 meta/info.json:")
    update_info_json(args.dataset_root, args.dry_run)

    if args.dry_run:
        print("\ndry-run 完成，没有修改任何文件。")
    else:
        print(
            "\n完成。注意 meta/stats.json 和 meta/episodes/ 里的统计信息没有更新，"
            "包含新列的 min/max/mean 统计需要你自己用 LeRobot 的统计工具重新算。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
