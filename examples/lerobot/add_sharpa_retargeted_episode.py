#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
把一份外部算好的 Manus 重定向结果（宽表 parquet，每行一帧，列名形如
{side}_thumb_CMC_FE / {side}_solved_base_x 等）合并进 LeRobot 数据集里
对应 episode 的 parquet，新增四列：

    observation.sharpa_joints_left / right    （22 个关节角 + valid，共 23 float32）
    observation.sharpa_base_left / right      （x,y,z,qx,qy,qz,qw,valid，共 8 float32）

关节角直接原样拷贝（已经是 Sharpa Wave 22 DOF 的弧度值，列名顺序与
add_sharpa_joints.py 文档里的 MJCF 顺序一致，无需重排）。

底座位姿要做一次坐标转换：源文件里的 {side}_solved_base_* 是 IK 联合优化出的
Sharpa 根身体位姿，定义在 stage 系（世界系，跟 observation.controller_*、
observation.head_pose 同一个系）。这里把它转换到头显左鱼眼相机系
（head_left），链条跟 add_wrist_pose.py / add_sharpa_joints.py 的
sharpa_base_*_in_head_* 完全一样，只是只算 head_left 这一路、列名简化成
sharpa_base_{side}（不带 _in_head_left 后缀）：

    T_headsetLocal->head_left = T_picoCam->head_left . T_headsetLocal->picoCam
        （标定见 calib_data/pico_camera 与 calib_data/pico_to_head）
    T_base->head_left(t) = T_headsetLocal->head_left . inv(T_headsetLocal->stage(t))
                            . T_base->stage(t)

有效性：valid = 该帧手套数据非全零（沿用 observation.hand_{side} 的判定）
∧ head_pose 有效 ∧ 同侧 controller 有效 —— 跟 add_sharpa_joints.py 里
sharpa_base_*_in_head_* 的门控一致（controller 无效时插件沿用缓存锚点，
世界系位姿会 stale，见该脚本文件头注释）。

这个脚本只处理单个 episode（源文件本来就是单 episode 导出），不是全量
数据集批处理；写完这一集之后，dataset/meta/info.json 里新增的四个特征
描述只对这一集的 parquet 成立，其余 episode 的 parquet 还没有这几列 ——
脚本会打印一条提醒，别当成全量已完成。

用法
----
    python3 add_sharpa_retargeted_episode.py \\
        --dataset-root /home/nvidia/IsaacTeleop/dataset \\
        --retargeted-parquet /home/nvidia/IsaacTeleop/episode_000000_sharpa_joints.parquet \\
        --episode-index 0

    # 干跑，只核对能不能对齐、算完但不写回：
    python3 add_sharpa_retargeted_episode.py ... --dry-run
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

# 头部相机标定加载、批量位姿代数、parquet 列写入，都跟 add_wrist_pose.py /
# add_sharpa_joints.py 同一套链条，直接复用，不重新发明。
from add_wrist_pose import (
    CALIB_DIR,
    add_fixed_size_list_column,
    batch_stage_to_cam,
    compose_batch,
    load_headset_to_head_cams,
)

SIDES = ("left", "right")

# 源宽表列名顺序 = Sharpa Wave MJCF 关节顺序（跟 add_sharpa_joints.py 文档
# 一致），直接决定 sharpa_joints_{side} 里 22 个角度的排布。
JOINT_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)

POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "valid"]


def find_episode_file(dataset_root: Path, episode_index: int) -> Path:
    """在 meta/episodes/chunk-*/file-*.parquet 里查 episode_index 对应的
    data/chunk_index、data/file_index，拼出该 episode 的数据 parquet 路径。"""
    for meta_path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(meta_path, columns=[
            "episode_index", "data/chunk_index", "data/file_index"])
        eps = table.column("episode_index").to_numpy()
        hit = np.where(eps == episode_index)[0]
        if len(hit) == 0:
            continue
        row = hit[0]
        chunk_idx = table.column("data/chunk_index")[row].as_py()
        file_idx = table.column("data/file_index")[row].as_py()
        return (dataset_root / "data" / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.parquet")
    sys.exit(f"ERROR: 在 {dataset_root / 'meta' / 'episodes'} 里没找到 "
              f"episode_index={episode_index}")


def load_retargeted(path: Path) -> dict:
    """宽表源 parquet -> {frame_index, side: {joints (N,22), base_t (N,3), base_q_xyzw (N,4)}}。"""
    df = pq.read_table(path).to_pandas()
    if "frame_index" not in df.columns:
        sys.exit(f"ERROR: {path} 缺 frame_index 列，没法跟数据集对齐")
    out: dict = {"frame_index": df["frame_index"].to_numpy(np.int64)}
    for side in SIDES:
        joint_cols = [f"{side}_{suf}" for suf in JOINT_SUFFIXES]
        missing = [c for c in joint_cols if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: {path} 缺列 {missing}")
        base_cols = [f"{side}_solved_base_{a}" for a in
                     ("x", "y", "z", "qx", "qy", "qz", "qw")]
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: {path} 缺列 {missing}")
        joints = df[joint_cols].to_numpy(np.float64)
        base = df[base_cols].to_numpy(np.float64)
        out[side] = {
            "joints": joints,
            "base_t": base[:, 0:3],
            "base_q_xyzw": base[:, 3:7],
        }
    return out


def align_to_target(retargeted: dict, target_frame_index: np.ndarray) -> dict:
    """把源数据按 target 的 frame_index 顺序重排；源缺的帧算不存在（下游按
    valid=0 处理）。要求源覆盖 target 的每一帧，否则直接报错——静默补零
    会把"没算"和"算出来是零"混为一谈。"""
    src_idx = retargeted["frame_index"]
    pos = {int(fi): i for i, fi in enumerate(src_idx)}
    missing = [int(fi) for fi in target_frame_index if int(fi) not in pos]
    if missing:
        sys.exit(f"ERROR: 源文件缺目标 episode 的 frame_index {missing[:10]}"
                  f"{'...' if len(missing) > 10 else ''}（共缺 {len(missing)} 帧）")
    order = np.array([pos[int(fi)] for fi in target_frame_index])
    aligned: dict = {}
    for side in SIDES:
        aligned[side] = {k: v[order] for k, v in retargeted[side].items()}
    return aligned


def build_sharpa_joints(joints: np.ndarray, hand_valid: np.ndarray) -> np.ndarray:
    """(N,22) 弧度 + (N,) 手套有效位 -> (N,23) float32，跟 add_sharpa_joints.py
    的 sharpa_joints_{side} 同语义：valid=0 的帧角度整行清零。"""
    n = joints.shape[0]
    out = np.zeros((n, 23), dtype=np.float32)
    out[hand_valid, :22] = joints[hand_valid].astype(np.float32)
    out[hand_valid, 22] = 1.0
    return out


def build_sharpa_base_in_cam(
    base_t: np.ndarray, base_q_xyzw: np.ndarray,
    head_pose: np.ndarray, ctrl: np.ndarray,
    T_headset_to_cam: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """stage 系底座 (N,3)/(N,4 xyzw) -> 目标相机系下的 (N,8) float32
    [x,y,z,qx,qy,qz,qw,valid]。

    valid = head_pose 有效 ∧ controller 有效（跟 add_sharpa_joints.py 里
    sharpa_base_*_in_head_* 的门控一致，见该脚本文件头关于 stale 锚点的
    注释）。源数据本身没有携带"这帧算没算"的标记，姿态转换对所有行都算，
    只在最后按这两个条件把 valid 位清零、位姿清零。
    """
    R_base = Rotation.from_quat(base_q_xyzw).as_matrix()
    t_base = base_t

    R_sc, t_sc, head_valid = batch_stage_to_cam(head_pose, T_headset_to_cam)
    R_res, t_res = compose_batch(R_sc, t_sc, R_base, t_base)

    ctrl_valid = ctrl[:, 7] > 0.5
    valid = head_valid & ctrl_valid

    n = R_res.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    if valid.any():
        q = Rotation.from_matrix(R_res[valid]).as_quat()
        out[valid, 0:3] = t_res[valid]
        out[valid, 3:7] = q
        out[valid, 7] = 1.0
    stats = {"head_valid": int(head_valid.sum()), "ctrl_valid": int(ctrl_valid.sum()),
              "valid": int(valid.sum())}
    return out, stats


def update_info_json(dataset_root: Path, episode_index: int, dry_run: bool) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    for side in SIDES:
        info["features"][f"observation.sharpa_joints_{side}"] = {
            "dtype": "float32",
            "shape": [23],
            "names": [f"{side}_{suf}" for suf in JOINT_SUFFIXES] + ["valid"],
        }
        info["features"][f"observation.sharpa_base_{side}"] = {
            "dtype": "float32",
            "shape": [8],
            "names": POSE_NAMES,
        }
    if dry_run:
        print(f"  (dry-run，不写 {info_path})")
        return
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  已更新: {info_path}")
    print(f"  注意：这四个特征目前只有 episode_index={episode_index} 的 parquet "
          f"里有实际数据，其余 episode 还没跑这个脚本——特征声明是全局的，"
          f"但数据是这一集独有的，消费时自己注意。")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path,
                     default=Path("/home/nvidia/IsaacTeleop/dataset"),
                     help="LeRobot 数据集根目录")
    ap.add_argument("--retargeted-parquet", type=Path,
                     default=Path("/home/nvidia/IsaacTeleop/episode_000000_sharpa_joints.parquet"),
                     help="外部算好的 Manus 重定向宽表 parquet（单 episode）")
    ap.add_argument("--episode-index", type=int, default=0,
                     help="这份重定向数据对应数据集里的哪个 episode_index")
    ap.add_argument("--cam-side", choices=("left", "right"), default="left",
                     help="sharpa_base_{left|right} 转换到哪个头显鱼眼相机系"
                          "（head_left 或 head_right），默认 head_left")
    ap.add_argument("--pico-intrinsics", type=Path,
                     default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz")
    ap.add_argument("--pico-to-head", type=Path,
                     default=CALIB_DIR / "pico_to_head" / "extrinsics.npz")
    ap.add_argument("--dry-run", action="store_true",
                     help="只对齐、计算、打印统计，不写回 parquet / info.json")
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: {args.dataset_root} 看起来不是 LeRobot 数据集根目录"
              f"（缺 meta/info.json）", file=sys.stderr)
        return 1
    if not args.retargeted_parquet.exists():
        print(f"ERROR: 找不到 {args.retargeted_parquet}", file=sys.stderr)
        return 1

    target_path = find_episode_file(args.dataset_root, args.episode_index)
    print(f"episode_index={args.episode_index} -> {target_path.relative_to(args.dataset_root)}")

    print("加载标定:")
    hl, hr = load_headset_to_head_cams(args.pico_intrinsics, args.pico_to_head)
    T_headset_to_cam = {"left": hl, "right": hr}[args.cam_side]
    print(f"  头显 -> head_{args.cam_side} 外参已加载（{args.pico_to_head}）")

    print(f"\n读取重定向源文件: {args.retargeted_parquet}")
    retargeted = load_retargeted(args.retargeted_parquet)
    print(f"  {len(retargeted['frame_index'])} 帧")

    table = pq.read_table(target_path)
    n = table.num_rows
    target_frame_index = table.column("frame_index").to_numpy()
    aligned = align_to_target(retargeted, target_frame_index)
    print(f"  已按目标 episode 的 {n} 帧 frame_index 对齐")

    head_pose = np.stack(
        table.column("observation.head_pose").to_numpy(zero_copy_only=False)
    ).astype(np.float64)

    for side in SIDES:
        rows = np.stack(
            table.column(f"observation.hand_{side}").to_numpy(zero_copy_only=False)
        ).astype(np.float32)
        hand_valid = np.any(rows != 0.0, axis=1)

        joints_col = build_sharpa_joints(aligned[side]["joints"], hand_valid)
        print(f"  [{side}] sharpa_joints: {int(hand_valid.sum())}/{n} 有效")

        ctrl = np.stack(
            table.column(f"observation.controller_{side}").to_numpy(zero_copy_only=False)
        ).astype(np.float64)
        base_col, stats = build_sharpa_base_in_cam(
            aligned[side]["base_t"], aligned[side]["base_q_xyzw"],
            head_pose, ctrl, T_headset_to_cam)
        print(f"  [{side}] sharpa_base (head_{args.cam_side} 系): "
              f"head_valid={stats['head_valid']}/{n}, "
              f"ctrl_valid={stats['ctrl_valid']}/{n}, "
              f"最终 valid={stats['valid']}/{n}")

        if not args.dry_run:
            table = add_fixed_size_list_column(
                table, f"observation.sharpa_joints_{side}", joints_col)
            table = add_fixed_size_list_column(
                table, f"observation.sharpa_base_{side}", base_col)

    if args.dry_run:
        print("\ndry-run 完成，没有修改任何文件。")
        return 0

    pq.write_table(table, target_path)
    print(f"\n已写回: {target_path}")

    print("\n更新 meta/info.json:")
    update_info_json(args.dataset_root, args.episode_index, dry_run=False)

    print("\n完成。meta/stats.json 和 meta/episodes/ 的统计信息没有更新，"
          "如果下游需要新列的 min/max/mean 统计，自己用 LeRobot 的统计工具重新算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
