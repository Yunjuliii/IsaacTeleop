#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
给 record_cameras.py 录的 LeRobot 数据集，把 observation.hand_{left|right}
（Manus 手套的 25/24 关节位姿）逐帧重定向成 Sharpa Wave 手的 22 个关节角，
作为新列写回数据集的 parquet。

IK 链：数据集行 -> MANO 21 关键点 -> Pinocchio FreeFlyer + Pink 差分 IK
-> 22 个手指关节角（弧度）。直接驱动 robotic_grounding 的
SharpaHandKinematics（不经 SharpaHandRetargeter 封装），并做两处关键改动：

  1. 所有 frame task 的姿态 cost 置零，纯位置 IK。库的姿态目标契约是
     MANO 约定（腕的 120° 修正常数按 MANO 标定，指尖姿态目标甚至零修正
     直接用输入四元数），而本数据集的关节姿态是 Manus 骨架约定，二者差
     一个 ~155° 的常数。错约定的姿态目标会把优化器拉进一个"底座平移
     ~24cm、手身扭转 ~73°"的局部极小：指尖残差好看（2-5cm）但指根/掌部
     错 8-11cm，手身整体摆歪。位置目标不涉及任何姿态约定，天然免疫。
  2. FreeFlyer 底座种子逐帧用 Kabsch 给出：把模型中立形状的 11 个任务
     位点刚体对齐到该帧的人手关键点（纯位置、零拟合常数），从正确的
     姿态盆地起步。这也是必须绕过 SharpaHandRetargeter 的原因——它会用
     输入腕四元数（Manus 约定，即错误盆地）强行覆盖种子。

  3. 任务权重换成"贴合抓握"设计（细节见 SharpaSolver.__init__ 的注释）：
     每指梯形位置路标（指腹段最强、指尖最弱）、修正库表 MP 位点错配一节
     的 bug、拇指改为对齐指腹段而非硬钉指尖。

  实测（dataset_stage ep1 抽样，1->2->3 逐步）：底座-腕距离 24cm -> 0.5cm；
  四指中/远节段方向偏差 11°/12° -> 5°/5°，指根残差 1.3 -> 0.5cm，拇-食
  口径误差 +0.6 -> -0.3cm；拇指远节方向偏差 48° -> 3°、IP 从死勾 84° 回到
  跟随真实弯曲；代价是指尖残差 0.0 -> 0.3cm（力量抓握无感）。

坐标系
------
数据集两种 hand_frame 都支持，按 info.json 里该列的 shape 自动识别：

  168 = 24 关节 x 7（local，腕局部系，wrist 被去掉）
      wrist 在自身坐标系下是恒等位姿，重建 26 关节输入时补回
      [0,0,0, 0,0,0,1] 即可。
  175 = 25 关节 x 7（stage，世界系，含 wrist）

两种都能直接喂：IK 是纯位置任务 + 每帧 Kabsch 定种子，解对输入的刚体
变换等变——角度只取决于关节相对彼此的几何，对全局坐标系的选取不敏感。
local 帧还有个额外的好处：手形信号来自手套本身，跟手柄（wrist 锚）追踪
丢不丢完全无关，所以这里不需要再按 controller_valid 过滤。

新增列（每行 23 个 float32：22 个关节角 + valid）：
    observation.sharpa_joints_left
    observation.sharpa_joints_right

同时导出 IK 联合优化出的 FreeFlyer 底座位姿（"优化后的 Sharpa 腕"，即
MJCF 根身体 {side}_hand_C_MC 的位姿。IK 为了让指尖贴合人手指尖会把底座
相对腕平移/旋转，这个量逐帧变化，回放时用它替代"底座=人腕"的假设；
修好姿态目标后它离人腕只有 ~0.5-2cm，剩余偏移就是两手的真实形状差）。
底座与输入同系：local 帧数据集下是 Manus 腕系的 ΔT，stage 帧数据集下
直接就是世界（stage）系位姿，回放挂载无需再组合任何链条：

新增列（每行 8 个 float32：x,y,z, qx,qy,qz,qw, valid）：
    observation.sharpa_base_left
    observation.sharpa_base_right

local 帧的回放合成（世界系）：
    T_world_sharpaBase(t) = T_world_manusWrist(t) · sharpa_base(t)
其中 T_world_manusWrist 需要经 controller 链条锚定（aim·kHandOffset，
或 wrist_in_world·C），见 calibrate_grip_to_aim.py。

stage 帧数据集还会额外写出底座在两个头部鱼眼相机系下的位姿（标定加载与
位姿代数复用 add_wrist_pose.py，链条也一致）：

    T_cam_base(t) = T_headsetLocal->head_x · inv(T_headsetLocal->stage(t))
                    · sharpa_base(t)

新增列（每行 8 个 float32，仅 stage 帧数据集）：
    observation.sharpa_base_{left|right}_in_head_{left|right}

相机系列的 valid = 手有效 ∧ head_pose 有效 ∧ 同侧 controller 有效。最后
一项是 stage 帧特有的：controller 无效时插件沿用缓存锚点，该帧手关节
（连同底座）冻结在旧位置，看起来正常实际是 stale（见 record_cameras.py
CONTROLLER_POSE_NAMES 的注释），世界系的 sharpa_base 列本身不做这个
门控（跟 hand_* 列同语义），消费时自行按 controller_valid 过滤。
local 帧数据集跳过这四列：local 的底座是腕系 ΔT，换相机系需要 controller
锚定链，那是回放端的事。

关节顺序与 MJCF 一致（names 写进 info.json）：
    {side}_thumb_CMC_FE, {side}_thumb_CMC_AA, {side}_thumb_MCP_FE,
    {side}_thumb_MCP_AA, {side}_thumb_IP,
    {side}_{index|middle|ring}_{MCP_FE, MCP_AA, PIP, DIP},
    {side}_pinky_{CMC, MCP_FE, MCP_AA, PIP, DIP}

valid=0 表示该帧手套数据整行为零（手不在/未追踪，record_cameras.py 的
约定），此时 22 个角度也整行清零——跟 CONTROLLER_POSE_NAMES 的约定一致，
别把零值当成"手是张开的"来用。

依赖
----
需要 isaacteleop[grounding] 的 IK 依赖（pin, pin-pink, daqp,
loop-rate-limiters）和带 robotic_grounding 的 isaacteleop wheel：

    pip install -r src/core/python/requirements-grounding.txt

用法
----
    python3 add_sharpa_joints.py --dataset-root ~/datasets/my_task
    python3 add_sharpa_joints.py --dataset-root ~/datasets/my_task --dry-run

    # 下游要消费指尖空间位置（而非关节角）时，可把 IK 权重聚焦到任务手指，
    # 换更小的指尖残差。关节角当学习信号用时别开——聚焦会带来限位削顶和
    # 站立偏置，时间信号质量反而变差（详见 FOCUS_TIP_COST 处的注释）：
    python3 add_sharpa_joints.py --dataset-root ~/datasets/my_task \
        --focus-fingers thumb,index,middle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# 头部相机系的标定加载和批量位姿代数与 add_wrist_pose.py 完全同链，直接复用。
from add_wrist_pose import (
    CALIB_DIR,
    batch_stage_to_cam,
    compose_batch,
    load_headset_to_head_cams,
    pack_pose_valid,
)

# IK 内核直接来自 robotic_grounding（原因见文件头：需要控制 FreeFlyer 种子，
# SharpaHandRetargeter 封装会用输入腕四元数覆盖它）。
import pinocchio as pin
from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics

SIDES = ("left", "right")

# 数据集 hand 列的两种布局；见 record_cameras.py 的 HAND_FRAME_{STAGE,LOCAL}。
# local 帧的 24 个关节按 OpenXR 顺序对应索引 2..25（wrist=1 被去掉），
# stage 帧的 25 个对应 1..25。两者都跳过 PALM(0)。
DIM_LOCAL = 24 * 7   # 168
DIM_STAGE = 25 * 7   # 175

# 数据集行 -> MANO 21 关键点（wrist, thumb1-4, index1-4, middle1-4, ring1-4,
# pinky1-4）的行号。跳过非拇指手指的 metacarpal，同 SharpaHandRetargeter 的
# _OPENXR_TO_MANO_INDICES（OpenXR 索引 [1,2..5, 7..10, 12..15, 17..20, 22..25]）。
# stage 行号 = OpenXR 索引 - 1；local 行号 = OpenXR 索引 - 2（wrist 恒等，缺席）。
_MANO_ROWS_STAGE = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14,
                    16, 17, 18, 19, 21, 22, 23, 24]
_MANO_ROWS_LOCAL = [r - 1 for r in _MANO_ROWS_STAGE[1:]]   # 不含 wrist


def resolve_mjcf(side: str) -> str:
    """Sharpa Wave MJCF 随 robotic_grounding wheel 分发；_nomesh 变体不引用
    24MB 的 STL 网格，Pinocchio 加载更快，IK 结果与带网格版一致。"""
    from importlib.resources import files
    return str(files("robotic_grounding") / "assets" / "xmls" / "sharpawave"
               / f"{side}_sharpawave_nomesh.xml")


def row_to_mano(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """数据集一行 (168,)/(175,) -> MANO 21 关键点 (positions (21,3), quats wxyz (21,4))。

    local 帧的 wrist 恒等位姿在此补回。姿态只用作 Pink 目标的占位（本脚本
    把所有姿态 cost 置零，见文件头），位置才是 IK 的全部信号。
    """
    joints = row.reshape(-1, 7).astype(np.float64)
    if joints.shape[0] == 25:            # stage：wrist 在第 0 行
        p = joints[_MANO_ROWS_STAGE, 0:3]
        q_xyzw = joints[_MANO_ROWS_STAGE, 3:7].copy()
    elif joints.shape[0] == 24:          # local：wrist 恒等，已被去掉
        p = np.zeros((21, 3))
        q_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (21, 1))
        p[1:] = joints[_MANO_ROWS_LOCAL, 0:3]
        q_xyzw[1:] = joints[_MANO_ROWS_LOCAL, 3:7]
    else:
        raise ValueError(f"hand 列每行应为 {DIM_LOCAL} 或 {DIM_STAGE} 个 float，"
                         f"得到 {row.shape[0]}")
    # float32 位姿复合会让四元数范数漂移，Pinocchio 要求单位范数。
    q_xyzw /= np.maximum(np.linalg.norm(q_xyzw, axis=1, keepdims=True), 1e-8)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    return p, q_wxyz


FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# --focus-fingers 的权重方案。注意：下面的实测数字量于姿态目标修复之前
# （当时默认指尖残差 ~5.9cm）；纯位置 IK 后默认残差已是 ~0cm，这个开关
# 大概率不再需要，留着只为兼容。
# 实测（旧 ep0 全 177 帧）这是个单调取舍：
#   boost 1.0->2.0 把重点指尖残差 5.9cm 压到 3.6cm，但 middle_PIP 与原始
#   弯曲的相关性 0.96->0.83、32% 的帧削顶在 100° 限位、张开手时还留着
#   ~50° 的站立偏置——空间贴合的收益全部由时间信号质量买单。
# 所以：关节角要拿去当学习信号的话【别用这个开关】，默认权重就是最优；
# 只有下游真的消费指尖空间位置时才值得开。非重点手指降权后角度基本不可用。
FOCUS_TIP_COST = 2.0      # 重点手指指尖（原 1.0，pinky 原 0.5）
DEMOTE_TIP_COST = 0.05    # 非重点手指指尖：降权但不清零，保留大致跟随
DEMOTE_MP_COST = 0.01     # 非重点手指 MP 位点（原 0.1）


class SharpaSolver:
    """一侧手的离线求解器：直接驱动 SharpaHandKinematics，逐帧喂行、收角度。

    纯位置 IK + 每帧 Kabsch 定 FreeFlyer 种子（原因见文件头）。手指角
    warm-start 跨帧延续，底座种子每帧重算。"""

    def __init__(self, side: str, max_iter: int, scale: float,
                 focus: tuple[str, ...] | None = None):
        self.side = side
        self._scale = scale
        self._kin = SharpaHandKinematics(
            side=side,
            robot_asset_path=resolve_mjcf(side),
            source_model="mano",
            use_relative_frames=False,
            max_iter=max_iter,
        )
        # 贴合抓握权重，替换库默认的"指尖 1.0 独大"方案（实测对比见文件头）：
        #   - 每指铺"梯形"位置路标：指腹段两端（MP@PIP、DP@DIP）最强、指根
        #     居中、指尖最弱——力量抓握传力靠指腹条带的方向和贴合深度，
        #     不靠指尖端点；
        #   - 修正库表 SHARPA_TO_MANO_MAPPING 的错位配对：MP 位点物理在
        #     PIP（离根 ~14.3cm），库表却配到 f1/指根（~10cm），其注释行
        #     显示本意是 f2——之前 ~4.4cm 的"掌部残差"就是这个错配；
        #   - 拇指：Sharpa 拇指比人手长 ~27%（掌骨段 6.5 vs 4.5cm），指尖
        #     强约束会把多余长度折进 IP（死勾 ~84°、远节方向偏 ~50°）；
        #     改为强约束指腹段（thumb2/thumb3）、弱化指尖后，多余长度被
        #     吸收进手掌内不可见的掌骨段，指尖只沿轴多探出 ~3mm。
        m = {f"{side}_hand_C_MC": ("wrist", 0.2, 0.0),
             f"{side}_thumb_CMC_VL_site": ("thumb1", 0.1, 0.0),
             f"{side}_thumb_MCP_VL_site": ("thumb2", 0.4, 0.0),
             f"{side}_thumb_DP_site": ("thumb3", 0.8, 0.0),
             f"{side}_thumb_tip_site": ("thumb4", 0.25, 0.0)}
        for fg in ("index", "middle", "ring"):
            m[f"{side}_{fg}_MCP_VL_site"] = (f"{fg}1", 0.3, 0.0)
            m[f"{side}_{fg}_MP_site"] = (f"{fg}2", 0.5, 0.0)
            m[f"{side}_{fg}_DP_site"] = (f"{fg}3", 0.5, 0.0)
            m[f"{side}_{fg}_tip_site"] = (f"{fg}4", 0.25, 0.0)
        m[f"{side}_pinky_MC_site"] = ("pinky1", 0.15, 0.0)
        m[f"{side}_pinky_MP_site"] = ("pinky2", 0.25, 0.0)
        m[f"{side}_pinky_DP_site"] = ("pinky3", 0.25, 0.0)
        m[f"{side}_pinky_tip_site"] = ("pinky4", 0.15, 0.0)
        self._kin.target_to_source = m
        self._kin.frame_tasks = self._kin.setup_frame_tasks()
        # 姿态目标契约是 MANO 约定，本数据是 Manus 约定（差 ~155° 常数），
        # 喂进去只会把解拉进扭转局部极小——全部置零，纯位置 IK。
        for task in self._kin.frame_tasks.values():
            task.set_orientation_cost(0.0)
        if focus:
            self._reweight_tasks(focus)

        self.joint_names: list[str] = list(
            self._kin.robot_finger_joint_names.values())
        self.n_joints = len(self.joint_names)

        # Kabsch 模板：中立位形（根=恒等）下全部任务位点在根系里的位置，
        # 以及每个位点对应的 MANO 关键点行号。用实例上改过的映射，不是库表。
        mapping = self._kin.target_to_source
        order = self._kin.source_joint_order
        self._site_mano_idx = np.array(
            [order.index(v[0]) for v in mapping.values()])
        model, data = self._kin.robot.model, self._kin.robot.data
        q0 = self._kin.robot.q0.copy()
        q0[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        self._kabsch_template = np.array([
            data.oMf[model.getFrameId(site)].translation.copy()
            for site in mapping
        ])
        self._qpos_prev: np.ndarray | None = None

    def _reweight_tasks(self, focus: tuple[str, ...]) -> None:
        """把 IK 注意力集中到 focus 里的手指上。

        权重表（SHARPA_TO_MANO_MAPPING）是 robotic_grounding 的模块常量，
        没有公开配置入口，所以直接改建好的 Pink FrameTask：重点手指指尖
        cost 提到 FOCUS_TIP_COST，其余手指降到 DEMOTE_*。wrist 任务
        不动——底座锚定不该跟着变。"""
        demote = [fg for fg in FINGERS if fg not in focus]
        for site, task in self._kin.frame_tasks.items():
            if any(f"_{fg}_" in site for fg in focus):
                if "tip" in site:
                    task.set_position_cost(FOCUS_TIP_COST)
            elif any(f"_{fg}_" in site for fg in demote):
                task.set_position_cost(
                    DEMOTE_TIP_COST if "tip" in site else DEMOTE_MP_COST)

    def reset(self) -> None:
        """episode 边界调用：清掉手指角的 warm-start，别让上一集的姿态
        泄漏到下一集的第一帧。底座种子本来就每帧重算，不受影响。"""
        self._qpos_prev = None

    def _kabsch_seed(self, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把中立模板刚体对齐到该帧的人手位点 -> (R (3,3), t (3,))。"""
        cm_m = self._kabsch_template.mean(axis=0)
        cm_h = targets.mean(axis=0)
        U, _, Vt = np.linalg.svd(
            (self._kabsch_template - cm_m).T @ (targets - cm_h))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return R, cm_h - R @ cm_m

    def solve_row(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一行 hand 数据 -> ((n_joints,) 角度, (7,) 底座位姿)。调用方保证 row 非全零。

        底座位姿是 IK 解的 FreeFlyer 前 7 维（xyz + xyzw 四元数），即
        Sharpa 根身体在输入坐标系下的最优位姿（local 帧 = Manus 腕系，
        stage 帧 = 世界系）。它和角度出自同一次联合优化，必须成对使用。"""
        mano_p, mano_q = row_to_mano(row)

        # Kabsch 目标要和 compute 内部的缩放一致（绕 wrist 缩放）。
        targets = mano_p[self._site_mano_idx]
        if self._scale != 1.0:
            targets = mano_p[0] + (targets - mano_p[0]) * self._scale
        R_seed, t_seed = self._kabsch_seed(targets)

        qpos = (self._kin.robot.q0.copy() if self._qpos_prev is None
                else self._qpos_prev.copy())
        qpos[0:3] = t_seed
        qpos[3:7] = Rotation.from_matrix(R_seed).as_quat()

        result = self._kin.compute(
            mano_p, mano_q, source_to_robot_scale=self._scale, qpos=qpos)
        q = np.asarray(result["q"], dtype=np.float64)
        self._qpos_prev = q.copy()
        return q[7:].astype(np.float32), q[:7].astype(np.float32)


# --------------------------------------------------------------------------- #
# parquet 读写（与 add_wrist_pose.py 相同的模式）
# --------------------------------------------------------------------------- #
def add_fixed_size_list_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    """values: (N, D) float32 -> fixed_size_list<float>[D] 列，追加到 table。"""
    flat = pa.array(values.reshape(-1), type=pa.float32())
    col = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    if name in table.column_names:
        table = table.drop([name])
    return table.append_column(name, col)


def process_file(path: Path, solvers: dict[str, SharpaSolver],
                 T_headset_to_head: dict[str, np.ndarray] | None,
                 dry_run: bool) -> dict[str, int]:
    table = pq.read_table(path)
    n = table.num_rows

    # 同一 parquet 文件里可能有多个 episode（data_file_size_mb 调大之后）；
    # warm-start 只能在同一集内延续，跨集要 reset。
    if "episode_index" in table.column_names:
        episode_idx = table.column("episode_index").to_numpy()
    else:
        episode_idx = np.zeros(n, dtype=np.int64)

    counts: dict[str, int] = {}
    for side, solver in solvers.items():
        rows = np.stack(
            table.column(f"observation.hand_{side}").to_numpy(zero_copy_only=False)
        ).astype(np.float32)
        # 整行为零 = 手不在（record_cameras.py 手 is_none 时写全零）。
        valid = np.any(rows != 0.0, axis=1)

        out = np.zeros((n, solver.n_joints + 1), dtype=np.float32)
        out_base = np.zeros((n, 8), dtype=np.float32)
        solver.reset()
        prev_ep = None
        for i in range(n):
            if episode_idx[i] != prev_ep:
                solver.reset()
                prev_ep = episode_idx[i]
            if not valid[i]:
                solver.reset()   # 断档之后别用断档前的姿态做 warm-start
                continue
            angles, base = solver.solve_row(rows[i])
            out[i, :-1] = angles
            out[i, -1] = 1.0
            out_base[i, :7] = base
            out_base[i, 7] = 1.0

        col = f"observation.sharpa_joints_{side}"
        counts[col] = int(valid.sum())
        if not dry_run:
            table = add_fixed_size_list_column(table, col, out)
            table = add_fixed_size_list_column(
                table, f"observation.sharpa_base_{side}", out_base)

        # stage 帧数据集：底座已是世界系，额外投到两个头部相机系。
        if T_headset_to_head is not None and rows.shape[1] == DIM_STAGE:
            head_pose = np.stack(
                table.column("observation.head_pose").to_numpy(zero_copy_only=False)
            ).astype(np.float64)
            ctrl = np.stack(
                table.column(f"observation.controller_{side}")
                .to_numpy(zero_copy_only=False)
            ).astype(np.float64)
            ctrl_valid = ctrl[:, 7] > 0.5
            base_valid = out_base[:, 7] > 0.5
            # 无效行是全零，零范数四元数会让 SciPy 抛异常（同 add_wrist_pose
            # 的处理），先替换成单位四元数，结果反正会被 valid 滤成零行。
            q_safe = out_base[:, 3:7].astype(np.float64).copy()
            q_safe[~base_valid] = [0.0, 0.0, 0.0, 1.0]
            R_base = Rotation.from_quat(q_safe).as_matrix()
            t_base = out_base[:, 0:3].astype(np.float64)
            for cam_side in ("left", "right"):
                R_sc, t_sc, head_valid = batch_stage_to_cam(
                    head_pose, T_headset_to_head[cam_side])
                R_res, t_res = compose_batch(R_sc, t_sc, R_base, t_base)
                cam_valid = base_valid & head_valid & ctrl_valid
                cam_col = f"observation.sharpa_base_{side}_in_head_{cam_side}"
                counts[cam_col] = int(cam_valid.sum())
                if not dry_run:
                    table = add_fixed_size_list_column(
                        table, cam_col, pack_pose_valid(R_res, t_res, cam_valid))

    if not dry_run:
        pq.write_table(table, path)

    counts["_rows"] = n
    return counts


def update_info_json(dataset_root: Path, solvers: dict[str, SharpaSolver],
                     stage_sides: set[str], dry_run: bool) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "valid"]
    for side, solver in solvers.items():
        info["features"][f"observation.sharpa_joints_{side}"] = {
            "dtype": "float32",
            "shape": [solver.n_joints + 1],
            "names": solver.joint_names + ["valid"],
        }
        # IK 底座位姿（与输入同系：local=Manus 腕系 ΔT，stage=世界系），
        # 与关节角同一次优化的产物。
        info["features"][f"observation.sharpa_base_{side}"] = {
            "dtype": "float32",
            "shape": [8],
            "names": pose_names,
        }
        if side in stage_sides:
            for cam_side in ("left", "right"):
                info["features"][
                    f"observation.sharpa_base_{side}_in_head_{cam_side}"
                ] = {
                    "dtype": "float32",
                    "shape": [8],
                    "names": pose_names,
                }

    if dry_run:
        print(f"  (dry-run，不写 {info_path})")
        return
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  已更新: {info_path}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="record_cameras.py 录的 LeRobot 数据集根目录")
    ap.add_argument("--max-iter", type=int, default=200,
                    help="每帧 IK 最大迭代数（默认 200，同重定向器默认值；"
                         "warm-start 下大多数帧远早于此收敛）")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="MANO 人手 -> Sharpa 机械手的尺度因子 "
                         "source_to_robot_scale（默认 1.0，同在线 demo）")
    ap.add_argument("--focus-fingers", type=str, default=None,
                    help="逗号分隔的手指名（thumb,index,middle,ring,pinky 的子集），"
                         "把 IK 权重集中到这些手指。注意这是空间贴合换时间信号的"
                         "取舍：重点指尖残差变小，但重点手指关节角会出现限位削顶"
                         "和站立偏置、与真实弯曲的相关性下降，非重点手指角度基本"
                         "不可用。关节角当学习信号用时别开这个，默认权重更好；"
                         "只有下游消费指尖空间位置时才值得开。两只手用同一份设置")
    ap.add_argument("--pico-intrinsics", type=Path,
                    default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz",
                    help="Pico 标定包 npz（同 add_wrist_pose.py），用于 stage 帧"
                         "数据集的 sharpa_base 头部相机系列")
    ap.add_argument("--pico-to-head", type=Path,
                    default=CALIB_DIR / "pico_to_head" / "extrinsics.npz",
                    help="pico 相机 -> 头部鱼眼的外参 npz（同 add_wrist_pose.py）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计有效帧数并抽样测 IK 速度，不写回 parquet / info.json")
    args = ap.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: {args.dataset_root} 看起来不是 LeRobot 数据集根目录"
              f"（缺 meta/info.json）", file=sys.stderr)
        return 1

    with open(args.dataset_root / "meta" / "info.json") as f:
        features = json.load(f)["features"]
    missing = [s for s in SIDES if f"observation.hand_{s}" not in features]
    if missing:
        print(f"ERROR: 数据集缺 observation.hand_{{{','.join(missing)}}} 列——"
              f"是不是用 --no-manus / --no-hand 录的？", file=sys.stderr)
        return 1
    stage_sides: set[str] = set()
    for side in SIDES:
        dim = features[f"observation.hand_{side}"]["shape"][0]
        frame = {DIM_LOCAL: "local(腕局部)", DIM_STAGE: "stage(世界)"}.get(dim)
        if frame is None:
            print(f"ERROR: observation.hand_{side} 的 shape 是 [{dim}]，"
                  f"既不是 local({DIM_LOCAL}) 也不是 stage({DIM_STAGE})",
                  file=sys.stderr)
            return 1
        if dim == DIM_STAGE:
            stage_sides.add(side)
        print(f"  observation.hand_{side}: [{dim}] -> {frame} 帧")

    # stage 帧才有世界系底座可投相机；标定加载失败与其静默跳过不如直接报错。
    T_headset_to_head: dict[str, np.ndarray] | None = None
    if stage_sides:
        for p in (args.pico_intrinsics, args.pico_to_head):
            if not p.exists():
                print(f"ERROR: 缺少相机标定文件 {p}（stage 帧数据集需要它来写 "
                      f"sharpa_base 的头部相机系列；参数 --pico-intrinsics / "
                      f"--pico-to-head 可改路径）", file=sys.stderr)
                return 1
        hl, hr = load_headset_to_head_cams(args.pico_intrinsics, args.pico_to_head)
        T_headset_to_head = {"left": hl, "right": hr}
        print(f"  头部相机标定已加载，将写出 sharpa_base_*_in_head_* 列")
    else:
        print(f"  local 帧数据集：跳过头部相机系列（底座是腕系 ΔT，投相机需要"
              f"锚定链，见文件头说明）")

    focus: tuple[str, ...] | None = None
    if args.focus_fingers:
        focus = tuple(fg.strip() for fg in args.focus_fingers.split(",") if fg.strip())
        bad = [fg for fg in focus if fg not in FINGERS]
        if bad:
            print(f"ERROR: --focus-fingers 里有未知手指名 {bad}，"
                  f"合法值: {','.join(FINGERS)}", file=sys.stderr)
            return 1

    print("\n加载 Sharpa 重定向器（Pinocchio + Pink IK）:")
    solvers = {}
    for side in SIDES:
        t0 = time.monotonic()
        solvers[side] = SharpaSolver(side, args.max_iter, args.scale, focus)
        print(f"  [{side}] {Path(resolve_mjcf(side)).name} "
              f"({solvers[side].n_joints} DOF, {time.monotonic() - t0:.1f}s)")
    if focus:
        print(f"  IK 聚焦手指: {', '.join(focus)}（其余手指已降权，"
              f"其输出角度仅供参考）")

    parquet_files = sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        print(f"ERROR: {args.dataset_root / 'data'} 下没找到 parquet 文件",
              file=sys.stderr)
        return 1

    if args.dry_run:
        # 不跑全量 IK：统计有效帧 + 拿第一个文件的前几十帧测速估个总时长。
        print(f"\ndry-run：{len(parquet_files)} 个 parquet 文件")
        totals: dict[str, int] = {}
        total_rows = 0
        for path in parquet_files:
            table = pq.read_table(path, columns=[
                f"observation.hand_{s}" for s in SIDES])
            total_rows += table.num_rows
            for side in SIDES:
                rows = np.stack(table.column(f"observation.hand_{side}")
                                .to_numpy(zero_copy_only=False))
                key = f"observation.sharpa_joints_{side}"
                totals[key] = totals.get(key, 0) + int(np.any(rows != 0, axis=1).sum())
        for name, cnt in totals.items():
            pct = 100.0 * cnt / total_rows if total_rows else 0.0
            print(f"  {name}: {cnt}/{total_rows} ({pct:.1f}%)")

        bench_table = pq.read_table(parquet_files[0])
        bench = np.stack(bench_table.column("observation.hand_right")
                         .to_numpy(zero_copy_only=False)).astype(np.float32)
        bench = bench[np.any(bench != 0, axis=1)][:30]
        if len(bench):
            solvers["right"].reset()
            t0 = time.monotonic()
            for row in bench:
                solvers["right"].solve_row(row)
            per = (time.monotonic() - t0) / len(bench)
            est = per * sum(totals.values())
            print(f"\n  IK 速度: {per * 1e3:.1f} ms/帧 -> 全量约 {est / 60:.1f} 分钟")
        print("\ndry-run 完成，没有修改任何文件。")
        return 0

    print(f"\n共 {len(parquet_files)} 个 parquet 文件")
    totals = {}
    total_rows = 0
    t_start = time.monotonic()
    for k, path in enumerate(parquet_files):
        t0 = time.monotonic()
        counts = process_file(path, solvers, T_headset_to_head, dry_run=False)
        rows = counts.pop("_rows")
        total_rows += rows
        for name, v in counts.items():
            totals[name] = totals.get(name, 0) + v
        elapsed = time.monotonic() - t_start
        eta = elapsed / (k + 1) * (len(parquet_files) - k - 1)
        print(f"  [{k + 1}/{len(parquet_files)}] "
              f"{path.relative_to(args.dataset_root)}: {rows} 行 "
              f"({time.monotonic() - t0:.1f}s, 剩余约 {eta / 60:.1f} 分钟)")

    print(f"\n共 {total_rows} 行，各列有效帧数:")
    for name, cnt in totals.items():
        pct = 100.0 * cnt / total_rows if total_rows else 0.0
        print(f"  {name}: {cnt}/{total_rows} ({pct:.1f}%)")

    print("\n更新 meta/info.json:")
    update_info_json(args.dataset_root, solvers, stage_sides, dry_run=False)

    print("\n完成。注意 meta/stats.json 和 meta/episodes/ 里的统计信息没有更新，"
          "包含新列的 min/max/mean 统计需要你自己用 LeRobot 的统计工具重新算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
