#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
测量 N₀：Manus raw skeleton 0 号节点(wrist)的变换，验证"manus wrist =
aim + 固定偏移"这个形式是否成立。

原理
----
插件合成 stage 帧腕关节的公式是（manus_hand_tracking_plugin.cpp）：

    joint_WRIST(t) = aim(t) · kHandOffset · N₀(t)

其中 N₀ 是 Manus Core 在自己世界空间里报告的 wrist 节点位姿（SDK 以
world 模式初始化，N₀ 未必恒等，甚至可能带着手套 IMU 朝向随时间变化）。
同帧读 joint_WRIST 和 aim 即可反解：

    N₀(t) = inv(aim(t) · kHandOffset) · joint_WRIST(t)

判定
----
按三个动作段分别统计 N₀ 的散布：

  A 静止           —— 基线噪声
  B 整臂刚性转动   —— 手+手柄一起转，腕不屈伸。若 N₀ 随之变化，
                      说明它带着 Manus 世界系的绝对朝向（IMU），
                      "aim+固定偏移"不成立且有漂移问题。
  C 只做腕屈伸     —— 手相对手柄转。若 N₀ 只在此段变化，说明它编码
                      了手套相对手柄的真实朝向（信息量比声称的多）。

三段都恒定（旋转 <2°、平移 <5mm）→ N₀ 恒定，"aim+固定偏移"成立，
有效偏移 = kHandOffset · N₀_mean，脚本会直接打印出来。

用法
----
戴好手套、controller 绑好、CloudXR 客户端已连接后运行：

    python3 measure_n0.py                # 双手，A/B/C 各 6/10/10 秒
    python3 measure_n0.py --seconds 8,12,12

时序数据存 calib_data/n0/n0_<时间戳>.npz 供离线复查。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from add_wrist_pose import pose_to_matrix
from record_cameras import DEFAULT_AIM_TO_WRIST, MANUS_PLUGIN_DIR, DEFAULT_WRIST_SOURCE

CALIB_DIR = Path(__file__).parent / "calib_data"
SIDES = ("left", "right")
POLL_HZ = 60.0

SEGMENTS = [
    ("A_still",  "【A 静止】双手自然放稳，保持不动"),
    ("B_rigid",  "【B 整臂刚性转动】手+手柄整体转动手臂，手腕不要屈伸"),
    ("C_flex",   "【C 腕屈伸】前臂尽量不动，只做手腕屈伸/侧偏"),
]


def khand_offset(side: str) -> np.ndarray:
    spec = DEFAULT_AIM_TO_WRIST[side]
    return pose_to_matrix(np.asarray(spec["position"], dtype=np.float64),
                          np.asarray(spec["quaternion"], dtype=np.float64))


def build_session():
    """HandsSource + ControllersSource + Manus 插件的最小会话。"""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import (
        ControllersSource, HandsSource)
    from isaacteleop.retargeting_engine.interface import OutputCombiner
    from isaacteleop.teleop_session_manager import (
        PluginConfig, TeleopSession, TeleopSessionConfig)

    if not list(MANUS_PLUGIN_DIR.glob("*/plugin.yaml")):
        sys.exit(f"ERROR: {MANUS_PLUGIN_DIR} 下没有 <plugin>/plugin.yaml —— "
                 f"Manus 插件不会启动（同 record_cameras.py 的检查）")

    hands = HandsSource(name="hands")
    controllers = ControllersSource(name="controllers")
    pipeline = OutputCombiner({
        "hand_left":        hands.output(HandsSource.LEFT),
        "hand_right":       hands.output(HandsSource.RIGHT),
        "controller_left":  controllers.output(ControllersSource.LEFT),
        "controller_right": controllers.output(ControllersSource.RIGHT),
    })
    config = TeleopSessionConfig(
        app_name="MeasureN0", pipeline=pipeline,
        plugins=[PluginConfig(plugin_name="manus_hand_plugin",
                              plugin_root_id="manus",
                              search_paths=[MANUS_PLUGIN_DIR])])
    session = TeleopSession(config)
    while True:
        try:
            session.__enter__()
            return session
        except RuntimeError as exc:
            if "Failed to get OpenXR system" not in str(exc):
                raise
            print("等待 CloudXR 客户端连接...", flush=True)
            time.sleep(2.0)
            from isaacteleop.teleop_session_manager import TeleopSession as TS
            session = TS(config)


def extract_n0(result, side: str, K_inv_cache: dict):
    """一帧里反解一侧的 N₀。返回 4x4 或 None（该帧数据无效）。"""
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInputIndex as CI, HandInputIndex as HI, HandJointIndex as HJ)

    hand = result[f"hand_{side}"]
    ctrl = result[f"controller_{side}"]
    if hand.is_none or ctrl.is_none or not bool(ctrl[CI.AIM_IS_VALID]):
        return None
    valid = np.asarray(hand[HI.JOINT_VALID])
    if not valid[HJ.WRIST]:
        return None
    pos = np.asarray(hand[HI.JOINT_POSITIONS], dtype=np.float64)[HJ.WRIST]
    ori = np.asarray(hand[HI.JOINT_ORIENTATIONS], dtype=np.float64)[HJ.WRIST]
    if not np.any(pos) and not np.any(ori[:3]):
        return None
    T_wrist = pose_to_matrix(pos, ori)
    T_aim = pose_to_matrix(np.asarray(ctrl[CI.AIM_POSITION], dtype=np.float64),
                           np.asarray(ctrl[CI.AIM_ORIENTATION], dtype=np.float64))
    return K_inv_cache[side] @ np.linalg.inv(T_aim) @ T_wrist


def stats(mats: list[np.ndarray]) -> tuple[np.ndarray, float, float, float, float]:
    """返回 (均值4x4, 旋转偏差mean°, 旋转偏差max°, 平移偏差mean mm, max mm)。"""
    ts = np.array([m[:3, 3] for m in mats])
    rots = Rotation.from_matrix(np.array([m[:3, :3] for m in mats]))
    r_mean = rots.mean()
    dev = np.degrees((r_mean.inv() * rots).magnitude())
    t_mean = ts.mean(axis=0)
    dt = np.linalg.norm(ts - t_mean, axis=1) * 1000
    M = np.eye(4); M[:3, :3] = r_mean.as_matrix(); M[:3, 3] = t_mean
    return M, dev.mean(), dev.max(), dt.mean(), dt.max()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=str, default="6,10,10",
                    help="三个动作段的时长，逗号分隔（默认 6,10,10）")
    args = ap.parse_args()
    durations = [float(s) for s in args.seconds.split(",")]
    if len(durations) != 3:
        sys.exit("ERROR: --seconds 需要 3 个数，如 6,10,10")

    # 与 record_cameras 一致：插件子进程从环境读 wrist source。
    os.environ.setdefault("MANUS_WRIST_SOURCE", DEFAULT_WRIST_SOURCE)

    K_inv = {s: np.linalg.inv(khand_offset(s)) for s in SIDES}
    session = build_session()
    period = 1.0 / POLL_HZ

    # 等到至少一侧出现有效手数据再开始计时。
    print("等待 Manus 手套数据...", flush=True)
    t_wait = time.monotonic()
    while True:
        result = session.step()
        if any(extract_n0(result, s, K_inv) is not None for s in SIDES):
            break
        if time.monotonic() - t_wait > 30:
            print("WARNING: 30 秒无有效手数据——检查手套供电/插件是否启动、"
                  "controller 是否在追踪。继续等待...", flush=True)
            t_wait = time.monotonic()
        time.sleep(period)
    print("手套就绪。\n")

    samples: dict[str, dict[str, list]] = {s: {seg: [] for seg, _ in SEGMENTS}
                                           for s in SIDES}
    try:
        for (seg, prompt), dur in zip(SEGMENTS, durations):
            print(f"\n{prompt} —— {dur:.0f} 秒，3 秒后开始...")
            time.sleep(3.0)
            print("  开始")
            t_end = time.monotonic() + dur
            while time.monotonic() < t_end:
                result = session.step()
                for side in SIDES:
                    n0 = extract_n0(result, side, K_inv)
                    if n0 is not None:
                        samples[side][seg].append(n0)
                time.sleep(period)
            print("  完成")
    finally:
        session.__exit__(None, None, None)

    out = {}
    print("\n" + "=" * 72)
    for side in SIDES:
        all_mats = [m for seg, _ in SEGMENTS for m in samples[side][seg]]
        if len(all_mats) < 30:
            print(f"[{side}] 有效帧不足({len(all_mats)})，跳过")
            continue
        print(f"\n[{side}] N₀ 统计（相对全程均值的散布）:")
        M_all, _, _, _, _ = stats(all_mats)
        r_all = Rotation.from_matrix(M_all[:3, :3])
        for seg, _ in SEGMENTS:
            mats = samples[side][seg]
            if not mats:
                print(f"  {seg:8s}: 无数据")
                continue
            ts = np.array([m[:3, 3] for m in mats])
            rots = Rotation.from_matrix(np.array([m[:3, :3] for m in mats]))
            dev = np.degrees((r_all.inv() * rots).magnitude())
            dt = np.linalg.norm(ts - M_all[:3, 3], axis=1) * 1000
            print(f"  {seg:8s}: {len(mats):4d} 帧  旋转偏差 mean {dev.mean():6.2f}° "
                  f"max {dev.max():6.2f}°   平移偏差 mean {dt.mean():5.1f}mm max {dt.max():5.1f}mm")
        q = r_all.as_quat()
        print(f"  N₀ 均值: t={np.round(M_all[:3,3],4)} quat_xyzw={np.round(q,4)}")
        _, rm, rx, tm, tx = stats(all_mats)
        if rx < 2.0 and tx < 5.0:
            eff = khand_offset(side) @ M_all
            qe = Rotation.from_matrix(eff[:3, :3]).as_quat()
            print(f"  ✅ N₀ 恒定 → 'aim+固定偏移'成立。有效偏移 kHandOffset·N₀ =")
            print(f"     t={np.round(eff[:3,3],4)} quat_xyzw={np.round(qe,4)}")
        else:
            print(f"  ❌ N₀ 不恒定（对照 B/C 段判断成因：B 段变→带绝对 IMU 朝向；"
                  f"仅 C 段变→编码腕相对手柄的朝向）")
        out[f"N0_mean_{side}"] = M_all
        for seg, _ in SEGMENTS:
            if samples[side][seg]:
                out[f"N0_{side}_{seg}"] = np.stack(samples[side][seg])

    out_dir = CALIB_DIR / "n0"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"n0_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, **out)
    print(f"\n已保存时序: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
