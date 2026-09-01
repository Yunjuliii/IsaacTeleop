#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
测 Pico controller 的 grip->aim 常数变换，并合成"数据集腕系 -> Manus 腕系"
的常数偏移 C。

背景
----
数据集里有两个不同的 wrist：

  wrist_in_world(t)   = T_grip->stage(t) . T_wrist->ctrl     (pivot+rotation 标定)
  manusWrist_world(t) = T_grip->stage(t) . T_grip->aim . kHandOffset
                        (manus_hand_tracking_plugin.cpp 锚手的方式)

observation.hand_* (local 帧) 及其重定向产物（Sharpa 关节角、底座 ΔT）都
定义在 Manus 腕系下；回放时要把它们挂回世界系，得走 manusWrist 这条链。
两条链只差一个常数：

  C = inv(T_wrist->ctrl) . T_grip->aim . kHandOffset
  T_world_manusWrist(t) = wrist_in_world(t) . C

T_wrist->ctrl（标定 npz）和 kHandOffset（插件源码常数，即 record_cameras.py
的 DEFAULT_AIM_TO_WRIST）都已知，唯一缺的是 T_grip->aim——controller 固件
定义的刚体常数，同一帧读 grip 和 aim 两个位姿即可测出，与戴法无关。

用法
----
开着 CloudXR 客户端、两只 controller 都在追踪时运行：

    python3 calibrate_grip_to_aim.py                 # 默认采 5 秒
    python3 calibrate_grip_to_aim.py --seconds 10
    python3 calibrate_grip_to_aim.py --no-compose-c  # 只测 grip->aim，不合成 C

输出 calib_data/grip_to_aim/grip_to_aim_<时间戳>.npz，字段（每侧）：
    T_grip_to_aim_{left,right}   (4,4)  grip->aim
    C_{left,right}               (4,4)  数据集腕系 -> Manus 腕系（除非 --no-compose-c）
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from add_wrist_pose import load_wrist_to_ctrl, pose_to_matrix
from record_cameras import DEFAULT_AIM_TO_WRIST

CALIB_DIR = Path(__file__).parent / "calib_data"
SIDES = ("left", "right")
POLL_HZ = 60.0


def collect_samples(seconds: float) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """开一个只含 ControllersSource 的最小会话，采集每侧 (grip 4x4, aim 4x4) 样本。

    只保留 grip 和 aim 同帧都 valid 的样本——两个位姿出自同一个刚体追踪，
    有效时它们的相对变换应当是常数。
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner
    from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex as CI
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

    controllers = ControllersSource(name="controllers")
    pipeline = OutputCombiner({
        "controller_left":  controllers.output(ControllersSource.LEFT),
        "controller_right": controllers.output(ControllersSource.RIGHT),
    })
    config = TeleopSessionConfig(app_name="GripToAimCalib", pipeline=pipeline, plugins=[])

    # 客户端没连上时 TeleopSession 进不去（XR_ERROR_FORM_FACTOR_UNAVAILABLE），
    # 跟 record_cameras.ManusHandBuffer.__enter__ 一样重试等待。
    session = TeleopSession(config)
    while True:
        try:
            session.__enter__()
            break
        except RuntimeError as exc:
            if "Failed to get OpenXR system" not in str(exc):
                raise
            print("等待 CloudXR 客户端连接...", flush=True)
            time.sleep(2.0)
            session = TeleopSession(config)

    samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {s: [] for s in SIDES}
    period = 1.0 / POLL_HZ
    t_end = time.monotonic() + seconds
    try:
        print(f"采集 {seconds:.0f} 秒（两只 controller 保持在追踪范围内，姿态随意）...")
        while time.monotonic() < t_end:
            result = session.step()
            for side in SIDES:
                ctrl = result[f"controller_{side}"]
                if ctrl.is_none:
                    continue
                if not (bool(ctrl[CI.GRIP_IS_VALID]) and bool(ctrl[CI.AIM_IS_VALID])):
                    continue
                grip = pose_to_matrix(
                    np.asarray(ctrl[CI.GRIP_POSITION], dtype=np.float64),
                    np.asarray(ctrl[CI.GRIP_ORIENTATION], dtype=np.float64))
                aim = pose_to_matrix(
                    np.asarray(ctrl[CI.AIM_POSITION], dtype=np.float64),
                    np.asarray(ctrl[CI.AIM_ORIENTATION], dtype=np.float64))
                samples[side].append((grip, aim))
            time.sleep(period)
    finally:
        session.__exit__(None, None, None)
    return samples


def average_grip_to_aim(samples: list[tuple[np.ndarray, np.ndarray]],
                        side: str) -> np.ndarray:
    """逐帧算 inv(grip)·aim 再取均值。固件常数应当几乎不散布——
    散布大说明两个位姿不是同刚体（数据有问题），直接报错而不是硬平均。"""
    rels = np.stack([np.linalg.inv(g) @ a for g, a in samples])
    ts = rels[:, :3, 3]
    rots = Rotation.from_matrix(rels[:, :3, :3])
    t_mean = ts.mean(axis=0)
    r_mean = rots.mean()

    t_spread = np.linalg.norm(ts - t_mean, axis=1).max() * 1000  # mm
    r_spread = np.degrees((r_mean.inv() * rots).magnitude().max())
    print(f"  [{side}] {len(samples)} 帧: 平移散布 max {t_spread:.2f} mm, "
          f"旋转散布 max {r_spread:.3f}°")
    if t_spread > 2.0 or r_spread > 0.5:
        sys.exit(f"ERROR: [{side}] grip->aim 逐帧散布过大（应为固件常数、接近零）。"
                 f"检查 controller 追踪是否稳定后重测。")

    T = np.eye(4)
    T[:3, :3] = r_mean.as_matrix()
    T[:3, 3] = t_mean
    return T


def hand_offset_matrix(side: str) -> np.ndarray:
    """kHandOffset（aim->manusWrist），取自 DEFAULT_AIM_TO_WRIST——它就是
    manus_hand_tracking_plugin.cpp 里 kLeft/RightHandOffset 的镜像拷贝。"""
    spec = DEFAULT_AIM_TO_WRIST[side]
    return pose_to_matrix(np.asarray(spec["position"], dtype=np.float64),
                          np.asarray(spec["quaternion"], dtype=np.float64))


def fmt_pose(T: np.ndarray) -> str:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return (f"t=[{T[0,3]:+.4f} {T[1,3]:+.4f} {T[2,3]:+.4f}] "
            f"q_xyzw=[{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="采集时长（默认 5 秒，60Hz 轮询）")
    ap.add_argument("--min-samples", type=int, default=60,
                    help="每侧最少有效帧数（默认 60）")
    ap.add_argument("--no-compose-c", action="store_true",
                    help="只测 grip->aim，不加载腕标定、不合成 C")
    args = ap.parse_args()

    samples = collect_samples(args.seconds)
    print("\ngrip->aim（应为固件常数）:")
    out: dict[str, np.ndarray] = {}
    for side in SIDES:
        if len(samples[side]) < args.min_samples:
            sys.exit(f"ERROR: [{side}] 有效帧只有 {len(samples[side])} "
                     f"(< {args.min_samples})。controller 是否开机并在追踪范围内？")
        T_ga = average_grip_to_aim(samples[side], side)
        print(f"  [{side}] {fmt_pose(T_ga)}")
        out[f"T_grip_to_aim_{side}"] = T_ga

    if not args.no_compose_c:
        print("\n合成 C = inv(T_wrist->ctrl) . T_grip->aim . kHandOffset:")
        controller_dir = CALIB_DIR / "controller"
        for side in SIDES:
            p, R = load_wrist_to_ctrl(controller_dir, side)
            T_wrist_ctrl = np.eye(4)
            T_wrist_ctrl[:3, :3] = R
            T_wrist_ctrl[:3, 3] = p
            C = np.linalg.inv(T_wrist_ctrl) @ out[f"T_grip_to_aim_{side}"] \
                @ hand_offset_matrix(side)
            out[f"C_{side}"] = C
            print(f"  [{side}] {fmt_pose(C)}")

    out_dir = CALIB_DIR / "grip_to_aim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"grip_to_aim_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, **out)
    print(f"\n已保存: {path}")
    print("回放合成: T_world_sharpaBase(t) = wrist_in_world(t) . C . ΔT(t)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
