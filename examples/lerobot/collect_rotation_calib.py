#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Collect PICO controller quaternion data for wrist→controller rotation calibration.

原理
----
每个 Phase 中，将手腕某条轴对齐到世界 +Y（朝上）并静止。
此时手柄四元数 q_ctrl 给出 R_ctrl_stage，由此得到 R_wrist_ctrl 的一列：

    R_wrist_ctrl * e_k  =  R_ctrl_stage^T · [0, 1, 0]

三个 Phase → 三列 → SVD 投影到最近旋转矩阵 SO(3)。

三个姿势（每手分别做）：

  Phase 1  手背朝上（Dorsal up）   手背正对天花板
  Phase 2  手刀朝上（Ulnar up）    小拇指侧朝上
  Phase 3  手指朝上（Distal up）   手指指尖朝上

输出
----
  calib_rotation_{side}_{timestamp}.npz
  ├── {side}_phase{1,2,3}_cols   — (N, 3) 每帧计算的列向量
  ├── R_wrist_ctrl_{side}        — (3, 3) 最终旋转矩阵
  └── quat_wrist_ctrl_{side}     — (4,)   对应四元数 [qx, qy, qz, qw]

用法
----
    python3 collect_rotation_calib.py              # 右手（默认）
    python3 collect_rotation_calib.py --side left
    python3 collect_rotation_calib.py --side both
    python3 collect_rotation_calib.py --seconds 8  # 每个姿势采集 8 秒
    python3 collect_rotation_calib.py --out ~/calib  # 输出目录
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# 路径常量（与 record_cameras.py 保持一致）
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANUS_PLUGIN_DIR = _REPO_ROOT / "install" / "plugins"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
POLL_HZ = 60  # 手柄数据轮询频率
WORLD_UP = np.array([0.0, 1.0, 0.0])  # 世界 +Y 方向
DEFAULT_SECS = 5  # 每个姿势的采集时长（秒）

# ---------------------------------------------------------------------------
# Phase 定义
# ---------------------------------------------------------------------------
PHASES: list[dict] = [
    dict(
        idx=1,
        tag="dorsal",
        zh="手背朝上",
        en="Back of hand faces UP (+Y)",
        instruction=(
            "将手腕转到手背（背面）朝向天花板。\n"
            "前臂保持水平，手腕不动。\n"
            "就位后按 SPACE 开始计时。"
        ),
    ),
    dict(
        idx=2,
        tag="ulnar",
        zh="手刀朝上",
        en="Little-finger edge faces UP (+Y)",
        instruction=(
            "将手腕转到小拇指一侧（尺侧/手刀）朝向天花板。\n"
            "前臂保持水平，手腕不动。\n"
            "就位后按 SPACE 开始计时。"
        ),
    ),
    dict(
        idx=3,
        tag="distal",
        zh="手指朝上",
        en="Fingertips point UP (+Y)",
        instruction=(
            "手指伸直，让指尖朝向天花板。\n"
            "前臂保持水平，手腕不动。\n"
            "就位后按 SPACE 开始计时。"
        ),
    ),
]


# ---------------------------------------------------------------------------
# 键盘
# ---------------------------------------------------------------------------
def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# 手柄数据缓冲（不加载 Manus 插件，只需 OpenXR 运行时提供手柄位姿）
# ---------------------------------------------------------------------------
class ControllerBuffer:
    """
    TeleopSession 的轻量包装，只暴露手柄 grip pose。
    不加载 Manus 插件（旋转标定不需要手套数据）。

    latest() → (left, right)，各为 float32[8]：
      [x, y, z, qx, qy, qz, qw, valid]
    """

    def __init__(self) -> None:
        from isaacteleop.retargeting_engine.deviceio_source_nodes import (
            ControllersSource,
        )
        from isaacteleop.retargeting_engine.interface import OutputCombiner
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex
        from isaacteleop.teleop_session_manager import (
            TeleopSession,
            TeleopSessionConfig,
        )

        self._CI = ControllerInputIndex
        self._TeleopSession = TeleopSession

        controllers = ControllersSource(name="controllers")
        pipeline = OutputCombiner(
            {
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
            }
        )
        self._cfg = TeleopSessionConfig(
            app_name="RotationCalib",
            pipeline=pipeline,
            plugins=[],  # 旋转标定不需要 Manus 插件
        )
        self._session = TeleopSession(self._cfg)
        self._lock = threading.Lock()
        self._left = np.zeros(8, dtype=np.float32)
        self._right = np.zeros(8, dtype=np.float32)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "ControllerBuffer":
        while True:
            try:
                self._session.__enter__()
                break
            except RuntimeError as exc:
                if "Failed to get OpenXR system" not in str(exc):
                    raise
                print("[Controller] 等待 CloudXR 客户端连接 ...", flush=True)
                time.sleep(2.0)
                self._session = self._TeleopSession(self._cfg)

        self._thread = threading.Thread(target=self._run, daemon=True, name="ctrl-poll")
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        try:
            self._session.__exit__(*args)
        except Exception as exc:
            if "signal 2" in str(exc) or "Interrupt" in str(exc):
                pass  # Ctrl+C 导致的插件退出，正常
            else:
                raise

    # ── 内部 ─────────────────────────────────────────────────────────────
    def _extract(self, ctrl) -> np.ndarray:
        out = np.zeros(8, dtype=np.float32)
        if ctrl.is_none:
            return out
        CI = self._CI
        if bool(ctrl[CI.GRIP_IS_VALID]):
            out[0:3] = np.asarray(ctrl[CI.GRIP_POSITION], dtype=np.float32)
            out[3:7] = np.asarray(ctrl[CI.GRIP_ORIENTATION], dtype=np.float32)
            out[7] = 1.0
        return out

    def _run(self) -> None:
        period = 1.0 / POLL_HZ
        deadline = time.monotonic() + period
        while not self._stop.is_set():
            try:
                result = self._session.step()
                left = self._extract(result["controller_left"])
                right = self._extract(result["controller_right"])
                with self._lock:
                    self._left = left
                    self._right = right
                if left[7] > 0.5 or right[7] > 0.5:
                    self._ready.set()
            except Exception as exc:
                self._error = exc
                print(f"\n[Controller] 轮询线程异常：{exc}", file=sys.stderr)
                return
            if self._stop.wait(timeout=max(deadline - time.monotonic(), 0.001)):
                return
            deadline += period
            if deadline < time.monotonic():
                deadline = time.monotonic() + period

    # ── 公共接口 ─────────────────────────────────────────────────────────
    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        if self._error:
            raise RuntimeError("手柄轮询线程已退出") from self._error
        with self._lock:
            return self._left.copy(), self._right.copy()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)


# ---------------------------------------------------------------------------
# 标定计算
# ---------------------------------------------------------------------------
def compute_R_wrist_ctrl(
    phase_cols: list[np.ndarray], side: str = "right"
) -> np.ndarray:
    """
    从三个 Phase 的列向量均值重建 R_wrist_ctrl，投影到 SO(3)。

    phase_cols: list of 3 arrays, each (N, 3)
    side: "left" or "right"
    returns: (3, 3) float64 rotation matrix

    注意：左手 Phase 2（尺侧朝上）与右手方向相反，会导致三列构成左手坐标系
    （det ≈ -1）。需对 Phase 2 列向量取反以修正为右手坐标系，再做 SVD 投影。
    """
    cols = []
    for raw in phase_cols:
        mean_col = raw.mean(axis=0)
        norm = np.linalg.norm(mean_col)
        if norm < 1e-6:
            raise ValueError("列向量接近零向量，该 Phase 的数据无效")
        cols.append(mean_col / norm)

    # 左手尺侧轴与右手互为镜像，取反后构成右手坐标系
    if side == "left":
        cols[1] = -cols[1]

    R_raw = np.column_stack(cols)  # (3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # 修正反射（理论上不应触发）
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_error_deg(R_raw_cols: list[np.ndarray]) -> float:
    """计算三列的最大正交性误差（度），评估数据质量。"""
    cols = [c.mean(axis=0) for c in R_raw_cols]
    cols = [c / np.linalg.norm(c) for c in cols]
    # 三对点积，理想值为 0
    dots = [
        abs(np.dot(cols[0], cols[1])),
        abs(np.dot(cols[0], cols[2])),
        abs(np.dot(cols[1], cols[2])),
    ]
    return float(np.degrees(np.arcsin(max(dots))))


# ---------------------------------------------------------------------------
# Phase 采集
# ---------------------------------------------------------------------------
AVG_FRAMES = 30  # 每次按 SPACE 采集并平均的帧数（≈ 0.5 s at 60 Hz）


def _current_col(buf: ControllerBuffer, side: str) -> np.ndarray | None:
    """返回当前帧对应的 R_wrist_ctrl 列估计，无效时返回 None。"""
    left, right = buf.latest()
    pose = right if side == "right" else left
    if pose[7] < 0.5:
        return None
    q = pose[3:7].astype(np.float64)
    col = Rotation.from_quat(q).inv().apply(WORLD_UP)
    return col / np.linalg.norm(col)


def _angle_to_col(col_cur: np.ndarray, ref_col: np.ndarray) -> float:
    """两个单位向量之间的夹角（度），取绝对值因为轴方向有歧义。"""
    dot = float(np.clip(np.dot(col_cur, ref_col / np.linalg.norm(ref_col)), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(dot))))


def collect_phase(
    buf: ControllerBuffer, phase: dict, side: str, prev_cols: list[np.ndarray]
) -> np.ndarray:
    """
    手动触发采集一个 Phase。用 select 做非阻塞键盘检测，避免多线程竞争。

    - 实时显示当前轴与前几个 Phase 轴的夹角（目标 90°）。
    - 按 SPACE：采集 AVG_FRAMES 帧取均值。
    - 按 r：重新提示（已在同一循环中，继续调整即可）。

    返回 (AVG_FRAMES, 3) float64。
    """
    print(f"\n{'─' * 60}")
    print(f"  Phase {phase['idx']}/3  ─  {phase['zh']}  ({phase['en']})")
    print(f"{'─' * 60}")
    print(phase["instruction"])
    if prev_cols:
        tags = [f"Phase{i + 1}" for i in range(len(prev_cols))]
        print(
            f"\n  实时显示与 {', '.join(tags)} 的夹角，调整到接近 90° 后按 SPACE 采样。"
        )
    else:
        print("\n  对准后按 SPACE 采样（本 Phase 无角度约束）。")
    print("  按 r 重采本 Phase（重新对准后再按 SPACE）。\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while True:
            # ── 实时显示 ──────────────────────────────────────────────
            col_cur = _current_col(buf, side)
            if col_cur is not None and prev_cols:
                parts = []
                for i, pc in enumerate(prev_cols):
                    angle = _angle_to_col(col_cur, pc)
                    diff = abs(angle - 90.0)
                    flag = "✓" if diff < 5 else ("△" if diff < 15 else "✗")
                    parts.append(f"Phase{i + 1}: {angle:5.1f}° [{flag}]")
                line = "  " + "   ".join(parts) + "   (目标 90°)   SPACE=采样"
            elif col_cur is not None:
                line = "  对准目标轴后按 SPACE 采样"
            else:
                line = "  ⚠ 手柄信号无效"
            print(f"\r{line:<72}", end="", flush=True)

            # ── 非阻塞键盘检测（50 ms 超时）────────────────────────────
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1)

            if ch == "\x03":  # Ctrl+C
                print()
                raise KeyboardInterrupt

            if ch == " ":
                print()  # 结束实时行
                # ── 采集 AVG_FRAMES 帧并平均 ──────────────────────────
                cols: list[np.ndarray] = []
                for _ in range(AVG_FRAMES):
                    left, right = buf.latest()
                    pose = right if side == "right" else left
                    if pose[7] > 0.5:
                        q = pose[3:7].astype(np.float64)
                        c = Rotation.from_quat(q).inv().apply(WORLD_UP)
                        cols.append(c / np.linalg.norm(c))
                    time.sleep(1.0 / POLL_HZ)

                if len(cols) < AVG_FRAMES // 2:
                    print(
                        f"  ✗ 有效帧不足（{len(cols)}/{AVG_FRAMES}），请检查手柄追踪后重新按 SPACE"
                    )
                    continue

                arr = np.array(cols, dtype=np.float64)
                mean_col = arr.mean(axis=0)
                mean_col /= np.linalg.norm(mean_col)

                angle_info = ""
                if prev_cols:
                    angles = [f"{_angle_to_col(mean_col, pc):.1f}°" for pc in prev_cols]
                    angle_info = "  夹角：" + " / ".join(
                        f"vs Phase{i + 1}={a}" for i, a in enumerate(angles)
                    )
                print(f"  ✓ Phase {phase['idx']} 采集完成{angle_info}")
                return arr

            if ch in ("r", "R"):
                print("\n  ↩ 重新对准，调整好后再按 SPACE...\n")
                # 继续同一 while 循环即可，无需重启线程

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def save_results(
    out_dir: Path, side: str, phase_cols: list[np.ndarray], R: np.ndarray
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calib_rotation_{side}_{timestamp}.npz"

    save_dict: dict[str, np.ndarray] = {}
    for i, (ph, cols) in enumerate(zip(PHASES, phase_cols)):
        save_dict[f"{side}_phase{i + 1}_{ph['tag']}_cols"] = cols
    save_dict[f"R_wrist_ctrl_{side}"] = R
    save_dict[f"quat_wrist_ctrl_{side}"] = Rotation.from_matrix(R).as_quat()  # xyzw

    np.savez(path, **save_dict)
    return path


# ---------------------------------------------------------------------------
# 打印结果
# ---------------------------------------------------------------------------
def print_result(side: str, phase_cols: list[np.ndarray], R: np.ndarray) -> None:
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # xyzw
    err = rotation_error_deg(phase_cols)

    # 三种常用欧拉角顺序
    euler_xyz = rot.as_euler("xyz", degrees=True)
    euler_zyx = rot.as_euler("zyx", degrees=True)
    euler_zyz = rot.as_euler("zyz", degrees=True)

    print(f"\n{'═' * 60}")
    print(f"  标定结果  —  {side.upper()} 手")
    print(f"{'═' * 60}")
    print(f"  R_wrist_ctrl_{side}:")
    for row in R:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print("\n  四元数 [qx, qy, qz, qw]:")
    print(f"    [{q[0]:+.6f}  {q[1]:+.6f}  {q[2]:+.6f}  {q[3]:+.6f}]")
    print("\n  欧拉角（度）：")
    print(
        f"    XYZ  :  rx={euler_xyz[0]:+7.2f}°  ry={euler_xyz[1]:+7.2f}°  rz={euler_xyz[2]:+7.2f}°"
    )
    print(
        f"    ZYX  :  rz={euler_zyx[0]:+7.2f}°  ry={euler_zyx[1]:+7.2f}°  rx={euler_zyx[2]:+7.2f}°"
    )
    print(
        f"    ZYZ  :  rz={euler_zyz[0]:+7.2f}°  ry={euler_zyz[1]:+7.2f}°  rz={euler_zyz[2]:+7.2f}°"
    )
    print(f"\n  正交性误差（三轴对齐质量）：{err:.2f}°", end="")
    if err < 5.0:
        print("  ✓ 良好")
    elif err < 15.0:
        print("  ⚠ 一般（建议重新采集）")
    else:
        print("  ✗ 较差（三轴对齐误差过大，请重新采集）")


# ---------------------------------------------------------------------------
# 单侧标定流程
# ---------------------------------------------------------------------------
def calibrate_one_side(buf: ControllerBuffer, side: str, out_dir: Path) -> None:
    assert side in ("left", "right")
    print(f"\n{'━' * 60}")
    print(f"  开始标定：{side.upper()} 手")
    print("  共 3 个姿势，按 SPACE 采样，按 r 重采当前姿势")
    print(f"{'━' * 60}")

    phase_cols: list[np.ndarray] = []
    for phase in PHASES:
        # 把已采集的 Phase 均值列向量传入，用于实时角度显示
        prev_mean_cols = [arr.mean(axis=0) for arr in phase_cols]
        cols = collect_phase(buf, phase, side, prev_mean_cols)
        phase_cols.append(cols)

    R = compute_R_wrist_ctrl(phase_cols, side=side)
    print_result(side, phase_cols, R)

    path = save_results(out_dir, side, phase_cols, R)
    print(f"\n  已保存到：{path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="right",
        help="标定哪只手（默认：right）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "calib_data" / "controller",
        help="输出目录（默认：calib_data/controller/，与此脚本同级）",
    )
    args = ap.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    print("=" * 60)
    print("  手腕→手柄旋转矩阵标定")
    print(f"  标定侧：{args.side}")
    print("=" * 60)
    print()

    try:
        with ControllerBuffer() as buf:
            print("等待手柄连接（最多 30 秒）...", end="", flush=True)
            if not buf.wait_ready(timeout=30.0):
                print("\n错误：30 秒内未收到有效手柄数据", file=sys.stderr)
                return 1
            print(" ✓")

            for side in sides:
                calibrate_one_side(buf, side, args.out)
                if len(sides) > 1 and side != sides[-1]:
                    print("\n换另一只手，准备好后按 SPACE 继续...", end="", flush=True)
                    while getch() != " ":
                        pass

    except KeyboardInterrupt:
        print("\n\n用户中断。")
        return 1

    print("\n标定完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
