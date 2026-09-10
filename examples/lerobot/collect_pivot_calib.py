#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
转轴标定（Pivot Calibration）——求手腕相对于手柄的平移偏移 p_wrist_ctrl。

原理
----
将手腕上某个骨性标志点（如尺骨茎突）抵在固定锚点上，绕该点自由转动手腕。
此时锚点在世界坐标系中的位置是常数：

    R_ctrl_stage(t) · p_wrist_ctrl + p_ctrl_stage(t) = c  （常数）

对任意两帧 (0, i) 做差，消掉未知的 c：

    (R_ctrl_i − R_ctrl_0) · p_wrist_ctrl = p_ctrl_0 − p_ctrl_i

将 N 帧数据堆叠成超定线性方程组，最小二乘求解 p_wrist_ctrl（3D 向量，单位 m）。

操作步骤
--------
1. 把手腕上的骨性点（尺骨茎突或手腕背侧中心）抵在桌角、墙角等固定锚点
2. 按 SPACE 开始录制
3. 在锚点不离开的前提下，向各方向尽量多转动手腕（前后、左右、旋转都要有）
4. 按 SPACE 停止
5. 脚本求解并打印 p_wrist_ctrl，左右手分别做

质量评估
--------
- 残差 RMS（mm）：越小越好，<3 mm 为良好
- 转动多样性：矩阵 A 的条件数，越小越好（<50 为良好）

输出
----
  calib_pivot_{side}_{timestamp}.npz
  ├── p_wrist_ctrl_{side}         — (3,) float64，单位 m
  ├── pivot_stage_{side}          — (3,) float64，锚点在世界坐标系中的位置估计
  ├── residual_rms_mm_{side}      — 标量，残差 RMS（mm）
  ├── R_ctrl_frames_{side}        — (N, 3, 3) 所有帧的旋转矩阵
  └── p_ctrl_frames_{side}        — (N, 3) 所有帧的手柄位置

用法
----
    python3 collect_pivot_calib.py              # 右手（默认）
    python3 collect_pivot_calib.py --side left
    python3 collect_pivot_calib.py --side both
    python3 collect_pivot_calib.py --out ~/calib
"""

from __future__ import annotations

import argparse
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
# 路径常量
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

POLL_HZ = 60  # 手柄轮询频率


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
# 手柄数据缓冲（无 Manus 插件）
# ---------------------------------------------------------------------------
class ControllerBuffer:
    """TeleopSession 的轻量包装，只暴露手柄 grip pose。

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
            app_name="PivotCalib",
            pipeline=pipeline,
            plugins=[],
        )
        self._session = TeleopSession(self._cfg)
        self._lock = threading.Lock()
        self._left = np.zeros(8, dtype=np.float32)
        self._right = np.zeros(8, dtype=np.float32)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

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
                pass
            else:
                raise

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
def solve_pivot(R_frames: np.ndarray, p_frames: np.ndarray) -> dict:
    """
    给定 N 帧手柄位姿，最小二乘求解 p_wrist_ctrl。

    Parameters
    ----------
    R_frames : (N, 3, 3)  每帧的旋转矩阵 R_ctrl_stage
    p_frames : (N, 3)     每帧的手柄位置 p_ctrl_stage

    Returns
    -------
    dict with keys:
        p_wrist_ctrl   : (3,) 手柄坐标系下手腕偏移（m）
        pivot_stage    : (3,) 锚点在世界坐标系中的估计位置（m）
        residual_rms_mm: float，残差 RMS（mm）
        condition_number: float，矩阵 A 的条件数
    """
    N = len(R_frames)
    if N < 10:
        raise ValueError(f"帧数太少（{N}），至少需要 10 帧")

    R0 = R_frames[0]  # 参考帧旋转
    p0 = p_frames[0]  # 参考帧位置

    # 构造超定线性方程组：A · p_wrist_ctrl = b
    # 每帧贡献 3 行：(R_i − R_0) · p = p_0 − p_i
    A = (R_frames[1:] - R0).reshape(-1, 3)  # (3(N-1), 3)
    b = (p0 - p_frames[1:]).reshape(-1)  # (3(N-1),)

    p_wrist_ctrl, residuals_sq, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    condition_number = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")

    # 残差 RMS（mm）
    b_pred = A @ p_wrist_ctrl
    rms_m = float(np.sqrt(np.mean((b_pred - b) ** 2)))

    # 锚点世界坐标（平均值）
    pivots = R_frames @ p_wrist_ctrl + p_frames  # (N, 3)
    pivot_stage = pivots.mean(axis=0)

    return {
        "p_wrist_ctrl": p_wrist_ctrl,
        "pivot_stage": pivot_stage,
        "residual_rms_mm": rms_m * 1000.0,
        "condition_number": condition_number,
    }


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
def collect_one_side(
    buf: ControllerBuffer, side: str, avg_frames: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    手动触发采集一侧手的转轴标定数据。

    每次按 SPACE：手腕静止后触发，对当前姿态连续采 avg_frames 帧取均值，
    作为一个样本点。按 e 结束采集并求解。

    Parameters
    ----------
    avg_frames : 每次按键采集并平均的帧数（默认 10 帧 ≈ 167ms），用于降噪

    Returns
    -------
    R_frames : (N, 3, 3)
    p_frames : (N, 3)
    """
    assert side in ("left", "right")

    print(f"\n{'━' * 60}")
    print(f"  转轴标定  —  {side.upper()} 手")
    print(f"{'━' * 60}")
    print(f"""
  操作说明：
    1. 找到手腕上的骨性点（推荐：尺骨茎突，即手腕小拇指侧的突出骨点）
    2. 将该点抵在桌角、墙角或其他固定锚点上，保持接触
    3. 转到一个新姿态，手腕静止后按 SPACE 采样（每次采 {avg_frames} 帧均值）
    4. 换一个姿态，再按 SPACE — 重复，覆盖各方向：
         • 腕屈 / 腕伸（前后弯）
         • 尺偏 / 桡偏（左右偏）
         • 旋前 / 旋后（前臂旋转）
    5. 至少采 15 个姿态，按 e 结束并求解
       （随时按 u 撤销上一个样本）
""")

    R_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []

    print("  SPACE = 采样    e = 结束求解    u = 撤销上一个\n")

    while True:
        n = len(R_list)
        print(f"  [{n:3d} 个姿态] 就位后按 SPACE，结束按 e ...", end="", flush=True)

        ch = getch()
        print()  # 换行

        if ch == "e":
            if n < 10:
                print(f"  ⚠ 样本数太少（{n}），至少需要 10 个，继续采集")
                continue
            break

        if ch == "u":
            if R_list:
                R_list.pop()
                p_list.pop()
                print(f"  ↩ 已撤销，当前 {len(R_list)} 个姿态")
            else:
                print("  没有可撤销的样本")
            continue

        if ch != " ":
            continue

        # 采 avg_frames 帧，取均值
        qs: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for _ in range(avg_frames):
            left, right = buf.latest()
            pose = right if side == "right" else left
            if pose[7] > 0.5:
                qs.append(pose[3:7].astype(np.float64))
                ps.append(pose[0:3].astype(np.float64))
            time.sleep(1.0 / POLL_HZ)

        if len(qs) < avg_frames // 2:
            print(f"  ✗ 有效帧不足（{len(qs)}/{avg_frames}），跳过，请检查手柄追踪")
            continue

        # 平均位置
        p_mean = np.mean(ps, axis=0)
        # 平均旋转（用 scipy Rotation.mean）
        R_mean = Rotation.from_quat(qs).mean().as_matrix()

        R_list.append(R_mean)
        p_list.append(p_mean)
        print(
            f"  ✓ 采样 {n + 1:3d}  pos=({p_mean[0] * 1000:+.1f}, "
            f"{p_mean[1] * 1000:+.1f}, {p_mean[2] * 1000:+.1f}) mm"
        )

    return np.array(R_list), np.array(p_list)


# ---------------------------------------------------------------------------
# 打印 + 保存
# ---------------------------------------------------------------------------
def print_result(side: str, result: dict) -> None:
    p = result["p_wrist_ctrl"]
    rms = result["residual_rms_mm"]
    cond = result["condition_number"]

    print(f"\n{'═' * 60}")
    print(f"  标定结果  —  {side.upper()} 手")
    print(f"{'═' * 60}")
    print(f"  p_wrist_ctrl_{side}（手柄坐标系下的手腕偏移）：")
    print(f"    x = {p[0] * 1000:+8.2f} mm")
    print(f"    y = {p[1] * 1000:+8.2f} mm")
    print(f"    z = {p[2] * 1000:+8.2f} mm")
    print(f"    ‖p‖ = {np.linalg.norm(p) * 1000:.2f} mm")
    print()
    print(f"  残差 RMS：{rms:.2f} mm", end="")
    if rms < 3.0:
        print("  ✓ 良好")
    elif rms < 8.0:
        print("  ⚠ 一般（可接受，建议检查锚点是否有滑动）")
    else:
        print("  ✗ 较差（锚点可能滑动，或转动多样性不足）")

    print(f"  矩阵条件数：{cond:.1f}", end="")
    if cond < 50:
        print("  ✓ 转动多样性充足")
    elif cond < 200:
        print("  ⚠ 转动多样性一般（建议补充各方向转动）")
    else:
        print("  ✗ 转动多样性不足（可能只绕一个轴转）")


def save_results(
    out_dir: Path, side: str, result: dict, R_frames: np.ndarray, p_frames: np.ndarray
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calib_pivot_{side}_{timestamp}.npz"
    np.savez(
        path,
        **{
            f"p_wrist_ctrl_{side}": result["p_wrist_ctrl"],
            f"pivot_stage_{side}": result["pivot_stage"],
            f"residual_rms_mm_{side}": np.array(result["residual_rms_mm"]),
            f"condition_number_{side}": np.array(result["condition_number"]),
            f"R_ctrl_frames_{side}": R_frames,
            f"p_ctrl_frames_{side}": p_frames,
        },
    )
    return path


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
    print("  手腕平移偏移标定（转轴标定 / Pivot Calibration）")
    print(f"  标定侧：{args.side}")
    print("=" * 60)

    try:
        with ControllerBuffer() as buf:
            print("\n等待手柄连接（最多 30 秒）...", end="", flush=True)
            if not buf.wait_ready(timeout=30.0):
                print("\n错误：30 秒内未收到有效手柄数据", file=sys.stderr)
                return 1
            print(" ✓")

            for i, side in enumerate(sides):
                R_frames, p_frames = collect_one_side(buf, side)

                print("\n  计算中...", flush=True)
                result = solve_pivot(R_frames, p_frames)

                print_result(side, result)
                path = save_results(args.out, side, result, R_frames, p_frames)
                print(f"\n  已保存到：{path}")

                if i < len(sides) - 1:
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
