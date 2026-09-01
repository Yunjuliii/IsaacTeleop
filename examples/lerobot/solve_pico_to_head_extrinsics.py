#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
求 Pico 左相机光心 到 head_left / head_right 鱼眼相机光心之间的刚性变换。

前提：Pico 头显跟这两个鱼眼相机在采集期间是刚性固定的，三者同时（或至少
在设备都没挪动的前提下）看向同一个固定不动的标定板/marker。

--board charuco（推荐）用 ChArUco 板代替单张 ArUco marker：单 marker 只有 4 个
共面角点，solvePnP 解出来的深度（沿 marker 法线方向）噪声很大——这台机器上
实测过，只是换个位置摆 marker，两次标定结果能差 1-2 cm。ChArUco 板一次给几十
个跨越整块板子的角点，深度约束好得多。板子用 generate_charuco_board.py 生成，
--board 两种模式的参数要跟 capture_head_aruco.py 采集时用的一致。

原理
----
每个相机各自对同一块标定板求解 PnP，得到标定板在该相机坐标系下的位姿
T_board->cam（板上的点变换到相机坐标系；用单 marker 时 T_board 就是 T_marker）：

    p_cam = R · p_marker + t

marker 是三者共同的参考系，于是：

    T_pico->head = T_marker->head · inv(T_marker->pico)

展开成 R, t：

    R_result = R_head · R_pico^T
    t_result = t_head - R_result · t_pico

p_head = R_result · p_pico + t_result，即把一个在 Pico 相机坐标系下的点变换到
head 相机坐标系。

数据来源
--------
  Pico 一帧 + 内参 : rcvpic.py 自动存的
      calib_data/pico_camera/pico_frame.png
      calib_data/pico_camera/left_intrinsics.npz
  head_left/right 图像 : capture_head_aruco.py 存的
      calib_data/pico_to_head/head_left_XXXX.png
      calib_data/pico_to_head/head_right_XXXX.png
  head 内参（去畸变用）: calibrate_fisheye.py 存的
      calib_data/fisheye_intrinsics/{left,right}_intrinsics.npz

注意：Pico 传来的内参假定图像已经是校正过的（没有额外畸变系数随包传来），
按针孔模型、畸变系数记 0 处理；head 鱼眼图像会先用 K,D 做去畸变，再在去畸变
图上跑 ArUco 检测和 PnP。

用法
----
    # 单 marker（旧方式，噪声较大，见上面的说明）
    python3 solve_pico_to_head_extrinsics.py --board marker --frame-index 0 \\
        --dict 5X5_100 --marker-id 0 --marker-length 0.15

    # ChArUco 板（推荐）：几何参数要跟 generate_charuco_board.py /
    # capture_head_aruco.py 用的一致
    python3 solve_pico_to_head_extrinsics.py --board charuco \\
        --squares-x 7 --squares-y 5 --square-size 0.035 --marker-size 0.026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from charuco_common import (
    ARUCO_DICTS,
    MIN_CHARUCO_CORNERS,
    add_charuco_args,
    build_board,
    detect_charuco_pose,
)

CALIB_DIR = Path(__file__).parent / "calib_data"


def detect_marker_pose(
    img_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    aruco_dict,
    marker_id: int,
    marker_length: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (R 3x3, t 3,) —— marker 在该相机坐标系下的位姿。检测不到就报错退出。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or marker_id not in ids.flatten():
        print(f"ERROR: {label} 没检测到 ID={marker_id} 的 marker", file=sys.stderr)
        sys.exit(1)

    idx = list(ids.flatten()).index(marker_id)

    # cv2.aruco.estimatePoseSingleMarkers() 在新版 OpenCV（这台机器的 .venv 是
    # 4.13.0）里被移除了，手动用 solvePnP 等价实现：marker 四个角点在自身坐标系
    # 下按 detectMarkers 的角点顺序（左上、右上、右下、左下）摆成 z=0 平面。
    half = marker_length / 2.0
    obj_points = np.array(
        [
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ],
        dtype=np.float64,
    )

    ok, rvec, tvec = cv2.solvePnP(obj_points, corners[idx][0], K, dist)
    if not ok:
        print(f"ERROR: {label} solvePnP 求解失败", file=sys.stderr)
        sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    print(f"  {label}: t = {t}")
    return R, t


def detect_board_pose(
    img_bgr: np.ndarray, board, K: np.ndarray, dist: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """ChArUco 等价物：返回 (R 3x3, t 3,)。检测不到/角点不够就报错退出。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    R, t, n = detect_charuco_pose(gray, board, K, dist)
    if R is None:
        print(
            f"ERROR: {label} 只检测到 {n} 个 ChArUco 角点（需要 >= "
            f"{MIN_CHARUCO_CORNERS}），标定板没完全入镜或角度太偏",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  {label}: t = {t}  ({n} 个角点)")
    return R, t


def compose(
    R_a: np.ndarray, t_a: np.ndarray, R_b_inv: np.ndarray, t_b_inv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T_result = T_a . T_b_inv，两个都作用在同一个 marker 参考系上。"""
    R = R_a @ R_b_inv
    t = t_a + R_a @ t_b_inv
    return R, t


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="capture_head_aruco.py 存的第几对 head_left/right 图像",
    )
    ap.add_argument(
        "--pico-frame", type=Path, default=CALIB_DIR / "pico_camera" / "pico_frame.png"
    )
    ap.add_argument(
        "--pico-intrinsics",
        type=Path,
        default=CALIB_DIR / "pico_camera" / "left_intrinsics.npz",
    )
    ap.add_argument("--head-dir", type=Path, default=CALIB_DIR / "pico_to_head")
    ap.add_argument(
        "--head-left-intrinsics",
        type=Path,
        default=CALIB_DIR / "fisheye_intrinsics" / "left_intrinsics.npz",
    )
    ap.add_argument(
        "--head-right-intrinsics",
        type=Path,
        default=CALIB_DIR / "fisheye_intrinsics" / "right_intrinsics.npz",
    )
    ap.add_argument(
        "--board",
        choices=("marker", "charuco"),
        default="marker",
        help="'marker': single ArUco marker (legacy, noisy depth). "
        "'charuco': ChArUco board (recommended, see module docstring). "
        "Must match what capture_head_aruco.py captured.",
    )
    ap.add_argument(
        "--marker-id", type=int, default=0, help="only used with --board marker"
    )
    ap.add_argument(
        "--marker-length",
        type=float,
        default=None,
        help="打印后实测的 marker 黑色方块边长，单位米。--board marker 时必填",
    )
    add_charuco_args(
        ap
    )  # --dict shared; --squares-x/y, --square/marker-size are charuco-only
    ap.add_argument(
        "--out", type=Path, default=CALIB_DIR / "pico_to_head" / "extrinsics.npz"
    )
    args = ap.parse_args()

    if args.board == "marker":
        if args.marker_length is None:
            print("ERROR: --board marker 需要 --marker-length", file=sys.stderr)
            return 1
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
        board = None
    else:
        board, aruco_dict = build_board(args)

    # ---------------------------------------------------------------- Pico
    pico_img = cv2.imread(str(args.pico_frame))
    if pico_img is None:
        print(
            f"ERROR: 读不到 {args.pico_frame}，先用 rcvpic.py 截一帧（要能看到标定板）",
            file=sys.stderr,
        )
        return 1
    pico_data = np.load(args.pico_intrinsics)
    K_pico = pico_data["K"]
    dist_pico = np.zeros(5)  # Pico 传来的图像假定已校正，无额外畸变

    # ------------------------------------------------------------ head L/R
    left_path = args.head_dir / f"head_left_{args.frame_index:04d}.png"
    right_path = args.head_dir / f"head_right_{args.frame_index:04d}.png"
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    if left_img is None or right_img is None:
        print(
            f"ERROR: 读不到 {left_path} 或 {right_path}，先用 capture_head_aruco.py 截图",
            file=sys.stderr,
        )
        return 1

    left_intr = np.load(args.head_left_intrinsics)
    right_intr = np.load(args.head_right_intrinsics)

    def scale_K(K, calib_size, actual_size):
        """calib_size / actual_size 都是 (w, h)。fisheye_intrinsics 的 K 是在
        calib_size 下标定出来的；如果相机现在跑的输出分辨率变了（比如换成了
        原生 2560x1984），直接把这个 K 套到新尺寸的图上会整个错位——nvvidconv
        做的是等比单独缩放（非裁剪），所以 fx/cx 按宽度比例、fy/cy 按高度比例
        线性缩放即可，不用重新标定。D 是无量纲的，原样复用。"""
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
        print(
            f"  内参标定分辨率 {cw}x{ch} 与图像实际分辨率 {aw}x{ah} 不一致，"
            f"已按比例缩放 K（D 不变）"
        )
        return K

    def undistort(img, K, D, calib_size):
        h, w = img.shape[:2]
        K = scale_K(K, tuple(calib_size), (w, h))
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR), new_K

    left_undist, K_left_new = undistort(
        left_img, left_intr["K"], left_intr["D"], left_intr["image_size"]
    )
    right_undist, K_right_new = undistort(
        right_img, right_intr["K"], right_intr["D"], right_intr["image_size"]
    )
    dist_zero = np.zeros(5)

    # ----------------------------------------------------------- 位姿求解
    print(
        f"检测各相机下{'标定板' if args.board == 'charuco' else 'marker'}"
        f"的位姿 (board -> camera):"
    )
    if args.board == "marker":
        R_pico, t_pico = detect_marker_pose(
            pico_img,
            K_pico,
            dist_pico,
            aruco_dict,
            args.marker_id,
            args.marker_length,
            "Pico",
        )
        R_left, t_left = detect_marker_pose(
            left_undist,
            K_left_new,
            dist_zero,
            aruco_dict,
            args.marker_id,
            args.marker_length,
            "head_left",
        )
        R_right, t_right = detect_marker_pose(
            right_undist,
            K_right_new,
            dist_zero,
            aruco_dict,
            args.marker_id,
            args.marker_length,
            "head_right",
        )
    else:
        R_pico, t_pico = detect_board_pose(pico_img, board, K_pico, dist_pico, "Pico")
        R_left, t_left = detect_board_pose(
            left_undist, board, K_left_new, dist_zero, "head_left"
        )
        R_right, t_right = detect_board_pose(
            right_undist, board, K_right_new, dist_zero, "head_right"
        )

    # T_pico->cam_x = T_marker->cam_x . inv(T_marker->pico)
    R_pico_inv = R_pico.T
    t_pico_inv = -R_pico.T @ t_pico

    R_pl, t_pl = compose(R_left, t_left, R_pico_inv, t_pico_inv)
    R_pr, t_pr = compose(R_right, t_right, R_pico_inv, t_pico_inv)

    def to_4x4(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    T_pico_left = to_4x4(R_pl, t_pl)
    T_pico_right = to_4x4(R_pr, t_pr)

    print("\n========== T_pico -> head_left ==========")
    print(T_pico_left)
    print(f"平移 (m): {t_pl}")

    print("\n========== T_pico -> head_right ==========")
    print(T_pico_right)
    print(f"平移 (m): {t_pr}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, T_pico_to_head_left=T_pico_left, T_pico_to_head_right=T_pico_right
    )
    print(f"\n已保存: {args.out}")

    if args.board == "marker":
        print(
            "\n注意：这是单帧单张 ArUco marker 的 PnP 解，深度方向噪声较大——"
            "建议换几个 marker 摆放位置/角度多测几次，看结果是否稳定收敛，"
            "不稳定的话需要多次取平均，或者改用 --board charuco（推荐，见本脚本"
            "文档开头的说明）。"
        )
    else:
        print(
            "\n注意：这仍然是单帧单次 PnP 解——ChArUco 板把深度方向的噪声压低了"
            "很多，但不是零。建议换几个位置/角度多测几次，看 T_pico->head_left/"
            "right 是否稳定收敛（尤其是两次之间 head_left 和 head_right 的相对"
            "基线应该完全不变，这是最灵敏的一致性检查），不稳定的话取平均。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
