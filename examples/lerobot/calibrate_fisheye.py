#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
鱼眼相机内参标定 —— 对 collect_fisheye_calib.py 采集的棋盘格图像跑
cv2.fisheye.calibrate，左右两路分别标定（各自独立的单目内参，不是双目外参）。

用法
----
    python3 calibrate_fisheye.py --dir calib_data/fisheye_intrinsics/left --rows 10 --cols 7 --square-size 0.025
    python3 calibrate_fisheye.py --dir calib_data/fisheye_intrinsics/right --rows 10 --cols 7 --square-size 0.025

输出
----
  <dir>/../<dir名>_intrinsics.npz
  ├── K              — (3,3) 相机内参矩阵
  ├── D              — (4,1) 鱼眼畸变系数 k1,k2,k3,k4
  ├── rms            — 重投影误差 (像素)，理想 < 1.0
  └── image_size     — (w, h)
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True,
                    help="collect_fisheye_calib.py 输出的 left/ 或 right/ 目录")
    ap.add_argument("--rows", type=int, default=10, help="棋盘格行方向格子数")
    ap.add_argument("--cols", type=int, default=7, help="棋盘格列方向格子数")
    ap.add_argument("--square-size", type=float, default=0.025, help="格子实测边长，米")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="要排除的文件名（不含路径），比如某张图导致 Ill-conditioned "
                         "matrix 报错时用这个跳过它，不用物理删文件")
    args = ap.parse_args()

    pattern = (args.cols - 1, args.rows - 1)  # 内角点数

    objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(str(args.dir / "*.png")))
    if not images:
        print(f"ERROR: 没在 {args.dir} 找到 png 图像", file=sys.stderr)
        return 1

    image_size = None
    used, skipped = 0, 0
    for fname in images:
        if Path(fname).name in args.exclude:
            print(f"  排除 {Path(fname).name} (--exclude)")
            skipped += 1
            continue
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  跳过 {Path(fname).name}: 未检测到角点")
            skipped += 1
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (3, 3), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

    print(f"\n用于标定: {used} 张，跳过: {skipped} 张")
    if used < 10:
        print("WARNING: 有效图像少于 10 张，标定结果可能不可靠，建议补拍。", file=sys.stderr)

    N = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3)) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3)) for _ in range(N)]

    calib_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size,
            K, D, rvecs, tvecs, calib_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
    except cv2.error as e:
        print(f"ERROR: 标定失败（可能是某张图病态，比如离得太近/太偏）: {e}", file=sys.stderr)
        print("  找出并删掉可疑的那几张图，重新跑。", file=sys.stderr)
        return 1

    print(f"\nRMS 重投影误差: {rms:.4f} 像素  (理想 < 1.0，> 2.0 建议重标)")
    print(f"K =\n{K}")
    print(f"D (k1,k2,k3,k4) =\n{D.ravel()}")

    out_path = args.dir.parent / f"{args.dir.name}_intrinsics.npz"
    np.savez(out_path, K=K, D=D, rms=rms, image_size=np.array(image_size))
    print(f"\n已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
