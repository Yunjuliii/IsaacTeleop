# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ChArUco board helpers shared by generate_charuco_board.py, capture_head_aruco.py
and solve_pico_to_head_extrinsics.py.

Why ChArUco instead of a single ArUco marker
---------------------------------------------
A single planar ArUco marker gives solvePnP only 4 coplanar corners. That's
enough to constrain rotation well (edge directions are sharp), but translation
-- especially depth, along the marker's normal -- is weakly constrained: a
couple of pixels of corner noise can move the solved depth by 1-2 cm. Measured
on this rig: moving the marker between two otherwise-identical calibration runs
changed the derived head_left/head_right baseline (which is physically fixed,
so it should NOT change) by ~17 mm.

A ChArUco board packs many markers + an interleaved chessboard pattern into one
target. cv2.aruco.CharucoDetector interpolates sub-pixel chessboard corners
from whichever markers it sees, so a single solvePnP call uses 6-30+ points
spread across the board instead of 4 in one small square -- far better
depth conditioning, and it degrades gracefully if part of the board is
occluded or out of frame (unlike a single marker, which is all-or-nothing).

用法
----
Import build_board()/detect_charuco_pose() from capture/solve scripts, and add
the shared CLI flags with add_charuco_args(). Physical dimensions
(--square-size/--marker-size) must match what you actually printed --
generate_charuco_board.py's printout doubles as the source of truth for those.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50, "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50, "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}

# Below this many detected chessboard corners, matchImagePoints()/solvePnP()
# either fails outright or is barely better-conditioned than a single ArUco
# marker was -- the whole point of switching boards. 6 is a floor, not a
# target; more of the board in frame is always better.
MIN_CHARUCO_CORNERS = 6


def add_charuco_args(ap: argparse.ArgumentParser) -> None:
    """Adds the board-geometry flags shared by every ChArUco-consuming script.

    Defaults describe a 7x5-square board on A4/Letter with a healthy margin,
    printed at 3.5 cm/square -- adjust to whatever you actually print, and keep
    every script's flags in sync (they must describe the same physical board).
    """
    ap.add_argument("--dict", default="5X5_100", choices=sorted(ARUCO_DICTS),
                    help="ArUco dictionary the board's markers are drawn from")
    ap.add_argument("--squares-x", type=int, default=7,
                    help="chessboard squares across (default: 7)")
    ap.add_argument("--squares-y", type=int, default=5,
                    help="chessboard squares down (default: 5)")
    ap.add_argument("--square-size", type=float, default=0.035,
                    help="chessboard square edge length, metres, AS PRINTED "
                         "(default: 0.035 = 35 mm)")
    ap.add_argument("--marker-size", type=float, default=0.026,
                    help="ArUco marker edge length, metres, AS PRINTED. Must be "
                         "smaller than --square-size (default: 0.026 = 26 mm, "
                         "OpenCV's usual ~0.75x ratio)")


def build_board(args: argparse.Namespace) -> tuple["cv2.aruco.CharucoBoard", "cv2.aruco.Dictionary"]:
    if args.marker_size >= args.square_size:
        raise ValueError(
            f"--marker-size ({args.marker_size}) must be smaller than "
            f"--square-size ({args.square_size}) -- the marker has to fit "
            f"inside its chessboard square with a white border around it."
        )
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_size, args.marker_size, aruco_dict)
    return board, aruco_dict


def detect_charuco(gray: np.ndarray, board: "cv2.aruco.CharucoBoard"
                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (charuco_corners, charuco_ids); either is None if nothing found."""
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    return charuco_corners, charuco_ids


def detect_charuco_pose(gray: np.ndarray, board: "cv2.aruco.CharucoBoard",
                        K: np.ndarray, D: np.ndarray,
                        min_corners: int = MIN_CHARUCO_CORNERS
                        ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Detects the board and solves its pose in one shot.

    Returns (R 3x3, t 3,, n_corners). R/t are None when fewer than
    min_corners chessboard corners were found or solvePnP failed to converge
    -- n_corners is still returned so callers can report why.
    """
    charuco_corners, charuco_ids = detect_charuco(gray, board)
    n = 0 if charuco_corners is None else len(charuco_corners)
    if charuco_corners is None or n < min_corners:
        return None, None, n

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < min_corners:
        return None, None, n

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
    if not ok:
        return None, None, n

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten(), n
