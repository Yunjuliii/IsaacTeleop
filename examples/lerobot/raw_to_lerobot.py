# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a raw recording from ``record_cameras_gstreamer.py`` into a LeRobot dataset.

The recorder writes one MP4 per camera plus two tables: the camera frames with
their capture times, and the pose stream with its own. This pairs them up and
hands the result to LeRobot. It runs offline, so nothing here is on a deadline
and a mistake costs a re-run rather than a re-recording.

The MP4s are not re-encoded. LeRobot's ``save_episode`` accepts an
already-encoded file for each camera when its streaming encoder hands one over,
so a shim stands in for that encoder and returns the recorder's own files. That
avoids a second generation of lossy compression, and it is why the per-frame
image data passed to ``add_frame`` is a shared dummy array: the pixels are
already on disk, only the shape is checked.

Pose alignment
--------------
Each camera frame is paired with the pose stream sample nearest to

    reference_time(frame) - lag_frames / fps

``reference_time`` defaults to when the frame arrived in the recorder, which is
what ``record_cameras.py`` compared against, so the same ``camera_lag_frames``
means the same thing it always did. ``--lag-reference stamp`` measures from the
camera's own capture time instead; the two differ by transport latency, which
was measured at about 10 ms for the head cameras and 43 ms for the wrist, so
the same lag value does not carry over between them.

A frame with no pose sample close enough is an error rather than a dropped row.
Dropping is what the single-pass recorder did, but it could only do that
because it dropped the video frame in the same breath. Here the MP4 already
exists, so removing row N would leave it pointing at video frame N+1 and
silently corrupt every later row.

Usage::

    python3 raw_to_lerobot.py --raw ~/dataset_0904_raw
    python3 raw_to_lerobot.py --raw ~/dataset_0904_raw --out ~/dataset_0904 --lag-frames 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from record_cameras import (
    CONTROLLER_POSE_DIM,
    CONTROLLER_POSE_NAMES,
    HAND_POLL_HZ,
    HEAD_POSE_DIM,
    HEAD_POSE_NAMES,
    NVENC_MIN_GOP,
    hand_feature_dim,
    hand_joint_names,
)


class PreEncodedVideoShim:
    """Stands in for LeRobot's streaming encoder and returns existing MP4s.

    Implements the surface ``DatasetWriter`` uses: ``start_episode``,
    ``feed_frame``, ``finish_episode``, ``cancel_episode`` and
    ``_dropped_frames``. Frames fed to it are discarded, because the encoded
    file already exists.

    LeRobot *moves* the file it is handed and then deletes the directory that
    held it, so each episode is copied into a throwaway directory per camera
    first. Handing over the recording itself would destroy it.
    """

    def __init__(self, tmp_root: Path):
        self._tmp_root = tmp_root
        self._sources: dict[str, Path] = {}
        self._stats: dict[str, dict | None] = {}
        self._staged: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode = 0

    def stage(self, video_key: str, source: Path, stats: dict | None) -> None:
        """Register the file and stats to hand over for the next episode."""
        self._sources[video_key] = source
        self._stats[video_key] = stats

    def start_episode(self, video_keys, temp_dir, depth_video_keys=None) -> None:
        self._staged = {}
        for key in video_keys:
            # Own parent directory per camera: LeRobot rmtree()s the parent
            # after moving the file out of it.
            dest_dir = self._tmp_root / f"ep{self._episode:06d}" / key
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "episode.mp4"
            shutil.copy2(self._sources[key], dest)
            self._staged[key] = dest

    def feed_frame(self, video_key: str, image) -> None:
        return None

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        result = {k: (v, self._stats.get(k)) for k, v in self._staged.items()}
        self._staged = {}
        self._episode += 1
        return result

    def cancel_episode(self) -> None:
        for path in self._staged.values():
            shutil.rmtree(path.parent, ignore_errors=True)
        self._staged = {}

    def close(self) -> None:
        """Called by ``DatasetWriter.finalize`` through ``flush_pending_videos``."""
        self.cancel_episode()


def video_stats(path: Path) -> dict | None:
    """Per-channel statistics over an encoded video, as LeRobot computes them.

    Mirrors the streaming encoder's own stats path exactly, including the
    downsample, so a converted dataset normalises the same way one recorded
    straight through LeRobot would. Values stay on the 0-255 scale; the caller
    divides, matching what ``save_episode`` does with encoder-supplied stats.
    """
    import av
    from lerobot.datasets.compute_stats import (
        RunningQuantileStats,
        auto_downsample_height_width,
    )

    tracker = RunningQuantileStats()
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
            img = auto_downsample_height_width(img)
            channels = img.shape[0]
            tracker.update(img.transpose(1, 2, 0).reshape(-1, channels))
            count += 1
    return tracker.get_statistics() if count >= 2 else None


def match_poses(
    targets: np.ndarray, pose_t: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest pose sample per target time. Returns (indices, absolute errors)."""
    if pose_t.size == 0:
        return np.zeros(targets.size, dtype=int), np.full(targets.size, np.inf)
    right = np.searchsorted(pose_t, targets)
    left = np.clip(right - 1, 0, pose_t.size - 1)
    right = np.clip(right, 0, pose_t.size - 1)
    take_right = np.abs(pose_t[right] - targets) < np.abs(pose_t[left] - targets)
    idx = np.where(take_right, right, left)
    return idx, np.abs(pose_t[idx] - targets)


def load_episode(ep_dir: Path) -> dict:
    meta = json.loads((ep_dir / "episode.json").read_text())
    frames = pq.read_table(ep_dir / "frames.parquet").to_pydict()
    poses = pq.read_table(ep_dir / "poses.parquet").to_pydict()
    return {"meta": meta, "frames": frames, "poses": poses, "dir": ep_dir}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, required=True, help="raw recording root")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="LeRobot dataset root (default: the raw root without its '_raw' suffix)",
    )
    ap.add_argument(
        "--lag-frames",
        type=float,
        default=None,
        help="how far back to look for the pose matching each frame, in frames "
        "at the recording fps (default: the recorder's camera_lag_frames hint)",
    )
    ap.add_argument(
        "--lag-reference",
        choices=("recv", "stamp"),
        default="recv",
        help="measure the lag back from when the frame arrived ('recv', the "
        "default and what record_cameras.py compared against) or from the "
        "camera's own capture time ('stamp')",
    )
    ap.add_argument(
        "--ref-camera",
        default=None,
        help="camera whose time defines each row (default: the first camera, "
        "normally a head camera, which is what the policy is trained to see)",
    )
    ap.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="convert even when some frames have no pose within tolerance, "
        "using the nearest sample. Off by default because a silently "
        "mismatched pose is worse than a refused conversion.",
    )
    ap.add_argument("--state-dim", type=int, default=None)
    ap.add_argument("--action-dim", type=int, default=None)
    ap.add_argument("--robot-type", default=None)
    ap.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="episode directory names to convert (default: all, in order)",
    )
    ap.add_argument(
        "--data-file-size-mb",
        type=float,
        default=0.001,
        help="roll to a new data/video file above this size; the tiny default "
        "gives one file per episode so a crash cannot orphan earlier ones",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.json",
        help="JSON config, read only for state_dim / action_dim / robot_type",
    )
    args = ap.parse_args()

    raw = args.raw.expanduser()
    session_path = raw / "session.json"
    if not session_path.exists():
        print(f"ERROR: {session_path} not found; is {raw} a raw recording?", file=sys.stderr)
        return 1
    session = json.loads(session_path.read_text())

    cfg: dict = {}
    if args.config.exists():
        cfg = json.loads(args.config.read_text())

    def get(cli_val, key, default):
        return cli_val if cli_val is not None else cfg.get(key, default)

    out = args.out
    if out is None:
        name = raw.name[:-4] if raw.name.endswith("_raw") else raw.name + "_lerobot"
        out = raw.parent / name
    out = Path(out).expanduser()

    fps = int(session["fps"])
    cam_names = list(session["cameras"])
    use_hand = bool(session.get("use_hand", False))
    collect_hands = bool(session.get("collect_hands", False))
    hand_frame = session.get("hand_frame", "stage")
    lag_frames = (
        args.lag_frames
        if args.lag_frames is not None
        else float(session.get("camera_lag_frames_hint", 0.0))
    )
    lag_s = lag_frames / fps
    ref_camera = args.ref_camera or cam_names[0]
    if ref_camera not in cam_names:
        print(f"ERROR: --ref-camera {ref_camera!r} not in {cam_names}", file=sys.stderr)
        return 1
    # Same rule the single-pass recorder used: at least one and a half pose
    # poll intervals, so a target landing between two samples is not rejected.
    tol = max(0.5 / fps, 1.5 / HAND_POLL_HZ)

    state_dim = int(get(args.state_dim, "state_dim", 6))
    action_dim = int(get(args.action_dim, "action_dim", 6))
    robot_type = get(args.robot_type, "robot_type", "ego_centric")

    ep_dirs = (
        [raw / name for name in args.episodes]
        if args.episodes
        else sorted(p for p in raw.glob("episode_*") if p.is_dir())
    )
    ep_dirs = [p for p in ep_dirs if (p / "episode.json").exists()]
    if not ep_dirs:
        print(f"ERROR: no complete episodes under {raw}", file=sys.stderr)
        return 1

    if out.exists() and any(out.iterdir()):
        print(f"ERROR: {out} already exists and is not empty.", file=sys.stderr)
        print("  Remove it or pass --out elsewhere; this tool does not append.", file=sys.stderr)
        return 1

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolution = session.get("resolution", {})
    features: dict = {}
    for name in cam_names:
        w, h = resolution.get(name, [640, 480])
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"motor_{i}" for i in range(state_dim)],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"motor_{i}" for i in range(action_dim)],
    }
    if use_hand:
        features["observation.head_pose"] = {
            "dtype": "float32",
            "shape": (HEAD_POSE_DIM,),
            "names": HEAD_POSE_NAMES,
        }
        for side in ("left", "right"):
            features[f"observation.controller_{side}"] = {
                "dtype": "float32",
                "shape": (CONTROLLER_POSE_DIM,),
                "names": CONTROLLER_POSE_NAMES,
            }
        if collect_hands:
            hand_dim = hand_feature_dim(hand_frame)
            names = hand_joint_names(hand_frame)
            for side in ("left", "right"):
                features[f"observation.hand_{side}"] = {
                    "dtype": "float32",
                    "shape": (hand_dim,),
                    "names": names,
                }

    tmp_root = out.parent / f".{out.name}_convert_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=f"teleop/{robot_type}",
        fps=fps,
        features=features,
        root=out,
        use_videos=True,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264", crf=23, g=NVENC_MIN_GOP
        ),
        data_files_size_in_mb=args.data_file_size_mb,
        video_files_size_in_mb=args.data_file_size_mb,
        metadata_buffer_size=1,
    )
    shim = PreEncodedVideoShim(tmp_root)
    dataset.writer._streaming_encoder = shim

    print(f"Raw     : {raw}  ({len(ep_dirs)} episode(s))")
    print(f"Output  : {out}")
    print(
        f"Lag     : {lag_frames:g} frames ({lag_s * 1e3:.0f} ms) back from "
        f"{args.lag_reference} time of {ref_camera}"
    )
    print(f"Hands   : {hand_frame if collect_hands else 'none'}\n")

    zeros_state = np.zeros(state_dim, dtype=np.float32)
    zeros_action = np.zeros(action_dim, dtype=np.float32)
    dummy_images = {
        name: np.zeros(features[f"observation.images.{name}"]["shape"], dtype=np.uint8)
        for name in cam_names
    }

    converted = 0
    skipped: list[str] = []
    for ep_dir in ep_dirs:
        ep = load_episode(ep_dir)
        meta, frames, poses = ep["meta"], ep["frames"], ep["poses"]
        n = int(meta["frames"])
        task = meta["task"]

        bad = {k: v for k, v in meta.get("mp4_frames", {}).items() if v != n}
        if bad:
            print(f"{ep_dir.name}: SKIPPED, video/table mismatch {bad} vs {n} rows")
            skipped.append(ep_dir.name)
            continue
        missing_files = [c for c in cam_names if not (ep_dir / f"{c}.mp4").exists()]
        if missing_files:
            print(f"{ep_dir.name}: SKIPPED, missing video for {missing_files}")
            skipped.append(ep_dir.name)
            continue

        if use_hand:
            pose_t = np.asarray(poses.get("t", []), dtype=np.float64)
            ref = np.asarray(frames[f"{args.lag_reference}.{ref_camera}"], dtype=np.float64)
            idx, err = match_poses(ref[:n] - lag_s, pose_t, tol)
            over = int((err > tol).sum())
            if over and not args.allow_unmatched:
                worst = float(err[err > tol].max())
                print(
                    f"{ep_dir.name}: SKIPPED, {over}/{n} frame(s) have no pose "
                    f"within {tol * 1e3:.0f} ms (worst {worst * 1e3:.0f} ms). "
                    f"Re-run with --allow-unmatched to use the nearest sample."
                )
                skipped.append(ep_dir.name)
                continue
            if over:
                print(
                    f"{ep_dir.name}: WARNING, {over}/{n} frame(s) matched beyond "
                    f"{tol * 1e3:.0f} ms tolerance"
                )
            arrays = {
                key: np.asarray(poses[key], dtype=np.float32)
                for key in poses
                if key != "t"
            }

        for name in cam_names:
            shim.stage(
                f"observation.images.{name}",
                ep_dir / f"{name}.mp4",
                video_stats(ep_dir / f"{name}.mp4"),
            )

        for i in range(n):
            row: dict = {"task": task}
            for name in cam_names:
                row[f"observation.images.{name}"] = dummy_images[name]
            row["observation.state"] = zeros_state
            row["action"] = zeros_action
            if use_hand:
                j = int(idx[i])
                row["observation.head_pose"] = arrays["head_pose"][j]
                row["observation.controller_left"] = arrays["controller_left"][j]
                row["observation.controller_right"] = arrays["controller_right"][j]
                if collect_hands:
                    row["observation.hand_left"] = arrays["hand_left"][j]
                    row["observation.hand_right"] = arrays["hand_right"][j]
            dataset.add_frame(row)

        dataset.save_episode()
        converted += 1
        extra = ""
        if use_hand and pose_t.size:
            extra = f", pose match median {np.median(err) * 1e3:.1f} ms"
        print(f"{ep_dir.name}: {n} frames -> episode {converted - 1}{extra}")

    dataset.finalize()
    shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nConverted {converted} episode(s) into {out}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")
    return 0 if converted else 1


if __name__ == "__main__":
    sys.exit(main())
