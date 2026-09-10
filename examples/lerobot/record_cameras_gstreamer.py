# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record raw episodes from ROS 2 camera topics + Manus glove, without LeRobot.

This is the capture half of what ``record_cameras.py`` does in one pass. It
writes one MP4 per camera through GStreamer's hardware encoder plus two sidecar
tables, and stops there. Turning a recording into a LeRobot dataset is
``raw_to_lerobot.py``, run afterwards with no real-time deadline.

Why it is split that way
------------------------
Two problems in the single-pass recorder both came from doing dataset work
inside the capture loop.

Frames went missing because PyAV holds the GIL while it opens ``h264_nvenc``.
Opening four encoders took 775 ms with the GIL held for 522 ms of it, and every
later episode still froze the interpreter for 143-277 ms. The rclpy executor
cannot run during a freeze, and the camera subscriptions keep only the newest
message, so DDS discarded roughly 25 frames per camera at the start of every
episode. Nothing counted them: they never reached Python. GStreamer encodes on
its own threads, which takes the encoder off the GIL entirely.

Saving was slow because LeRobot's ``save_episode`` waits on its encoder threads
with a 120 second timeout, and that timeout was being hit. Here the MP4s are
finalised when recording stops, which was measured at 135 ms for four files.

The pose stream is written raw rather than matched to frames
------------------------------------------------------------
``record_cameras.py`` picks, for each camera frame, the pose from
``camera_lag_frames`` in the past, and drops the whole row when no pose is
close enough. That bakes one guess at the lag into the data forever. Here the
camera frames and the 60 Hz pose stream are written as two tables with their
own timestamps, and the converter does the alignment. Re-tuning the lag is then
a re-run of the converter instead of a re-recording, and no row is ever dropped
at capture time.

Layout produced::

    <root>/
      session.json                     # fps, cameras, encoder settings, hand frame
      episode_000000/
        head_left.mp4  head_right.mp4  wrist_left.mp4  wrist_right.mp4
        frames.parquet                 # one row per video frame, with capture times
        poses.parquet                  # the 60 Hz pose stream, with its own times
        episode.json                   # task, counts, per-camera gap report
      episode_000001/
      ...

Usage::

    python3 record_cameras_gstreamer.py            # uses config.json alongside
    python3 record_cameras_gstreamer.py --no-hand  # cameras only

Controls:  s = start episode · e = end · y = save · n = discard · Ctrl+C = quit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rclpy
from rclpy.executors import SingleThreadedExecutor

# Everything below comes from the single-pass recorder so the two share one
# copy of the capture logic. Importing it does not pull in LeRobot: those
# imports live inside its main().
from record_cameras import (
    CONTROLLER_POSE_DIM,
    CONTROLLER_POSE_NAMES,
    DEFAULT_CAMERAS,
    DEFAULT_CAMERA_LAG_FRAMES,
    DEFAULT_WRIST_SOURCE,
    HAND_FRAMES,
    HAND_FRAME_STAGE,
    HAND_HISTORY_SECONDS,
    HAND_POLL_HZ,
    HEAD_POSE_DIM,
    HEAD_POSE_NAMES,
    MANUS_PLUGIN_DIR,
    CameraBuffer,
    ManusHandBuffer,
    getch,
    hand_feature_dim,
    hand_joint_names,
)
from gst_mp4_writer import GstMp4Writer, available_encoder

# Frames buffered per camera between the ROS callback and the recording loop.
# Much deeper than the single-pass recorder's 8 because this loop consumes in
# order rather than skipping to the newest frame: depth here buys recovery from
# a scheduling hiccup instead of just bounding staleness. 60 frames is 2 s per
# camera, about 55 MB across four cameras at 640x480.
FRAME_QUEUE_DEPTH = 60

# Depth of the DDS-side queue. The single-pass recorder used 1, which is why a
# GIL freeze turned straight into lost frames: with only the newest message
# kept, anything arriving while Python was stuck was overwritten in the
# middleware. One second of slack costs little and makes the Python-side queue
# the only place a frame can be dropped, where it is counted.
QOS_DEPTH = 30


def _fixed_list(values: list[np.ndarray], dim: int) -> pa.Array:
    """Pack equal-length float32 vectors into a fixed-size list column."""
    if not values:
        return pa.FixedSizeListArray.from_arrays(
            pa.array(np.zeros(0, dtype=np.float32)), dim
        )
    flat = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values])
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)


def _gap_report(stamps: np.ndarray, fps: int) -> dict:
    """Count frames the camera never delivered, from its own capture times.

    A hole here is a frame the sensor pipeline lost before this process saw it.
    Reported rather than hidden, because the sidecar records true capture times
    while the MP4 is written at a constant rate: without this number, a gap
    would look like ordinary motion on playback.
    """
    if stamps.size < 2:
        return {"frames": int(stamps.size), "missing": 0, "gaps": 0, "max_gap_s": 0.0}
    d = np.diff(stamps)
    period = 1.0 / fps
    # Round each interval to whole frame periods; anything above one is a hole.
    steps = np.maximum(np.round(d / period).astype(int), 1)
    return {
        "frames": int(stamps.size),
        "missing": int(steps.sum() - steps.size),
        "gaps": int((steps > 1).sum()),
        "max_gap_s": float(d.max()),
    }


def _pose_counts(poses: list[tuple], t_start: float, t_end: float) -> dict:
    """Split the pose stream into pre-roll and in-episode, and rate the latter.

    Only samples inside the recording window belong in a rate. The pre-roll is
    real data the converter needs, but counting it against the episode's own
    duration inflates the reported rate by however much history the buffer held.
    """
    if not poses:
        return {"preroll": 0, "inside": 0, "rate": 0.0}
    t = np.asarray([p[0] for p in poses], dtype=np.float64)
    inside = int(((t >= t_start) & (t <= t_end)).sum())
    duration = t_end - t_start
    return {
        "preroll": int((t < t_start).sum()),
        "inside": inside,
        "rate": round(inside / duration, 1) if duration > 0 else 0.0,
    }


def record_episode(
    buf: CameraBuffer,
    hand_buf: ManusHandBuffer | None,
    writers: dict[str, GstMp4Writer],
    cam_names: list[str],
    fps: int,
    stop_evt: threading.Event,
    result: dict,
) -> None:
    """Pull synchronised frame groups until stopped, encoding and logging each.

    Fills ``result`` in place so the caller can read it after joining. Frames
    are consumed in arrival order, not newest-first: with the encoder already
    open there is no long stall to skip past, so a brief scheduling delay
    should be absorbed by the queue rather than turned into a silent hole.
    """
    frame_timeout = 2.0 / fps
    stamps: dict[str, list[float]] = {n: [] for n in cam_names}
    recvs: dict[str, list[float]] = {n: [] for n in cam_names}
    poses: list[tuple] = []
    # Zero, not "now": the first drain should take everything the buffer holds,
    # so the pose stream starts before the first frame. The converter looks
    # backwards in time to match a frame, and would otherwise find nothing to
    # match the opening frames against.
    last_pose_t = 0.0
    drops_before = sum(q.dropped for q in buf._queues.values())

    buf.flush()
    t_start = time.time()
    while not stop_evt.is_set():
        frames = buf.next_frames_fifo(cam_names, timeout=frame_timeout)
        if frames is None:
            continue
        for name, (img, stamp, t_recv) in frames.items():
            writers[name].write(img)
            stamps[name].append(stamp)
            recvs[name].append(t_recv)
        if hand_buf is not None:
            # Drain every tick. The buffer only holds HAND_HISTORY_SECONDS, so
            # a slower drain would lose samples outright.
            new = hand_buf.drain_since(last_pose_t)
            if new:
                poses.extend(new)
                last_pose_t = new[-1][0]

    t_end = time.time()
    if hand_buf is not None:
        # Samples that landed between the final tick and the stop key.
        new = hand_buf.drain_since(last_pose_t)
        poses.extend(new)

    result.update(
        n=len(stamps[cam_names[0]]) if cam_names else 0,
        stamps=stamps,
        recvs=recvs,
        poses=poses,
        drops=sum(q.dropped for q in buf._queues.values()) - drops_before,
        t_start=t_start,
        t_end=t_end,
    )


def write_episode(
    ep_dir: Path,
    result: dict,
    cam_names: list[str],
    fps: int,
    task: str,
    hand_frame: str,
    collect_hands: bool,
    mp4_frames: dict[str, int],
) -> dict:
    """Write frames.parquet, poses.parquet and episode.json. Returns the meta."""
    n = result["n"]
    cols: dict[str, pa.Array] = {
        "frame_index": pa.array(np.arange(n, dtype=np.int32)),
    }
    gaps = {}
    for name in cam_names:
        stamp = np.asarray(result["stamps"][name][:n], dtype=np.float64)
        recv = np.asarray(result["recvs"][name][:n], dtype=np.float64)
        cols[f"stamp.{name}"] = pa.array(stamp)
        cols[f"recv.{name}"] = pa.array(recv)
        gaps[name] = _gap_report(stamp, fps)
    pq.write_table(
        pa.table(cols), ep_dir / "frames.parquet", compression="snappy"
    )

    poses = result["poses"]
    hand_dim = hand_feature_dim(hand_frame)
    pose_cols = {
        "t": pa.array(np.asarray([p[0] for p in poses], dtype=np.float64)),
        "head_pose": _fixed_list([p[3] for p in poses], HEAD_POSE_DIM),
        "controller_left": _fixed_list([p[4] for p in poses], CONTROLLER_POSE_DIM),
        "controller_right": _fixed_list([p[5] for p in poses], CONTROLLER_POSE_DIM),
    }
    if collect_hands:
        pose_cols["hand_left"] = _fixed_list([p[1] for p in poses], hand_dim)
        pose_cols["hand_right"] = _fixed_list([p[2] for p in poses], hand_dim)
    pq.write_table(
        pa.table(pose_cols), ep_dir / "poses.parquet", compression="snappy"
    )

    duration = result["t_end"] - result["t_start"]
    pose_counts = _pose_counts(poses, result["t_start"], result["t_end"])
    meta = {
        "task": task,
        "fps": fps,
        "frames": n,
        "duration_s": round(duration, 3),
        "achieved_fps": round(n / duration, 2) if duration > 0 else 0.0,
        "wall_start": result["t_start"],
        "wall_end": result["t_end"],
        "cameras": cam_names,
        # Frames actually handed to each encoder. Compared against `frames` by
        # the converter: a mismatch means the MP4 and the table disagree on
        # what row N is, which no amount of later alignment can repair.
        "mp4_frames": mp4_frames,
        "queue_drops": result["drops"],
        "camera_gaps": gaps,
        "poses": len(poses),
        # Split out, because the total is not a rate. The first drain of an
        # episode deliberately takes everything the pose buffer still holds, up
        # to HAND_HISTORY_SECONDS before recording started, so the converter has
        # samples to look back at when it aligns the opening frames. Dividing
        # the total by the episode duration counts that pre-roll as if it
        # happened during the episode and reports a rate well above the real
        # poll rate.
        "poses_preroll": pose_counts["preroll"],
        "poses_in_episode": pose_counts["inside"],
        "pose_rate_hz": pose_counts["rate"],
        "hand_frame": hand_frame,
        "collect_hands": collect_hands,
    }
    (ep_dir / "episode.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.json",
        help="JSON config file (default: config.json next to this script)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="output directory for raw episodes; defaults to config 'raw_root', "
        "else config 'root' with a '_raw' suffix",
    )
    ap.add_argument("--task", default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument(
        "--bitrate",
        type=int,
        default=None,
        help="target average bits/s per camera (default: 4000000)",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=None,
        help="0-51 target quality on the same scale as x264's CRF, applied in "
        "variable-bitrate mode (default: 23). Use --cbr for constant bitrate.",
    )
    ap.add_argument(
        "--cbr",
        action="store_true",
        help="encode at a constant bitrate instead of targeting a quality",
    )
    ap.add_argument(
        "--iframe-interval",
        type=int,
        default=None,
        help="frames between IDR frames (default: 4, matching the datasets the "
        "PyAV recorder produced; larger is smaller on disk but slower to seek)",
    )
    ap.add_argument("--cameras", nargs="*", default=None, help="name=topic pairs")
    ap.add_argument(
        "--no-hand",
        action="store_true",
        help="skip glove + head pose + controller capture entirely",
    )
    ap.add_argument(
        "--no-manus",
        action="store_true",
        help="keep head pose + controller capture but skip just the glove",
    )
    ap.add_argument("--wrist-source", choices=("controller", "hand_tracking", "auto"), default=None)
    ap.add_argument("--hand-frame", choices=HAND_FRAMES, default=None)
    ap.add_argument("--manus-plugin-dir", type=Path, default=MANUS_PLUGIN_DIR)
    args = ap.parse_args()

    cfg: dict = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = json.load(f)
    elif args.config != ap.get_default("config"):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 1

    def get(cli_val, key, default):
        return cli_val if cli_val is not None else cfg.get(key, default)

    root = args.root
    if root is None:
        raw_root = cfg.get("raw_root")
        if raw_root:
            root = Path(raw_root)
        elif cfg.get("root"):
            root = Path(str(cfg["root"]) + "_raw")
    if root is None:
        print(
            "ERROR: output root not set (use --root, or set 'raw_root' or "
            "'root' in config.json)",
            file=sys.stderr,
        )
        return 1
    root = Path(root).expanduser()

    task = get(args.task, "task", None)
    if not task:
        print("ERROR: task not set (use --task or config 'task')", file=sys.stderr)
        return 1
    fps = get(args.fps, "fps", 30)
    bitrate = get(args.bitrate, "bitrate", 4_000_000)
    iframe_interval = get(args.iframe_interval, "iframe_interval", 4)
    quality = None if args.cbr else get(args.quality, "quality", 23)
    hand_frame = get(args.hand_frame, "hand_frame", HAND_FRAME_STAGE)
    wrist_source = get(args.wrist_source, "wrist_source", DEFAULT_WRIST_SOURCE)
    aim_to_wrist = cfg.get("aim_to_wrist")
    camera_lag_frames = cfg.get("camera_lag_frames", DEFAULT_CAMERA_LAG_FRAMES)

    use_hand = not args.no_hand
    collect_hands = use_hand and not args.no_manus
    if hand_frame not in HAND_FRAMES:
        print(f"ERROR: hand_frame must be one of {HAND_FRAMES}", file=sys.stderr)
        return 1
    if collect_hands:
        os.environ.setdefault("MANUS_WRIST_SOURCE", wrist_source)

    cameras = DEFAULT_CAMERAS
    if args.cameras:
        cameras = dict(c.split("=", 1) for c in args.cameras)
    cam_names = list(cameras)

    try:
        encoder_name = available_encoder()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ ROS
    rclpy.init()
    buf = CameraBuffer(cameras, queue_depth=FRAME_QUEUE_DEPTH, qos_depth=QOS_DEPTH)
    executor = SingleThreadedExecutor()
    executor.add_node(buf)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    def teardown() -> None:
        executor.shutdown()
        spin.join(timeout=5.0)
        buf.destroy_node()
        rclpy.shutdown()

    print(f"Waiting for frames on: {', '.join(cameras.values())}")
    if not buf.wait_for_all(cam_names, timeout=30.0):
        missing = set(cameras) - set(buf.snapshot())
        print(f"ERROR: no frames from {sorted(missing)}", file=sys.stderr)
        teardown()
        return 1

    snap = buf.snapshot()
    sizes = {n: (img.shape[1], img.shape[0]) for n, (img, _, _) in snap.items()}
    for name in sorted(sizes):
        print(f"  {name}: {sizes[name][0]}x{sizes[name][1]}")

    print(f"\nMeasuring publish rates over 3s (target {fps} Hz)...")
    rates = buf.measure_rates(3.0)
    slow = {n: r for n, r in rates.items() if r < fps * 0.9}
    for name in sorted(rates):
        print(f"  {name:14} {rates[name]:6.1f} Hz{'   <-- TOO SLOW' if name in slow else ''}")
    if slow:
        print("\nWARNING: cameras above are not keeping up.")
        print("Continue anyway? [y/N] ", end="", flush=True)
        if getch().lower() != "y":
            print()
            teardown()
            return 1
        print()

    root.mkdir(parents=True, exist_ok=True)
    session = {
        "task": task,
        "fps": fps,
        "cameras": cameras,
        "resolution": {n: list(sizes[n]) for n in cam_names},
        "encoder": encoder_name,
        "bitrate": bitrate,
        "quality": quality,
        "iframe_interval": iframe_interval,
        "hand_frame": hand_frame,
        "hand_joint_names": hand_joint_names(hand_frame) if collect_hands else [],
        "head_pose_names": HEAD_POSE_NAMES,
        "controller_pose_names": CONTROLLER_POSE_NAMES,
        "hand_poll_hz": HAND_POLL_HZ,
        "collect_hands": collect_hands,
        "use_hand": use_hand,
        "wrist_source": wrist_source if collect_hands else None,
        "aim_to_wrist": aim_to_wrist,
        # Carried through as a hint only. Nothing here applies it; the
        # converter does, and can be re-run with a different value.
        "camera_lag_frames_hint": camera_lag_frames,
    }
    (root / "session.json").write_text(json.dumps(session, indent=2))

    print(f"\nTask    : {task}")
    print(f"Root    : {root}")
    print(f"Encoder : {encoder_name}")
    print(
        f"Rate    : {'CBR' if quality is None else f'VBR, quality {quality}'}, "
        f"{bitrate / 1e6:.1f} Mbps target, IDR every {iframe_interval} frames"
    )
    if use_hand:
        print(f"Hand    : {hand_frame}" if collect_hands else "Hand    : disabled (--no-manus)")
    else:
        print("Hand    : disabled (--no-hand)")
    print("\nControls:  s = start recording   e = end recording")
    print("           y = save episode       n = discard episode")
    print("           Ctrl+C = quit\n")

    class _NoOpCtx:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            pass

    hand_ctx = (
        ManusHandBuffer(
            args.manus_plugin_dir, hand_frame, aim_to_wrist, collect_hands=collect_hands
        )
        if use_hand
        else _NoOpCtx()
    )

    # Continue an existing recording rather than overwriting it.
    existing = sorted(p.name for p in root.glob("episode_*") if p.is_dir())
    next_idx = int(existing[-1].split("_")[1]) + 1 if existing else 0
    if existing:
        print(f"Continuing after {len(existing)} existing episode(s).\n")

    with hand_ctx as hand_buf:
        if use_hand and hand_buf is not None:
            what = "Manus glove" if collect_hands else "head/controller"
            print(f"Waiting for {what} data (up to 30 s)...")
            if hand_buf.wait_ready(timeout=30.0):
                print(f"  {what} ready.\n")
            else:
                print(f"  WARNING: no {what} data yet; poses will be zeros.\n")

        try:
            while True:
                # Ends with a newline, not a carriage return. The encoder's
                # V4L2 layer writes "Opening in BLOCKING MODE" straight to the
                # terminal from its own thread, and a rewritable prompt line
                # ends up spliced together with it.
                print("Press 's' to start a new episode...", flush=True)
                if getch() != "s":
                    continue

                ep_dir = root / f"episode_{next_idx:06d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                writers = {
                    n: GstMp4Writer(
                        ep_dir / f"{n}.mp4",
                        sizes[n][0],
                        sizes[n][1],
                        fps,
                        bitrate=bitrate,
                        iframe_interval=iframe_interval,
                        quality=quality,
                    )
                    for n in cam_names
                }
                t_open = time.perf_counter()
                try:
                    for w in writers.values():
                        w.open()
                except RuntimeError as exc:
                    print(f"\nERROR: could not start encoder: {exc}", file=sys.stderr)
                    for w in writers.values():
                        w.abort()
                    shutil.rmtree(ep_dir, ignore_errors=True)
                    continue
                open_ms = (time.perf_counter() - t_open) * 1e3

                print(
                    f"\nRecording episode {next_idx}... "
                    f"(encoders up in {open_ms:.0f} ms) press 'e' to stop"
                )
                stop_evt = threading.Event()
                result: dict = {}
                failure: list[BaseException] = []

                def run() -> None:
                    try:
                        record_episode(
                            buf, hand_buf, writers, cam_names, fps, stop_evt, result
                        )
                    except BaseException as exc:  # surfaced below, not swallowed
                        failure.append(exc)
                        stop_evt.set()

                t = threading.Thread(target=run, daemon=True)
                t.start()
                while not stop_evt.is_set():
                    if getch() == "e":
                        stop_evt.set()
                t.join()

                # Finalise now rather than on 'y': an encoder problem should be
                # visible before the save decision, and the MP4 frame counts are
                # part of what makes that decision informed.
                mp4_frames: dict[str, int] = {}
                close_errors: list[str] = []
                t_close = time.perf_counter()
                for name, w in writers.items():
                    try:
                        mp4_frames[name] = w.close()
                    except RuntimeError as exc:
                        close_errors.append(str(exc))
                        mp4_frames[name] = w.frames_written
                close_ms = (time.perf_counter() - t_close) * 1e3

                if failure:
                    print(f"\nERROR during recording: {failure[0]!r}", file=sys.stderr)
                    for w in writers.values():
                        w.abort()
                    shutil.rmtree(ep_dir, ignore_errors=True)
                    print("Episode discarded. Stopping the session.")
                    break

                n = result.get("n", 0)
                dur = result["t_end"] - result["t_start"]
                print(
                    f"Episode {next_idx}: {n} frames ({dur:.1f}s, "
                    f"{n / dur:.1f} fps achieved); files closed in {close_ms:.0f} ms"
                )
                for err in close_errors:
                    print(f"  ERROR: {err}", file=sys.stderr)
                if result.get("drops"):
                    print(
                        f"  WARNING: {result['drops']} frame(s) dropped from the "
                        f"camera queue; the recording loop fell more than "
                        f"{FRAME_QUEUE_DEPTH} frames behind."
                    )
                mismatched = {k: v for k, v in mp4_frames.items() if v != n}
                if mismatched:
                    print(
                        f"  ERROR: video/table frame count mismatch {mismatched} "
                        f"vs {n} rows; row N does not match video frame N."
                    )
                for name in cam_names:
                    rep = _gap_report(
                        np.asarray(result["stamps"][name][:n], dtype=np.float64), fps
                    )
                    if rep["missing"]:
                        print(
                            f"  NOTE: {name} missing {rep['missing']} frame(s) at "
                            f"the source in {rep['gaps']} gap(s), longest "
                            f"{rep['max_gap_s'] * 1e3:.0f} ms"
                        )
                if hand_buf is not None:
                    pc = _pose_counts(
                        result.get("poses", []), result["t_start"], result["t_end"]
                    )
                    print(
                        f"  Poses: {pc['inside']} samples in episode "
                        f"({pc['rate']:.1f} Hz, expected {HAND_POLL_HZ}) "
                        f"+ {pc['preroll']} kept from before it started"
                    )
                    if dur > HAND_HISTORY_SECONDS and pc["rate"] < HAND_POLL_HZ * 0.8:
                        print(
                            "  WARNING: pose stream is well under its poll rate; "
                            "samples were evicted before being drained."
                        )

                print("Save this episode?  y = save   n = discard", end="  ", flush=True)
                while True:
                    ch = getch()
                    if ch == "y":
                        if n == 0:
                            shutil.rmtree(ep_dir, ignore_errors=True)
                            print("\nEpisode was empty, nothing saved.")
                            break
                        meta = write_episode(
                            ep_dir,
                            result,
                            cam_names,
                            fps,
                            task,
                            hand_frame,
                            collect_hands,
                            mp4_frames,
                        )
                        size_mb = sum(
                            p.stat().st_size for p in ep_dir.iterdir()
                        ) / 1e6
                        print(
                            f"\nSaved {ep_dir.name}: {meta['frames']} frames, "
                            f"{size_mb:.1f} MB."
                        )
                        next_idx += 1
                        break
                    if ch == "n":
                        shutil.rmtree(ep_dir, ignore_errors=True)
                        print(f"\nDiscarded episode {next_idx}.")
                        break

        except KeyboardInterrupt:
            print("\n\nCtrl+C — stopping.")

    teardown()
    print(f"\nRaw recording in {root}")
    print(f"Convert with: python3 raw_to_lerobot.py --raw {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
