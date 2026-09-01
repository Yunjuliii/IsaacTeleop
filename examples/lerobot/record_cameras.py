# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record a LeRobot v2.1 dataset from ROS 2 camera topics + Manus glove hand tracking.

Requires four ROS 2 image topics to be publishing (e.g. via sensing_wrist_gstreamer):
  /head/left/image_raw   /head/right/image_raw
  /wrist/left/image_raw  /wrist/right/image_raw

Manus glove data and Pico head pose are captured via TeleopSession (no ROS required).
Use --no-hand to skip glove + head + controller capture entirely when no CloudXR
client is connected at all. Use --no-manus to keep head pose + controller capture
but skip just the Manus glove (e.g. gloves are off/charging but the headset and
controllers are still on).

Head pose (observation.head_pose) is recorded in STAGE space as [x, y, z, qx, qy, qz, qw].
Hand poses (observation.hand_{left|right}) are also in STAGE space.
To obtain ego-centric hand poses in post-processing:
  hand_ego = inv(head_pose) @ hand_pose_stage

The Manus wrist is positioned by the Pico controllers, not by optical hand
tracking -- see DEFAULT_WRIST_SOURCE for why. Every hand joint is therefore
rigidly anchored to the controller aim pose internally; the controller grip pose
is recorded in observation.controller_{left|right} (more natural for policy
learning). When a controller drops out, that hand's joints stop tracking the
operator even though they keep changing. Check the validity flags before trusting
a frame.

Format: one parquet + one mp4 per camera per episode.

Layout produced::

    <root>/
      meta/info.json
      meta/stats.json
      meta/episodes.jsonl
      meta/tasks.jsonl
      data/chunk-000/episode_000000.parquet
      ...
      videos/observation.images.<cam>/chunk-000/episode_000000.mp4
      ...

Usage::

    python3 record_cameras.py           # uses config.json in same directory
    python3 record_cameras.py --root ~/datasets/my_task --task "pick up the cube"
    python3 record_cameras.py --no-hand   # skip Manus glove + head pose + controllers
    python3 record_cameras.py --no-manus  # keep head pose + controllers, skip only the glove

Controls:  s = start episode · e = end · y = save · n = discard · Ctrl+C = quit
"""

import argparse
import collections
import json
import os
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_CAMERAS = {
    "head_left": "/head/left/image_raw",
    "head_right": "/head/right/image_raw",
    "wrist_left": "/wrist/left/image_raw",
    "wrist_right": "/wrist/right/image_raw",
}

# Frames buffered per camera between the ROS callback and the recording loop.
FRAME_QUEUE_DEPTH = 8

# Hand poll rate. record_episode() samples the buffer once per 30 Hz camera
# frame, so 60 Hz keeps every sample under half a camera period from the time it
# asks for, without spending GIL on work nobody reads — see ManusHandBuffer._run()
# for why not faster.
HAND_POLL_HZ = 60

# How much hand history to keep. Must comfortably exceed the camera lag being
# compensated, since a lookup older than the deque simply fails.
HAND_HISTORY_SECONDS = 2.0

# How far the cameras lag every other sensor, in frames.
#
# Measured end to end on this rig: exposure to Argus output is 62 ms for the head
# cameras and 60 ms for the wrist (vendor figures), plus 9.6 ms to the ROS
# callback and 5-13 ms waiting in FrameQueue for the slowest camera in the group
# — 80-85 ms for the head, 105-108 ms for the wrist, so 2.4 to 3.2 frames at
# 30 fps. The controller poses arrive over CloudXR, whose runtime extrapolates
# them to the requested instant, so they are effectively current.
#
# One number for all cameras because the recorder writes one row: the hand
# columns can only be contemporary with one of them, and the head cameras are
# what the policy is trained to look at.
DEFAULT_CAMERA_LAG_FRAMES = 3

# Smallest GOP this board's NVENC will open. Measured: g=2 and g=3 both fail
# avcodec_open2 with EINVAL, g=4 and up succeed. Software h264 accepts any value,
# so this is safe to apply unconditionally.
NVENC_MIN_GOP = 4

# --------------------------------------------------------------------------- #
# Manus / hand-tracking constants
# --------------------------------------------------------------------------- #
# 25 OpenXR hand joints (WRIST=1 … LITTLE_TIP=25, skipping PALM=0).
# Names follow the anatomy used in the OpenXR hand-tracking extension and the
# LeRobot EgoWorld convention: observation.hand_{left|right}, flat float32[175].
HAND_JOINT_NAMES = [
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
]
# 7 values per joint: position (x,y,z in metres) + orientation quaternion (qx,qy,qz,qw).
# One name per joint (not per scalar) — the 7 floats for each joint share the joint label.
HAND_POSE_DIM = 7  # x, y, z, qx, qy, qz, qw

# --------------------------------------------------------------------------- #
# Hand reference frame
# --------------------------------------------------------------------------- #
# What the plugin injects is always STAGE space, built as
#     joint_stage[j] = aim_pose * aim_to_wrist * manus_local[j]
# where aim_pose comes from the controller and aim_to_wrist is a constant
# calibrated for the glove mount (kLeft/RightHandOffset in
# manus_hand_tracking_plugin.cpp). Every joint therefore carries the controller's
# world position baked in.
#
# HAND_FRAME_LOCAL undoes that: inv(joint[WRIST]) * joint[j] cancels aim_pose and
# aim_to_wrist exactly, leaving manus_local[j] -- pure hand shape, independent of
# where the arm was. Note that dropping the wrist COLUMN alone would not achieve
# this: the surviving joints stay in stage space and the wrist is recoverable
# from any one of them, so the anchor would still be in the data.
#
# Under LOCAL the wrist is identity by construction, so it is dropped rather than
# stored as 7 constant floats that invite being mistaken for a real pose.
HAND_FRAME_STAGE = "stage"
HAND_FRAME_LOCAL = "local"
HAND_FRAMES = (HAND_FRAME_STAGE, HAND_FRAME_LOCAL)

# Default aim-to-wrist transform, mirroring kLeftHandOffset / kRightHandOffset in
# manus_hand_tracking_plugin.cpp. Used only when config.json overrides it; leaving
# it unset passes the plugin's own composition through untouched, which is exact.
# 2026-08-25: mirrors the rig-calibrated constants (inv(T_grip->aim) . T_wrist->ctrl
# . C0 -- see the comment above kLeftHandOffset in the cpp). The stage-frame WRIST
# joint now lands on the calibrated anatomical wrist, not the vendor's nominal
# held-controller guess.
DEFAULT_AIM_TO_WRIST = {
    "left": {
        "position": [0.044895, -0.123515, 0.054711],
        "quaternion": [0.35957026, 0.05137509, 0.43123836, 0.82589546],
    },  # qx, qy, qz, qw
    "right": {
        "position": [-0.043748, -0.124802, 0.052615],
        "quaternion": [0.33336273, -0.06644838, -0.48965076, 0.80292966],
    },
}


def hand_joint_names(frame: str) -> list[str]:
    """Joint names for the given frame; LOCAL drops the identity wrist."""
    return HAND_JOINT_NAMES if frame == HAND_FRAME_STAGE else HAND_JOINT_NAMES[1:]


def hand_feature_dim(frame: str) -> int:
    """25 x 7 = 175 in STAGE, 24 x 7 = 168 in LOCAL."""
    return len(hand_joint_names(frame)) * HAND_POSE_DIM


# --------------------------------------------------------------------------- #
# Pose algebra
# --------------------------------------------------------------------------- #
# Poses are flat [x, y, z, qx, qy, qz, qw]; quaternions are xyzw, matching both
# the OpenXR wire format and scipy's Rotation.from_quat.
def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * b -- apply b in a's frame. Mirrors oxr_utils::multiply_poses."""
    Ra = Rotation.from_quat(a[3:7])
    out = np.empty(HAND_POSE_DIM, dtype=np.float32)
    out[0:3] = a[0:3] + Ra.apply(b[0:3])
    out[3:7] = (Ra * Rotation.from_quat(b[3:7])).as_quat()
    return out


def _invert(a: np.ndarray) -> np.ndarray:
    Ri = Rotation.from_quat(a[3:7]).inv()
    out = np.empty(HAND_POSE_DIM, dtype=np.float32)
    out[0:3] = -Ri.apply(a[0:3])
    out[3:7] = Ri.as_quat()
    return out


def to_wrist_local(hand_stage: np.ndarray) -> np.ndarray:
    """STAGE (25 x 7) -> wrist-local (24 x 7), wrist dropped.

    inv(wrist) * joint[j]. Both aim_pose and aim_to_wrist cancel, so the result
    is the glove's own local pose regardless of what the wrist source was doing
    -- which is the point: it is exactly the part of the signal that does not
    depend on controller tracking.
    """
    joints = hand_stage.reshape(-1, HAND_POSE_DIM)
    inv_wrist = _invert(joints[0])
    return np.stack([_compose(inv_wrist, j) for j in joints[1:]]).reshape(-1)


def to_stage(hand_local: np.ndarray, aim: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """wrist-local (24 x 7) -> STAGE (25 x 7) under a caller-supplied offset.

    Re-anchors as aim * offset * local[j], with the wrist prepended as
    aim * offset (identity in local space). Used only when config.json overrides
    the offset; the plugin's own composition is passed through otherwise, since
    round-tripping it through here would add float error for nothing.
    """
    root = _compose(aim, offset)
    joints = hand_local.reshape(-1, HAND_POSE_DIM)
    return np.concatenate([root, *(_compose(root, j) for j in joints)])


# Head pose: position (x,y,z) + orientation quaternion (qx,qy,qz,qw), all in STAGE space.
HEAD_POSE_DIM = 7
HEAD_POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]

# Controller grip pose, STAGE space: position(3) + quaternion(4) + validity(1).
# Aim pose is not recorded; it is still extracted internally when config.json
# supplies an aim_to_wrist override (the plugin anchors hand joints on aim).
#
# The validity flag is not redundant with zeroing. Once the controller is the
# only world anchor -- which it is under MANUS_WRIST_SOURCE=controller -- an
# all-zero pose is bit-identical to a legitimate pose at the stage origin, and a
# model trained on that learns to regress toward the origin on dropout. It is
# also exactly the condition that makes the plugin's own fallback bail out
# (get_controller_wrist_pose returns false on !aim_valid), so a zero here means
# that frame's hand joints are stale, not merely that the controller is missing.
CONTROLLER_POSE_DIM = 8
CONTROLLER_POSE_NAMES = [
    "grip_x",
    "grip_y",
    "grip_z",
    "grip_qx",
    "grip_qy",
    "grip_qz",
    "grip_qw",
    "grip_valid",
]

# Plugin ROOT, not the plugin's own directory: PluginManager::discover_plugins()
# scans the SUBDIRECTORIES of each search path for <subdir>/plugin.yaml
# (plugin_manager.cpp:98). Pointing this at install/plugins/manus finds nothing --
# that directory holds only files -- and TeleopSession then skips the plugin
# silently (teleop_session.py:1073), so the glove process never starts and every
# observation.hand_* row is zeros. Matches PLUGIN_ROOT_DIR in the other examples.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANUS_PLUGIN_DIR = _REPO_ROOT / "install" / "plugins"

# Which device positions the Manus wrist. Read by the plugin subprocess from the
# environment; see wrist_source_from_env() in manus_hand_tracking_plugin.cpp.
#
# "controller" rather than the plugin's own "auto" default, because auto never
# reaches the controller on this rig. The plugin matches optical XDevs by the
# exact serials "Head Device (0)"/"Head Device (1)", which is what a Quest 3
# reports; CloudXR on this Pico enumerates "Push Hand Tracker (Left/Right)",
# "Derived EXT Hand Interaction Right" and friends instead, so every
# xrCreateHandTrackerEXT comes back XR_ERROR_FEATURE_UNSUPPORTED. Worse, when
# optical partially works it degrades to valid-but-untracked poses, and
# inject_hand_data() gates its controller fallback on validity rather than
# tracking -- so the whole hand freezes in stage space instead of falling back.
# Measured over one episode: 18-47% of frames frozen under auto, up to 1.5 s at a
# stretch, versus 0.7% and at most one frame under controller.
DEFAULT_WRIST_SOURCE = "controller"


# --------------------------------------------------------------------------- #
# Manus hand buffer
# --------------------------------------------------------------------------- #
class ManusHandBuffer:
    """Runs TeleopSession with the Manus plugin in a background thread.

    Exposes the latest left/right hand pose arrays via :meth:`latest`, shaped
    (hand_feature_dim(hand_frame),). Values are [x, y, z, qx, qy, qz, qw] per
    joint: 25 joints WRIST → LITTLE_TIP under STAGE, 24 joints
    THUMB_METACARPAL → LITTLE_TIP under LOCAL.
    """

    def __init__(
        self,
        plugin_dir: Path = MANUS_PLUGIN_DIR,
        hand_frame: str = HAND_FRAME_STAGE,
        aim_to_wrist: dict | None = None,
        collect_hands: bool = True,
    ):
        """collect_hands=False keeps head pose + controller capture (still needs a
        connected CloudXR client) but does not load the Manus plugin at all, so the
        glove subprocess never starts -- for when the gloves are off/charging but
        the headset and controllers are still tracked. observation.hand_* is not
        produced in this mode; see the --no-manus caller in main()."""
        from isaacteleop.retargeting_engine.deviceio_source_nodes import (
            HandsSource,
            HeadSource,
            ControllersSource,
        )
        from isaacteleop.retargeting_engine.interface import OutputCombiner
        from isaacteleop.retargeting_engine.tensor_types import (
            HandInputIndex,
            HeadPoseIndex,
            ControllerInputIndex,
        )
        from isaacteleop.teleop_session_manager import (
            PluginConfig,
            TeleopSession,
            TeleopSessionConfig,
        )

        self._collect_hands = collect_hands

        if collect_hands:
            # A plugin the manager cannot discover is skipped without a word, so the
            # first sign of a wrong search path is an entire session of zeroed hand
            # columns. Check the layout the manager actually requires, up front.
            if not list(plugin_dir.glob("*/plugin.yaml")):
                raise FileNotFoundError(
                    f"no <plugin>/plugin.yaml under {plugin_dir}. This must be the plugin "
                    f"ROOT (e.g. {MANUS_PLUGIN_DIR}), not an individual plugin's directory: "
                    f"PluginManager scans each search path's subdirectories. Without it the "
                    f"Manus process never starts and observation.hand_* records all zeros."
                )

        self._TeleopSession = TeleopSession
        self._HandInputIndex = HandInputIndex
        self._HeadPoseIndex = HeadPoseIndex
        self._ControllerInputIndex = ControllerInputIndex

        head = HeadSource(name="head")
        controllers = ControllersSource(name="controllers")
        outputs = {
            "head": head.output("head"),
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
        plugins = []
        if collect_hands:
            hands = HandsSource(name="hands")
            outputs["hand_left"] = hands.output(HandsSource.LEFT)
            outputs["hand_right"] = hands.output(HandsSource.RIGHT)
            plugins.append(
                PluginConfig(
                    plugin_name="manus_hand_plugin",
                    plugin_root_id="manus",
                    search_paths=[plugin_dir],
                )
            )
        pipeline = OutputCombiner(outputs)
        config = TeleopSessionConfig(
            app_name="LeRobotRecorder",
            pipeline=pipeline,
            plugins=plugins,
        )
        if hand_frame not in HAND_FRAMES:
            raise ValueError(
                f"hand_frame must be one of {HAND_FRAMES}, got {hand_frame!r}"
            )
        self._hand_frame = hand_frame
        self._feature_dim = hand_feature_dim(hand_frame)
        # None means "leave the plugin's own composition alone". Only a config
        # override populates this, and only STAGE can use it -- re-anchoring is
        # meaningless once the frame is wrist-local.
        self._aim_to_wrist = None
        if aim_to_wrist is not None and hand_frame == HAND_FRAME_STAGE:
            self._aim_to_wrist = {
                side: np.asarray(
                    list(spec["position"]) + list(spec["quaternion"]), dtype=np.float32
                )
                for side, spec in aim_to_wrist.items()
            }

        self._config = config
        self._session = TeleopSession(config)
        self._lock = threading.Lock()
        self._left = np.zeros(self._feature_dim, dtype=np.float32)
        self._right = np.zeros(self._feature_dim, dtype=np.float32)
        self._head_pose = np.zeros(HEAD_POSE_DIM, dtype=np.float32)
        self._controller_left = np.zeros(CONTROLLER_POSE_DIM, dtype=np.float32)
        self._controller_right = np.zeros(CONTROLLER_POSE_DIM, dtype=np.float32)
        # Timestamped history, so a row can be built from the sample that is
        # contemporary with its camera frame rather than the newest one. Sized
        # for HAND_HISTORY_SECONDS at the poll rate; at 60 Hz that is a few
        # hundred small tuples, and running short is worse than the memory --
        # a lookup that falls off the end of the deque discards the whole row.
        self._history: collections.deque = collections.deque(
            maxlen=int(HAND_POLL_HZ * HAND_HISTORY_SECONDS)
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # -- context manager so callers can use `with ManusHandBuffer() as buf:` --
    def __enter__(self):
        # XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35): CloudXR runtime is up but no
        # client (phone / headset) has connected yet.  Retry until one does.
        while True:
            try:
                self._session.__enter__()
                break
            except RuntimeError as exc:
                if "Failed to get OpenXR system" not in str(exc):
                    raise
                print("[Manus] Waiting for CloudXR client to connect...", flush=True)
                time.sleep(2.0)
                # TeleopSession is not reusable after a failed __enter__; recreate it.
                self._session = self._TeleopSession(self._config)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="manus-hand-poll"
        )
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Wedged inside a native tracker call. Tearing the session down now
                # would destroy the hand trackers out from under it, so leak instead.
                print(
                    "[Manus] poll thread did not exit; leaking session.",
                    file=sys.stderr,
                )
                return False
        try:
            self._session.__exit__(*args)
        except Exception as exc:
            # The Manus plugin subprocess receives SIGINT when the user presses
            # Ctrl+C. The plugin manager re-raises this as PluginCrashException
            # with signal 2 — that's a clean shutdown, not a real crash. Suppress
            # it so that dataset.finalize() still runs after this context exits.
            msg = str(exc)
            if "signal 2" in msg or "Interrupt" in msg:
                print(
                    "[Manus] plugin exited on Ctrl+C (signal 2), ignoring.",
                    file=sys.stderr,
                )
            else:
                raise

    # -- internal -------------------------------------------------------
    def _extract_hand(self, hand, side: str, aim: np.ndarray) -> np.ndarray:
        """Pack one HandInput into the configured frame.

        Returns (175,) under STAGE, (168,) under LOCAL. ``aim`` is that hand's
        controller pose, needed only to re-anchor under a config-supplied offset.
        """
        if hand.is_none:
            return np.zeros(self._feature_dim, dtype=np.float32)
        HI = self._HandInputIndex
        # shape (26, 3) and (26, 4) — index 0 is PALM, 1-25 are the joints we want.
        positions = np.asarray(hand[HI.JOINT_POSITIONS], dtype=np.float32)
        orientations = np.asarray(hand[HI.JOINT_ORIENTATIONS], dtype=np.float32)
        pos = positions[1:26]  # (25, 3)  xyz metres
        ori = orientations[1:26]  # (25, 4)  quaternion xyzw
        # Interleave as [x,y,z, qx,qy,qz,qw] × 25 = 175 floats
        stage = np.concatenate([pos, ori], axis=1).reshape(-1)

        if self._hand_frame == HAND_FRAME_LOCAL:
            return to_wrist_local(stage).astype(np.float32)
        if self._aim_to_wrist is None:
            return stage
        # Custom offset: strip the plugin's anchor, re-anchor on ours. Without a
        # valid aim there is nothing to anchor to, and reusing the plugin's stale
        # root under a different offset would be neither frame -- zero instead.
        if aim[7] < 0.5:
            return np.zeros(self._feature_dim, dtype=np.float32)
        return to_stage(
            to_wrist_local(stage), aim[0:7], self._aim_to_wrist[side]
        ).astype(np.float32)

    def _extract_head(self, head) -> np.ndarray:
        """Pack HeadPose into a flat float32 array of shape (7,): [x,y,z, qx,qy,qz,qw]."""
        if head.is_none:
            return np.zeros(HEAD_POSE_DIM, dtype=np.float32)
        HI = self._HeadPoseIndex
        if not bool(head[HI.IS_VALID]):
            return np.zeros(HEAD_POSE_DIM, dtype=np.float32)
        pos = np.asarray(head[HI.POSITION], dtype=np.float32)  # (3,) xyz metres
        ori = np.asarray(head[HI.ORIENTATION], dtype=np.float32)  # (4,) qx qy qz qw
        return np.concatenate([pos, ori])

    def _extract_controller(self, ctrl) -> np.ndarray:
        """Pack one ControllerInput grip pose into a flat float32 array of shape (8,).

        Layout: grip_pos(3) + grip_ori(4) + grip_valid.

        An invalid pose is zeroed AND flagged; read the flag, not the zeros --
        see CONTROLLER_POSE_NAMES for why the zeros alone are ambiguous.
        """
        out = np.zeros(CONTROLLER_POSE_DIM, dtype=np.float32)
        if ctrl.is_none:
            return out
        CI = self._ControllerInputIndex
        if bool(ctrl[CI.GRIP_IS_VALID]):
            out[0:3] = np.asarray(
                ctrl[CI.GRIP_POSITION], dtype=np.float32
            )  # xyz metres
            out[3:7] = np.asarray(
                ctrl[CI.GRIP_ORIENTATION], dtype=np.float32
            )  # qx qy qz qw
            out[7] = 1.0
        return out

    def _extract_aim(self, ctrl) -> np.ndarray:
        """Pack one ControllerInput aim pose for internal hand re-anchoring only.

        Used exclusively when config.json supplies aim_to_wrist; the plugin
        anchors every hand joint on aim, so re-anchoring with a custom offset
        still needs aim. Not recorded in the dataset.
        """
        out = np.zeros(CONTROLLER_POSE_DIM, dtype=np.float32)
        if ctrl.is_none:
            return out
        CI = self._ControllerInputIndex
        if bool(ctrl[CI.AIM_IS_VALID]):
            out[0:3] = np.asarray(ctrl[CI.AIM_POSITION], dtype=np.float32)
            out[3:7] = np.asarray(ctrl[CI.AIM_ORIENTATION], dtype=np.float32)
            out[7] = 1.0
        return out

    def _run(self) -> None:
        # Paced, not free-running. The Manus plugin is a separate process pinned at
        # 90 Hz (src/plugins/manus/app/main.cpp:117) that caches the skeleton, so
        # polling faster only re-reads identical bytes — and it costs the whole
        # recording: no deviceio pybind binding releases the GIL
        # (deviceio_session/python/session_bindings.cpp:137 has no call_guard) and
        # step() never blocks, so an unthrottled loop free-runs at ~1800 Hz holding
        # ~49% of the GIL. That starves this process's ROS executor and LeRobot
        # encoder feed: measured here, unthrottled collapsed the record loop from
        # 29 Hz to 7.5 Hz and silently dropped ~9% of video frames on top; at
        # 60 Hz all four cameras came back lossless.
        period = 1.0 / HAND_POLL_HZ
        deadline = time.monotonic() + period

        while not self._stop.is_set():
            try:
                result = self._session.step()
                # Grip pose is what gets recorded.  When aim_to_wrist is set,
                # hand re-anchoring still needs the aim pose (the plugin builds
                # every joint as aim * offset * local), so extract aim separately
                # for that internal path only.
                ctrl_left = self._extract_controller(result["controller_left"])
                ctrl_right = self._extract_controller(result["controller_right"])
                if self._collect_hands:
                    hand_l, hand_r = result["hand_left"], result["hand_right"]
                    if self._aim_to_wrist is not None:
                        aim_left = self._extract_aim(result["controller_left"])
                        aim_right = self._extract_aim(result["controller_right"])
                    else:
                        aim_left, aim_right = ctrl_left, ctrl_right
                    left = self._extract_hand(hand_l, "left", aim_left)
                    right = self._extract_hand(hand_r, "right", aim_right)
                else:
                    # No HandsSource in the pipeline at all (see __init__); stay zero.
                    hand_l = hand_r = None
                    left, right = self._left, self._right
                head = self._extract_head(result["head"])
                # time.time(), not the monotonic clock this loop paces itself
                # with: the lookup key has to be the clock ROS message headers
                # use, and rclpy's default is system time. Pacing stays on
                # monotonic below, where only elapsed time matters.
                sample_t = time.time()
                with self._lock:
                    self._left = left
                    self._right = right
                    self._head_pose = head
                    self._controller_left = ctrl_left
                    self._controller_right = ctrl_right
                    self._history.append(
                        (sample_t, left, right, head, ctrl_left, ctrl_right)
                    )
                if self._collect_hands:
                    # Gate on real hand data, not on step() merely returning. Setting
                    # this unconditionally made "Manus glove ready." print even when
                    # both hands were absent for the whole run, which is exactly the
                    # symptom a missing glove process produces -- so the one check
                    # that should have caught it instead vouched for it.
                    if not (hand_l.is_none and hand_r.is_none):
                        self._ready.set()
                else:
                    # No hands in the pipeline to gate on; head or controller data
                    # arriving is the only signal a client is actually connected.
                    # _extract_head zeros the whole array (not just a flag) when
                    # invalid, so "any nonzero" is the validity check here.
                    if np.any(head) or ctrl_left[7] > 0.5 or ctrl_right[7] > 0.5:
                        self._ready.set()
            except Exception as exc:
                # Remember it: latest() must not keep handing back a frozen pose
                # that would be stamped onto every remaining row of the episode.
                self._error = exc
                print(f"[Manus] step error: {exc}", file=sys.stderr)
                return

            # Absolute schedule so hand samples don't drift against the camera
            # trigger, and wait() doubles as the shutdown check so Ctrl+C doesn't
            # have to sit out a full period. The floor keeps the throttle alive if
            # a step ever overruns its budget — without it, contention shortens the
            # sleep, which feeds back into more contention.
            if self._stop.wait(timeout=max(deadline - time.monotonic(), 0.001)):
                return
            deadline += period
            now = time.monotonic()
            if deadline < now:
                deadline = now + period  # fell behind; resync rather than burst

    # -- public ---------------------------------------------------------
    def latest(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (left, right, head_pose, ctrl_left, ctrl_right) copies.

        left, right:  float32 of shape (175,) — 25 hand joints × 7 values each.
        head_pose:    float32 of shape (7,)   — [x, y, z, qx, qy, qz, qw] in STAGE space.
        ctrl_left/right: float32 of shape (8,) — grip pose [x, y, z, qx, qy, qz, qw, valid].

        Raises if the poll thread died — a dead thread is otherwise
        indistinguishable from a healthy one and would silently freeze the
        columns at their last value for the rest of the episode.
        """
        if self._error is not None:
            raise RuntimeError("Manus poll thread died") from self._error
        with self._lock:
            return (
                self._left.copy(),
                self._right.copy(),
                self._head_pose.copy(),
                self._controller_left.copy(),
                self._controller_right.copy(),
            )

    def get_at(self, t: float, tol: float):
        """Return (sample_time, left, right, head, ctrl_l, ctrl_r) nearest to t.

        Returns None when the closest sample is further than tol away, which
        means either the history has not filled to t yet or the poll thread
        stalled across it. Returning None rather than the nearest match keeps a
        row from being built out of a pose that belongs to a different moment --
        a dropped row costs one frame, a wrong row is silently bad training data.

        Raises if the poll thread died, for the same reason latest() does.
        """
        if self._error is not None:
            raise RuntimeError("Manus poll thread died") from self._error
        with self._lock:
            if not self._history:
                return None
            # Linear scan from the newest end: the target is a fixed lag behind
            # now, so the match is always a few samples in from the back.
            best = None
            best_dt = None
            for entry in reversed(self._history):
                dt = abs(entry[0] - t)
                if best_dt is None or dt < best_dt:
                    best, best_dt = entry, dt
                elif entry[0] < t:
                    # Walking further back only increases the distance.
                    break
            if best_dt is None or best_dt > tol:
                return None
            ts, left, right, head, cl, cr = best
        return ts, left.copy(), right.copy(), head.copy(), cl.copy(), cr.copy()

    def history_span(self) -> float:
        """Seconds between the oldest and newest samples held."""
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            return self._history[-1][0] - self._history[0][0]

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until at least one frame has arrived, or timeout."""
        return self._ready.wait(timeout)


# --------------------------------------------------------------------------- #
# Frame queue
# --------------------------------------------------------------------------- #
class FrameQueue:
    """Bounded leaky queue: when full the OLDEST frame is dropped, newest wins.

    A deque rather than queue.Queue because next_frames() has to push frames back
    to the FRONT when a sibling camera times out mid-collection. With queue.Queue
    there was no way to do that, so every missed tick silently discarded the
    frames already pulled from the other cameras.

    Every drop is counted. A dropped frame is not cosmetic: the recorder writes
    timestamp = frame_index / fps, so a hole in the stream makes that column lie.
    """

    def __init__(self, depth: int):
        self._items: collections.deque = collections.deque()
        self._depth = depth
        self._cv = threading.Condition()
        self.dropped = 0
        self.skipped = 0

    def push(self, item) -> None:
        with self._cv:
            self._items.append(item)
            while len(self._items) > self._depth:
                self._items.popleft()
                self.dropped += 1
            self._cv.notify()

    def push_front(self, item) -> None:
        with self._cv:
            self._items.appendleft(item)
            while len(self._items) > self._depth:
                self._items.pop()
                self.dropped += 1
            self._cv.notify()

    def pop(self, timeout: float):
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            return self._items.popleft()

    def pop_latest(self, timeout: float):
        """Return the NEWEST frame, discarding any backlog behind it."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            self.skipped += len(self._items) - 1
            item = self._items.pop()
            self._items.clear()
            return item

    def clear(self) -> None:
        """Discard the backlog."""
        with self._cv:
            self._items.clear()


# --------------------------------------------------------------------------- #
# ROS node
# --------------------------------------------------------------------------- #
class CameraBuffer(Node):
    """Per-camera queues that always expose the latest received frame.

    Each camera gets its own FrameQueue. next_frames() discards any backlog
    and returns the newest frame per camera via pop_latest(), keeping the
    camera timestamps as close as possible to the controller data sampled
    via hand_buf.latest() in the same recording tick.
    """

    def __init__(self, cameras: dict[str, str]):
        super().__init__("lerobot_collector")
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[np.ndarray, float, float]] = {}
        self._counts: dict[str, int] = {n: 0 for n in cameras}
        self._queues: dict[str, FrameQueue] = {
            n: FrameQueue(FRAME_QUEUE_DEPTH) for n in cameras
        }
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for name, topic in cameras.items():
            self.create_subscription(
                Image, topic, lambda m, n=name: self._cb(m, n), qos
            )
        self.get_logger().info(f"Subscribed to {len(cameras)} camera topics")

    def _cb(self, msg: Image, name: str) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"{name}: expected rgb8, got {msg.encoding}", once=True
            )
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        row = msg.width * 3
        if msg.step != row:
            buf = buf.reshape(msg.height, msg.step)[:, :row].reshape(-1)
        img = buf.reshape(msg.height, msg.width, 3)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Arrival on the SAME clock as header.stamp, not time.monotonic(): the
        # difference between the two is the transport cost (nvvidconv,
        # videoconvert, gscam2 publish, DDS), and subtracting a monotonic
        # reading from a system-clock stamp would measure the offset between
        # two clock domains instead.
        t_recv = self.get_clock().now().nanoseconds * 1e-9
        frame = (img, stamp, t_recv)
        with self._lock:
            self._latest[name] = frame
            self._counts[name] += 1
        self._queues[name].push(frame)

    def next_frames(self, names: list[str], timeout: float) -> dict | None:
        """Return the newest available frame per camera, or None on timeout.

        Uses pop_latest() so that any backlog accumulated while the recording
        loop was busy is discarded and only the freshest frame is returned.
        This keeps the camera timestamp as close as possible to the controller
        data sampled via hand_buf.latest() in the same recording tick.

        If any camera times out, frames already popped from other cameras are
        pushed back so the next call can retry the full group together.
        """
        frames: dict[str, tuple[np.ndarray, float, float]] = {}
        deadline = time.monotonic() + timeout
        for name in names:
            remaining = deadline - time.monotonic()
            frame = self._queues[name].pop_latest(remaining) if remaining > 0 else None
            if frame is None:
                for done, held in frames.items():
                    self._queues[done].push_front(held)
                return None
            frames[name] = frame
        return frames

    def flush(self) -> None:
        """Discard buffered backlogs so the next episode starts from live frames."""
        for q in self._queues.values():
            q.clear()

    def snapshot(self) -> dict[str, tuple[np.ndarray, float, float]]:
        with self._lock:
            return dict(self._latest)

    def measure_rates(self, duration: float = 3.0) -> dict[str, float]:
        with self._lock:
            start = dict(self._counts)
        t0 = time.monotonic()
        time.sleep(duration)
        with self._lock:
            end = dict(self._counts)
        dt = time.monotonic() - t0
        return {n: (end[n] - start.get(n, 0)) / dt for n in end}

    def wait_for_all(self, names: list[str], timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(n in self.snapshot() for n in names):
                return True
            time.sleep(0.1)
        return False


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record_episode(
    buf: CameraBuffer,
    hand_buf: ManusHandBuffer | None,
    dataset: LeRobotDataset,
    cam_names: list[str],
    task: str,
    fps: int,
    state_dim: int,
    action_dim: int,
    stop_evt: threading.Event,
    camera_lag_frames: float = DEFAULT_CAMERA_LAG_FRAMES,
    collect_hands: bool = True,
) -> tuple[int, int, int, int]:
    """Record frames until stop_evt is set.

    Returns (frame_count, dropped, skipped, unmatched).

    The hand and controller columns are taken from camera_lag_frames frames in
    the past, so each row pairs an image with the pose that was true when that
    image was exposed rather than when it finished arriving.
    """
    frame_timeout = 2.0 / fps
    lag_s = camera_lag_frames / fps
    # At least one and a half poll intervals: samples land 1/HAND_POLL_HZ apart,
    # so a tolerance of half a camera frame would reject perfectly good matches
    # whenever the target falls between two of them. Still far tighter than the
    # ~100 ms error this compensates, so a genuinely wrong pose is rejected.
    match_tol = max(0.5 / fps, 1.5 / HAND_POLL_HZ)
    now = lambda: buf.get_clock().now().nanoseconds * 1e-9
    buf.flush()
    drops_before = sum(q.dropped for q in buf._queues.values())
    skips_before = sum(q.skipped for q in buf._queues.values())
    frame_i = 0
    unmatched = 0

    # The lookup reaches lag_s into the past, so an episode that starts before
    # the history covers it would drop its opening rows one by one.
    if hand_buf is not None:
        deadline = time.monotonic() + 5.0
        while (
            hand_buf.history_span() < lag_s + 0.1
            and time.monotonic() < deadline
            and not stop_evt.is_set()
        ):
            time.sleep(0.02)

    while not stop_evt.is_set():
        frames = buf.next_frames(cam_names, timeout=frame_timeout)
        if frames is None:
            continue

        frame_data: dict = {"task": task}
        for name, (img, _stamp, _t_recv) in frames.items():
            frame_data[f"observation.images.{name}"] = img
        frame_data["observation.state"] = np.zeros(state_dim, dtype=np.float32)
        frame_data["action"] = np.zeros(action_dim, dtype=np.float32)

        if hand_buf is not None:
            # The image in this row was exposed lag_s ago, so pair it with the
            # pose from lag_s ago rather than the newest one. Sampling the newest
            # is what put the hand columns ahead of the image in the first place.
            target_t = now() - lag_s
            sample = hand_buf.get_at(target_t, tol=match_tol)
            if sample is None:
                # No contemporary pose: drop the row instead of pairing the image
                # with whatever happens to be nearest. Counted so a run that
                # quietly loses half its frames cannot look like a clean one.
                unmatched += 1
                continue
            t_hand, left, right, head_pose, ctrl_left, ctrl_right = sample
            if collect_hands:
                frame_data["observation.hand_left"] = left
                frame_data["observation.hand_right"] = right
            frame_data["observation.head_pose"] = head_pose
            frame_data["observation.controller_left"] = ctrl_left
            frame_data["observation.controller_right"] = ctrl_right

        dataset.add_frame(frame_data)
        frame_i += 1

    drops = sum(q.dropped for q in buf._queues.values()) - drops_before
    skips = sum(q.skipped for q in buf._queues.values()) - skips_before
    return frame_i, drops, skips, unmatched


def find_all_zero_rows(
    dataset: LeRobotDataset, collect_hands: bool
) -> dict[str, list[int]]:
    """Scan the just-recorded episode's still-buffered frames for rows that are
    a literal all-zero vector in head_pose / controller_left / controller_right
    / hand_left / hand_right.

    This is not the same thing as CONTROLLER_POSE_NAMES' valid=0 convention: a
    single dropped controller tick writing one zeroed-out row is normal and
    already handled downstream (e.g. add_wrist_pose.py filters on valid). What
    this catches is a device that produced *nothing* for the whole row --
    e.g. the headset/controller/glove was never connected or lost tracking
    entirely -- which no per-frame valid flag protects against for head_pose
    (it has no valid column of its own) and which is worth failing loudly on
    rather than silently saving an episode full of zeros.
    """
    buf = dataset.writer.episode_buffer
    keys = [
        "observation.head_pose",
        "observation.controller_left",
        "observation.controller_right",
    ]
    if collect_hands:
        keys += ["observation.hand_left", "observation.hand_right"]
    bad: dict[str, list[int]] = {}
    for key in keys:
        rows = buf.get(key)
        if not rows:
            continue
        zero_idx = [i for i, row in enumerate(rows) if not np.any(row)]
        if zero_idx:
            bad[key] = zero_idx
    return bad


# --------------------------------------------------------------------------- #
def getch() -> str:
    """Read one keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


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
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--state-dim", type=int, default=None)
    ap.add_argument("--action-dim", type=int, default=None)
    ap.add_argument("--robot-type", default=None)
    ap.add_argument(
        "--encoder",
        default=None,
        help="ffmpeg encoder (default: h264_nvenc; 'h264' is software "
        "libx264 and cannot keep up with four cameras)",
    )
    ap.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="name=topic pairs; defaults to the four head/wrist cameras",
    )
    ap.add_argument(
        "--no-hand",
        action="store_true",
        help="skip glove + head pose + controller capture entirely "
        "(e.g. when no CloudXR client is connected at all)",
    )
    ap.add_argument(
        "--no-manus",
        action="store_true",
        help="keep head pose + controller capture but skip just the Manus "
        "glove (e.g. gloves are off/charging but headset+controllers "
        "are still on); implied by --no-hand",
    )
    ap.add_argument(
        "--camera-lag-frames",
        type=float,
        default=None,
        help="how many camera frames the cameras lag every other sensor; "
        "the hand and controller columns are taken that far back so "
        f"they match the image (default: {DEFAULT_CAMERA_LAG_FRAMES}, "
        "0 disables compensation). Fractional values are allowed and "
        f"often needed: at {HAND_POLL_HZ} Hz hand/head/controller poll "
        "vs. camera fps, half a camera frame is one poll tick, so "
        "e.g. 3.5 is a legitimate answer, not just 3 or 4",
    )
    ap.add_argument(
        "--manus-plugin-dir",
        type=Path,
        default=MANUS_PLUGIN_DIR,
        help=f"plugin ROOT holding manus/plugin.yaml, not manus/ itself "
        f"(default: {MANUS_PLUGIN_DIR})",
    )
    ap.add_argument(
        "--wrist-source",
        choices=("controller", "hand_tracking", "auto"),
        default=None,
        help="Which device positions the Manus wrist; sets MANUS_WRIST_SOURCE "
        f"for the plugin process (default: {DEFAULT_WRIST_SOURCE})",
    )
    ap.add_argument(
        "--hand-frame",
        choices=HAND_FRAMES,
        default=None,
        help=f"'{HAND_FRAME_STAGE}': world poses, {hand_feature_dim(HAND_FRAME_STAGE)} "
        f"floats/hand. '{HAND_FRAME_LOCAL}': wrist-relative hand shape, "
        f"{hand_feature_dim(HAND_FRAME_LOCAL)} floats/hand (wrist dropped, it is "
        f"identity). Default: {HAND_FRAME_STAGE}",
    )
    ap.add_argument(
        "--data-file-size-mb",
        type=float,
        default=None,
        help="roll to a new data/video file once the current one would "
        "exceed this size (default: 0.001MB -- smaller than any real "
        "episode's data, so save_episode() always rolls to a fresh "
        "file and closes/footer-finalizes the previous one right "
        "away, i.e. one file per episode). LeRobot's own default is "
        "100MB/200MB, which packs many episodes into one file behind "
        "a single ParquetWriter that only writes its footer on "
        "rollover/finalize -- if the process dies before that, every "
        "episode still buffered in that file is lost, not just the "
        "one being recorded (this bit us: 42 episodes gone from one "
        "mid-session crash). Raise this (e.g. to 100) once the "
        "pipeline is trusted not to crash mid-session and you'd "
        "rather have fewer, bigger files for training.",
    )
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

    root = Path(get(args.root, "root", None)).expanduser()
    task = get(args.task, "task", None)
    fps = get(args.fps, "fps", 30)
    state_dim = get(args.state_dim, "state_dim", 6)
    action_dim = get(args.action_dim, "action_dim", 6)
    robot_type = get(args.robot_type, "robot_type", "sensing_gmsl2_rig")
    # h264_nvenc, not software h264: libx264 measured 39.7 frames/s on this board
    # against the 120 four cameras need, and worse in practice because LeRobot
    # passes g=2, making every other frame a keyframe. The shortfall showed up as
    # frames dropped from the encoder queue and as backlog that aged each
    # recorded frame by up to 77 ms. NVENC measured 292 frames/s.
    #
    # An earlier note here said pyav cannot open h264_nvenc on Jetson because
    # avcodec_open2 fails. That was an incomplete JetPack install, not the codec;
    # it opens and encodes here now. Use --encoder h264 to go back to software.
    encoder = get(args.encoder, "encoder", "h264_nvenc")
    data_file_size_mb = get(args.data_file_size_mb, "data_file_size_mb", 0.001)
    wrist_source = get(args.wrist_source, "wrist_source", DEFAULT_WRIST_SOURCE)
    camera_lag_frames = get(
        args.camera_lag_frames, "camera_lag_frames", DEFAULT_CAMERA_LAG_FRAMES
    )
    hand_frame = get(args.hand_frame, "hand_frame", HAND_FRAME_STAGE)
    # Config-only: seven numbers per hand is not a command line. Absent means
    # "keep the plugin's own kLeft/RightHandOffset", which is exact; supplying it
    # makes the recorder strip that anchor and re-apply this one.
    aim_to_wrist = cfg.get("aim_to_wrist")
    use_hand = not args.no_hand
    # --no-hand implies --no-manus: no session running means no glove plugin either.
    collect_hands = use_hand and not args.no_manus

    if hand_frame not in HAND_FRAMES:
        print(
            f"ERROR: hand_frame must be one of {HAND_FRAMES}, got {hand_frame!r}",
            file=sys.stderr,
        )
        return 1
    if aim_to_wrist is not None:
        if hand_frame != HAND_FRAME_STAGE:
            print(
                f"ERROR: 'aim_to_wrist' only applies to hand_frame='{HAND_FRAME_STAGE}'; "
                f"under '{HAND_FRAME_LOCAL}' the anchor is removed entirely, so an "
                f"offset would have nothing to act on.",
                file=sys.stderr,
            )
            return 1
        for side in ("left", "right"):
            spec = aim_to_wrist.get(side)
            if spec is None:
                print(
                    f"ERROR: 'aim_to_wrist' must define both 'left' and 'right'; "
                    f"missing {side!r}.",
                    file=sys.stderr,
                )
                return 1
            if (
                len(spec.get("position", [])) != 3
                or len(spec.get("quaternion", [])) != 4
            ):
                print(
                    f"ERROR: aim_to_wrist.{side} needs position[3] and quaternion[4] "
                    f"(qx, qy, qz, qw).",
                    file=sys.stderr,
                )
                return 1
            n = float(np.linalg.norm(spec["quaternion"]))
            if abs(n - 1.0) > 1e-3:
                # Silently normalising would hide a transposed or wxyz-ordered
                # quaternion, which stays unit-norm and produces a plausible but
                # wrong hand for the entire dataset.
                print(
                    f"ERROR: aim_to_wrist.{side}.quaternion has norm {n:.6f}, expected 1. "
                    f"Check the order is [qx, qy, qz, qw].",
                    file=sys.stderr,
                )
                return 1

    # Set before TeleopSession forks the plugin, which inherits this environment.
    # An explicit MANUS_WRIST_SOURCE in the caller's environment still wins, so
    # `MANUS_WRIST_SOURCE=auto python3 record_cameras.py` remains a one-liner.
    if collect_hands:
        os.environ.setdefault("MANUS_WRIST_SOURCE", wrist_source)

    if root is None or str(root) in ("", "None"):
        print(
            "ERROR: dataset root not set (use --root or set 'root' in config.json)",
            file=sys.stderr,
        )
        return 1
    if not task:
        print(
            "ERROR: task not set (use --task or set 'task' in config.json)",
            file=sys.stderr,
        )
        return 1

    cameras = DEFAULT_CAMERAS
    if args.cameras:
        cameras = dict(c.split("=", 1) for c in args.cameras)

    # ------------------------------------------------------------------ ROS
    rclpy.init()
    buf = CameraBuffer(cameras)
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
    if not buf.wait_for_all(list(cameras), timeout=30.0):
        missing = set(cameras) - set(buf.snapshot())
        print(f"ERROR: no frames from {sorted(missing)}", file=sys.stderr)
        print(
            "  Check that all cameras are publishing and that RMW_IMPLEMENTATION /\n"
            "  CYCLONEDDS_URI match the camera bringup.",
            file=sys.stderr,
        )
        teardown()
        return 1

    snap = buf.snapshot()
    for name, (img, _, _) in sorted(snap.items()):
        print(f"  {name}: {img.shape[1]}x{img.shape[0]}")

    print(f"\nMeasuring publish rates over 3s (target {fps} Hz)...")
    rates = buf.measure_rates(3.0)
    slow = {n: r for n, r in rates.items() if r < fps * 0.9}
    for name in sorted(rates):
        flag = "   <-- TOO SLOW" if name in slow else ""
        print(f"  {name:14} {rates[name]:6.1f} Hz{flag}")
    if slow:
        print("\nWARNING: cameras above are not keeping up.")
        print("Continue anyway? [y/N] ", end="", flush=True)
        if getch().lower() != "y":
            print()
            teardown()
            return 1
        print()

    # ---------------------------------------------------------- Dataset setup
    features: dict = {}
    for name, (img, _, _) in snap.items():
        h, w = img.shape[:2]
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
        features["observation.controller_left"] = {
            "dtype": "float32",
            "shape": (CONTROLLER_POSE_DIM,),
            "names": CONTROLLER_POSE_NAMES,
        }
        features["observation.controller_right"] = {
            "dtype": "float32",
            "shape": (CONTROLLER_POSE_DIM,),
            "names": CONTROLLER_POSE_NAMES,
        }
        if collect_hands:
            # Shape and names both follow hand_frame, so info.json stays self-describing:
            # a 168-wide column whose first name is "thumb_metacarpal" is unambiguously
            # wrist-local, and nothing downstream has to be told which mode produced it.
            hand_dim = hand_feature_dim(hand_frame)
            hand_names = hand_joint_names(hand_frame)
            features["observation.hand_left"] = {
                "dtype": "float32",
                "shape": (hand_dim,),
                "names": hand_names,
            }
            features["observation.hand_right"] = {
                "dtype": "float32",
                "shape": (hand_dim,),
                "names": hand_names,
            }

    # LeRobot defaults to g=2 so training can seek to any frame while decoding at
    # most one other. This board's NVENC rejects that outright -- avcodec_open2
    # fails with EINVAL for g=2 and g=3, and succeeds from g=4 up -- which is why
    # h264_nvenc looked unopenable here while a bare encoder opened fine. Four is
    # the smallest value it takes, so it keeps seeking cheap (decode at most three
    # extra frames) and costs about 19% more bitrate than a long GOP.
    rgb_encoder = RGBEncoderConfig(vcodec=encoder, crf=23, g=NVENC_MIN_GOP)
    dataset = LeRobotDataset.create(
        repo_id=f"teleop/{robot_type}",
        fps=fps,
        features=features,
        root=root,
        use_videos=True,
        streaming_encoding=True,
        rgb_encoder=rgb_encoder,
        # Deliberately smaller than any single episode's data/video, so every
        # save_episode() rolls to a fresh file and closes (footer-finalizes) the
        # previous one immediately. See --data-file-size-mb help for why.
        data_files_size_in_mb=data_file_size_mb,
        video_files_size_in_mb=data_file_size_mb,
        # LeRobot separately buffers episode *index* rows (meta/episodes/*.parquet,
        # the start/end frame + chunk/file pointers save_episode() needs to find an
        # episode's data again) and only writes that buffer out every 10 episodes
        # by default. A crash between flushes orphans however many already-closed,
        # perfectly valid data/video files came before it -- the raw frames are
        # still on disk, but nothing points at them. Flush every episode so the
        # index never lags behind what data_files_size_in_mb already made durable.
        metadata_buffer_size=1,
    )

    cam_names = list(cameras.keys())
    print(f"\nTask    : {task}")
    print(f"Root    : {root}")
    print(f"Encoder : {encoder}")
    print(
        f"Data/video file size cap: {data_file_size_mb}MB "
        f"(one episode per file below this size)"
    )
    if use_hand:
        if collect_hands:
            print(
                f"Hand    : Manus glove ({hand_feature_dim(hand_frame)} floats/hand × 2, "
                f"frame={hand_frame}"
                + (", custom aim_to_wrist" if aim_to_wrist else "")
                + ")"
            )
            print(
                f"Wrist   : {os.environ.get('MANUS_WRIST_SOURCE', 'auto')} "
                f"(confirm against the plugin's 'wrist source:' line below)"
            )
        else:
            print(
                "Hand    : disabled (--no-manus); head pose + controllers still recorded"
            )
        print(f"Head    : Pico head pose ({HEAD_POSE_DIM} floats, STAGE space)")
        print(f"Ctrl    : grip pose + validity ({CONTROLLER_POSE_DIM} floats × 2)")
        if camera_lag_frames:
            print(
                f"Lag     : hand/controller taken {camera_lag_frames} frames "
                f"({camera_lag_frames * 1000.0 / fps:.0f} ms) back to match the cameras"
            )
        else:
            print(
                "Lag     : compensation disabled; hand/controller are the newest "
                "samples, so they lead the images"
            )
    else:
        print("Hand    : disabled (--no-hand)")
        print("Head    : disabled (--no-hand)")
        print("Ctrl    : disabled (--no-hand)")
    print("\nControls:  s = start recording   e = end recording")
    print("           y = save episode       n = discard episode")
    print("           Ctrl+C = quit and save dataset\n")

    # --------------------------------------------------------- Manus hand buffer
    # Use a no-op context when hand capture is disabled so the episode loop is
    # identical in both cases.
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

    with hand_ctx as hand_buf:
        if use_hand and hand_buf is not None:
            if collect_hands:
                print("Waiting for Manus glove data (up to 30 s)...")
                if hand_buf.wait_ready(timeout=30.0):
                    print("  Manus glove ready.\n")
                else:
                    print(
                        "  WARNING: no glove data yet; hand features will be zeros "
                        "until the glove connects.\n"
                    )
            else:
                print("Waiting for head/controller data (up to 30 s)...")
                if hand_buf.wait_ready(timeout=30.0):
                    print("  Head/controller ready.\n")
                else:
                    print(
                        "  WARNING: no head/controller data yet; those columns "
                        "will be zeros until a client connects.\n"
                    )

        # ------------------------------------------------------ Episode loop
        try:
            while True:
                print("Press 's' to start a new episode...", end="\r", flush=True)
                ch = getch()
                if ch != "s":
                    continue

                ep_idx = dataset.meta.total_episodes
                print(f"\nRecording episode {ep_idx}... press 'e' to stop")

                stop_evt = threading.Event()
                result: dict = {}
                t = threading.Thread(
                    target=lambda: result.update(
                        zip(
                            ("n", "drops", "skips", "unmatched"),
                            record_episode(
                                buf,
                                hand_buf,
                                dataset,
                                cam_names,
                                task,
                                fps,
                                state_dim,
                                action_dim,
                                stop_evt,
                                camera_lag_frames,
                                collect_hands,
                            ),
                        )
                    ),
                    daemon=True,
                )
                t.start()

                while not stop_evt.is_set():
                    ch = getch()
                    if ch == "e":
                        stop_evt.set()

                t.join()
                n = result.get("n", 0)
                drops = result.get("drops", 0)
                skips = result.get("skips", 0)
                unmatched = result.get("unmatched", 0)

                # LeRobot's streaming encoder drops video frames when its queue is
                # full but still writes the parquet row, so rows can outnumber mp4
                # frames per camera and the episode is silently desynced. Read the
                # counters before save_episode(); start_episode() resets them.
                enc = getattr(dataset.writer, "_streaming_encoder", None)
                enc_drops = {
                    k: v for k, v in getattr(enc, "_dropped_frames", {}).items() if v
                }

                print(f"Episode {ep_idx}: {n} frames ({n / fps:.1f}s at {fps} fps)")
                # pop_latest() counted these all along but nobody ever read the
                # counter, which is how a backlog stayed invisible: a stalled loop
                # leaves the queues permanently full, and every frame recorded
                # after that is stale by the standing backlog. Non-zero here means
                # the loop is running behind the cameras; the average is how many
                # frames deep each camera queue sat beyond the one that was kept.
                if skips:
                    per_cam = skips / max(n * len(cam_names), 1)
                    print(
                        f"  NOTE: {skips} stale frames discarded "
                        f"({per_cam:.2f} per camera per row): the recording loop "
                        f"is running behind the cameras."
                    )
                if unmatched:
                    print(
                        f"  WARNING: {unmatched} rows discarded with no pose within "
                        f"half a frame of {camera_lag_frames} frames ago. The hand "
                        f"poll thread is not keeping up, or the lag exceeds the "
                        f"{HAND_HISTORY_SECONDS:.0f}s of history kept."
                    )
                if drops:
                    print(
                        f"  WARNING: {drops} synced frame groups dropped (queue overflow)"
                    )
                if enc_drops:
                    for cam, c in sorted(enc_drops.items()):
                        print(f"  ERROR: encoder dropped {c} frame(s) for {cam}")
                    print(
                        "  This episode is DESYNCED: parquet rows outnumber video "
                        "frames, so row N no longer matches video frame N."
                    )

                zero_rows = find_all_zero_rows(dataset, collect_hands)
                if zero_rows:
                    for col, idx in sorted(zero_rows.items()):
                        preview = idx[:5]
                        more = f" (+{len(idx) - 5} more)" if len(idx) > 5 else ""
                        print(
                            f"  ERROR: {col} is all-zero for {len(idx)} row(s), "
                            f"e.g. frame {preview}{more} -- device gave no data "
                            f"at all, not just an isolated tracking dropout."
                        )

                print(
                    "Save this episode?  y = save   n = discard", end="  ", flush=True
                )
                while True:
                    ch = getch()
                    if ch == "y":
                        if zero_rows:
                            print(
                                "\n  Cannot save: all-zero rows above would silently "
                                "corrupt this episode. Press 'n' to discard.",
                                end="  ",
                                flush=True,
                            )
                            continue
                        if n > 0:
                            dataset.save_episode()
                            print(
                                f"\nSaved episode {ep_idx}. "
                                f"Total: {dataset.meta.total_episodes}."
                            )
                        else:
                            dataset.clear_episode_buffer(delete_images=True)
                            print("\nEpisode was empty, nothing saved.")
                        break
                    if ch == "n":
                        dataset.clear_episode_buffer(delete_images=True)
                        print(
                            f"\nDiscarded episode {ep_idx}. "
                            f"Total: {dataset.meta.total_episodes}."
                        )
                        break

        except KeyboardInterrupt:
            print("\n\nCtrl+C — finalizing dataset...")
        except Exception as exc:
            # Anything that blows up here (most likely dataset.save_episode(),
            # e.g. an encoder thread crash) used to propagate straight past
            # dataset.finalize() below -- the ParquetWriter for the current
            # data/video file never got its footer written, and since it was
            # shared across every episode buffered into that file, ALL of them
            # went from "on disk" to "unreadable" at once, not just the one
            # being saved. --data-file-size-mb keeps that blast radius to one
            # episode, but we still need finalize() to actually run so
            # already-completed files get closed out cleanly instead of the
            # process just dying here. The recording session ends either way:
            # whatever broke (e.g. the encoder threads) is not something this
            # loop can safely keep recording through.
            print(f"\n\nERROR during recording: {exc!r}")
            print(
                "Stopping the session so already-saved episodes stay intact "
                "-- finalizing dataset..."
            )

    try:
        dataset.finalize()
    except Exception as exc:
        print(f"ERROR: dataset.finalize() failed too: {exc!r}", file=sys.stderr)
        print(
            "Whatever was already rolled into its own closed file (see "
            "--data-file-size-mb) should still be openable; check "
            f"{root} for the most recent chunk.",
            file=sys.stderr,
        )
    teardown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
