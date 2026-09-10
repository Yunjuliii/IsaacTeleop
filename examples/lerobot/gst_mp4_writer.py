# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-MP4-per-camera H.264 writer built on GStreamer's hardware encoder.

Why this exists instead of LeRobot's PyAV streaming encoder
-----------------------------------------------------------
PyAV does not release the GIL while ``avcodec_open2`` runs, and opening
``h264_nvenc`` on this board is slow. Measured here, opening four encoders
concurrently took 775 ms and held the GIL for 522 ms of that; every later
episode still froze the whole interpreter for 143-277 ms. During the freeze the
rclpy executor cannot run, and because the camera subscriptions use
``KEEP_LAST`` with a shallow depth, the frames that arrive in that window are
dropped inside DDS. They never reach Python, so every drop counter in the
recorder stayed at zero while roughly 25 frames per camera went missing at the
start of each episode.

GStreamer does its encoding on its own threads, so none of that touches the
GIL. Same board, same four 640x480 streams:

    open 4 encoders     73 ms   (GIL held 3 ms)
    max stall in run     7 ms
    finalise 4 files   135 ms
    push per frame    0.39 ms median, over four cameras
    total CPU          0.35 cores including colour conversion and encode

Colour conversion runs in the pipeline rather than in CuPy. ``camera_viz`` uses
a CUDA kernel because its frames are already GPU-resident; frames here arrive
from ROS as CPU ``rgb8``, so a GPU round trip would only add an upload and a
download. Feeding ``nvvidconv`` pre-padded RGBA to move the conversion onto the
VIC was measured too and was worse: 6.20 ms per push versus 0.39 ms, for the
same 0.3 cores, because the padding copy runs under the GIL and the buffer is a
third larger.

Timestamps
----------
The MP4 is written at a constant frame rate: presentation time is
``frame_index / fps``, not the camera's capture time. Frame N of the MP4 is
therefore always row N of the sidecar table, which is the invariant LeRobot
needs. The real capture times are not thrown away, they are recorded per frame
in the sidecar so a gap stays visible and can be accounted for offline.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

# PyGObject ships with the distro, not in this project's venv, and the venv is
# built with --no-system-site-packages. Import it from the system path rather
# than requiring a venv install, which would have to be rebuilt against the
# distro's own GLib.
_DIST_PACKAGES = "/usr/lib/python3/dist-packages"

# Encoder elements in priority order. nvv4l2h264enc is the Jetson V4L2 M2M
# NVENC path; nvh264enc is the desktop GstCUDA one; x264enc is a software
# fallback that exists so a board with no hardware encoder still records, not
# because it is expected to keep up with four cameras.
_ENCODER_CANDIDATES = ("nvv4l2h264enc", "nvh264enc", "x264enc")

_GST_LOCK = threading.Lock()
_GST = None


def _gst():
    """Import and initialise GStreamer once per process."""
    global _GST
    with _GST_LOCK:
        if _GST is not None:
            return _GST
        try:
            import gi
        except ImportError:
            if _DIST_PACKAGES not in sys.path:
                sys.path.append(_DIST_PACKAGES)
            try:
                import gi
            except ImportError as exc:
                raise RuntimeError(
                    "GstMp4Writer needs PyGObject. On Ubuntu/L4T: "
                    "`sudo apt install python3-gi gstreamer1.0-tools "
                    "gstreamer1.0-plugins-good gstreamer1.0-plugins-bad`."
                ) from exc
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        _GST = Gst
        return _GST


def available_encoder() -> str:
    """Name of the best H.264 encoder element present, or raise."""
    Gst = _gst()
    for name in _ENCODER_CANDIDATES:
        if Gst.ElementFactory.find(name) is not None:
            return name
    raise RuntimeError(
        f"no usable H.264 encoder element (tried {_ENCODER_CANDIDATES}). "
        "On Jetson install `nvidia-l4t-gstreamer`; elsewhere install "
        "gst-plugins-ugly for x264enc."
    )


# The V4L2 layer under `nvv4l2h264enc` prints "Opening in BLOCKING MODE" to
# file descriptor 1 once per encoder, so four times per episode. It is not a
# Python print, so `redirect_stdout` cannot catch it, and it is not emitted
# inside `set_state` either: temporarily pointing fd 1 at /dev/null across the
# whole of open(), including a synchronous wait for the READY transition, was
# tried and still let all of it through, because the plugin writes later from
# its own thread. Callers should assume it will appear and not print prompts
# that a stray line can overwrite. It is harmless; real failures arrive on the
# pipeline bus, which open() and write() both check.

def _element_properties(name: str) -> set[str]:
    """Property names the installed build of an element actually exposes."""
    Gst = _gst()
    factory = Gst.ElementFactory.find(name)
    if factory is None:
        return set()
    element = factory.create(None)
    if element is None:
        return set()
    return {spec.name for spec in element.list_properties()}


def _encoder_args(
    name: str, *, bitrate: int, iframe_interval: int, quality: int | None
) -> str:
    """Build the encoder element string, dropping properties this build lacks.

    The property set moves between L4T releases: `maxperf-enable` exists on
    R35-era `nvv4l2h264enc` and is gone on this board's, while `preset-id` and
    `maxbitrate` are the Thor-era replacements for `preset-level` and
    `peak-bitrate`. Naming an absent property makes `parse_launch` fail
    outright, so ask the element what it has rather than branching on a version
    string.
    """
    if name == "nvv4l2h264enc":
        # control-rate 0 is VBR and 1 is CBR. Under VBR, `cq` is a target
        # quality on the same 0-51 scale x264 calls CRF, so `quality` carries
        # over from what the PyAV path expressed as crf.
        want: dict[str, object] = {
            "iframeinterval": iframe_interval,
            # An I-frame that is not an IDR does not reset the reference list,
            # so a decoder seeking to it can still need earlier frames. Making
            # every I-frame an IDR is what makes frame N independently
            # seekable, which is the whole point of a short interval.
            "idrinterval": iframe_interval,
            "insert-sps-pps": "true",
            # Baseline is the element default and costs bitrate for no benefit
            # here; 4 is High.
            "profile": 4,
            "maxperf-enable": "true",
        }
        if quality is None:
            want.update({"control-rate": 1, "bitrate": bitrate})
        else:
            want.update(
                {
                    "control-rate": 0,
                    "bitrate": bitrate,
                    "maxbitrate": bitrate * 3,
                    "peak-bitrate": bitrate * 3,
                    "cq": quality,
                }
            )
        have = _element_properties(name)
        return " ".join(
            [name] + [f"{k}={v}" for k, v in want.items() if k in have]
        )
    if name == "nvh264enc":
        return (
            f"{name} bitrate={bitrate // 1000} gop-size={iframe_interval} "
            f"rc-mode=vbr preset=hq"
        )
    if name == "x264enc":
        opts = f"{name} key-int-max={iframe_interval} speed-preset=veryfast"
        if quality is None:
            return f"{opts} bitrate={bitrate // 1000}"
        return f"{opts} pass=qual quantizer={quality}"
    raise RuntimeError(f"unknown encoder element {name!r}")


class GstMp4Writer:
    """Encode RGB frames to one H.264 MP4 through a private GStreamer pipeline.

    ``appsrc(RGB) ! videoconvert ! nvvidconv ! nvv4l2h264enc ! h264parse !
    mp4mux ! filesink``

    One instance owns one file and one camera. Not thread-safe: call
    :meth:`write` from a single thread.
    """

    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: int,
        *,
        bitrate: int = 4_000_000,
        iframe_interval: int = 4,
        quality: int | None = 23,
        queue_seconds: float = 4.0,
    ):
        """``quality`` is a 0-51 target on the same scale as x264's CRF, applied
        in VBR mode; pass ``None`` for plain CBR at ``bitrate``."""
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self._bitrate = bitrate
        self._iframe_interval = iframe_interval
        self._quality = quality
        self._frame_bytes = width * height * 3
        # Bound appsrc's queue instead of leaving it unlimited, so an encoder
        # that genuinely cannot keep up shows up as backpressure on write()
        # rather than as unbounded memory growth. With block=true a full queue
        # blocks the caller; PyGObject drops the GIL around the signal
        # emission, so blocking here does not freeze the ROS executor.
        self._max_bytes = int(self._frame_bytes * fps * queue_seconds)
        self._duration_ns = int(1e9 / fps)

        self._pipeline = None
        self._appsrc = None
        self._frames = 0
        self._encoder_name: str | None = None

    # -- lifecycle ------------------------------------------------------
    def open(self) -> None:
        """Build the pipeline and bring it to PLAYING. Raises on failure."""
        if self._pipeline is not None:
            raise RuntimeError(f"{self.path.name}: already open")
        Gst = _gst()
        self._encoder_name = available_encoder()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_str = " ! ".join(
            [
                (
                    f"appsrc name=src is-live=false format=time block=true "
                    f"max-bytes={self._max_bytes} "
                    f"caps=video/x-raw,format=RGB,width={self.width},"
                    f"height={self.height},framerate={self.fps}/1"
                ),
                # RGB is not one of nvvidconv's sysmem input formats, so the
                # RGB->NV12 step happens here. videoconvert runs on the
                # pipeline's own thread and is multi-threaded, so its cost does
                # not land on the recording loop.
                "videoconvert",
                "video/x-raw,format=NV12",
                "nvvidconv",
                "video/x-raw(memory:NVMM),format=NV12",
                _encoder_args(
                    self._encoder_name,
                    bitrate=self._bitrate,
                    iframe_interval=self._iframe_interval,
                    quality=self._quality,
                ),
                "h264parse",
                "mp4mux",
                f'filesink location="{self.path}" sync=false',
            ]
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        if self._appsrc is None:
            self._pipeline = None
            raise RuntimeError(f"{self.path.name}: pipeline has no appsrc")

        rc = self._pipeline.set_state(Gst.State.PLAYING)
        # ASYNC is the normal answer here and must not be treated as failure:
        # the sink cannot preroll until a buffer arrives, and the first buffer
        # only arrives once write() is called. Waiting for SUCCESS deadlocks
        # until the timeout every single time. A pipeline that is genuinely
        # broken either fails to parse above, reports FAILURE here, or posts a
        # bus error, which write() checks before every frame.
        if rc == Gst.StateChangeReturn.FAILURE:
            err = self._take_error() or "pipeline failed to start"
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            raise RuntimeError(f"{self.path.name}: {err}")
        err = self._take_error()
        if err is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            raise RuntimeError(f"{self.path.name}: {err}")

    def write(self, rgb: np.ndarray) -> None:
        """Queue one HxWx3 uint8 RGB frame. Raises if the pipeline has failed."""
        if self._pipeline is None:
            raise RuntimeError(f"{self.path.name}: writer is not open")
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"{self.path.name}: expected uint8 "
                f"{(self.height, self.width, 3)}, got {rgb.dtype} {rgb.shape}"
            )
        Gst = _gst()
        err = self._take_error()
        if err is not None:
            raise RuntimeError(f"{self.path.name}: {err}")

        buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(rgb).tobytes())
        buf.pts = self._frames * self._duration_ns
        buf.dts = buf.pts
        buf.duration = self._duration_ns
        rc = self._appsrc.emit("push-buffer", buf)
        if rc != Gst.FlowReturn.OK:
            raise RuntimeError(f"{self.path.name}: appsrc rejected frame ({rc})")
        self._frames += 1

    def close(self, timeout: float = 20.0) -> int:
        """Flush, finalise the MP4 and return the number of frames written.

        Sends end-of-stream and waits for it to reach the sink. Without that
        wait ``mp4mux`` never writes its index and the file is unplayable.
        """
        if self._pipeline is None:
            return self._frames
        Gst = _gst()
        self._appsrc.emit("end-of-stream")
        msg = self._pipeline.get_bus().timed_pop_filtered(
            int(timeout * Gst.SECOND), Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsrc = None
        if msg is None:
            raise RuntimeError(
                f"{self.path.name}: encoder did not finish within {timeout:.0f} s; "
                f"the file is likely truncated"
            )
        if msg.type == Gst.MessageType.ERROR:
            gerror, debug = msg.parse_error()
            raise RuntimeError(f"{self.path.name}: {gerror.message} ({debug})")
        return self._frames

    def abort(self) -> None:
        """Tear the pipeline down and delete the file. Never raises."""
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(_gst().State.NULL)
            except Exception:
                pass
            self._pipeline = None
            self._appsrc = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    # -- internal -------------------------------------------------------
    def _take_error(self) -> str | None:
        """Return a pending bus error, or None. Does not block."""
        if self._pipeline is None:
            return None
        Gst = _gst()
        msg = self._pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if msg is None:
            return None
        gerror, debug = msg.parse_error()
        return f"{gerror.message} ({debug})"

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def encoder(self) -> str | None:
        return self._encoder_name
