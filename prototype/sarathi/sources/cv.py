"""Sources backed by OpenCV's VideoCapture: local webcam, RTSP, and video files.

RTSP covers both the Raspberry Pi rig and the office WiFi IP camera - they are
the same protocol, so they are the same class with a different URL.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from ..types import Frame, SourceInfo
from ..util.log import get_logger
from .base import FrameSource, SourceError

log = get_logger(__name__)


class CvSource(FrameSource):
    """Shared plumbing for anything OpenCV can open."""

    kind = "cv"

    def __init__(self, source_id: str, target: int | str, *, api: int = cv2.CAP_ANY) -> None:
        super().__init__(source_id)
        self._target = target
        self._api = api
        self._cap: cv2.VideoCapture | None = None

    def _configure(self, cap: cv2.VideoCapture) -> None:
        """Hook for subclasses to set properties after opening."""

    def open(self) -> SourceInfo:
        cap = cv2.VideoCapture(self._target, self._api)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"cannot open {self.kind} source {self._target!r}")
        self._configure(cap)
        self._cap = cap

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._info = SourceInfo(
            source_id=self.source_id,
            kind=self.kind,
            width=width,
            height=height,
            nominal_fps=fps,
            is_networked=self.kind == "rtsp",
            detail={"target": str(self._target)},
        )
        log.info("opened %s %s (%dx%d @ %.1f fps)", self.kind, self._target, width, height, fps)
        return self._info

    def grab(self) -> Frame | None:
        if self._cap is None:
            raise SourceError("grab() before open()")
        ok, image = self._cap.read()
        received = time.monotonic()
        if not ok or image is None:
            return self._on_read_failure()
        return Frame(
            image=image,
            seq=self._next_seq(),
            # No transport-time information is available from VideoCapture, so
            # capture and receive coincide. For RTSP this understates true
            # latency; see docs/07-benchmarks.md for how end-to-end latency is
            # actually measured (external millisecond-clock method).
            ts_capture=received,
            ts_received=received,
            source_id=self.source_id,
        )

    def _on_read_failure(self) -> Frame | None:
        raise SourceError(f"{self.kind} read failed")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamSource(CvSource):
    """Local USB / built-in camera. Used for the Mac-side prototype loop."""

    kind = "webcam"

    def __init__(
        self,
        source_id: str = "webcam",
        index: int = 0,
        *,
        width: int | None = 1280,
        height: int | None = 720,
        fps: float | None = 30.0,
    ) -> None:
        # AVFoundation is the working backend on macOS; CAP_ANY picks a slower
        # path and sometimes ignores resolution requests.
        api = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        super().__init__(source_id, index, api=api)
        self._want = (width, height, fps)

    def _configure(self, cap: cv2.VideoCapture) -> None:
        width, height, fps = self._want
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        # One-frame driver buffer. Without this the driver hands us frames that
        # are already several frames old whenever inference runs slower than
        # capture.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class RtspSource(CvSource):
    """RTSP stream - Raspberry Pi rig or a fixed WiFi IP camera.

    FFmpeg defaults are tuned for smooth playback, which means it will happily
    buffer a second of video to avoid a stutter. That is exactly the wrong
    trade here, so the buffering is disabled through the capture options.
    """

    kind = "rtsp"

    def __init__(
        self,
        source_id: str,
        url: str,
        *,
        transport: str = "tcp",
        timeout_us: int = 5_000_000,
    ) -> None:
        # These options are read by the FFmpeg backend at VideoCapture
        # construction time, from the environment. There is no per-capture API
        # for them in the Python bindings, so it has to be set here.
        opts = "|".join(
            [
                f"rtsp_transport;{transport}",
                "fflags;nobuffer",
                "flags;low_delay",
                f"stimeout;{timeout_us}",
                "reorder_queue_size;0",
            ]
        )
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        super().__init__(source_id, url, api=cv2.CAP_FFMPEG)

    def _configure(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class FileSource(CvSource):
    """Recorded video. The backbone of reproducible benchmarking.

    Two modes matter:

    * `realtime=True` paces playback to the file's own frame rate, so the
      scheduler, gating and speech budget behave as they would on a live feed.
      This is the mode for evaluating *guidance* quality.
    * `realtime=False` runs as fast as the pipeline allows, for measuring raw
      throughput and accuracy. Guidance timing is meaningless in this mode.
    """

    kind = "file"

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        *,
        realtime: bool = True,
        loop: bool = False,
    ) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            raise SourceError(f"video file not found: {p}")
        super().__init__(source_id, str(p), api=cv2.CAP_FFMPEG)
        self._realtime = realtime
        self._loop = loop
        self._t0: float | None = None

    def open(self) -> SourceInfo:
        info = super().open()
        self._t0 = None
        return info

    def grab(self) -> Frame | None:
        if self._cap is None:
            raise SourceError("grab() before open()")

        # Presentation timestamp must be read *before* the frame, because
        # POS_MSEC advances past the frame we are about to receive.
        pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        ok, image = self._cap.read()
        if not ok or image is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._t0 = None
                return self.grab()
            return None  # clean end of stream

        if self._realtime:
            if self._t0 is None:
                self._t0 = time.monotonic() - pos_ms / 1000.0
            target = self._t0 + pos_ms / 1000.0
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        received = time.monotonic()
        return Frame(
            image=image,
            seq=self._next_seq(),
            ts_capture=received,
            ts_received=received,
            source_id=self.source_id,
            meta={"pos_ms": pos_ms},
        )

    def _on_read_failure(self) -> Frame | None:
        return None
