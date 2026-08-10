"""Frame source abstraction.

The product has to accept video from anything: the phone's own camera, an
ESP32-CAM on a cap, a Raspberry Pi rig, an office IP camera, or a recorded
file during benchmarking. Everything downstream of this module is written
against `FrameSource` and never learns which of those it is talking to.

Two policies are enforced here, once, for every source:

**Drop, don't queue.** `LatestFrame` runs the blocking grab on its own thread
and keeps exactly one frame - the newest. If inference takes 120 ms and the
camera produces frames every 33 ms, the three frames that arrived meanwhile are
discarded rather than backing up. A queue would give smooth-looking throughput
while the guidance drifted further and further behind reality, which for this
product is a safety bug, not a performance one.

**Reconnect, don't die.** Networked sources drop out. WiFi roams, an ESP32
browns out under load. The wrapper reconnects with backoff and reports the gap
instead of terminating the session, because a blind user cannot be expected to
notice the app went quiet and restart it.
"""

from __future__ import annotations

import abc
import threading
import time
from typing import Any

from ..types import Frame, SourceInfo
from ..util.log import get_logger

log = get_logger(__name__)


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed unrecoverably."""


class FrameSource(abc.ABC):
    """A blocking, single-threaded producer of frames.

    Implementations only need to handle the happy path plus raising on
    failure. Threading, staleness and reconnection are handled by
    `LatestFrame`, so each concrete source stays small and testable.
    """

    #: Short transport name, surfaced in logs, benchmarks and SourceInfo.
    kind: str = "unknown"

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._seq = 0
        self._info: SourceInfo | None = None

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            raise SourceError(f"source {self.source_id!r} queried before open()")
        return self._info

    @abc.abstractmethod
    def open(self) -> SourceInfo:
        """Acquire the underlying device or connection. Raises `SourceError`."""

    @abc.abstractmethod
    def grab(self) -> Frame | None:
        """Block until the next frame is available.

        Returns None on clean end-of-stream (a file running out). Raises
        `SourceError` on a failure that reconnecting might fix.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device or connection. Must be safe to call twice."""

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class LatestFrame:
    """Threaded single-slot buffer over a `FrameSource`.

    Usage:
        with LatestFrame(MjpegSource(...)) as cam:
            while True:
                frame = cam.get(timeout=1.0)

    `get()` returns the newest frame captured so far and will not return the
    same frame twice - it blocks until something newer arrives, so a fast
    consumer does not spin re-processing one image.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        reconnect: bool = True,
        max_backoff_s: float = 5.0,
    ) -> None:
        self._source = source
        self._reconnect = reconnect
        self._max_backoff_s = max_backoff_s

        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: Frame | None = None
        self._latest_seq_taken = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False
        self._error: BaseException | None = None

        # Counters the benchmark harness reads to report real capture health.
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnects = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "LatestFrame":
        self._source.open()
        self._thread = threading.Thread(
            target=self._run, name=f"cap-{self._source.source_id}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._source.close()

    def __enter__(self) -> "LatestFrame":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- consumer side -----------------------------------------------------

    def get(self, timeout: float = 1.0, *, allow_repeat: bool = False) -> Frame | None:
        """Return the newest unseen frame, or None on timeout / end of stream."""
        deadline = time.monotonic() + timeout
        with self._new_frame:
            while True:
                have_new = self._latest is not None and (
                    allow_repeat or self._latest.seq != self._latest_seq_taken
                )
                if have_new:
                    frame = self._latest
                    assert frame is not None
                    self._latest_seq_taken = frame.seq
                    return frame
                if self._ended:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)

    @property
    def ended(self) -> bool:
        """True once the underlying stream finished and will produce no more."""
        with self._lock:
            return self._ended

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def info(self) -> SourceInfo:
        return self._source.info

    # -- producer thread ---------------------------------------------------

    def _run(self) -> None:
        backoff = 0.25
        while not self._stop.is_set():
            try:
                frame = self._source.grab()
            except SourceError as exc:
                if not self._reconnect:
                    self._finish(exc)
                    return
                log.warning(
                    "source %s failed (%s); reconnecting in %.2fs",
                    self._source.source_id,
                    exc,
                    backoff,
                )
                self._source.close()
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self._max_backoff_s)
                try:
                    self._source.open()
                    self.reconnects += 1
                    backoff = 0.25
                except SourceError as reopen_exc:
                    log.warning("reconnect to %s failed: %s", self._source.source_id, reopen_exc)
                continue
            except Exception as exc:  # noqa: BLE001 - surfaced to the consumer
                self._finish(exc)
                return

            if frame is None:  # clean end of stream
                self._finish(None)
                return

            with self._new_frame:
                # If the consumer never took the previous frame, it is being
                # dropped right now. Counting this is how we show in the docs
                # that the pipeline is capture-bound or inference-bound.
                if self._latest is not None and self._latest.seq != self._latest_seq_taken:
                    self.frames_dropped += 1
                self._latest = frame
                self.frames_captured += 1
                self._new_frame.notify_all()

        self._finish(None)

    def _finish(self, exc: BaseException | None) -> None:
        with self._new_frame:
            self._ended = True
            self._error = exc
            self._new_frame.notify_all()
