"""Tests for the source layer.

These run without any camera hardware: a synthetic `FrameSource` exercises the
threading, drop-don't-queue and reconnect behaviour, and a generated video file
exercises the real OpenCV path.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from sarathi.sources import SourceError, create_source, infer_kind
from sarathi.sources.base import FrameSource, LatestFrame
from sarathi.types import Frame, SourceInfo


class FakeSource(FrameSource):
    """Emits `count` frames at a fixed interval, optionally failing partway."""

    def __init__(self, count: int = 10, interval: float = 0.01, fail_at: int | None = None):
        super().__init__("fake")
        self._count = count
        self._interval = interval
        self._fail_at = fail_at
        self.opens = 0

    def open(self) -> SourceInfo:
        self.opens += 1
        self._info = SourceInfo("fake", "fake", 4, 4, 100.0)
        return self._info

    def grab(self) -> Frame | None:
        if self._seq >= self._count:
            return None
        if self._fail_at is not None and self._seq == self._fail_at:
            self._fail_at = None  # fail once, then recover
            raise SourceError("simulated dropout")
        time.sleep(self._interval)
        now = time.monotonic()
        seq = self._next_seq()
        return Frame(np.full((4, 4, 3), seq % 256, np.uint8), seq, now, now, "fake")

    def close(self) -> None:
        pass


def test_latest_frame_yields_every_frame_to_a_fast_consumer():
    with LatestFrame(FakeSource(count=5)) as cam:
        seqs = []
        while (frame := cam.get(timeout=1.0)) is not None:
            seqs.append(frame.seq)
    assert seqs == [1, 2, 3, 4, 5]


def test_latest_frame_drops_stale_frames_for_a_slow_consumer():
    """A consumer slower than the source must skip ahead, not fall behind."""
    with LatestFrame(FakeSource(count=40, interval=0.002)) as cam:
        first = cam.get(timeout=1.0)
        assert first is not None
        time.sleep(0.15)  # let a pile of frames arrive and be discarded
        second = cam.get(timeout=1.0)
        assert second is not None
        # The consumer skipped ahead rather than receiving frame 2.
        assert second.seq > first.seq + 1
        assert cam.frames_dropped > 0


def test_get_does_not_return_the_same_frame_twice():
    with LatestFrame(FakeSource(count=3, interval=0.01)) as cam:
        first = cam.get(timeout=1.0)
        assert first is not None
        again = cam.get(timeout=1.0)
        assert again is None or again.seq != first.seq


def test_source_reconnects_after_a_transient_failure():
    source = FakeSource(count=6, fail_at=3)
    with LatestFrame(source, max_backoff_s=0.05) as cam:
        deadline = time.monotonic() + 5.0
        while not cam.ended and time.monotonic() < deadline:
            cam.get(timeout=0.1)
        assert cam.reconnects >= 1
        assert source.opens >= 2


def test_end_of_stream_is_clean_not_an_error():
    with LatestFrame(FakeSource(count=2)) as cam:
        while cam.get(timeout=1.0) is not None:
            pass
        assert cam.ended
        assert cam.error is None


def test_get_times_out_without_blocking_forever():
    class Silent(FakeSource):
        def grab(self):
            time.sleep(10)
            return None

    cam = LatestFrame(Silent()).start()
    try:
        started = time.monotonic()
        assert cam.get(timeout=0.1) is None
        assert time.monotonic() - started < 1.0
    finally:
        cam.stop()


# -- registry ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("rtsp://10.0.0.5:8554/cam", "rtsp"),
        ("http://192.168.4.1:81/stream", "mjpeg"),
        ("clip.mp4", "file"),
        ("0", "webcam"),
    ],
)
def test_infer_kind(target, expected):
    assert infer_kind(target) == expected


def test_infer_kind_rejects_the_ambiguous():
    with pytest.raises(SourceError):
        infer_kind("some-random-string")


def test_create_source_rejects_unknown_kind():
    with pytest.raises(SourceError, match="unknown source kind"):
        create_source({"kind": "telepathy"})


def test_create_source_reports_bad_options_clearly():
    with pytest.raises(SourceError, match="bad options"):
        create_source({"kind": "rtsp", "url": "rtsp://x", "nonsense": 1})


# -- real OpenCV path --------------------------------------------------------


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    for i in range(15):
        writer.write(np.full((48, 64, 3), i * 15, np.uint8))
    writer.release()
    return path


def test_file_source_reads_a_real_video_to_completion(sample_video):
    source = create_source({"kind": "file", "path": str(sample_video), "realtime": False})
    with LatestFrame(source) as cam:
        count = 0
        while cam.get(timeout=2.0) is not None:
            count += 1
        assert cam.ended and cam.error is None
    assert count > 0


def test_file_source_realtime_mode_paces_playback(sample_video):
    """15 frames at 30 fps must take about half a second, not zero."""
    source = create_source({"kind": "file", "path": str(sample_video), "realtime": True})
    started = time.monotonic()
    with LatestFrame(source) as cam:
        while cam.get(timeout=2.0) is not None:
            pass
    elapsed = time.monotonic() - started
    assert 0.25 < elapsed < 2.0


def test_missing_file_fails_at_construction_not_mid_stream(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        create_source({"kind": "file", "path": str(tmp_path / "nope.mp4")})


def test_threads_are_cleaned_up(sample_video):
    before = threading.active_count()
    source = create_source({"kind": "file", "path": str(sample_video), "realtime": False})
    with LatestFrame(source) as cam:
        cam.get(timeout=2.0)
    time.sleep(0.1)
    assert threading.active_count() <= before
