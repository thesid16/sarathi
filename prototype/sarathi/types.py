"""Core data types shared across the whole pipeline.

These are deliberately plain and dependency-free. Every stage - sources,
perception, guidance - speaks in these types, which is what makes the stages
independently swappable.

Coordinate conventions used everywhere in this codebase:
  * Boxes are (x1, y1, x2, y2) in *pixels* of the frame they came from,
    origin top-left.
  * Bearing is in degrees, 0 = straight ahead, negative = left, positive
    = right. This is converted to clock-face positions only at the very
    last moment, in the phrasing layer.
  * Distance is in metres. `None` means "not estimated", which is different
    from "far away" and must never be silently treated as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceInfo:
    """Static description of where frames are coming from."""

    source_id: str
    kind: str  # "webcam" | "mjpeg" | "rtsp" | "file"
    width: int
    height: int
    nominal_fps: float
    # True when frames arrive over a network link, which means latency is
    # variable and staleness checks matter. Local cameras are effectively
    # instant; an ESP32-CAM over WiFi is not.
    is_networked: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Frame:
    """One captured image plus the timing metadata we need for latency work.

    `ts_capture` is the best available estimate of when light hit the sensor.
    For local cameras that is close to `ts_received`; for a networked source we
    can usually only bound it, so the two differ and the gap *is* the transport
    latency we report in benchmarks.
    """

    image: np.ndarray  # HxWx3, BGR, uint8
    seq: int
    ts_capture: float
    ts_received: float
    source_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def transport_latency_ms(self) -> float:
        return (self.ts_received - self.ts_capture) * 1000.0

    def age_ms(self, now: float | None = None) -> float:
        """How stale this frame is right now.

        The scheduler uses this to throw away frames that queued up behind a
        slow inference pass. Announcing a chair that was there 800 ms ago is
        worse than saying nothing.
        """
        return ((now if now is not None else time.monotonic()) - self.ts_capture) * 1000.0


class Hazard(Enum):
    """How much a class matters when deciding what is worth saying.

    Ranking detections purely by confidence or size produces a system that
    narrates walls while ignoring a descending staircase. This enum is the
    coarse prior that stops that; the saliency engine refines it with
    geometry.
    """

    CRITICAL = 3  # drop-offs, stairs down, open manhole, moving vehicle
    HIGH = 2  # head-height and shin-height obstacles, kerbs, poles
    MEDIUM = 1  # furniture, doors, people
    LOW = 0  # context objects - cup, book, plant


@dataclass
class Detection:
    """One detected object in one frame."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    class_id: int
    label: str
    # Populated by the tracker; None until an object has been seen twice.
    track_id: int | None = None
    # Populated by the distance stage. Metres. None = not estimated.
    distance_m: float | None = None
    # How the distance was obtained, so the docs and the phrasing layer can
    # treat a depth-net reading differently from a rough geometric guess.
    distance_source: str | None = None  # "geometric" | "depth_net" | "fused"
    bearing_deg: float | None = None
    hazard: Hazard = Hazard.LOW

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class PerceptionResult:
    """Everything perception concluded about one frame."""

    frame_seq: int
    ts_capture: float
    detections: list[Detection] = field(default_factory=list)
    # Coarse relative depth map, if a depth pass ran on this frame. Low-res.
    depth_map: np.ndarray | None = None
    # Per-stage wall-clock cost in milliseconds, for the benchmark harness.
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


class Urgency(Enum):
    """Controls whether an utterance can interrupt one already playing."""

    AMBIENT = 0  # spoken only if nothing else is queued
    NORMAL = 1
    URGENT = 2  # pre-empts current speech, preceded by an earcon


@dataclass
class Utterance:
    """Something the system wants to say, before it becomes audio."""

    text: str
    urgency: Urgency = Urgency.NORMAL
    # Dedup key - the anti-repeat cooldown is keyed on this, not on the text,
    # so "chair 2 o'clock 1.5 m" and "chair 2 o'clock 1.2 m" collapse to one
    # subject and do not both get spoken.
    topic: str | None = None
    earcon: str | None = None
    lang: str = "en"
