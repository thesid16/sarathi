"""Multi-object tracking.

Detection alone cannot drive speech. Without stable identity the system has no
way to know that the chair it is looking at is the same chair it mentioned two
seconds ago, so it either repeats itself endlessly or goes silent using a
timer that also suppresses genuinely new objects. Tracking is what makes
"don't repeat yourself" mean "don't repeat this object" rather than "don't say
this sentence".

Two further things fall out of having tracks, both of which matter more than
the tracking itself:

**Confirmation.** A detector run at 5-8 Hz produces single-frame false
positives. Announcing them is worse than missing them - a system that
occasionally shouts "car!" at a shadow gets switched off, and then it detects
nothing at all. A track must be seen `min_hits` times before anything is said
about it.

**Closing speed.** A car 8 m away approaching at 6 m/s is a different fact from
a car parked 8 m away, and only a track can tell them apart.

The tracker is intentionally simple: greedy IoU matching, per class, no Kalman
filter. At 5-8 Hz with a walking user, objects move a few pixels between
frames and IoU is sufficient. A motion model would add state, tuning and
failure modes to solve a problem this product does not have.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..types import Detection, Hazard


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """One object followed across frames."""

    track_id: int
    label: str
    class_id: int
    box: tuple[float, float, float, float]
    score: float
    hazard: Hazard
    distance_m: float | None
    bearing_deg: float | None
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    #: (timestamp, distance) samples, newest last. Bounded - only recent
    #: history is relevant to closing speed.
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    @property
    def closing_speed_mps(self) -> float | None:
        """Metres per second of approach. Positive means getting nearer.

        Least-squares slope over the recent history rather than a two-point
        difference: distance estimates are noisy, and a single bad frame
        should not read as a two-metre lunge.
        """
        if len(self.history) < 3:
            return None
        times = [t for t, _ in self.history]
        dists = [d for _, d in self.history]
        span = times[-1] - times[0]
        if span < 0.2:
            return None
        mean_t = sum(times) / len(times)
        mean_d = sum(dists) / len(dists)
        denom = sum((t - mean_t) ** 2 for t in times)
        if denom <= 0:
            return None
        slope = sum((t - mean_t) * (d - mean_d) for t, d in zip(times, dists)) / denom
        return -slope  # distance shrinking = positive closing speed

    def time_to_contact_s(self) -> float | None:
        """Seconds until arrival at current closing speed, if approaching."""
        speed = self.closing_speed_mps
        if speed is None or speed <= 0.05 or self.distance_m is None:
            return None
        return self.distance_m / speed


class Tracker:
    """Greedy per-class IoU tracker."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        min_hits: int = 2,
        max_misses: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        """Advance the tracker one frame. Returns the confirmed, live tracks."""
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))

        # Score every plausible pairing, then take them greedily best-first.
        # Matching only within a class stops a person's box from inheriting the
        # identity of the chair it walked in front of.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                overlap = iou(track.box, det.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, ti, di))
        pairs.sort(reverse=True)

        for overlap, ti, di in pairs:
            if ti not in unmatched_tracks or di not in unmatched_dets:
                continue
            self._absorb(self.tracks[ti], detections[di], ts)
            detections[di].track_id = self.tracks[ti].track_id
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(di)

        for ti in unmatched_tracks:
            self.tracks[ti].misses += 1

        for di in sorted(unmatched_dets):
            track = self._spawn(detections[di], ts)
            detections[di].track_id = track.track_id
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.misses == 0 and t.confirmed(self.min_hits)]

    # -- internals ---------------------------------------------------------

    def _spawn(self, det: Detection, ts: float) -> Track:
        track = Track(
            track_id=self._next_id,
            label=det.label,
            class_id=det.class_id,
            box=det.box,
            score=det.score,
            hazard=det.hazard,
            distance_m=det.distance_m,
            bearing_deg=det.bearing_deg,
            first_seen=ts,
            last_seen=ts,
        )
        self._next_id += 1
        if det.distance_m is not None:
            track.history.append((ts, det.distance_m))
        return track

    def _absorb(self, track: Track, det: Detection, ts: float) -> None:
        track.box = det.box
        track.score = det.score
        track.hazard = det.hazard
        track.bearing_deg = det.bearing_deg
        track.last_seen = ts
        track.hits += 1
        track.misses = 0
        if det.distance_m is not None:
            # Light smoothing. Distance drives what gets spoken, and an
            # unsmoothed estimate makes the same object drift between "two
            # metres" and "three metres" on consecutive frames.
            if track.distance_m is None:
                track.distance_m = det.distance_m
            else:
                track.distance_m = 0.6 * track.distance_m + 0.4 * det.distance_m
            track.history.append((ts, track.distance_m))
        else:
            track.distance_m = None

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1


def lateral_offset_m(distance_m: float, bearing_deg: float) -> float:
    """How far to the side of the walking line an object sits, in metres.

    The number that decides whether something is in the way. A pole 30 degrees
    off at eight metres is four metres to the side and irrelevant; the same
    30 degrees at one metre is half a metre away and about to be walked into.
    Bearing alone cannot tell those apart, which is why saliency uses this
    instead.
    """
    return distance_m * math.tan(math.radians(bearing_deg))
