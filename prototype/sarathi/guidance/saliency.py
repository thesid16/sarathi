"""Deciding what is worth saying.

This is the hardest part of the product and the one most likely to decide
whether anyone keeps using it. A detector at 5 Hz on a street produces
hundreds of true detections a minute. Speaking them is not assistance, it is
noise, and the documented reason people abandon assistive vision tools is that
they talk too much. Restraint here is the feature.

Ranking is by four things, and the weighting between them is data in the
config rather than constants in code, because it will be tuned against
recorded walks:

**Lateral offset, not bearing.** A pole thirty degrees off at eight metres is
four metres to the side and irrelevant. The same thirty degrees at one metre
is half a metre away and about to be walked into. Bearing cannot tell those
apart; metres to the side of the walking line can.

**Proximity, non-linearly.** The difference between 1 m and 2 m matters far
more than between 5 m and 6 m.

**Hazard class.** A descending staircase outranks a sofa whatever the geometry
says. Geometry alone produces a system that narrates walls and misses stairs.

**Closing speed.** A car 8 m away approaching at 6 m/s is not the same fact as
a car parked 8 m away.

Then two filters that do most of the work of not being annoying: a per-object
cooldown so the same thing is not repeated, and a hard utterance budget so the
system physically cannot talk over itself. Both can be overridden, but only by
a genuine escalation - something that has become much closer since it was last
mentioned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..perception.tracking import Track, lateral_offset_m
from ..types import Hazard, Urgency


@dataclass
class SaliencyConfig:
    """Tunable weights. Defaults are a starting point, not measured optima."""

    #: Beyond this, nothing is announced unprompted. Six metres is roughly
    #: four walking paces - far enough to react, near enough to be relevant.
    max_distance_m: float = 6.0
    #: Half-width of the walking corridor. Shoulder width plus a margin.
    corridor_half_width_m: float = 0.7
    #: Distance at which proximity scores 0.5. The curve is smooth and never
    #: reaches zero, so a critical hazard at the edge of range still carries
    #: some weight instead of losing its proximity term at exactly max range.
    proximity_half_m: float = 2.5

    #: Proximity and path multiply rather than being averaged: something is
    #: worth announcing when it is near AND in the way, not when it scores
    #: moderately on both. Averaging them produced a system where a chair
    #: squarely in the walking line ranked below the announcement threshold
    #: while a cup on a table nearly cleared it.
    hazard_bonus: float = 0.45
    closing_bonus: float = 0.35

    #: Below this combined score, say nothing.
    score_floor: float = 0.55
    #: Low-hazard classes are context, not navigation. They answer "what is on
    #: the table?" when asked and are never announced unprompted - otherwise
    #: the walking loop narrates cutlery.
    announce_low_hazard: bool = False
    #: Do not mention the same tracked object again within this window.
    repeat_cooldown_s: float = 8.0
    #: Nor the same class of thing, however many instances there are.
    class_cooldown_s: float = 3.0
    #: Hard floor on the gap between utterances.
    min_utterance_gap_s: float = 1.5

    #: A critical hazard nearer than this, in the walking corridor, interrupts.
    urgent_distance_m: float = 2.5
    #: So does anything arriving sooner than this.
    urgent_ttc_s: float = 2.0
    #: Even urgent alerts respect this, or they machine-gun.
    urgent_cooldown_s: float = 2.0

    #: Objects whose distance could not be estimated still deserve a chance to
    #: be mentioned, but should rank below anything measured.
    unknown_distance_proximity: float = 0.35


@dataclass
class Ranked:
    """A track with its score and the reasoning that produced it."""

    track: Track
    score: float
    urgency: Urgency
    in_path: bool
    lateral_m: float | None
    #: Per-term contributions. Kept so a surprising announcement can be
    #: explained rather than guessed at, and so the docs can show real
    #: examples of why something was or was not said.
    terms: dict[str, float] = field(default_factory=dict)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


class SaliencyEngine:
    """Ranks tracks and decides which, if any, to speak about."""

    def __init__(self, config: SaliencyConfig | None = None) -> None:
        self.config = config or SaliencyConfig()
        self._last_spoken_track: dict[int, float] = {}
        self._last_spoken_class: dict[str, float] = {}
        self._last_distance_spoken: dict[int, float] = {}
        self._last_utterance_ts: float = -1e9

    # -- scoring -----------------------------------------------------------

    def score(self, track: Track) -> Ranked:
        cfg = self.config
        distance = track.distance_m

        if distance is None:
            proximity = cfg.unknown_distance_proximity
            lateral = None
            # Without distance, bearing is all there is. Treat the forward
            # cone generously rather than pretending to know the offset.
            path = _clamp01(1.0 - abs(track.bearing_deg or 0.0) / 45.0) * 0.8
        else:
            proximity = 1.0 / (1.0 + (distance / cfg.proximity_half_m) ** 2)
            lateral = lateral_offset_m(distance, track.bearing_deg or 0.0)
            path = math.exp(-((lateral / cfg.corridor_half_width_m) ** 2))

        # Hazard levels are not linear in consequence: you bump into a sofa,
        # you fall down a staircase. Scaling super-linearly keeps a critical
        # hazard ahead of a merely bulky object that happens to be nearer -
        # linear weighting let a sofa at 1.2 m outrank stairs down at 2.5 m.
        hazard = (track.hazard.value / Hazard.CRITICAL.value) ** 1.5

        closing_speed = track.closing_speed_mps
        closing = _clamp01((closing_speed or 0.0) / 2.0)

        # Multiplicative core: near AND in the way. Hazard class and closing
        # speed are bonuses on top, so a descending staircase still outranks a
        # sofa without either term being able to carry an object that is
        # neither near nor in the path.
        relevance = proximity * path
        score = _clamp01(relevance + cfg.hazard_bonus * hazard + cfg.closing_bonus * closing)

        beyond_range = distance is not None and distance > cfg.max_distance_m
        too_trivial = track.hazard is Hazard.LOW and not cfg.announce_low_hazard
        if beyond_range or too_trivial:
            score = 0.0

        in_path = lateral is not None and abs(lateral) <= cfg.corridor_half_width_m
        urgency = self._urgency(track, in_path, distance)

        return Ranked(
            track=track,
            score=score,
            urgency=urgency,
            in_path=in_path,
            lateral_m=lateral,
            terms={
                "proximity": proximity,
                "path": path,
                "hazard": hazard,
                "closing": closing,
            },
        )

    def _urgency(self, track: Track, in_path: bool, distance: float | None) -> Urgency:
        cfg = self.config
        ttc = track.time_to_contact_s()
        if ttc is not None and ttc <= cfg.urgent_ttc_s and in_path:
            return Urgency.URGENT
        if (
            track.hazard is Hazard.CRITICAL
            and in_path
            and distance is not None
            and distance <= cfg.urgent_distance_m
        ):
            return Urgency.URGENT
        if track.hazard in {Hazard.CRITICAL, Hazard.HIGH} and in_path:
            return Urgency.NORMAL
        return Urgency.AMBIENT

    def rank(self, tracks: list[Track]) -> list[Ranked]:
        return sorted((self.score(t) for t in tracks), key=lambda r: r.score, reverse=True)

    # -- selection ---------------------------------------------------------

    def select(self, tracks: list[Track], now: float) -> Ranked | None:
        """The one thing worth saying right now, or nothing."""
        cfg = self.config
        ranked = self.rank(tracks)

        for candidate in ranked:
            if candidate.score < cfg.score_floor and candidate.urgency is not Urgency.URGENT:
                # Ranked descending, so nothing below this will qualify either.
                break
            if not self._passes_cooldown(candidate, now):
                continue
            if not self._passes_budget(candidate, now):
                continue
            self._mark_spoken(candidate, now)
            return candidate
        return None

    def _passes_cooldown(self, candidate: Ranked, now: float) -> bool:
        cfg = self.config
        track = candidate.track
        # Computed once: it overrides both cooldowns below. Letting it override
        # only the per-object one leaves the per-class one silencing a car that
        # has halved its distance, purely because a different car was mentioned
        # a moment ago.
        escalated = self._escalated(candidate)

        last = self._last_spoken_track.get(track.track_id)
        if last is not None:
            window = (
                cfg.urgent_cooldown_s
                if candidate.urgency is Urgency.URGENT
                else cfg.repeat_cooldown_s
            )
            if now - last < window and not escalated:
                return False

        last_class = self._last_spoken_class.get(track.label)
        if (
            last_class is not None
            and now - last_class < cfg.class_cooldown_s
            and candidate.urgency is not Urgency.URGENT
            and not escalated
        ):
            return False
        return True

    def _escalated(self, candidate: Ranked) -> bool:
        """Has this object become materially more dangerous since we spoke?

        Without this the cooldown is actively harmful: a car announced at six
        metres would stay silent while it closed to one. Halving the distance
        is a different fact about the world and earns the right to interrupt.
        """
        track = candidate.track
        previous = self._last_distance_spoken.get(track.track_id)
        if previous is None or track.distance_m is None:
            return False
        if track.distance_m <= previous * 0.5:
            return True
        ttc = track.time_to_contact_s()
        return ttc is not None and ttc <= self.config.urgent_ttc_s

    def _passes_budget(self, candidate: Ranked, now: float) -> bool:
        if candidate.urgency is Urgency.URGENT:
            return True
        return now - self._last_utterance_ts >= self.config.min_utterance_gap_s

    def _mark_spoken(self, candidate: Ranked, now: float) -> None:
        track = candidate.track
        self._last_spoken_track[track.track_id] = now
        self._last_spoken_class[track.label] = now
        if track.distance_m is not None:
            self._last_distance_spoken[track.track_id] = track.distance_m
        self._last_utterance_ts = now

    def reset(self) -> None:
        self._last_spoken_track.clear()
        self._last_spoken_class.clear()
        self._last_distance_spoken.clear()
        self._last_utterance_ts = -1e9
