"""Deciding when NOT to run the model.

Everything else in this project makes inference better. This module makes it
happen less often, which is the only reason the app can run for a day.

The ordering of the gates is the design. They run cheapest-first, so a frame
that will be skipped is skipped for the least possible cost:

1. **Staleness** - free. A frame that queued behind a slow inference describes
   a scene the user has walked past.
2. **Motion gate** - about 0.1 ms. A stationary user pointed at a static scene
   needs no inference at all, and indoors that is most frames.
3. **Rate limit** - free. Even when things are changing, there is no value in
   running faster than the user can act on.
4. **Thermal** - cached. Degrade before the OS throttles, not after.

The gate is run at a *higher* rate than inference on purpose. Frame
differencing on a 64x64 downscale costs microseconds; the detector costs tens
of milliseconds. Sampling motion cheaply and often is what lets the inference
rate drop to idle without the system being slow to notice the user has started
walking again.

One safeguard that matters more than it looks: a keepalive. Pure motion gating
can wedge - a miscalibrated threshold, a camera that returns a frozen frame, a
scene that changes in ways the downscale cannot see - and a blind user would
have no way to tell the difference between "nothing to report" and "stopped
working". So inference runs at a floor rate regardless of what the gates say.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..types import Frame
from ..util.log import get_logger

log = get_logger(__name__)


class Activity(Enum):
    """What the user appears to be doing, inferred from the scene."""

    IDLE = "idle"
    MOVING = "moving"


class Skip(Enum):
    """Why a frame was not processed. Recorded so the docs can show real
    proportions rather than claimed ones."""

    RAN = "ran"
    STALE = "stale"
    STATIC = "static"
    RATE = "rate_limited"
    THERMAL = "thermal"


@dataclass
class Decision:
    run: bool
    reason: Skip
    activity: Activity
    target_hz: float
    #: True when this pass is a keepalive rather than a motion-triggered one.
    keepalive: bool = False


# -- motion ------------------------------------------------------------------


class MotionGate:
    """Frame-difference gate on a small greyscale downscale.

    Deliberately crude. The question is not "what moved" but "is this worth
    looking at properly", and a 64x64 mean absolute difference answers that for
    a few microseconds. Anything more sophisticated would cost a meaningful
    fraction of the inference it is meant to avoid.
    """

    def __init__(self, *, threshold: float = 0.012, size: int = 64) -> None:
        self.threshold = threshold
        self.size = size
        self._previous: np.ndarray | None = None
        self.last_score: float = 1.0

    def score(self, image: np.ndarray) -> float:
        """Normalised mean absolute difference from the previous frame."""
        small = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        current = small.astype(np.float32) / 255.0

        if self._previous is None:
            self._previous = current
            # The first frame has nothing to compare against. Treat it as
            # changed so the pipeline always produces one result at startup
            # rather than sitting silent until something moves.
            self.last_score = 1.0
            return self.last_score

        self.last_score = float(np.abs(current - self._previous).mean())
        self._previous = current
        return self.last_score

    def changed(self, image: np.ndarray) -> bool:
        return self.score(image) >= self.threshold

    def reset(self) -> None:
        self._previous = None
        self.last_score = 1.0


# -- thermal -----------------------------------------------------------------


class ThermalReader:
    """Headroom before the platform starts throttling us.

    Returns 0.0 (cool) to 1.0 (about to be throttled). Android has
    PowerManager.getThermalHeadroom for exactly this; the desktop
    implementations here exist so the policy can be developed and tested
    before the port.
    """

    def pressure(self) -> float:
        return 0.0


class NullThermalReader(ThermalReader):
    """Always cool. Default under test and on platforms with no signal."""


class MacThermalReader(ThermalReader):
    """Reads macOS CPU speed limit via `pmset -g therm`.

    Not equivalent to Android's headroom - it reports throttling that is
    already happening rather than how close we are to it - so it is a
    development stand-in, not the real policy input. The Android reader is what
    the shipped behaviour will be tuned against.
    """

    def __init__(self, cache_s: float = 2.0) -> None:
        self.cache_s = cache_s
        self._value = 0.0
        self._checked_at = -1e9
        self._available = shutil.which("pmset") is not None

    def pressure(self) -> float:
        if not self._available:
            return 0.0
        now = time.monotonic()
        if now - self._checked_at < self.cache_s:
            return self._value
        self._checked_at = now
        try:
            out = subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2
            ).stdout
        except (OSError, subprocess.SubprocessError):
            self._available = False
            return 0.0
        for line in out.splitlines():
            if "CPU_Speed_Limit" in line:
                _, _, value = line.partition("=")
                try:
                    limit = float(value.strip())
                except ValueError:
                    break
                self._value = max(0.0, min(1.0, (100.0 - limit) / 100.0))
                return self._value
        self._value = 0.0
        return self._value


# -- scheduler ---------------------------------------------------------------


@dataclass
class SchedulerConfig:
    #: Inference rate while the user is moving.
    max_inference_hz: float = 8.0
    #: Inference rate while stationary and the scene is static.
    idle_inference_hz: float = 1.0
    #: Floor rate that runs regardless of gating. The safeguard against a
    #: wedged gate leaving the user silently unguided.
    keepalive_hz: float = 0.2
    #: Frames older than this at decision time are discarded.
    max_frame_age_ms: float = 250.0

    motion_enabled: bool = True
    motion_threshold: float = 0.012
    motion_size: int = 64
    #: How long the scene must stay static before dropping to the idle rate.
    settle_s: float = 2.0

    thermal_enabled: bool = True
    #: Above this pressure, start shedding frame rate.
    thermal_soft: float = 0.30
    #: Above this, drop to the idle rate whatever the user is doing.
    thermal_hard: float = 0.70

    depth_hz: float = 2.0
    #: Depth only earns its cost while the user is actually moving.
    depth_when_idle: bool = False


@dataclass
class SchedulerStats:
    considered: int = 0
    ran: int = 0
    keepalives: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    #: Rolling mean of measured inference cost, used to report duty cycle.
    inference_ms: float = 0.0
    _inference_samples: int = 0

    def note_skip(self, reason: Skip) -> None:
        self.skipped[reason.value] = self.skipped.get(reason.value, 0) + 1

    def note_inference(self, ms: float) -> None:
        self._inference_samples += 1
        n = self._inference_samples
        self.inference_ms += (ms - self.inference_ms) / n

    @property
    def skip_rate(self) -> float:
        return 0.0 if self.considered == 0 else 1.0 - self.ran / self.considered

    def summary(self) -> str:
        lines = [
            f"frames considered {self.considered}",
            f"inference ran     {self.ran}  ({100 * (1 - self.skip_rate):.1f}%)",
            f"  of which keepalive {self.keepalives}",
            f"mean inference    {self.inference_ms:.1f} ms",
        ]
        for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            share = 100 * count / max(1, self.considered)
            lines.append(f"skipped {reason:<13} {count:>6}  ({share:.1f}%)")
        return "\n".join(lines)


class Scheduler:
    """Decides, per frame, whether to spend inference on it."""

    #: Accept frames arriving up to 5% early against the rate target. See the
    #: rate-limit branch in `decide` for why a strict comparison is wrong.
    _RATE_TOLERANCE = 0.95

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        thermal: ThermalReader | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.thermal = thermal or NullThermalReader()
        self.gate = MotionGate(
            threshold=self.config.motion_threshold, size=self.config.motion_size
        )
        self.stats = SchedulerStats()

        self._last_inference = -1e9
        self._last_depth = -1e9
        self._last_motion_at = -1e9
        self._activity = Activity.MOVING  # assume moving until proven otherwise

    @property
    def activity(self) -> Activity:
        return self._activity

    def target_hz(self, pressure: float) -> float:
        cfg = self.config
        base = cfg.max_inference_hz if self._activity is Activity.MOVING else cfg.idle_inference_hz
        if not cfg.thermal_enabled or pressure <= cfg.thermal_soft:
            return base
        if pressure >= cfg.thermal_hard:
            return cfg.idle_inference_hz
        # Linear shed between soft and hard, so the rate falls off gradually
        # instead of stepping down in a way the user would hear.
        span = max(1e-6, cfg.thermal_hard - cfg.thermal_soft)
        fraction = (pressure - cfg.thermal_soft) / span
        return base - (base - cfg.idle_inference_hz) * fraction

    def decide(self, frame: Frame, now: float | None = None) -> Decision:
        cfg = self.config
        now = time.monotonic() if now is None else now
        self.stats.considered += 1

        # 1. Staleness. Free, and acting on a stale frame is worse than not
        #    acting: it describes somewhere the user no longer is.
        if frame.age_ms(now) > cfg.max_frame_age_ms:
            self.stats.note_skip(Skip.STALE)
            return Decision(False, Skip.STALE, self._activity, 0.0)

        # 2. Motion. Cheap enough to run on nearly every frame, which is what
        #    lets the inference rate idle down without being slow to wake.
        moved = True
        if cfg.motion_enabled:
            moved = self.gate.changed(frame.image)
            if moved:
                self._last_motion_at = now
            self._activity = (
                Activity.MOVING if now - self._last_motion_at < cfg.settle_s else Activity.IDLE
            )

        pressure = self.thermal.pressure() if cfg.thermal_enabled else 0.0
        hz = self.target_hz(pressure)
        since = now - self._last_inference

        # 3. Keepalive. Checked before the gates so nothing can suppress it.
        if cfg.keepalive_hz > 0 and since >= 1.0 / cfg.keepalive_hz:
            self._last_inference = now
            self.stats.ran += 1
            self.stats.keepalives += 1
            return Decision(True, Skip.RAN, self._activity, hz, keepalive=True)

        # 4. Rate limit, with a small tolerance for early arrival.
        #
        #    Frames never land on exact intervals. Compared strictly, a camera
        #    running at twice the target rate drops to two-thirds of it: the
        #    frame at +0.199999 s is rejected, so the next candidate is at
        #    +0.3 s, and the pattern alternates. The tolerance costs nothing
        #    and recovers the rate that was actually asked for.
        if hz <= 0 or since < (1.0 / hz) * self._RATE_TOLERANCE:
            reason = (
                Skip.THERMAL
                if cfg.thermal_enabled and pressure > cfg.thermal_soft
                else Skip.RATE
            )
            self.stats.note_skip(reason)
            return Decision(False, reason, self._activity, hz)

        # 5. Static scene. Last because it is the most expensive check, and
        #    because a frame that fails the rate limit never needed it.
        if cfg.motion_enabled and not moved and self._activity is Activity.IDLE:
            self.stats.note_skip(Skip.STATIC)
            return Decision(False, Skip.STATIC, self._activity, hz)

        self._last_inference = now
        self.stats.ran += 1
        return Decision(True, Skip.RAN, self._activity, hz)

    def should_run_depth(self, now: float | None = None) -> bool:
        """Whether the Tier 2 depth pass has earned its cost this frame."""
        cfg = self.config
        now = time.monotonic() if now is None else now
        if cfg.depth_hz <= 0:
            return False
        if not cfg.depth_when_idle and self._activity is Activity.IDLE:
            return False
        if now - self._last_depth < 1.0 / cfg.depth_hz:
            return False
        self._last_depth = now
        return True

    def note_inference(self, duration_ms: float) -> None:
        self.stats.note_inference(duration_ms)

    def reset(self) -> None:
        self.gate.reset()
        self.stats = SchedulerStats()
        self._last_inference = -1e9
        self._last_depth = -1e9
        self._last_motion_at = -1e9
        self._activity = Activity.MOVING
