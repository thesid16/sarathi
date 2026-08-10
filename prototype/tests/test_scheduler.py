"""Tests for the power-aware scheduler.

The properties asserted here are the battery story. If motion gating stops
working, nothing fails and no test goes red on its own - the app simply runs
the detector on every frame and dies at lunchtime. So the skip behaviour is
pinned down explicitly, including the safeguard that stops gating from
silencing the system entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from sarathi.runtime import (
    Activity,
    MotionGate,
    Scheduler,
    SchedulerConfig,
    Skip,
    ThermalReader,
)
from sarathi.types import Frame


def frame(image: np.ndarray, ts: float, seq: int = 1) -> Frame:
    return Frame(image=image, seq=seq, ts_capture=ts, ts_received=ts, source_id="test")


def still(value: int = 40, size: int = 240) -> np.ndarray:
    return np.full((size, size, 3), value, np.uint8)


def busy(seed: int, size: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


class FakeThermal(ThermalReader):
    def __init__(self, value: float = 0.0):
        self.value = value

    def pressure(self) -> float:
        return self.value


# -- motion gate -------------------------------------------------------------


def test_the_first_frame_always_counts_as_changed():
    """Otherwise the pipeline sits silent at startup until something moves."""
    assert MotionGate().changed(still()) is True


def test_an_identical_frame_is_not_changed():
    gate = MotionGate()
    gate.changed(still())
    assert gate.changed(still()) is False
    assert gate.last_score == pytest.approx(0.0, abs=1e-6)


def test_a_completely_different_frame_is_changed():
    gate = MotionGate()
    gate.changed(busy(1))
    assert gate.changed(busy(2)) is True


def test_a_small_change_stays_below_the_threshold():
    """Sensor noise must not wake the detector."""
    gate = MotionGate(threshold=0.05)
    gate.changed(still(40))
    assert gate.changed(still(42)) is False  # ~0.008 normalised


def test_the_threshold_is_what_decides():
    """The same change reads as motion or not, depending only on threshold."""
    a, b = still(40), still(60)  # a normalised difference of about 0.078
    loose, tight = MotionGate(threshold=0.01), MotionGate(threshold=0.5)
    loose.changed(a)
    tight.changed(a)
    assert loose.changed(b) is True
    assert tight.changed(b) is False


def test_reset_forgets_the_previous_frame():
    gate = MotionGate()
    gate.changed(still())
    gate.reset()
    assert gate.changed(still()) is True


# -- staleness ---------------------------------------------------------------


def test_a_stale_frame_is_dropped_before_anything_else_is_computed():
    scheduler = Scheduler(SchedulerConfig(max_frame_age_ms=250))
    decision = scheduler.decide(frame(busy(1), ts=0.0), now=1.0)
    assert decision.run is False and decision.reason is Skip.STALE


def test_a_fresh_frame_is_not_dropped_as_stale():
    scheduler = Scheduler(SchedulerConfig(max_frame_age_ms=250))
    assert scheduler.decide(frame(busy(1), ts=0.0), now=0.05).run is True


# -- gating ------------------------------------------------------------------


def test_a_static_scene_stops_costing_inference():
    """The single biggest battery win: a stationary user needs no detector."""
    cfg = SchedulerConfig(settle_s=1.0, keepalive_hz=0.0, max_inference_hz=100.0)
    scheduler = Scheduler(cfg)
    image = still()

    now = 0.0
    for _ in range(40):
        scheduler.decide(frame(image, now), now)
        now += 0.1

    assert scheduler.activity is Activity.IDLE
    # Once settled, the static gate becomes the dominant reason for skipping.
    assert scheduler.stats.skipped["static"] >= scheduler.stats.skipped.get("rate_limited", 0)
    assert scheduler.stats.skip_rate > 0.7


def test_a_changing_scene_keeps_running_inference():
    cfg = SchedulerConfig(settle_s=1.0, keepalive_hz=0.0, max_inference_hz=100.0)
    scheduler = Scheduler(cfg)
    now = 0.0
    for i in range(20):
        scheduler.decide(frame(busy(i), now), now)
        now += 0.1
    assert scheduler.activity is Activity.MOVING
    assert scheduler.stats.ran >= 18


def test_motion_after_stillness_wakes_the_detector_up():
    cfg = SchedulerConfig(settle_s=0.5, keepalive_hz=0.0, max_inference_hz=100.0)
    scheduler = Scheduler(cfg)
    now = 0.0
    for _ in range(20):
        scheduler.decide(frame(still(), now), now)
        now += 0.1
    assert scheduler.activity is Activity.IDLE

    decision = scheduler.decide(frame(busy(99), now), now)
    assert decision.run is True
    assert decision.activity is Activity.MOVING


def test_gating_can_be_switched_off_entirely():
    cfg = SchedulerConfig(motion_enabled=False, keepalive_hz=0.0, max_inference_hz=100.0)
    scheduler = Scheduler(cfg)
    now = 0.0
    for _ in range(10):
        scheduler.decide(frame(still(), now), now)
        now += 0.1
    assert scheduler.stats.skipped.get("static", 0) == 0


# -- rate limiting -----------------------------------------------------------


def test_inference_is_capped_at_the_target_rate():
    cfg = SchedulerConfig(max_inference_hz=5.0, keepalive_hz=0.0, motion_enabled=False)
    scheduler = Scheduler(cfg)
    now = 0.0
    for i in range(100):  # 10 s of 10 Hz frames
        scheduler.decide(frame(busy(i), now), now)
        now += 0.1
    assert 45 <= scheduler.stats.ran <= 55  # about 5 per second


def test_the_idle_rate_is_lower_than_the_moving_rate():
    cfg = SchedulerConfig(max_inference_hz=8.0, idle_inference_hz=1.0)
    scheduler = Scheduler(cfg)
    assert scheduler.target_hz(0.0) == 8.0
    scheduler._activity = Activity.IDLE
    assert scheduler.target_hz(0.0) == 1.0


# -- keepalive ---------------------------------------------------------------


def test_a_wedged_gate_cannot_silence_the_system():
    """A blind user cannot tell 'nothing to report' from 'stopped working'."""
    cfg = SchedulerConfig(keepalive_hz=1.0, settle_s=0.5, max_inference_hz=8.0)
    scheduler = Scheduler(cfg)
    image = still()
    now = 0.0
    for _ in range(100):  # 10 s of a perfectly frozen scene
        scheduler.decide(frame(image, now), now)
        now += 0.1
    assert scheduler.stats.keepalives >= 8
    assert scheduler.stats.ran >= 8


def test_keepalive_passes_are_flagged_as_such():
    cfg = SchedulerConfig(keepalive_hz=2.0, max_inference_hz=0.001, motion_enabled=False)
    scheduler = Scheduler(cfg)
    decisions = []
    now = 0.0
    for _ in range(10):
        decisions.append(scheduler.decide(frame(busy(1), now), now))
        now += 0.3
    assert any(d.keepalive for d in decisions if d.run)


# -- thermal -----------------------------------------------------------------


def test_cool_hardware_runs_at_full_rate():
    scheduler = Scheduler(SchedulerConfig(max_inference_hz=8.0), FakeThermal(0.0))
    assert scheduler.target_hz(0.0) == 8.0


def test_rate_sheds_gradually_as_pressure_rises():
    """A step change in rate is audible; a ramp is not."""
    cfg = SchedulerConfig(max_inference_hz=8.0, idle_inference_hz=1.0,
                          thermal_soft=0.3, thermal_hard=0.7)
    scheduler = Scheduler(cfg, FakeThermal(0.0))
    rates = [scheduler.target_hz(p) for p in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9)]
    assert rates[0] == rates[1] == 8.0
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] == 1.0
    assert 1.0 < rates[3] < 8.0  # genuinely intermediate, not a step


def test_severe_pressure_drops_to_the_idle_rate():
    cfg = SchedulerConfig(max_inference_hz=8.0, idle_inference_hz=1.0, thermal_hard=0.7)
    scheduler = Scheduler(cfg, FakeThermal(0.95))
    assert scheduler.target_hz(0.95) == 1.0


def test_thermal_throttling_is_reported_distinctly_from_plain_rate_limiting():
    """So the benchmark can tell 'we chose to' from 'the device made us'."""
    cfg = SchedulerConfig(max_inference_hz=8.0, keepalive_hz=0.0, motion_enabled=False)
    scheduler = Scheduler(cfg, FakeThermal(0.9))
    now = 0.0
    for i in range(30):
        scheduler.decide(frame(busy(i), now), now)
        now += 0.05
    assert scheduler.stats.skipped.get("thermal", 0) > 0


def test_thermal_management_can_be_disabled():
    cfg = SchedulerConfig(max_inference_hz=8.0, thermal_enabled=False)
    scheduler = Scheduler(cfg, FakeThermal(0.99))
    assert scheduler.target_hz(0.99) == 8.0


# -- depth tier --------------------------------------------------------------


def test_depth_runs_at_its_own_lower_rate():
    scheduler = Scheduler(SchedulerConfig(depth_hz=2.0))
    scheduler._activity = Activity.MOVING
    ran = sum(1 for i in range(100) if scheduler.should_run_depth(i * 0.1))
    assert 18 <= ran <= 22  # about 2 per second over 10 s


def test_depth_does_not_run_while_the_user_is_stationary():
    """Steps and kerbs do not appear when you are standing still."""
    scheduler = Scheduler(SchedulerConfig(depth_hz=2.0, depth_when_idle=False))
    scheduler._activity = Activity.IDLE
    assert not any(scheduler.should_run_depth(i * 0.5) for i in range(10))


def test_depth_can_be_turned_off():
    scheduler = Scheduler(SchedulerConfig(depth_hz=0.0))
    scheduler._activity = Activity.MOVING
    assert scheduler.should_run_depth(1.0) is False


# -- stats -------------------------------------------------------------------


def test_stats_report_the_skip_breakdown():
    cfg = SchedulerConfig(settle_s=0.5, keepalive_hz=0.0, max_inference_hz=100.0)
    scheduler = Scheduler(cfg)
    now = 0.0
    for _ in range(30):
        scheduler.decide(frame(still(), now), now)
        now += 0.1
    summary = scheduler.stats.summary()
    assert "frames considered 30" in summary
    assert "static" in summary
    assert 0.0 < scheduler.stats.skip_rate <= 1.0


def test_mean_inference_cost_is_averaged():
    scheduler = Scheduler()
    for ms in (10.0, 20.0, 30.0):
        scheduler.note_inference(ms)
    assert scheduler.stats.inference_ms == pytest.approx(20.0)
