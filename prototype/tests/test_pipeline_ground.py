"""Tests for how the pipeline confirms a floor hazard before speaking.

These exist because of a bug the caching introduced, and the bug is worth
stating: a ground reading stays valid for ~1.2 s while depth runs at ~2 Hz, so
the same single measurement is re-injected as a detection on every frame in
between. The tracker sees it repeatedly, min_hits is satisfied within two
frames, and the "confirmation" is one measurement wearing a disguise.

False "step down ahead" alerts are the worst failure this system has. Somebody
stops or stumbles for nothing, and then stops trusting it.
"""

from __future__ import annotations

import pytest

from sarathi.perception.ground import GroundReading
from sarathi.runtime import Pipeline, PipelineConfig


@pytest.fixture
def pipeline():
    # No models: this is about the confirmation logic, not inference.
    return Pipeline(PipelineConfig(detector=None, depth=None))


def drop(distance: float, fit: float = 0.95) -> GroundReading:
    return GroundReading(distance, "step_down", distance, fit, 60)


def clear(fit: float = 0.95) -> GroundReading:
    return GroundReading(9.0, None, None, fit, 60)


def test_one_depth_pass_is_not_enough_to_announce_a_drop(pipeline):
    reading = drop(3.0)
    pipeline._ground_at = 1.0
    pipeline._vote(reading, 1.0)
    assert pipeline._ground_confirmed(reading) is False


def test_two_passes_are_still_not_enough(pipeline):
    for i, ts in enumerate((1.0, 1.5)):
        pipeline._ground_at = ts
        pipeline._vote(drop(3.0), ts)
    assert pipeline._ground_confirmed(drop(3.0)) is False


def test_three_agreeing_passes_confirm(pipeline):
    for ts in (1.0, 1.5, 2.0):
        pipeline._ground_at = ts
        pipeline._vote(drop(3.0), ts)
    assert pipeline._ground_confirmed(drop(3.0)) is True


def test_passes_must_agree_about_where_not_just_that(pipeline):
    """Readings scattered from 1 m to 6 m are noise, not a step."""
    for ts, d in ((1.0, 1.2), (1.5, 3.4), (2.0, 6.0)):
        pipeline._ground_at = ts
        pipeline._vote(drop(d), ts)
    assert pipeline._ground_confirmed(drop(6.0)) is False


def test_a_clear_floor_reading_wipes_the_record(pipeline):
    """Evidence against is not merely the absence of evidence for."""
    for ts in (1.0, 1.5):
        pipeline._ground_at = ts
        pipeline._vote(drop(3.0), ts)
    pipeline._ground_at = 2.0
    pipeline._vote(clear(), 2.0)
    pipeline._ground_at = 2.5
    pipeline._vote(drop(3.0), 2.5)
    assert pipeline._ground_confirmed(drop(3.0)) is False


def test_stale_votes_expire(pipeline):
    for ts in (1.0, 1.5, 2.0):
        pipeline._ground_at = ts
        pipeline._vote(drop(3.0), ts)
    pipeline._ground_at = 30.0  # long after the window
    assert pipeline._ground_confirmed(drop(3.0)) is False


def test_step_up_is_never_surfaced_as_a_hazard(pipeline):
    """A step_up is far more often a wall or furniture, which the detector
    already covers. Announcing every vertical surface ahead is unusable."""
    reading = GroundReading(3.0, "step_up", 3.0, 0.95, 60)
    for ts in (1.0, 1.5, 2.0):
        pipeline._ground_at = ts
        pipeline._vote(reading, ts)
    assert pipeline._ground_confirmed(reading) is False


def test_a_confirmed_drop_becomes_a_critical_detection(pipeline):
    import numpy as np
    from sarathi.types import Frame, Hazard

    for ts in (1.0, 1.5, 2.0):
        pipeline._ground_at = ts
        pipeline._vote(drop(3.0), ts)
    frame = Frame(np.zeros((720, 1280, 3), np.uint8), 1, 0.0, 0.0, "t")
    det = pipeline._ground_hazard(drop(3.0), frame, pipeline.camera_for(frame))
    assert det is not None
    assert det.label == "step_down"
    assert det.hazard is Hazard.CRITICAL
    assert det.distance_m == pytest.approx(3.0)
    assert det.distance_source == "depth"
