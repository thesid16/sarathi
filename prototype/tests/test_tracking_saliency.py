"""Tests for tracking and the saliency engine.

These encode product behaviour, not just code behaviour. "Does not repeat
itself", "does not announce a one-frame false positive", "interrupts when a car
is closing" and "prefers the stairs over the sofa" are the properties that
decide whether the thing is usable, so they are asserted directly.
"""

from __future__ import annotations

import pytest

from sarathi.guidance import SaliencyConfig, SaliencyEngine
from sarathi.perception.tracking import Tracker, iou, lateral_offset_m
from sarathi.types import Detection, Hazard, Urgency


def det(x1, y1, x2, y2, *, label="chair", class_id=0, distance=None, bearing=0.0,
        hazard=Hazard.MEDIUM, score=0.9):
    return Detection(
        box=(x1, y1, x2, y2), score=score, class_id=class_id, label=label,
        distance_m=distance, bearing_deg=bearing, hazard=hazard,
    )


# -- geometry helpers --------------------------------------------------------


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_lateral_offset_distinguishes_near_and_far_at_the_same_bearing():
    """The reason saliency uses metres-to-the-side rather than degrees."""
    assert lateral_offset_m(8.0, 30.0) == pytest.approx(4.62, abs=0.02)
    assert lateral_offset_m(1.0, 30.0) == pytest.approx(0.577, abs=0.01)


# -- tracking ----------------------------------------------------------------


def test_a_track_needs_repeat_sightings_before_it_is_reported():
    """One-frame false positives must never reach speech."""
    tracker = Tracker(min_hits=2)
    assert tracker.update([det(0, 0, 10, 10)], 0.0) == []
    live = tracker.update([det(1, 1, 11, 11)], 0.2)
    assert len(live) == 1 and live[0].hits == 2


def test_identity_persists_across_frames():
    tracker = Tracker(min_hits=1)
    first = tracker.update([det(0, 0, 20, 20)], 0.0)[0]
    second = tracker.update([det(2, 2, 22, 22)], 0.2)[0]
    assert first.track_id == second.track_id


def test_detections_are_stamped_with_their_track_id():
    tracker = Tracker(min_hits=1)
    d = det(0, 0, 20, 20)
    tracker.update([d], 0.0)
    assert d.track_id is not None


def test_different_classes_never_share_an_identity():
    """A person walking in front of a chair must not inherit the chair's id."""
    tracker = Tracker(min_hits=1)
    chair = tracker.update([det(0, 0, 20, 20, label="chair", class_id=0)], 0.0)[0]
    live = tracker.update([det(0, 0, 20, 20, label="person", class_id=1)], 0.2)
    assert all(t.track_id != chair.track_id for t in live if t.label == "person")


def test_a_track_survives_brief_occlusion_then_expires():
    tracker = Tracker(min_hits=1, max_misses=2)
    tracker.update([det(0, 0, 20, 20)], 0.0)
    for i in range(2):
        assert tracker.update([], 0.2 * (i + 1)) == []
        assert tracker.tracks, "should still be alive during the miss window"
    tracker.update([], 1.0)
    assert tracker.tracks == []


def test_closing_speed_is_measured_from_the_distance_history():
    """Approaching 1 m/s: 5.0 -> 4.0 -> 3.0 over two seconds."""
    tracker = Tracker(min_hits=1)
    for i, d in enumerate([5.0, 4.5, 4.0, 3.5, 3.0]):
        tracker.update([det(0, 0, 20, 20, distance=d)], i * 0.5)
    speed = tracker.tracks[0].closing_speed_mps
    assert speed is not None and speed > 0.3  # smoothing damps it, sign is what matters


def test_a_stationary_object_is_not_reported_as_closing():
    tracker = Tracker(min_hits=1)
    for i in range(5):
        tracker.update([det(0, 0, 20, 20, distance=3.0)], i * 0.3)
    assert abs(tracker.tracks[0].closing_speed_mps or 0.0) < 0.05


def test_closing_speed_needs_enough_history_to_be_trusted():
    tracker = Tracker(min_hits=1)
    tracker.update([det(0, 0, 20, 20, distance=5.0)], 0.0)
    assert tracker.tracks[0].closing_speed_mps is None


def test_distance_is_smoothed_against_single_bad_frames():
    tracker = Tracker(min_hits=1)
    tracker.update([det(0, 0, 20, 20, distance=3.0)], 0.0)
    tracker.update([det(0, 0, 20, 20, distance=9.0)], 0.2)  # one wild estimate
    assert tracker.tracks[0].distance_m < 6.0


# -- saliency ----------------------------------------------------------------


def build(tracker, label, distance, bearing, hazard, *, class_id=0, frames=4, dt=0.3, ts0=0.0):
    """Run one stationary object through a tracker so it becomes confirmed."""
    for i in range(frames):
        tracker.update(
            [det(0, 0, 20, 20, label=label, class_id=class_id,
                 distance=distance, bearing=bearing, hazard=hazard)],
            ts0 + i * dt,
        )
    return tracker.tracks[-1]


def build_many(specs, *, frames=4, dt=0.3, ts0=0.0):
    """Build several objects through ONE tracker, so their ids are distinct.

    Every Tracker numbers its tracks from 1, so building two objects in two
    separate trackers hands them both track_id=1. The per-object cooldown then
    treats them as the same object - which silently made an earlier version of
    the class-cooldown test pass for entirely the wrong reason.

    specs: (label, distance, bearing, hazard, class_id). Boxes are spaced out
    so same-class objects stay separate tracks.
    """
    tracker = Tracker(min_hits=1)
    for i in range(frames):
        frame = [
            det(idx * 100, 0, idx * 100 + 20, 20, label=label, class_id=class_id,
                distance=distance, bearing=bearing, hazard=hazard)
            for idx, (label, distance, bearing, hazard, class_id) in enumerate(specs)
        ]
        tracker.update(frame, ts0 + i * dt)
    return tracker.tracks


def test_in_path_outranks_the_same_object_off_to_the_side():
    engine = SaliencyEngine()
    ahead = build(Tracker(min_hits=1), "pole", 2.0, 0.0, Hazard.HIGH)
    beside = build(Tracker(min_hits=1), "pole", 2.0, 50.0, Hazard.HIGH)
    assert engine.score(ahead).score > engine.score(beside).score
    assert engine.score(ahead).in_path and not engine.score(beside).in_path


def test_hazard_class_beats_geometry():
    """Stairs down at 3 m must outrank a sofa at 2 m."""
    engine = SaliencyEngine()
    stairs = build(Tracker(min_hits=1), "stairs_down", 3.0, 0.0, Hazard.CRITICAL)
    sofa = build(Tracker(min_hits=1), "sofa", 2.0, 0.0, Hazard.LOW)
    assert engine.score(stairs).score > engine.score(sofa).score


def test_nearer_outranks_further():
    engine = SaliencyEngine()
    near = build(Tracker(min_hits=1), "chair", 1.0, 0.0, Hazard.MEDIUM)
    far = build(Tracker(min_hits=1), "chair", 5.0, 0.0, Hazard.MEDIUM)
    assert engine.score(near).score > engine.score(far).score


def test_a_critical_hazard_in_the_path_is_urgent():
    engine = SaliencyEngine()
    manhole = build(Tracker(min_hits=1), "open_manhole", 1.8, 0.0, Hazard.CRITICAL)
    assert engine.score(manhole).urgency is Urgency.URGENT


def test_the_same_hazard_off_to_the_side_is_not_urgent():
    engine = SaliencyEngine()
    manhole = build(Tracker(min_hits=1), "open_manhole", 1.8, 60.0, Hazard.CRITICAL)
    assert engine.score(manhole).urgency is not Urgency.URGENT


def test_an_unmeasurable_distance_still_scores_but_ranks_below_a_measured_one():
    engine = SaliencyEngine()
    unknown = build(Tracker(min_hits=1), "chair", None, 0.0, Hazard.MEDIUM)
    known = build(Tracker(min_hits=1), "chair", 1.0, 0.0, Hazard.MEDIUM)
    assert 0.0 < engine.score(unknown).score < engine.score(known).score


def test_score_terms_are_exposed_for_explanation():
    engine = SaliencyEngine()
    ranked = engine.score(build(Tracker(min_hits=1), "pole", 2.0, 0.0, Hazard.HIGH))
    assert set(ranked.terms) == {"proximity", "path", "hazard", "closing"}


# -- selection: the not-being-annoying rules ---------------------------------


def test_nothing_is_said_about_a_distant_irrelevant_object():
    engine = SaliencyEngine()
    tracker = Tracker(min_hits=1)
    far = build(tracker, "cup", 5.5, 70.0, Hazard.LOW)
    assert engine.select([far], 10.0) is None


def test_an_object_is_not_announced_twice():
    engine = SaliencyEngine()
    tracker = Tracker(min_hits=1)
    chair = build(tracker, "chair", 1.5, 0.0, Hazard.HIGH)
    assert engine.select([chair], 10.0) is not None
    assert engine.select([chair], 12.0) is None  # inside the cooldown


def test_the_utterance_budget_prevents_talking_over_itself():
    engine = SaliencyEngine(SaliencyConfig(min_utterance_gap_s=1.5, class_cooldown_s=0.0))
    a, b = build_many([
        ("chair", 1.5, 0.0, Hazard.HIGH, 0),
        ("door", 1.6, 0.0, Hazard.HIGH, 1),
    ])
    assert engine.select([a, b], 10.0) is not None
    assert engine.select([a, b], 10.4) is None  # too soon
    assert engine.select([a, b], 12.0) is not None


def test_an_urgent_hazard_overrides_the_budget():
    """A budget that silences an imminent drop-off is a bug, not a feature."""
    engine = SaliencyEngine()
    chair, manhole = build_many([
        ("chair", 2.0, 0.0, Hazard.MEDIUM, 0),
        ("open_manhole", 1.5, 0.0, Hazard.CRITICAL, 1),
    ])
    assert engine.select([chair], 10.0) is not None
    urgent = engine.select([manhole], 10.1)  # well inside the budget
    assert urgent is not None and urgent.urgency is Urgency.URGENT


def test_a_closing_object_re_announces_despite_the_cooldown():
    """Announced at 6 m, now at 1.5 m - staying silent would be dangerous."""
    engine = SaliencyEngine()
    tracker = Tracker(min_hits=1)
    for i, d in enumerate([6.0, 6.0, 6.0]):
        tracker.update([det(0, 0, 20, 20, label="car", distance=d,
                            hazard=Hazard.CRITICAL)], i * 0.3)
    assert engine.select(tracker.tracks, 10.0) is not None

    for i, d in enumerate([4.0, 3.0, 2.0, 1.5]):
        tracker.update([det(0, 0, 20, 20, label="car", distance=d,
                            hazard=Hazard.CRITICAL)], 10.5 + i * 0.3)
    again = engine.select(tracker.tracks, 11.8)
    assert again is not None, "an object that halved its distance must be re-announced"


def test_a_second_object_of_the_same_class_is_damped():
    """Six chairs in a row should not become six sentences."""
    engine = SaliencyEngine(SaliencyConfig(min_utterance_gap_s=0.0))
    a, b = build_many([
        ("chair", 1.5, 0.0, Hazard.HIGH, 0),
        ("chair", 1.6, 10.0, Hazard.HIGH, 0),
    ])
    assert a.track_id != b.track_id, "two chairs must be two tracks"
    assert engine.select([a, b], 10.0) is not None
    assert engine.select([a, b], 10.5) is None  # class cooldown holds


def test_reset_clears_all_speech_history():
    engine = SaliencyEngine()
    chair = build(Tracker(min_hits=1), "chair", 1.5, 0.0, Hazard.HIGH)
    assert engine.select([chair], 10.0) is not None
    engine.reset()
    assert engine.select([chair], 10.1) is not None


def test_select_returns_the_highest_ranked_eligible_track():
    engine = SaliencyEngine(SaliencyConfig(class_cooldown_s=0.0))
    sofa, stairs = build_many([
        ("sofa", 1.2, 0.0, Hazard.MEDIUM, 0),
        ("stairs_down", 2.5, 0.0, Hazard.CRITICAL, 1),
    ])
    chosen = engine.select([sofa, stairs], 10.0)
    assert chosen is not None and chosen.track.label == "stairs_down"
