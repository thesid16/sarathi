"""Tests for the evaluation harness.

The metrics are the project's only defence against fooling itself, so the
properties that make them meaningful are asserted directly: a late warning is
not a hit, repeating yourself is not ten correct announcements, and mentioning
something real is not a false alarm.
"""

from __future__ import annotations

import pytest
import yaml

from sarathi.bench import SpokenEvent, TruthError, TruthEvent, evaluate, load_truth
from sarathi.types import Urgency


def said(t, label, text=None, urgency=Urgency.NORMAL):
    return SpokenEvent(t=t, label=label, text=text or f"{label} ahead", urgency=urgency)


# -- recall ------------------------------------------------------------------


def test_an_announcement_inside_the_lead_window_is_a_hit():
    truth = [TruthEvent(t=10.0, label="stairs_down", lead_s=4.0)]
    r = evaluate(truth, [said(7.5, "stairs_down")], duration_s=20)
    assert r.hits == 1
    assert r.hazard_recall == 1.0
    assert r.mean_lead_s == pytest.approx(2.5)


def test_an_announcement_that_arrives_too_late_is_not_a_hit():
    """A warning delivered as you step off the kerb is not a warning."""
    truth = [TruthEvent(t=10.0, label="stairs_down", lead_s=4.0, grace_s=0.5)]
    r = evaluate(truth, [said(12.0, "stairs_down")], duration_s=20)
    assert r.hits == 0
    assert r.hazard_recall == 0.0
    assert len(r.late) == 1 and not r.missed


def test_never_announced_is_reported_as_missed_not_late():
    truth = [TruthEvent(t=10.0, label="open_manhole")]
    r = evaluate(truth, [], duration_s=20)
    assert len(r.missed) == 1 and not r.late


def test_an_announcement_far_too_early_does_not_count():
    truth = [TruthEvent(t=30.0, label="pole", lead_s=4.0)]
    r = evaluate(truth, [said(2.0, "pole")], duration_s=40)
    assert r.hits == 0


def test_optional_events_do_not_affect_recall():
    truth = [
        TruthEvent(t=5.0, label="bin", required=False),
        TruthEvent(t=10.0, label="pole", required=True),
    ]
    r = evaluate(truth, [said(8.0, "pole")], duration_s=20)
    assert r.truth_required == 1
    assert r.hazard_recall == 1.0


# -- precision ---------------------------------------------------------------


def test_saying_something_that_was_not_there_is_a_false_alarm():
    truth = [TruthEvent(t=10.0, label="pole")]
    r = evaluate(truth, [said(8.0, "pole"), said(12.0, "open_manhole")], duration_s=20)
    assert r.utterance_precision == pytest.approx(0.5)
    assert len(r.spurious) == 1
    assert r.spurious[0].label == "open_manhole"


def test_mentioning_something_real_but_optional_is_not_a_false_alarm():
    """A real bin announced is chatter, not a lie. Chattiness is measured
    separately, by utterances per minute."""
    truth = [TruthEvent(t=5.0, label="bin", required=False)]
    r = evaluate(truth, [said(4.0, "bin")], duration_s=20)
    assert r.utterance_precision == 1.0
    assert not r.spurious


def test_repeating_yourself_is_not_ten_correct_announcements():
    """One-to-one matching. Otherwise precision would reward the chattiness it
    exists to penalise."""
    truth = [TruthEvent(t=10.0, label="stairs_down", lead_s=6.0)]
    spoken = [said(t, "stairs_down") for t in (5.0, 6.0, 7.0, 8.0, 9.0)]
    r = evaluate(truth, spoken, duration_s=20)
    assert r.hits == 1  # not 5
    assert r.spoken == 5


def test_the_earliest_qualifying_warning_is_the_one_credited():
    truth = [TruthEvent(t=10.0, label="car", lead_s=6.0)]
    r = evaluate(truth, [said(9.0, "car"), said(5.0, "car")], duration_s=20)
    assert r.mean_lead_s == pytest.approx(5.0)  # credited the 5.0s one


def test_perfect_silence_scores_perfect_precision_and_zero_recall():
    """The degenerate strategy has to be visible as one."""
    truth = [TruthEvent(t=t, label="pole") for t in (5, 10, 15)]
    r = evaluate(truth, [], duration_s=20)
    assert r.utterance_precision == 1.0
    assert r.hazard_recall == 0.0


def test_narrating_everything_scores_perfect_recall_and_poor_precision():
    """And so does the opposite degenerate strategy."""
    truth = [TruthEvent(t=10.0, label="pole", lead_s=20.0)]
    spoken = [said(float(t), "pole") for t in range(10)] + [
        said(float(t), "broccoli") for t in range(10)
    ]
    r = evaluate(truth, spoken, duration_s=20)
    assert r.hazard_recall == 1.0
    assert r.utterance_precision < 0.6


# -- chattiness --------------------------------------------------------------


def test_utterances_per_minute():
    r = evaluate([], [said(float(i), "pole") for i in range(10)], duration_s=120)
    assert r.utterances_per_min == pytest.approx(5.0)


def test_no_truth_and_no_speech_is_not_a_division_by_zero():
    r = evaluate([], [], duration_s=0)
    assert r.hazard_recall == 1.0
    assert r.utterance_precision == 1.0
    assert r.utterances_per_min == 0.0


# -- ground truth files ------------------------------------------------------


def test_loading_a_truth_file(tmp_path):
    p = tmp_path / "walk.yaml"
    p.write_text(yaml.safe_dump({
        "clip": "walk1.mp4",
        "duration_s": 95,
        "events": [
            {"t": 12.4, "label": "stairs_down", "distance_m": 2.5},
            {"t": 30.0, "label": "bin", "required": False},
        ],
    }))
    clip, duration, events = load_truth(p)
    assert clip == "walk1.mp4" and duration == 95
    assert len(events) == 2
    assert events[0].required and not events[1].required
    assert events[0].distance_m == 2.5


def test_a_truth_file_without_events_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"clip": "x"}))
    with pytest.raises(TruthError, match="events"):
        load_truth(p)


def test_an_event_missing_its_fields_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"events": [{"t": 1.0}]}))
    with pytest.raises(TruthError, match="`t` and `label`"):
        load_truth(p)


def test_missing_truth_file(tmp_path):
    with pytest.raises(TruthError, match="not found"):
        load_truth(tmp_path / "nope.yaml")


def test_summary_names_what_went_wrong():
    truth = [TruthEvent(t=10.0, label="open_manhole")]
    r = evaluate(truth, [said(3.0, "ghost")], clip="c.mp4", duration_s=20)
    text = r.summary()
    assert "MISSED" in text and "open_manhole" in text
    assert "SPURIOUS" in text and "ghost" in text
