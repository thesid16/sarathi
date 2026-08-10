"""Tests for on-demand text reading.

The engine itself is not under test - PaddleOCR's accuracy is PaddleOCR's
problem. What is under test is everything between raw OCR output and speech,
because a screen-reader user cannot skim past noise the way a sighted user
skims past a cluttered sign. Fragments arrive in detection order, which is
close to arbitrary; spoken aloud that becomes "204 Lab".
"""

from __future__ import annotations

import numpy as np
import pytest

from sarathi.models.adapters.rapid_ocr import RapidOcrReader
from sarathi.models.manifest import ModelManifest


class FakeEngine:
    """Stands in for RapidOCR: returns whatever quads it was given."""

    def __init__(self, result):
        self.result = result

    def __call__(self, image):
        return self.result, None


def quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def reader(result, **kwargs):
    manifest = ModelManifest.from_dict({
        "id": "t", "task": "ocr", "license": "Apache-2.0",
        "distribution": "bundled", "vendored_weights": True,
        "runtime": {"prototype": "rapidocr"},
    })
    r = RapidOcrReader.__new__(RapidOcrReader)
    from sarathi.models.base import Model
    Model.__init__(r, manifest, None, None)
    r._engine = FakeEngine(result)
    r.min_confidence = kwargs.get("min_confidence", 0.55)
    r.line_tolerance = kwargs.get("line_tolerance", 0.6)
    r.last_inference_ms = 0.0
    return r


IMG = np.zeros((100, 400, 3), np.uint8)


def test_fragments_on_one_line_are_joined_in_left_to_right_order():
    """Detection order is arbitrary; spoken order must not be."""
    r = reader([
        [quad(300, 40, 380, 80), "204", 0.99],
        [quad(40, 40, 160, 80), "Lab", 0.98],
    ])
    assert r.read(IMG) == [("Lab 204", pytest.approx(0.98))]


def test_lines_come_back_top_to_bottom():
    r = reader([
        [quad(40, 200, 300, 250), "Second Floor", 0.97],
        [quad(40, 40, 200, 90), "Lab 204", 0.99],
    ])
    assert [t for t, _ in r.read(IMG)] == ["Lab 204", "Second Floor"]


def test_low_confidence_fragments_are_dropped():
    """Reading out a hallucinated word is worse than reading nothing."""
    r = reader([
        [quad(40, 40, 200, 90), "Lab 204", 0.99],
        [quad(40, 200, 200, 250), "gibberish", 0.20],
    ])
    assert [t for t, _ in r.read(IMG)] == ["Lab 204"]


def test_the_confidence_reported_is_the_weakest_fragment_in_the_line():
    """A line is only as trustworthy as its worst piece."""
    r = reader([
        [quad(40, 40, 160, 80), "Lab", 0.99],
        [quad(300, 40, 380, 80), "204", 0.71],
    ])
    assert r.read(IMG)[0][1] == pytest.approx(0.71)


def test_slightly_skewed_text_still_merges_into_one_line():
    """Scene text is rarely level - a sign photographed at an angle drifts."""
    r = reader([
        [quad(40, 40, 160, 84), "Lab", 0.99],
        [quad(300, 48, 380, 92), "204", 0.99],
    ])
    assert len(r.read(IMG)) == 1


def test_clearly_separate_lines_do_not_merge():
    r = reader([
        [quad(40, 40, 160, 84), "Lab", 0.99],
        [quad(40, 300, 160, 344), "204", 0.99],
    ])
    assert len(r.read(IMG)) == 2


def test_empty_and_whitespace_results_are_ignored():
    r = reader([
        [quad(40, 40, 160, 84), "   ", 0.99],
        [quad(40, 200, 160, 244), "Real", 0.99],
    ])
    assert [t for t, _ in r.read(IMG)] == ["Real"]


def test_no_text_found_returns_nothing_rather_than_guessing():
    assert reader([]).read(IMG) == []
    assert reader(None).read(IMG) == []


def test_malformed_engine_output_is_survived():
    r = reader([[quad(1, 1, 2, 2)], "nonsense", [quad(40, 40, 160, 84), "Ok", 0.99]])
    assert [t for t, _ in r.read(IMG)] == ["Ok"]
