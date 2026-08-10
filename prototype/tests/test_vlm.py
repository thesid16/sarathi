"""Tests for the scene-description adapter.

Everything here runs without the 2.5 GB model, which is the point: the parts
that decide what a blind user hears are pure functions, and the part that
decides whether the model can be used at all is a header read. Neither needs a
GPU or a download to be checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sarathi.models.adapters.gemma_vlm import (
    MAX_CHARS,
    VISION_SECTION,
    has_vision_encoder,
    tidy,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The image shows a doorway ahead.", "A doorway ahead."),
        ("In this image, a car is parked.", "A car is parked."),
        ("I see stairs going down.", "Stairs going down."),
        ("**A bench** on the _left_.", "A bench on the left."),
        ("A clear path ahead.", "A clear path ahead."),
    ],
)
def test_openers_and_markup_are_stripped(raw: str, expected: str) -> None:
    """Announcements start with the thing, not with a preamble.

    "The image shows" costs about a second of speech and carries nothing. For
    someone navigating by ear that second is the whole latency budget.
    """
    assert tidy(raw) == expected


def test_only_the_first_line_survives() -> None:
    """Multi-paragraph answers are truncated, not concatenated.

    A model asked for one sentence sometimes writes a list. Reading the list
    aloud occupies the audio channel long enough to displace a hazard warning,
    which turns a convenience feature into a safety problem.
    """
    assert tidy("A door ahead.\n\nAlso, there is a window.") == "A door ahead."


def test_long_answers_are_cut_at_a_word_boundary() -> None:
    long = "A " + "very " * 200 + "long corridor."
    result = tidy(long)
    assert len(result) <= MAX_CHARS + 1  # the ellipsis
    assert result.endswith("…")
    assert not result.endswith("ver…")


def test_empty_input_yields_empty_output() -> None:
    assert tidy("   \n  ") == ""


def test_vision_encoder_detected_in_header(tmp_path: Path) -> None:
    weights = tmp_path / "with-vision.litertlm"
    weights.write_bytes(b"\x00" * 100 + VISION_SECTION + b"\x00" * 100)
    assert has_vision_encoder(weights)


def test_text_only_build_is_rejected(tmp_path: Path) -> None:
    """The exact shape of a real 1.9 GB mistake.

    `gemma-4-E2B-it-gpu.litertlm` holds one section,
    `tf_lite_artisan_text_decoder`, and no vision encoder. Nothing on its model
    card says so, it loads without complaint, and it fails only when an image
    is attached - eleven seconds into a request the user is waiting on.
    """
    weights = tmp_path / "text-only.litertlm"
    weights.write_bytes(b"\x00" * 100 + b"tf_lite_artisan_text_decoder" + b"\x00" * 100)
    assert not has_vision_encoder(weights)


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """Absent weights answer "no", rather than raising.

    The caller's next question is always "can I use this?", and a missing file
    and a text-only file are the same answer to it.
    """
    assert not has_vision_encoder(tmp_path / "nothing-here.litertlm")
