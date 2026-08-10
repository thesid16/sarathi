"""Turning a ranked track into words.

Phrasing is table-driven: `phrases/en.yaml` and `phrases/hi.yaml` hold every
string, and the same files are read by the Android app. Adding a language, or
fixing a word that sounds wrong when heard for the hundredth time, is a file
edit rather than a code change and a release.

Three rules the tables and this module enforce together:

**Short beats complete.** "Step down ahead" has reached the user before
"stairs descending, twelve o'clock, two point five metres" has finished its
first word. Urgent phrasings drop everything not needed to act.

**Round hard.** Spoken distance is useful at about half-metre granularity.
"One point four seven metres" claims accuracy the estimator does not have and
takes three times as long to say. Where the estimator reports low confidence,
the phrasing hedges - "about two metres" - rather than pretending.

**Object first.** It is the word the user is waiting for. Direction and
distance can arrive a beat later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..perception.distance import clock_position
from ..types import Hazard, Urgency, Utterance
from .saliency import Ranked

_REPO_ROOT = Path(__file__).resolve().parents[3]
PHRASE_DIR = _REPO_ROOT / "phrases"


class PhraseError(ValueError):
    """Raised when a phrase table is missing or malformed."""


@dataclass
class PhraseBook:
    """Every string the system can say, in one language."""

    lang: str
    templates: dict[str, str]
    bearing: dict[str, Any]
    distance: dict[str, Any]
    objects: dict[str, str]
    system: dict[str, str]
    bearing_style: str = "clock"
    hedge_above_uncertainty: float = 0.25

    @classmethod
    def load(cls, lang: str = "en", directory: str | Path | None = None) -> "PhraseBook":
        base = Path(directory or PHRASE_DIR)
        path = base / f"{lang}.yaml"
        if not path.exists():
            raise PhraseError(f"no phrase table for language {lang!r} at {path}")
        data = yaml.safe_load(path.read_text()) or {}

        for key in ("templates", "bearing", "distance"):
            if key not in data:
                raise PhraseError(f"{path}: missing required section {key!r}")
        steps = (data["distance"] or {}).get("steps")
        if not steps:
            raise PhraseError(f"{path}: distance.steps is required and must be non-empty")

        return cls(
            lang=str(data.get("lang", lang)),
            templates=dict(data["templates"]),
            bearing=dict(data["bearing"]),
            distance=dict(data["distance"]),
            objects=dict(data.get("objects") or {}),
            system=dict(data.get("system") or {}),
            bearing_style=str(data.get("bearing_style", "clock")),
            hedge_above_uncertainty=float(data.get("hedge_above_uncertainty", 0.25)),
        )

    # -- pieces ------------------------------------------------------------

    def object_name(self, label: str) -> str:
        """Spoken name, falling back to the class label.

        The overrides differ from class names on purpose: `stairs_down` is a
        label, "step down" is what a person needs to hear, and it is shorter.
        """
        return self.objects.get(label, label.replace("_", " "))

    def bearing_phrase(self, bearing_deg: float | None, *, ahead_band_deg: float = 12.0) -> str:
        """Direction as words. Straight ahead is said as 'ahead', not '12 o'clock'."""
        if bearing_deg is None or abs(bearing_deg) <= ahead_band_deg:
            return str(self.bearing.get("ahead", "ahead"))

        if self.bearing_style == "relative":
            table = self.bearing.get("relative") or {}
            if bearing_deg < 0:
                key = "slight_left" if bearing_deg > -35 else "left"
            else:
                key = "slight_right" if bearing_deg < 35 else "right"
            return str(table.get(key, table.get("ahead", "")))

        table = self.bearing.get("clock") or {}
        hour = clock_position(bearing_deg)
        # YAML keys may parse as ints; accept either.
        return str(table.get(hour, table.get(str(hour), self.bearing.get("ahead", ""))))

    def distance_phrase(self, distance_m: float | None, uncertainty: float = 0.0) -> str:
        """Distance as words, rounded to what is worth hearing."""
        if distance_m is None:
            return ""
        text = str(self.distance["steps"][-1]["text"])
        for step in self.distance["steps"]:
            if distance_m < float(step["max"]):
                text = str(step["text"])
                break
        if uncertainty > self.hedge_above_uncertainty:
            hedge = self.distance.get("hedge")
            if hedge:
                return f"{hedge} {text}"
        return text

    def system_phrase(self, key: str) -> str:
        return self.system.get(key, key.replace("_", " "))


class Phraser:
    """Builds utterances from ranked tracks."""

    def __init__(self, book: PhraseBook | None = None, *, lang: str = "en") -> None:
        self.book = book or PhraseBook.load(lang)

    @property
    def lang(self) -> str:
        return self.book.lang

    def utterance(self, ranked: Ranked, *, uncertainty: float = 0.0) -> Utterance:
        book = self.book
        track = ranked.track

        name = book.object_name(track.label)
        bearing = book.bearing_phrase(track.bearing_deg)
        ahead_word = str(book.bearing.get("ahead", "ahead"))
        is_ahead = bearing == ahead_word

        if ranked.urgency is Urgency.URGENT:
            # Strip to the minimum. Distance is dropped entirely: at urgent
            # range the user needs to stop, not to know whether it is 1.5 m or
            # 2 m, and every extra syllable is delay.
            text = book.templates["urgent"].format(object=name, bearing=bearing)
        else:
            distance = book.distance_phrase(track.distance_m, uncertainty)
            if is_ahead:
                key = "ahead_full" if distance else "ahead_no_distance"
            else:
                key = "full" if distance else "no_distance"
            text = book.templates[key].format(
                object=name, bearing=bearing, distance=distance
            )

        return Utterance(
            text=_tidy(text),
            urgency=ranked.urgency,
            # Keyed on the tracked object, not the sentence, so a chair whose
            # distance ticks from "two metres" to "one and a half" is still
            # recognised as the same subject and not repeated.
            topic=f"{track.label}#{track.track_id}",
            earcon=_earcon_for(track.hazard, ranked.urgency),
            lang=book.lang,
        )

    def system(self, key: str, urgency: Urgency = Urgency.NORMAL) -> Utterance:
        return Utterance(
            text=self.book.system_phrase(key),
            urgency=urgency,
            topic=f"system:{key}",
            lang=self.book.lang,
        )


def _earcon_for(hazard: Hazard, urgency: Urgency) -> str | None:
    """Which non-speech cue precedes the words, if any.

    A rising tone reaches the user several hundred milliseconds before a spoken
    word can. For something they are about to walk into, that gap is the whole
    point of having an earcon at all.
    """
    if urgency is Urgency.URGENT:
        return "alert" if hazard is Hazard.CRITICAL else "warn"
    return None


def _tidy(text: str) -> str:
    """Clean up the seams left by an empty slot in a template."""
    text = " ".join(text.split())
    text = text.replace(" ,", ",").replace(",,", ",")
    return text.strip().strip(",").strip()
