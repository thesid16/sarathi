"""Measuring whether the guidance is any good.

mAP is the wrong metric for this product. A detector with excellent mAP that
announces a wall while missing a staircase has failed, and one that misses half
the chairs but never misses a drop-off has succeeded. So the harness measures
what the user experiences:

**Hazard recall** - of the dangerous things that were actually there, how many
got announced, *in time to act on*. An alert that arrives as you step off the
kerb is a miss, so recall is measured against a lead-time window rather than
mere co-occurrence.

**Utterance precision** - of the things it said, how many were worth saying.
This is the metric that keeps the system honest about chattiness: it is trivial
to get perfect recall by narrating everything, and precision is what that
costs.

**Lead time** - how far ahead of the hazard the warning landed. The number that
decides whether recall was real.

**Utterances per minute** - the blunt chattiness measure. Users abandon tools
over this, so it is reported whether or not anyone asked.

Ground truth is a small YAML file per clip. Annotating a two-minute walk takes
a few minutes and is worth more than any amount of held-out mAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..types import Urgency


class TruthError(ValueError):
    """Raised when a ground-truth file is malformed."""


@dataclass(frozen=True)
class TruthEvent:
    """Something that was really there, and when it mattered."""

    t: float
    label: str
    #: False for things that are fine to mention but not required. Lets a clip
    #: record "there was a bin here" without penalising silence about it.
    required: bool = True
    distance_m: float | None = None
    #: How long before `t` an announcement still counts as useful. Default four
    #: seconds is roughly three walking paces of warning.
    lead_s: float = 4.0
    #: Grace after `t` - an announcement this late is recorded but not a hit.
    grace_s: float = 0.5

    def window(self) -> tuple[float, float]:
        return (self.t - self.lead_s, self.t + self.grace_s)


@dataclass(frozen=True)
class SpokenEvent:
    t: float
    label: str
    text: str
    urgency: Urgency = Urgency.NORMAL


@dataclass
class EvalResult:
    clip: str
    duration_s: float
    truth_required: int = 0
    hits: int = 0
    spoken: int = 0
    justified: int = 0
    lead_times: list[float] = field(default_factory=list)
    missed: list[TruthEvent] = field(default_factory=list)
    spurious: list[SpokenEvent] = field(default_factory=list)
    late: list[tuple[TruthEvent, float]] = field(default_factory=list)

    @property
    def hazard_recall(self) -> float:
        return 1.0 if self.truth_required == 0 else self.hits / self.truth_required

    @property
    def utterance_precision(self) -> float:
        return 1.0 if self.spoken == 0 else self.justified / self.spoken

    @property
    def mean_lead_s(self) -> float | None:
        return sum(self.lead_times) / len(self.lead_times) if self.lead_times else None

    @property
    def utterances_per_min(self) -> float:
        return 0.0 if self.duration_s <= 0 else self.spoken * 60.0 / self.duration_s

    def summary(self) -> str:
        lead = f"{self.mean_lead_s:.1f} s" if self.mean_lead_s is not None else "n/a"
        lines = [
            f"clip                 {self.clip}  ({self.duration_s:.0f} s)",
            f"hazard recall        {self.hazard_recall:.0%}  "
            f"({self.hits}/{self.truth_required} required hazards announced in time)",
            f"utterance precision  {self.utterance_precision:.0%}  "
            f"({self.justified}/{self.spoken} utterances corresponded to something real)",
            f"mean lead time       {lead}",
            f"chattiness           {self.utterances_per_min:.1f} utterances/min",
        ]
        if self.missed:
            lines.append("")
            lines.append("MISSED - these were there and nothing was said:")
            for event in self.missed:
                lines.append(f"    {event.t:6.1f}s  {event.label}")
        if self.late:
            lines.append("")
            lines.append("LATE - announced, but not in time to act on:")
            for event, at in self.late:
                lines.append(f"    {event.t:6.1f}s  {event.label}  (spoken at {at:.1f}s)")
        if self.spurious:
            lines.append("")
            lines.append("SPURIOUS - said, but nothing was there:")
            for spoken in self.spurious[:12]:
                lines.append(f"    {spoken.t:6.1f}s  {spoken.text}")
            if len(self.spurious) > 12:
                lines.append(f"    ... and {len(self.spurious) - 12} more")
        return "\n".join(lines)


def load_truth(path: str | Path) -> tuple[str, float, list[TruthEvent]]:
    """Read a ground-truth clip annotation."""
    p = Path(path).expanduser()
    if not p.exists():
        raise TruthError(f"ground truth not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if "events" not in data:
        raise TruthError(f"{p}: needs an `events` list")

    events: list[TruthEvent] = []
    for raw in data["events"]:
        if "t" not in raw or "label" not in raw:
            raise TruthError(f"{p}: every event needs `t` and `label`, got {raw}")
        events.append(
            TruthEvent(
                t=float(raw["t"]),
                label=str(raw["label"]),
                required=bool(raw.get("required", True)),
                distance_m=raw.get("distance_m"),
                lead_s=float(raw.get("lead_s", 4.0)),
                grace_s=float(raw.get("grace_s", 0.5)),
            )
        )
    return str(data.get("clip", p.stem)), float(data.get("duration_s", 0.0)), events


def evaluate(
    truth: list[TruthEvent],
    spoken: list[SpokenEvent],
    *,
    clip: str = "",
    duration_s: float = 0.0,
) -> EvalResult:
    """Score a run against ground truth.

    Matching is one-to-one: an utterance can justify at most one truth event
    and a truth event is satisfied by at most one utterance. Without that,
    repeating "step down ahead" ten times would count as ten correct
    announcements and the precision metric would reward exactly the chattiness
    it exists to penalise.
    """
    result = EvalResult(clip=clip, duration_s=duration_s)
    result.truth_required = sum(1 for e in truth if e.required)
    result.spoken = len(spoken)

    used_spoken: set[int] = set()
    matched_truth: set[int] = set()

    for ti, event in enumerate(truth):
        start, end = event.window()
        best: tuple[float, int] | None = None
        for si, said in enumerate(spoken):
            if si in used_spoken or said.label != event.label:
                continue
            if start <= said.t <= end:
                lead = event.t - said.t
                # Prefer the earliest qualifying utterance: the most useful
                # warning is the one that arrived first.
                if best is None or said.t < spoken[best[1]].t:
                    best = (lead, si)
        if best is not None:
            lead, si = best
            used_spoken.add(si)
            matched_truth.add(ti)
            if event.required:
                result.hits += 1
                result.lead_times.append(lead)
        elif event.required:
            # Was it said at all, just too late to be useful? That is a
            # different failure from never noticing, and worth separating.
            late_at = next(
                (s.t for s in spoken if s.label == event.label and s.t > end), None
            )
            if late_at is not None:
                result.late.append((event, late_at))
            else:
                result.missed.append(event)

    # An utterance is justified if it matched, or if it names something the
    # clip says was present at all - including non-required objects. Mentioning
    # a real bin is not a false alarm, it is only chatter.
    optional_labels = {e.label for e in truth}
    for si, said in enumerate(spoken):
        if si in used_spoken or said.label in optional_labels:
            result.justified += 1
        else:
            result.spurious.append(said)

    return result
