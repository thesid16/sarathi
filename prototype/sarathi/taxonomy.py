"""The class taxonomy: what the detector knows, and how well it can know it.

Loaded from `training/taxonomy/sarathi77.yaml`, which is the single source of
truth for class ids, spoken labels in both languages, hazard priors, and - the
part most projects leave out - which public dataset can actually supply each
class.

Recording coverage next to each class is deliberate. It means the honest
question ("which dangerous things can we not yet detect?") is answerable by
running a command rather than by remembering, and it makes the gap visible
before training instead of after.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from .types import Hazard

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = _REPO_ROOT / "training" / "taxonomy" / "sarathi77.yaml"

HAZARD_BY_NAME = {
    "critical": Hazard.CRITICAL,
    "high": Hazard.HIGH,
    "medium": Hazard.MEDIUM,
    "low": Hazard.LOW,
}

COVERAGE_LEVELS = ("good", "partial", "pending", "gap")


class TaxonomyError(ValueError):
    """Raised when the taxonomy file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class TaxonomyClass:
    id: int
    name: str
    hi: str
    group: str
    hazard: Hazard
    sources: tuple[str, ...]
    coverage: str
    note: str | None = None
    hi_review: bool = False

    def label(self, lang: str = "en") -> str:
        return self.hi if lang == "hi" else self.name.replace("_", " ")


class Taxonomy:
    """An ordered, validated set of classes."""

    def __init__(self, classes: list[TaxonomyClass], sources: dict[str, dict], version: str):
        self.classes = classes
        self.sources = sources
        self.version = version
        self._by_name = {c.name: c for c in classes}

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self) -> Iterator[TaxonomyClass]:
        return iter(self.classes)

    def __getitem__(self, key: int | str) -> TaxonomyClass:
        if isinstance(key, int):
            return self.classes[key]
        return self._by_name[key]

    def names(self, lang: str = "en") -> list[str]:
        """Label list in class-id order - the file a model manifest points at."""
        return [c.label(lang) for c in self.classes]

    def hazard_map(self) -> dict[int, Hazard]:
        return {c.id: c.hazard for c in self.classes}

    def by_coverage(self, coverage: str) -> list[TaxonomyClass]:
        return [c for c in self.classes if c.coverage == coverage]

    def blind_spots(self) -> list[TaxonomyClass]:
        """Dangerous classes with no confirmed data source.

        This is the number that belongs in the evaluation report and in any
        honest description of the product. A tool that silently fails on open
        manholes is worse than one that says it does.
        """
        return [
            c
            for c in self.classes
            if c.coverage in {"gap", "pending"}
            and c.hazard in {Hazard.CRITICAL, Hazard.HIGH}
        ]

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Taxonomy":
        p = Path(path or DEFAULT_TAXONOMY)
        if not p.exists():
            raise TaxonomyError(f"taxonomy not found: {p}")
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise TaxonomyError(f"{p}: invalid YAML: {exc}") from exc

        declared_sources = data.get("sources", {}) or {}
        raw_classes = data.get("classes") or []
        if not raw_classes:
            raise TaxonomyError(f"{p}: no classes defined")

        classes: list[TaxonomyClass] = []
        for entry in raw_classes:
            where = f"{p}: class {entry.get('name', entry.get('id', '?'))!r}"
            for key in ("id", "name", "hi", "group", "hazard", "coverage"):
                if key not in entry:
                    raise TaxonomyError(f"{where}: missing {key!r}")

            hazard_name = str(entry["hazard"]).lower()
            if hazard_name not in HAZARD_BY_NAME:
                raise TaxonomyError(
                    f"{where}: hazard must be one of {sorted(HAZARD_BY_NAME)}, "
                    f"got {hazard_name!r}"
                )
            coverage = str(entry["coverage"]).lower()
            if coverage not in COVERAGE_LEVELS:
                raise TaxonomyError(
                    f"{where}: coverage must be one of {COVERAGE_LEVELS}, got {coverage!r}"
                )

            sources = tuple(str(s) for s in entry.get("sources", []))
            unknown = [s for s in sources if s not in declared_sources]
            if unknown:
                raise TaxonomyError(
                    f"{where}: references undeclared source(s) {unknown}. "
                    f"Add them to the `sources:` block so their licence is recorded."
                )
            # A class with sources but marked `gap`, or no sources but marked
            # `good`, means the file has drifted from reality.
            if coverage == "good" and not sources:
                raise TaxonomyError(f"{where}: coverage 'good' but no sources listed")
            if coverage == "gap" and sources and all(
                declared_sources.get(s, {}).get("status") not in {None, "planned"} for s in sources
            ):
                pass  # `gap` with a speculative source is allowed - it names the intent

            classes.append(
                TaxonomyClass(
                    id=int(entry["id"]),
                    name=str(entry["name"]),
                    hi=str(entry["hi"]),
                    group=str(entry["group"]),
                    hazard=HAZARD_BY_NAME[hazard_name],
                    sources=sources,
                    coverage=coverage,
                    note=entry.get("note"),
                    hi_review=bool(entry.get("hi_review", False)),
                )
            )

        ids = [c.id for c in classes]
        if ids != list(range(len(ids))):
            missing = sorted(set(range(len(ids))) - set(ids))
            raise TaxonomyError(
                f"{p}: class ids must be contiguous from 0. "
                f"Got {len(ids)} classes; missing/duplicated around {missing[:5]}. "
                "A hole here silently shifts every label after it."
            )
        duplicates = [n for n, count in Counter(c.name for c in classes).items() if count > 1]
        if duplicates:
            raise TaxonomyError(f"{p}: duplicate class names: {duplicates}")

        return cls(classes, declared_sources, str(data.get("version", "0.0.0")))

    # -- reporting ---------------------------------------------------------

    def coverage_report(self) -> str:
        lines = [f"Sarathi taxonomy {self.version} - {len(self)} classes", ""]

        counts = Counter(c.coverage for c in self.classes)
        lines.append("Coverage")
        for level in COVERAGE_LEVELS:
            n = counts.get(level, 0)
            bar = "#" * round(30 * n / max(1, len(self)))
            lines.append(f"  {level:<8} {n:>3}  {bar}")

        lines += ["", "By hazard"]
        for hazard in (Hazard.CRITICAL, Hazard.HIGH, Hazard.MEDIUM, Hazard.LOW):
            members = [c for c in self.classes if c.hazard is hazard]
            weak = [c for c in members if c.coverage in {"gap", "pending"}]
            lines.append(
                f"  {hazard.name:<8} {len(members):>3} classes, {len(weak)} without confirmed data"
            )

        blind = self.blind_spots()
        lines += ["", f"Blind spots - {len(blind)} dangerous classes with no confirmed source:"]
        for c in blind:
            reason = c.note or f"sources: {', '.join(c.sources) or 'none'}"
            lines.append(f"  [{c.coverage:<7}] {c.hazard.name:<8} {c.name:<16} {reason}")

        needs_review = [c for c in self.classes if c.hi_review]
        if needs_review:
            lines += [
                "",
                f"Hindi labels flagged for native review: "
                f"{', '.join(c.name for c in needs_review)}",
            ]
        return "\n".join(lines)
