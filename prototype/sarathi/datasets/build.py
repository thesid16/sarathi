"""Assembling one training set out of many sources.

Writes YOLO-format labels (one text file per image, class-normalised xywh),
because every candidate detector can consume it and it survives being moved
between machines.

Three things this does that a plain conversion script would not, and each
exists because of a way these projects usually go wrong:

**Splits by source-and-scene, never at random.** A random split puts adjacent
frames of the same walk in both train and validation. The model then scores
beautifully on a validation set it has effectively memorised, and the number is
worthless. Frames are grouped by their parent directory - which for every
source here is a capture session - and whole groups go to one side or the
other.

**Reports the class distribution before training, not after.** A taxonomy of 77
classes assembled from public data is violently long-tailed. Discovering that
`open_manhole` has 40 instances after a training run has wasted the run.

**Generates attribution from the configs.** Every source here is CC BY or MIT
and requires it. Generated means it cannot drift out of date.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..taxonomy import Taxonomy
from ..util.log import get_logger
from .readers import Sample

log = get_logger(__name__)


@dataclass
class BuildStats:
    per_class: Counter[str] = field(default_factory=Counter)
    per_source: Counter[str] = field(default_factory=Counter)
    train_images: int = 0
    val_images: int = 0
    skipped_unknown_class: Counter[str] = field(default_factory=Counter)

    def report(self, taxonomy: Taxonomy) -> str:
        total = sum(self.per_class.values())
        lines = [
            f"images   {self.train_images} train  /  {self.val_images} val",
            f"boxes    {total}",
            "",
            "per source:",
        ]
        for source, count in self.per_source.most_common():
            lines.append(f"    {source:<26} {count:>8}")

        lines += ["", "per class (taxonomy order, zero-count classes shown):"]
        empty: list[str] = []
        for cls in taxonomy:
            count = self.per_class.get(cls.name, 0)
            if count == 0:
                empty.append(cls.name)
                continue
            share = 100 * count / max(1, total)
            bar = "#" * max(1, round(share / 2))
            lines.append(f"    {cls.name:<20} {count:>7}  {share:5.1f}%  {bar}")

        if empty:
            lines += [
                "",
                f"{len(empty)} classes have NO training data:",
                "    " + ", ".join(empty),
                "",
                "A class with no data is a class the model cannot detect. These must",
                "either get data, get dropped from the shipped label set, or be",
                "reported as known-absent in the evaluation - never left to look",
                "supported because they appear in the taxonomy.",
            ]

        thin = [
            (c.name, self.per_class[c.name])
            for c in taxonomy
            if 0 < self.per_class.get(c.name, 0) < 50
        ]
        if thin:
            lines += ["", "classes with fewer than 50 instances (expect poor recall):"]
            for name, count in sorted(thin, key=lambda kv: kv[1]):
                lines.append(f"    {name:<20} {count:>5}")

        if self.skipped_unknown_class:
            lines += ["", "labels not in the taxonomy (a config bug if unexpected):"]
            for name, count in self.skipped_unknown_class.most_common(10):
                lines.append(f"    {name:<20} {count:>5}")
        return "\n".join(lines)


def _group_key(sample: Sample) -> str:
    """Which capture session a sample belongs to.

    The parent directory is the session for every source here: WOTR keeps one
    folder per walk, Roboflow exports split by their own train/valid folders,
    and Mendeley names files by scene. Grouping on it stops near-duplicate
    frames straddling the split.
    """
    return f"{sample.source}/{sample.image_path.parent.name}"


def _assign_split(key: str, val_fraction: float) -> str:
    """Deterministic split from a hash of the group key.

    Deterministic rather than random so a rebuild produces the same split, and
    hashed rather than sequential so ordering does not bias it. Adding a new
    source cannot reshuffle the existing ones.
    """
    digest = hashlib.sha256(key.encode()).hexdigest()
    return "val" if (int(digest[:8], 16) % 10000) / 10000.0 < val_fraction else "train"


def build_dataset(
    samples: list[Sample],
    out_dir: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    val_fraction: float = 0.15,
    copy_images: bool = False,
) -> BuildStats:
    """Write a YOLO-format dataset. Returns the distribution to inspect first."""
    taxonomy = taxonomy or Taxonomy.load()
    out = Path(out_dir)
    stats = BuildStats()
    index = {cls.name: cls.id for cls in taxonomy}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Decide splits per group first, so every frame of a session lands together.
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    split_of = {key: _assign_split(key, val_fraction) for key in groups}

    seen_names: set[str] = set()
    for key, members in groups.items():
        split = split_of[key]
        for sample in members:
            lines: list[str] = []
            for box in sample.boxes:
                if box.label not in index:
                    stats.skipped_unknown_class[box.label] += 1
                    continue
                cx = (box.x1 + box.x2) / 2.0 / sample.width
                cy = (box.y1 + box.y2) / 2.0 / sample.height
                bw = (box.x2 - box.x1) / sample.width
                bh = (box.y2 - box.y1) / sample.height
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(
                    f"{index[box.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
                stats.per_class[box.label] += 1
            if not lines:
                continue

            # Names collide across sources; prefix with the source to keep them
            # unique without renaming anything on disk.
            stem = f"{sample.source}__{sample.image_path.stem}"
            n = 1
            while stem in seen_names:
                n += 1
                stem = f"{sample.source}__{sample.image_path.stem}_{n}"
            seen_names.add(stem)

            image_out = out / "images" / split / f"{stem}{sample.image_path.suffix}"
            if copy_images:
                shutil.copy2(sample.image_path, image_out)
            else:
                # Symlink by default: these datasets total tens of gigabytes and
                # copying them to rebuild a label set is pure waste.
                if image_out.exists() or image_out.is_symlink():
                    image_out.unlink()
                image_out.symlink_to(sample.image_path.resolve())

            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            stats.per_source[sample.source] += 1
            if split == "train":
                stats.train_images += 1
            else:
                stats.val_images += 1

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(taxonomy),
                "names": {cls.id: cls.name for cls in taxonomy},
            },
            sort_keys=False,
        )
    )
    return stats


def write_attribution(configs: list[dict], out_path: str | Path) -> None:
    """Generate the attribution notice from the dataset configs.

    Generated rather than maintained, so it cannot fall behind what was
    actually used. Every source in this project is CC BY, CC BY-SA or MIT and
    all three require attribution.
    """
    lines = [
        "# Dataset attribution",
        "",
        "Generated from `training/datasets/*.yaml`. Do not edit by hand.",
        "",
        "Sarathi's model weights are trained on the datasets below. Weights are",
        "distributed under AGPL-3.0; the datasets remain under their own terms.",
        "",
    ]
    for config in configs:
        for name, spec in (config.get("sources") or {"": config}).items():
            if not isinstance(spec, dict) or "licence" not in spec:
                continue
            lines.append(f"## {name or config.get('name', 'dataset')}")
            lines.append("")
            lines.append(f"- Licence: {spec['licence']}")
            if spec.get("url"):
                lines.append(f"- Source: {spec['url']}")
            if spec.get("doi"):
                lines.append(f"- DOI: {spec['doi']}")
            if spec.get("attribution"):
                lines.append(f"- Attribution: {str(spec['attribution']).strip()}")
            lines.append("")
    Path(out_path).write_text("\n".join(lines))
