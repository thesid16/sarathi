"""Assembling one training set out of many sources.

Writes YOLO-format labels (one text file per image, class-normalised xywh),
because every candidate detector can consume it and it survives being moved
between machines.

Three things this does that a plain conversion script would not, and each
exists because of a way these projects usually go wrong:

**Splits by contiguous blocks of frames, never at random.** A random split puts
adjacent frames of the same walk in both train and validation. The model then
scores beautifully on a validation set it has effectively memorised, and the
number is worthless. Frames are sorted and chunked into blocks, and whole
blocks go to one side or the other.

An earlier version grouped by parent directory on the assumption that each
source organises frames by capture session. WOTR does not - all 13,928 images
sit in one folder - so the whole dataset became a single group and a 15%
validation target came out at 79%. Blocks plus largest-first packing hold
regardless of how a source chooses to lay out its files.

**Reports the class distribution before training, not after.** A taxonomy of 77
classes assembled from public data is violently long-tailed. Discovering that
`open_manhole` has 40 instances after a training run has wasted the run.

**Generates attribution from the configs.** Every source here is CC BY or MIT
and requires it. Generated means it cannot drift out of date.
"""

from __future__ import annotations

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
    #: Counted separately, because mixing them is actively misleading. IDD
    #: contributes 32,280 auto_rickshaw boxes and the model will never see one
    #: of them - reporting a combined total would show a well-covered class
    #: that in fact has no training data at all.
    per_class_train: Counter[str] = field(default_factory=Counter)
    per_class_eval: Counter[str] = field(default_factory=Counter)
    per_class: Counter[str] = field(default_factory=Counter)
    per_source: Counter[str] = field(default_factory=Counter)
    train_images: int = 0
    val_images: int = 0
    eval_only_images: int = 0
    #: The class list this checkpoint is actually trained on, in model order.
    shipped: list[str] = field(default_factory=list)
    excluded_thin: list[str] = field(default_factory=list)
    skipped_unknown_class: Counter[str] = field(default_factory=Counter)
    #: In the taxonomy but below the instance threshold - deliberate, not a bug.
    not_shipped: Counter[str] = field(default_factory=Counter)

    def report(self, taxonomy: Taxonomy) -> str:
        train_total = sum(self.per_class_train.values())
        eval_total = sum(self.per_class_eval.values())
        lines = [
            f"images   {self.train_images} train  /  {self.val_images} val"
            + (f"  /  {self.eval_only_images} val_domain (held out by licence and design)"
               if self.eval_only_images else ""),
            f"boxes    {train_total} trainable  /  {eval_total} held out",
            "",
            "per source:",
        ]
        for source, count in self.per_source.most_common():
            lines.append(f"    {source:<26} {count:>8}")

        lines += [
            "",
            "per class - TRAIN counts drive the model; HELD-OUT counts do not:",
            f"    {'class':<20} {'train':>8} {'held out':>9}",
        ]
        empty: list[str] = []
        for cls in taxonomy:
            train = self.per_class_train.get(cls.name, 0)
            held = self.per_class_eval.get(cls.name, 0)
            if train == 0 and held == 0:
                empty.append(cls.name)
                continue
            share = 100 * train / max(1, train_total)
            bar = "#" * max(0, round(share / 2))
            flag = "  <- eval only" if train == 0 and held > 0 else ""
            lines.append(
                f"    {cls.name:<20} {train:>8} {held:>9}  {share:5.1f}% {bar}{flag}"
            )

        untrainable = [
            c.name for c in taxonomy if self.per_class_train.get(c.name, 0) == 0
        ]
        if untrainable:
            lines += [
                "",
                f"{len(untrainable)} classes have NO TRAINING data:",
                "    " + ", ".join(untrainable),
                "",
                "A class with no training data is a class the model cannot detect,",
                "whether or not it appears in a held-out set. These must either get",
                "data, get dropped from the shipped label set, or be reported as",
                "known-absent in the evaluation - never left to look supported",
                "because they appear in the taxonomy.",
            ]

        thin = [
            (c.name, self.per_class_train[c.name])
            for c in taxonomy
            if 0 < self.per_class_train.get(c.name, 0) < 50
        ]
        if thin:
            lines += ["", "classes with fewer than 50 instances (expect poor recall):"]
            for name, count in sorted(thin, key=lambda kv: kv[1]):
                lines.append(f"    {name:<20} {count:>5}")

        if self.shipped:
            lines += [
                "",
                f"shipped label set: {len(self.shipped)} classes "
                f"(of {len(taxonomy)} in the taxonomy)",
                "    " + ", ".join(self.shipped),
            ]
            if self.excluded_thin:
                lines += [
                    "",
                    f"{len(self.excluded_thin)} taxonomy classes are NOT in this "
                    "checkpoint's label set.",
                    "They must be reported as known-absent in the evaluation. A model",
                    "that claims a class it was never trained on will produce confident",
                    "nonsense rather than silence, which for a hazard class is the",
                    "wrong direction to fail in.",
                ]

        if self.not_shipped:
            lines += [
                "",
                "boxes dropped for being below the instance threshold "
                "(deliberate, not a bug):",
            ]
            for name, count in self.not_shipped.most_common(12):
                lines.append(f"    {name:<20} {count:>7}")

        if self.skipped_unknown_class:
            lines += ["", "labels not in the taxonomy (a config bug if unexpected):"]
            for name, count in self.skipped_unknown_class.most_common(10):
                lines.append(f"    {name:<20} {count:>5}")
        return "\n".join(lines)


def group_samples(samples: list[Sample], block_size: int = 200) -> dict[str, list[Sample]]:
    """Partition samples into blocks that must not be split across train/val.

    Grouping by directory alone is not enough. It assumes each source organises
    frames by capture session, and WOTR does not - all 13,928 images sit in one
    `JPEGImages` folder, so the whole dataset collapsed into a single group and
    landed entirely on one side of the split.

    Instead: sort each directory's frames by name and chunk them into blocks of
    consecutive files. Consecutive filenames are consecutive frames in every
    source here, so near-duplicates stay together, while there are still plenty
    of blocks to distribute. The only leakage is at block boundaries - two
    frames in every `block_size` - which is negligible and bounded, unlike a
    random split where it is total.
    """
    by_dir: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_dir[f"{sample.source}/{sample.image_path.parent.name}"].append(sample)

    groups: dict[str, list[Sample]] = {}
    for directory, members in by_dir.items():
        members.sort(key=lambda s: s.image_path.name)
        for start in range(0, len(members), block_size):
            groups[f"{directory}#{start // block_size:05d}"] = members[start : start + block_size]
    return groups


def assign_splits(
    groups: dict[str, list[Sample]], val_fraction: float
) -> dict[str, str]:
    """Distribute whole blocks to train/val, stratified by source.

    Stratified, not global. A global greedy pack looks reasonable and is
    quietly wrong here: every block is the same size, so ordering by size then
    key degenerates to alphabetical, and validation filled up entirely from
    whichever source sorts first. The result was a val set made purely of
    stair images - 2 of 26 classes evaluated - while the split ratio looked
    perfect at 15%.

    Taking the fraction from each source independently guarantees every source,
    and therefore every class, appears on both sides. That is the property that
    actually matters; hitting the global ratio exactly is not.

    Deterministic: blocks are ordered by key within each source, so a rebuild
    reproduces the split and it never depends on filesystem iteration order.
    """
    by_source: dict[str, list[str]] = defaultdict(list)
    for key in groups:
        by_source[key.split("/", 1)[0]].append(key)

    splits: dict[str, str] = {}
    for source, keys in sorted(by_source.items()):
        keys.sort()
        total = sum(len(groups[k]) for k in keys)
        target = total * val_fraction
        taken = 0
        chosen: set[str] = set()
        for key in keys:
            if taken >= target:
                break
            # Never take the last block: a source must keep something to train
            # on. With a single block this leaves `chosen` empty, which is the
            # correct answer - there is nothing to hold out.
            if len(chosen) >= len(keys) - 1:
                break
            chosen.add(key)
            taken += len(groups[key])

        # A source with more than one block must contribute to validation, or
        # its classes are measured on nothing.
        if not chosen and len(keys) > 1:
            chosen.add(keys[-1])
        elif not chosen:
            log.warning(
                "source %r produced a single block, so nothing can be held out from it. "
                "Its classes will not appear in validation.", source,
            )
        for key in keys:
            splits[key] = "val" if key in chosen else "train"
    return splits


def shipped_classes(
    samples: list[Sample], taxonomy: Taxonomy, min_instances: int
) -> list[str]:
    """Which classes the model will actually be trained to detect.

    A 77-class head where 51 outputs never fire is not free: it is a wider
    output tensor, more NMS work per frame, and a label set that claims support
    the model does not have. Worse, a class with 2 training instances will
    produce occasional confident nonsense rather than nothing, which for a
    hazard class is the bad direction to fail in.

    So the shipped label set is the classes with enough data to learn, and the
    taxonomy stays 77 because it is the product's vocabulary rather than this
    checkpoint's. Everything excluded is named in the evaluation as
    known-absent.
    """
    counts: Counter[str] = Counter()
    for sample in samples:
        if sample.role == "eval_only":
            continue
        for box in sample.boxes:
            counts[box.label] += 1
    return [c.name for c in taxonomy if counts.get(c.name, 0) >= min_instances]


def build_dataset(
    samples: list[Sample],
    out_dir: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    val_fraction: float = 0.15,
    copy_images: bool = False,
    block_size: int = 200,
    min_instances: int = 1,
) -> BuildStats:
    """Write a YOLO-format dataset. Returns the distribution to inspect first."""
    taxonomy = taxonomy or Taxonomy.load()
    out = Path(out_dir)
    stats = BuildStats()

    # Compacted ids: the model's class 0..M-1, not the taxonomy's 0..76. A
    # sparse head wastes capacity and inference on outputs that never fire.
    ship = shipped_classes(samples, taxonomy, min_instances)
    index = {name: i for i, name in enumerate(ship)}
    taxonomy_names = {c.name for c in taxonomy}
    stats.shipped = ship
    stats.excluded_thin = [
        c.name for c in taxonomy if c.name not in index
    ]

    # Clear previous output before writing.
    #
    # Rebuilding into a populated directory silently mixes label files from two
    # different id spaces. When the shipped class set changed from 77 to 26,
    # the old files kept referring to ids 43-47, and the trainer's response was
    # to "ignore corrupt label" and carry on - training on a quietly reduced
    # subset with no error and no exit code. A stale build directory has to be
    # impossible rather than merely discouraged.
    #
    # Only directories this function created are touched, identified by the
    # data.yaml it writes. Anything else is left alone rather than guessed at.
    ours = (out / "data.yaml").exists() or not out.exists()
    for split in ("train", "val", "val_domain"):
        for kind in ("images", "labels"):
            target = out / kind / split
            if ours and target.exists():
                for existing in target.iterdir():
                    if existing.is_file() or existing.is_symlink():
                        existing.unlink()
            target.mkdir(parents=True, exist_ok=True)
    if not ours:
        log.warning(
            "%s was not created by this builder (no data.yaml); leaving existing "
            "files in place. Stale labels from a previous id space will be "
            "silently ignored by the trainer - delete it by hand if unsure.",
            out,
        )

    # Sources marked eval_only never enter the training split, whatever the
    # ratio works out to. That is a licensing constraint for IDD and a
    # deliberate experiment besides: holding out Indian road footage measures
    # the domain gap rather than papering over it.
    trainable = [s for s in samples if s.role != "eval_only"]
    held_out = [s for s in samples if s.role == "eval_only"]

    groups = group_samples(trainable, block_size=block_size)
    split_of = assign_splits(groups, val_fraction)

    # Held-out sources get their OWN split, not merged into val.
    #
    # Merging them was a real methodology error. IDD is 41,451 images against
    # an ordinary val split of 2,655, so validation became 94% IDD - and IDD
    # contains only 7 of the 26 shipped classes. The result: 17 classes had no
    # validation data whatsoever, and the headline mAP silently measured
    # "Indian roads, COCO-ish classes" while claiming to measure the model.
    #
    # Two sets answer two different questions. `val` asks whether the model
    # learned what it was taught; `val_domain` asks whether that transfers to
    # the country the product is for. Averaging them answers neither.
    for key, members in group_samples(held_out, block_size=block_size).items():
        groups[key] = members
        split_of[key] = "val_domain"

    seen_names: set[str] = set()
    for key, members in groups.items():
        split = split_of[key]
        for sample in members:
            lines: list[str] = []
            for box in sample.boxes:
                # Count coverage BEFORE the shipping check. A held-out class
                # like auto_rickshaw is not in the model's label set and still
                # has to appear in the report - it is the reason IDD was held
                # out, and silently dropping it would hide exactly the gap the
                # hold-out exists to expose.
                if sample.role == "eval_only":
                    stats.per_class_eval[box.label] += 1
                else:
                    stats.per_class_train[box.label] += 1

                if box.label not in index:
                    if box.label in taxonomy_names:
                        # In the taxonomy, below the instance threshold.
                        stats.not_shipped[box.label] += 1
                    else:
                        # Not a taxonomy class at all - a label-map bug.
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
            elif split == "val_domain":
                stats.eval_only_images += 1
            else:
                stats.val_images += 1

    names = {i: name for i, name in enumerate(ship)}
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(ship),
                "names": names,
            },
            sort_keys=False,
        )
    )
    # The domain-gap set: same classes, different country. Evaluated
    # separately so a transfer failure is visible rather than averaged away.
    (out / "data_domain.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val_domain",
                "nc": len(ship),
                "names": names,
            },
            sort_keys=False,
        )
    )
    # The label file the model manifest will point at, in model order.
    (out / "labels.txt").write_text("\n".join(ship) + "\n")
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
