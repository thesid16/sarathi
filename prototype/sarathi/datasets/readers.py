"""Reading each source dataset into one common shape.

Every dataset arrives in a different format and speaks a different label
vocabulary. This module turns each into `Sample` objects carrying taxonomy
labels, and nothing downstream ever learns which dataset a sample came from.

The label maps live in `training/datasets/*.yaml`, not here. Mapping decisions
are judgement calls that deserve review - `tricycle` becoming `cycle_rickshaw`
is an approximation someone should be able to argue with - and judgement calls
belong in reviewable data rather than buried in a parser.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..util.log import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Box:
    label: str  # already remapped to a taxonomy class name
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            self.label,
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    @property
    def valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


@dataclass
class Sample:
    image_path: Path
    width: int
    height: int
    source: str
    boxes: list[Box] = field(default_factory=list)
    #: Registered depth map, where the dataset has one. Only Mendeley does,
    #: and it is the reason that dataset matters beyond its size.
    depth_path: Path | None = None


@dataclass
class ReadStats:
    files_seen: int = 0
    samples_kept: int = 0
    boxes_kept: int = 0
    dropped_unmapped: dict[str, int] = field(default_factory=dict)
    dropped_degenerate: int = 0
    missing_images: int = 0

    def note_unmapped(self, label: str) -> None:
        self.dropped_unmapped[label] = self.dropped_unmapped.get(label, 0) + 1

    def summary(self) -> str:
        lines = [
            f"  files seen      {self.files_seen}",
            f"  samples kept    {self.samples_kept}",
            f"  boxes kept      {self.boxes_kept}",
        ]
        if self.missing_images:
            lines.append(f"  ! missing images  {self.missing_images}")
        if self.dropped_degenerate:
            lines.append(f"  dropped degenerate boxes  {self.dropped_degenerate}")
        if self.dropped_unmapped:
            top = sorted(self.dropped_unmapped.items(), key=lambda kv: -kv[1])
            joined = ", ".join(f"{k}:{v}" for k, v in top[:8])
            lines.append(f"  dropped unmapped  {joined}")
        return "\n".join(lines)


def _find_image(annotation: Path, images_dir: Path | None, declared: str | None) -> Path | None:
    """Locate the image an annotation refers to.

    VOC files name their image, but the filename in the XML is unreliable in
    community datasets - it often still points at whatever the exporter was
    called locally. Falling back to matching stems is what makes these
    downloads usable at all.
    """
    candidates: list[Path] = []
    if declared:
        name = Path(declared).name
        if images_dir:
            candidates.append(images_dir / name)
        candidates.append(annotation.parent / name)
    for suffix in IMAGE_SUFFIXES:
        if images_dir:
            candidates.append(images_dir / (annotation.stem + suffix))
        candidates.append(annotation.parent / (annotation.stem + suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_voc(
    root: str | Path,
    label_map: dict[str, str | None],
    *,
    source: str = "voc",
    stats: ReadStats | None = None,
) -> Iterator[Sample]:
    """Read Pascal VOC XML annotations. Covers WOTR and the Roboflow exports."""
    root = Path(root)
    stats = stats if stats is not None else ReadStats()

    images_dir = next(
        (root / name for name in ("JPEGImages", "images") if (root / name).is_dir()), None
    )

    for xml_path in sorted(root.rglob("*.xml")):
        stats.files_seen += 1
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            log.debug("unparseable annotation %s: %s", xml_path.name, exc)
            continue

        declared = tree.findtext("filename")
        image_path = _find_image(xml_path, images_dir, declared)
        if image_path is None:
            stats.missing_images += 1
            continue

        size = tree.find("size")
        width = int(float(size.findtext("width") or 0)) if size is not None else 0
        height = int(float(size.findtext("height") or 0)) if size is not None else 0
        if width <= 0 or height <= 0:
            continue

        boxes: list[Box] = []
        for obj in tree.iter("object"):
            raw = (obj.findtext("name") or "").strip()
            if raw not in label_map:
                stats.note_unmapped(raw)
                continue
            mapped = label_map[raw]
            if mapped is None:  # deliberately dropped
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                box = Box(
                    mapped,
                    float(bnd.findtext("xmin") or 0),
                    float(bnd.findtext("ymin") or 0),
                    float(bnd.findtext("xmax") or 0),
                    float(bnd.findtext("ymax") or 0),
                ).clipped(width, height)
            except (TypeError, ValueError):
                continue
            if not box.valid:
                stats.dropped_degenerate += 1
                continue
            boxes.append(box)

        if not boxes:
            continue
        stats.samples_kept += 1
        stats.boxes_kept += len(boxes)
        yield Sample(image_path, width, height, source, boxes)


def read_mendeley_stairs(
    root: str | Path,
    label_map: dict[int, str],
    *,
    source: str = "mendeley_stairs",
    edge_height_frac: float = 0.045,
    stats: ReadStats | None = None,
) -> Iterator[Sample]:
    """Read the Mendeley stair dataset.

    Its annotations are *line segments*, not boxes: one line per stair edge,
    `cls x1 y1 x2 y2`, where cls 0 is convex (the tread rises toward you, a
    step up) and 1 is concave (it falls away, a step down).

    Converting a line to a box means giving it vertical extent. That is lossy,
    but a step edge genuinely is a thin horizontal thing, so a shallow box
    around the segment is a fair representation rather than a fudge.

    Coordinates are auto-detected as pixels or normalised, because community
    re-uploads of this dataset disagree and silently misreading normalised
    coordinates as pixels would put every box in the top-left corner.
    """
    root = Path(root)
    stats = stats if stats is not None else ReadStats()

    for label_path in sorted(root.rglob("*.txt")):
        if label_path.name in {"urls.tsv", "README.txt"}:
            continue
        stats.files_seen += 1

        image_path = next(
            (p for p in (label_path.with_suffix(".jpg"), label_path.with_suffix(".jpeg"))
             if p.exists()), None
        )
        if image_path is None:
            stats.missing_images += 1
            continue
        depth_path = label_path.with_suffix(".png")

        try:
            lines = label_path.read_text().strip().splitlines()
        except OSError:
            continue

        parsed: list[tuple[int, float, float, float, float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                parsed.append(
                    (int(float(parts[0])), *(float(v) for v in parts[1:5]))  # type: ignore[misc]
                )
            except ValueError:
                continue
        if not parsed:
            continue

        # The dataset is 512x512 throughout; read it from the file only if
        # something disagrees.
        width = height = 512
        coords = [v for _, *rest in parsed for v in rest]
        normalised = bool(coords) and max(coords) <= 1.5
        scale_x, scale_y = (width, height) if normalised else (1.0, 1.0)

        boxes: list[Box] = []
        pad = edge_height_frac * height / 2.0
        for cls, x1, y1, x2, y2 in parsed:
            mapped = label_map.get(cls)
            if mapped is None:
                stats.note_unmapped(str(cls))
                continue
            px1, px2 = sorted((x1 * scale_x, x2 * scale_x))
            py1, py2 = sorted((y1 * scale_y, y2 * scale_y))
            box = Box(mapped, px1, py1 - pad, px2, py2 + pad).clipped(width, height)
            if not box.valid:
                stats.dropped_degenerate += 1
                continue
            boxes.append(box)

        if not boxes:
            continue
        stats.samples_kept += 1
        stats.boxes_kept += len(boxes)
        yield Sample(
            image_path, width, height, source, boxes,
            depth_path=depth_path if depth_path.exists() else None,
        )
