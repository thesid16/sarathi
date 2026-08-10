"""Tests for dataset ingest.

The properties asserted here are the ones whose absence produces a model that
looks good and is not: a split that leaks adjacent frames between train and
val, a label map that silently drops half a class, and coordinates read in the
wrong units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sarathi.datasets import (
    ReadStats,
    Sample,
    build_dataset,
    read_mendeley_stairs,
    read_voc,
    write_attribution,
)
from sarathi.datasets.readers import Box
from sarathi.taxonomy import Taxonomy

TAX = Taxonomy.load()


def voc_xml(filename, width, height, objects):
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in objects
    )
    return (f"<annotation><filename>{filename}</filename>"
            f"<size><width>{width}</width><height>{height}</height></size>{objs}</annotation>")


def make_voc(root: Path, name: str, objects, width=640, height=480):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / f"{name}.xml").write_text(voc_xml(f"{name}.jpg", width, height, objects))


# -- VOC ---------------------------------------------------------------------


def test_voc_reads_and_remaps_labels(tmp_path):
    make_voc(tmp_path, "a", [("pole", 10, 20, 60, 200), ("tricycle", 100, 50, 200, 300)])
    stats = ReadStats()
    samples = list(read_voc(tmp_path, {"pole": "pole", "tricycle": "cycle_rickshaw"},
                            source="wotr", stats=stats))
    assert len(samples) == 1
    assert {b.label for b in samples[0].boxes} == {"pole", "cycle_rickshaw"}
    assert stats.boxes_kept == 2


def test_labels_mapped_to_none_are_dropped_deliberately(tmp_path):
    """WOTR's undirected 'stair' label is dropped rather than guessed at."""
    make_voc(tmp_path, "a", [("pole", 10, 20, 60, 200), ("stair", 0, 0, 50, 50)])
    samples = list(read_voc(tmp_path, {"pole": "pole", "stair": None}))
    assert [b.label for b in samples[0].boxes] == ["pole"]


def test_unmapped_labels_are_counted_not_silently_ignored(tmp_path):
    """A label map that misses half a class should be visible, not silent."""
    make_voc(tmp_path, "a", [("pole", 10, 20, 60, 200), ("mystery", 0, 0, 50, 50)])
    stats = ReadStats()
    list(read_voc(tmp_path, {"pole": "pole"}, stats=stats))
    assert stats.dropped_unmapped == {"mystery": 1}
    assert "mystery" in stats.summary()


def test_boxes_are_clipped_to_the_image(tmp_path):
    make_voc(tmp_path, "a", [("pole", -50, -50, 9999, 9999)], width=640, height=480)
    box = list(read_voc(tmp_path, {"pole": "pole"}))[0].boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 640, 480)


def test_degenerate_boxes_are_dropped(tmp_path):
    make_voc(tmp_path, "a", [("pole", 10, 20, 60, 200), ("pole", 50, 50, 50, 50)])
    stats = ReadStats()
    samples = list(read_voc(tmp_path, {"pole": "pole"}, stats=stats))
    assert len(samples[0].boxes) == 1
    assert stats.dropped_degenerate == 1


def test_an_annotation_with_no_matching_image_is_counted(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "orphan.xml").write_text(voc_xml("nope.jpg", 640, 480, [("pole", 1, 1, 9, 9)]))
    stats = ReadStats()
    assert list(read_voc(tmp_path, {"pole": "pole"}, stats=stats)) == []
    assert stats.missing_images == 1


def test_image_is_found_by_stem_when_the_xml_filename_is_wrong(tmp_path):
    """Community exports routinely keep the exporter's local filenames."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "real.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "real.xml").write_text(
        voc_xml("C:/Users/someone/Desktop/whatever.jpg", 640, 480, [("pole", 1, 1, 9, 9)])
    )
    assert len(list(read_voc(tmp_path, {"pole": "pole"}))) == 1


# -- Mendeley stair lines ----------------------------------------------------


def make_mendeley(root: Path, name: str, lines: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / f"{name}.png").write_bytes(b"\x89PNG")
    (root / f"{name}.txt").write_text(lines)


def test_stair_lines_become_boxes_with_vertical_extent(tmp_path):
    make_mendeley(tmp_path, "color_0", "1 100 300 400 305\n0 100 200 400 202\n")
    samples = list(read_mendeley_stairs(tmp_path, {0: "step_up", 1: "step_down"}))
    assert len(samples) == 1
    labels = [b.label for b in samples[0].boxes]
    assert labels == ["step_down", "step_up"]
    down = samples[0].boxes[0]
    assert down.x1 == 100 and down.x2 == 400
    assert down.y2 - down.y1 > 5  # a line got real height


def test_the_depth_map_is_carried_through(tmp_path):
    """The whole reason this dataset matters."""
    make_mendeley(tmp_path, "color_0", "1 100 300 400 305\n")
    sample = list(read_mendeley_stairs(tmp_path, {0: "step_up", 1: "step_down"}))[0]
    assert sample.depth_path is not None and sample.depth_path.suffix == ".png"


def test_normalised_coordinates_are_detected_not_misread(tmp_path):
    """Reading normalised coordinates as pixels puts every box in the corner."""
    make_mendeley(tmp_path, "color_0", "1 0.2 0.5 0.8 0.51\n")
    box = list(read_mendeley_stairs(tmp_path, {0: "step_up", 1: "step_down"}))[0].boxes[0]
    assert box.x1 == pytest.approx(0.2 * 512, abs=1)
    assert box.x2 == pytest.approx(0.8 * 512, abs=1)


def test_malformed_label_lines_are_skipped(tmp_path):
    make_mendeley(tmp_path, "color_0", "garbage\n1 100 300 400 305\n\n")
    assert len(list(read_mendeley_stairs(tmp_path, {1: "step_down"}))[0].boxes) == 1


# -- building ----------------------------------------------------------------


def sample_at(tmp_path, session: str, name: str, label="pole"):
    d = tmp_path / session
    d.mkdir(parents=True, exist_ok=True)
    img = d / f"{name}.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    return Sample(img, 640, 480, "src", [Box(label, 10, 20, 60, 200)])


def test_build_writes_yolo_labels_and_a_data_yaml(tmp_path):
    out = tmp_path / "out"
    stats = build_dataset([sample_at(tmp_path, "s1", "a")], out, TAX)
    assert (out / "data.yaml").exists()
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["nc"] == len(TAX)
    label_files = list((out / "labels").rglob("*.txt"))
    assert len(label_files) == 1
    cls, cx, cy, bw, bh = label_files[0].read_text().split()
    assert int(cls) == TAX["pole"].id
    assert 0 < float(cx) < 1 and 0 < float(bh) < 1
    assert stats.per_class["pole"] == 1


def test_frames_from_one_session_never_straddle_the_split(tmp_path):
    """The failure that makes a validation score meaningless."""
    samples = [sample_at(tmp_path, "walk1", f"f{i}") for i in range(30)]
    samples += [sample_at(tmp_path, "walk2", f"g{i}") for i in range(30)]
    build_dataset(samples, tmp_path / "out", TAX, val_fraction=0.5)

    def session_of(p: Path) -> str:
        return "walk1" if p.stem.split("__")[1].startswith("f") else "walk2"

    train = {session_of(p) for p in (tmp_path / "out/images/train").iterdir()}
    val = {session_of(p) for p in (tmp_path / "out/images/val").iterdir()}
    assert not (train & val), "a session appeared on both sides of the split"


def test_the_split_is_deterministic_across_rebuilds(tmp_path):
    samples = [sample_at(tmp_path, f"s{i}", "a") for i in range(12)]
    first = build_dataset(samples, tmp_path / "o1", TAX)
    second = build_dataset(samples, tmp_path / "o2", TAX)
    assert (first.train_images, first.val_images) == (second.train_images, second.val_images)


def test_colliding_filenames_from_different_sources_do_not_overwrite(tmp_path):
    a = sample_at(tmp_path, "s1", "0001")
    b = sample_at(tmp_path, "s2", "0001")
    b.source = "other"
    build_dataset([a, b], tmp_path / "out", TAX)
    assert len(list((tmp_path / "out/labels").rglob("*.txt"))) == 2


def test_labels_outside_the_taxonomy_are_reported(tmp_path):
    s = sample_at(tmp_path, "s1", "a")
    s.boxes[0].label = "not_a_taxonomy_class"
    stats = build_dataset([s], tmp_path / "out", TAX)
    assert stats.skipped_unknown_class["not_a_taxonomy_class"] == 1


def test_the_report_names_classes_with_no_data(tmp_path):
    """77 classes assembled from public data is violently long-tailed, and
    finding that out after a training run has wasted the run."""
    report = build_dataset([sample_at(tmp_path, "s1", "a")], tmp_path / "out", TAX).report(TAX)
    assert "have NO training data" in report
    assert "open_manhole" in report


def test_the_report_flags_thin_classes(tmp_path):
    report = build_dataset([sample_at(tmp_path, "s1", "a")], tmp_path / "out", TAX).report(TAX)
    assert "fewer than 50 instances" in report


# -- attribution -------------------------------------------------------------


def test_attribution_is_generated_from_the_configs(tmp_path):
    configs = [{
        "sources": {
            "rf_stairs_katti": {
                "licence": "CC-BY-4.0",
                "url": "https://universe.roboflow.com/katti/stairs-5yily",
                "attribution": "Stairs dataset, katti, CC BY 4.0",
            }
        }
    }]
    out = tmp_path / "ATTRIBUTION.md"
    write_attribution(configs, out)
    text = out.read_text()
    assert "CC-BY-4.0" in text and "katti" in text
    assert "Do not edit by hand" in text
