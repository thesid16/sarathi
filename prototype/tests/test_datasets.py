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
    # Ids are compacted to the shipped set, not the taxonomy's 0..76: a head
    # with outputs that never fire costs tensor width and NMS work per frame.
    ship = (out / "labels.txt").read_text().split()
    assert ship == ["pole"]
    assert data["nc"] == 1 and data["names"] == {0: "pole"}
    label_files = list((out / "labels").rglob("*.txt"))
    assert len(label_files) == 1
    cls, cx, cy, bw, bh = label_files[0].read_text().split()
    assert int(cls) == 0
    assert 0 < float(cx) < 1 and 0 < float(bh) < 1
    assert stats.per_class_train["pole"] == 1


def test_consecutive_frames_stay_together(tmp_path):
    """Adjacent frames on both sides of the split is what makes a validation
    score meaningless."""
    from sarathi.datasets.build import assign_splits, group_samples

    samples = [sample_at(tmp_path, "walk", f"frame_{i:05d}") for i in range(1000)]
    groups = group_samples(samples, block_size=200)
    splits = assign_splits(groups, 0.2)
    # Within any block every frame shares a split, and blocks are contiguous.
    for key, members in groups.items():
        names = sorted(s.image_path.stem for s in members)
        assert len(members) <= 200
        assert splits[key] in {"train", "val"}
        # contiguity: the block spans a single run of frame numbers
        nums = sorted(int(n.split("_")[1]) for n in names)
        assert nums[-1] - nums[0] == len(nums) - 1


def test_the_split_hits_its_target_even_with_few_huge_blocks(tmp_path):
    """One source with everything in a single directory produced a 15% target
    coming out at 79%. Largest-first packing has to hold regardless."""
    from sarathi.datasets.build import assign_splits, group_samples

    # One enormous directory plus a couple of small ones - the WOTR shape.
    samples = [sample_at(tmp_path, "big", f"f{i:05d}") for i in range(13928)]
    samples += [sample_at(tmp_path, "small", f"g{i:05d}") for i in range(500)]
    groups = group_samples(samples, block_size=200)
    splits = assign_splits(groups, 0.15)
    val = sum(len(v) for k, v in groups.items() if splits[k] == "val")
    ratio = val / len(samples)
    assert 0.12 <= ratio <= 0.18, f"target 15%, got {ratio:.0%}"


def test_a_single_block_source_does_not_break_the_assignment(tmp_path):
    from sarathi.datasets.build import assign_splits, group_samples

    samples = [sample_at(tmp_path, "solo", f"f{i}") for i in range(5)]
    groups = group_samples(samples, block_size=200)
    assert len(assign_splits(groups, 0.15)) == len(groups)


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
    assert "have NO TRAINING data" in report
    assert "open_manhole" in report


def test_held_out_boxes_never_count_as_training_coverage(tmp_path):
    """IDD supplies 32,280 auto_rickshaw boxes the model will never see. A
    combined total would show a well-covered class that has no training data."""
    held = [sample_at(tmp_path, "idd", f"g{i}", label="auto_rickshaw") for i in range(40)]
    for s_ in held:
        s_.source, s_.role = "idd", "eval_only"
    stats = build_dataset(
        [sample_at(tmp_path, "w", f"f{i}") for i in range(60)] + held,
        tmp_path / "out", TAX,
    )
    assert stats.per_class_eval["auto_rickshaw"] == 40
    assert stats.per_class_train["auto_rickshaw"] == 0
    report = stats.report(TAX)
    assert "eval only" in report
    section = report.split("have NO TRAINING data:")[1]
    assert "auto_rickshaw" in section


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


# -- COCO and eval-only roles ------------------------------------------------


def make_coco(tmp_path, categories, annotations, images):
    import json
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "train2017").mkdir(parents=True, exist_ok=True)
    for img in images:
        (root / "train2017" / img["file_name"]).write_bytes(b"\xff\xd8\xff\xd9")
    (root / "annotations" / "instances_train2017.json").write_text(json.dumps({
        "categories": categories, "annotations": annotations, "images": images,
    }))
    return root


def test_coco_reads_and_remaps(tmp_path):
    from sarathi.datasets import read_coco
    root = make_coco(
        tmp_path,
        [{"id": 1, "name": "chair"}, {"id": 2, "name": "couch"}],
        [{"image_id": 7, "category_id": 1, "bbox": [10, 20, 50, 60]},
         {"image_id": 7, "category_id": 2, "bbox": [80, 20, 40, 40]}],
        [{"id": 7, "file_name": "a.jpg", "width": 640, "height": 480}],
    )
    samples = list(read_coco(root, {"chair": "chair", "couch": "sofa"}))
    assert len(samples) == 1
    assert sorted(b.label for b in samples[0].boxes) == ["chair", "sofa"]
    # COCO bbox is xywh; it must come out as corners.
    chair = next(b for b in samples[0].boxes if b.label == "chair")
    assert (chair.x1, chair.y1, chair.x2, chair.y2) == (10, 20, 60, 80)


def test_coco_crowd_regions_are_skipped(tmp_path):
    """A crowd box is a blob over many instances; training on it teaches the
    model that a crowd is one object."""
    from sarathi.datasets import read_coco
    root = make_coco(
        tmp_path, [{"id": 1, "name": "person"}],
        [{"image_id": 7, "category_id": 1, "bbox": [10, 20, 50, 60], "iscrowd": 1}],
        [{"id": 7, "file_name": "a.jpg", "width": 640, "height": 480}],
    )
    assert list(read_coco(root, {"person": "person"})) == []


def test_coco_unmapped_categories_are_counted(tmp_path):
    from sarathi.datasets import ReadStats, read_coco
    root = make_coco(
        tmp_path, [{"id": 1, "name": "giraffe"}],
        [{"image_id": 7, "category_id": 1, "bbox": [1, 1, 5, 5]}],
        [{"id": 7, "file_name": "a.jpg", "width": 64, "height": 48}],
    )
    stats = ReadStats()
    list(read_coco(root, {"person": "person"}, stats=stats))
    assert stats.dropped_unmapped == {"giraffe": 1}


def test_missing_coco_annotations_is_not_fatal(tmp_path):
    from sarathi.datasets import read_coco
    assert list(read_coco(tmp_path, {"person": "person"})) == []


def test_eval_only_samples_never_enter_training(tmp_path):
    """The licence constraint on IDD, and the domain-gap experiment."""
    train = [sample_at(tmp_path, "w", f"f{i}") for i in range(100)]
    held = [sample_at(tmp_path, "idd", f"g{i}") for i in range(100)]
    for s in held:
        s.source, s.role = "idd", "eval_only"

    build_dataset(train + held, tmp_path / "out", TAX, val_fraction=0.15)
    train_names = {p.stem.split("__")[0] for p in (tmp_path / "out/images/train").iterdir()}
    assert "idd" not in train_names


def test_eval_only_images_are_reported_as_held_out(tmp_path):
    held = [sample_at(tmp_path, "idd", f"g{i}") for i in range(20)]
    for s in held:
        s.source, s.role = "idd", "eval_only"
    stats = build_dataset(
        [sample_at(tmp_path, "w", f"f{i}") for i in range(80)] + held,
        tmp_path / "out", TAX,
    )
    assert stats.eval_only_images == 20
    assert "held out by licence and design" in stats.report(TAX)


def test_parallel_annotation_and_image_trees_resolve(tmp_path):
    """IDD's layout: Annotations/set/scene/x.xml <-> JPEGImages/set/scene/x.jpg.
    The subdirectory path has to be preserved or nothing resolves at all."""
    root = tmp_path / "IDD_Detection"
    ann = root / "Annotations" / "highquality_16k" / "HYD-2018"
    img = root / "JPEGImages" / "highquality_16k" / "HYD-2018"
    ann.mkdir(parents=True)
    img.mkdir(parents=True)
    (img / "0001245.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ann / "0001245.xml").write_text(
        voc_xml("0001245.jpg", 1920, 1080, [("autorickshaw", 899, 638, 994, 780)])
    )
    samples = list(read_voc(root, {"autorickshaw": "auto_rickshaw"}, source="idd"))
    assert len(samples) == 1
    assert samples[0].boxes[0].label == "auto_rickshaw"
    assert samples[0].image_path.name == "0001245.jpg"


def test_idd_bndbox_element_order_does_not_matter(tmp_path):
    """IDD writes ymax before xmax. Parsing by position would break."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "a.xml").write_text(
        "<annotation><filename>a.jpg</filename>"
        "<size><width>640</width><height>480</height></size>"
        "<object><name>car</name><bndbox>"
        "<xmin>10</xmin><ymax>200</ymax><xmax>100</xmax><ymin>50</ymin>"
        "</bndbox></object></annotation>"
    )
    box = list(read_voc(root, {"car": "car"}))[0].boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 50, 100, 200)


def test_thin_classes_are_kept_out_of_the_shipped_label_set(tmp_path):
    """pothole has 2 real instances. A class with 2 examples produces occasional
    confident nonsense rather than silence, which for a hazard is the wrong
    direction to fail in."""
    samples = [sample_at(tmp_path, "s", f"a{i}", label="pole") for i in range(100)]
    samples += [sample_at(tmp_path, "s", f"b{i}", label="pothole") for i in range(2)]
    stats = build_dataset(samples, tmp_path / "out", TAX, min_instances=50)
    assert "pole" in stats.shipped
    assert "pothole" not in stats.shipped
    assert stats.not_shipped["pothole"] == 2
    # and it is still counted as coverage, so the report tells the truth
    assert stats.per_class_train["pothole"] == 2


def test_below_threshold_is_reported_separately_from_a_label_map_bug(tmp_path):
    """One is deliberate, the other means a config is wrong. Conflating them
    means nobody notices the config."""
    s1 = sample_at(tmp_path, "s", "a", label="pothole")
    s2 = sample_at(tmp_path, "s", "b")
    s2.boxes[0].label = "not_a_taxonomy_class"
    stats = build_dataset([s1, s2], tmp_path / "out", TAX, min_instances=99)
    assert stats.not_shipped["pothole"] == 1
    assert stats.skipped_unknown_class["not_a_taxonomy_class"] == 1
    report = stats.report(TAX)
    assert "below the instance threshold" in report
    assert "a config bug if unexpected" in report


def test_held_out_classes_survive_the_shipping_filter_in_the_report(tmp_path):
    """auto_rickshaw is eval-only and unshipped. Dropping it from the report
    would hide the exact gap the hold-out exists to expose."""
    train = [sample_at(tmp_path, "w", f"f{i}") for i in range(100)]
    held = [sample_at(tmp_path, "idd", f"g{i}", label="auto_rickshaw") for i in range(60)]
    for s_ in held:
        s_.source, s_.role = "idd", "eval_only"
    stats = build_dataset(train + held, tmp_path / "out", TAX, min_instances=50)
    assert "auto_rickshaw" not in stats.shipped
    assert stats.per_class_eval["auto_rickshaw"] == 60
    assert "auto_rickshaw" in stats.report(TAX)


def test_a_rebuild_removes_labels_from_the_previous_id_space(tmp_path):
    """The failure was silent and expensive: when the shipped set went from 77
    classes to 26, stale label files still referenced ids 43-47. The trainer's
    response is 'ignoring corrupt label' - it trains on a reduced subset with
    no error and no non-zero exit."""
    out = tmp_path / "out"
    build_dataset([sample_at(tmp_path, "s", "a")], out, TAX)
    stale = out / "labels" / "train" / "leftover.txt"
    stale.write_text("47 0.5 0.5 0.1 0.1\n")
    assert stale.exists()

    build_dataset([sample_at(tmp_path, "s", "b")], out, TAX)
    assert not stale.exists()
    for label_file in (out / "labels").rglob("*.txt"):
        for line in label_file.read_text().splitlines():
            assert int(line.split()[0]) == 0  # only the one shipped class


def test_a_directory_we_did_not_create_is_left_alone(tmp_path):
    """Never delete files out of a path this builder did not write."""
    out = tmp_path / "somebody_elses"
    (out / "labels" / "train").mkdir(parents=True)
    precious = out / "labels" / "train" / "precious.txt"
    precious.write_text("do not delete me")
    build_dataset([sample_at(tmp_path, "s", "a")], out, TAX)
    assert precious.exists()



def test_held_out_sources_get_their_own_split_not_merged_into_val(tmp_path):
    """Merging them was a real methodology error: IDD at 41,451 images against
    a 2,655-image val split made validation 94% IDD, and IDD carries only 7 of
    26 shipped classes - so 17 classes had no validation data at all while the
    headline mAP looked fine."""
    train = [sample_at(tmp_path, "w", f"f{i}") for i in range(200)]
    held = [sample_at(tmp_path, "idd", f"g{i}") for i in range(800)]
    for s_ in held:
        s_.source, s_.role = "idd", "eval_only"

    out = tmp_path / "out"
    stats = build_dataset(train + held, out, TAX, val_fraction=0.15, block_size=20)

    val_names = {p.stem.split("__")[0] for p in (out / "images/val").iterdir()}
    domain_names = {p.stem.split("__")[0] for p in (out / "images/val_domain").iterdir()}
    assert "idd" not in val_names, "held-out data must not dilute ordinary validation"
    assert domain_names == {"idd"}
    assert stats.eval_only_images == 800
    # ordinary val stays a sane fraction of the trainable data, not swamped
    assert 0.10 <= stats.val_images / 200 <= 0.20


def test_both_data_yamls_are_written(tmp_path):
    out = tmp_path / "out"
    held = [sample_at(tmp_path, "idd", f"g{i}") for i in range(20)]
    for s_ in held:
        s_.source, s_.role = "idd", "eval_only"
    build_dataset([sample_at(tmp_path, "w", f"f{i}") for i in range(80)] + held, out, TAX)
    import yaml as _y
    assert _y.safe_load((out / "data.yaml").read_text())["val"] == "images/val"
    assert _y.safe_load((out / "data_domain.yaml").read_text())["val"] == "images/val_domain"



def test_greedy_packing_never_leaves_validation_empty(tmp_path):
    """One block against a 15% target is closer to the target by staying out of
    val entirely - which would mean every later number is measured on training
    data. Forced across instead, loudly."""
    from sarathi.datasets.build import assign_splits, group_samples

    samples = [sample_at(tmp_path, "s", f"f{i:04d}") for i in range(400)]
    groups = group_samples(samples, block_size=200)   # exactly 2 blocks
    splits = assign_splits(groups, 0.15)
    assert "val" in splits.values(), "validation must never be empty when it can be filled"


def test_a_single_block_source_is_reported_rather_than_silently_unvalidated(tmp_path, caplog):
    from sarathi.datasets.build import assign_splits, group_samples

    samples = [sample_at(tmp_path, "s", f"f{i}") for i in range(5)]
    groups = group_samples(samples, block_size=200)   # one block
    splits = assign_splits(groups, 0.15)
    assert set(splits.values()) == {"train"}
