"""Tests for the class taxonomy.

The real taxonomy file is loaded and checked, not a fixture, because the file
is a deliverable in its own right: a hole in the class ids silently shifts
every label after it, and a class claiming coverage it does not have is exactly
the sort of quiet optimism this project is trying to avoid.
"""

from __future__ import annotations

import pytest
import yaml

from sarathi.taxonomy import Taxonomy, TaxonomyError
from sarathi.types import Hazard

MINIMAL = {
    "version": "1.0.0",
    "sources": {"wotr": {"licence": "MIT"}, "coco": {"licence": "CC-BY-4.0"}},
    "classes": [
        {"id": 0, "name": "pole", "hi": "खंभा", "group": "obstacle",
         "hazard": "high", "sources": ["wotr"], "coverage": "good"},
        {"id": 1, "name": "chair", "hi": "कुर्सी", "group": "furniture",
         "hazard": "medium", "sources": ["coco"], "coverage": "good"},
    ],
}


def write(tmp_path, data, name="tax.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return path


# -- the real file -----------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    return Taxonomy.load()


def test_real_taxonomy_loads_and_is_contiguous(real):
    assert len(real) == 77
    assert [c.id for c in real] == list(range(77))


def test_every_class_has_both_languages(real):
    for c in real:
        assert c.name and c.hi, f"{c.name} is missing a label"
        # Hindi labels must actually be Devanagari, not an untranslated copy.
        assert any("ऀ" <= ch <= "ॿ" for ch in c.hi), f"{c.name}: hi is not Devanagari"


def test_label_lists_are_ordered_by_class_id(real):
    names = real.names("en")
    assert len(names) == len(real)
    assert names[real["pole"].id] == "pole"
    assert real.names("hi")[real["chair"].id] == real["chair"].hi


def test_underscores_become_spaces_for_speech(real):
    assert real["open_manhole"].label("en") == "open manhole"


def test_the_dangerous_classes_are_marked_dangerous(real):
    for name in ("open_manhole", "stairs_down", "step_down", "auto_rickshaw", "car"):
        assert real[name].hazard is Hazard.CRITICAL, f"{name} should be critical"


def test_blind_spots_are_only_dangerous_and_uncovered(real):
    blind = real.blind_spots()
    assert blind, "if this is empty the coverage fields have drifted optimistic"
    for c in blind:
        assert c.coverage in {"gap", "pending"}
        assert c.hazard in {Hazard.CRITICAL, Hazard.HIGH}


def test_coverage_report_names_the_gaps(real):
    report = real.coverage_report()
    assert "Blind spots" in report
    assert "glass_door" in report  # a real, documented limitation


def test_every_referenced_source_has_a_recorded_licence(real):
    for c in real:
        for source in c.sources:
            assert "licence" in real.sources[source], f"{source} has no licence recorded"


# -- validation --------------------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(TaxonomyError, match="not found"):
        Taxonomy.load(tmp_path / "absent.yaml")


def test_non_contiguous_ids_are_rejected(tmp_path):
    """A hole shifts every label after it - silently mislabelling everything."""
    data = {**MINIMAL, "classes": [
        {**MINIMAL["classes"][0], "id": 0},
        {**MINIMAL["classes"][1], "id": 5},
    ]}
    with pytest.raises(TaxonomyError, match="contiguous"):
        Taxonomy.load(write(tmp_path, data))


def test_duplicate_names_are_rejected(tmp_path):
    data = {**MINIMAL, "classes": [
        {**MINIMAL["classes"][0], "id": 0, "name": "pole"},
        {**MINIMAL["classes"][1], "id": 1, "name": "pole"},
    ]}
    with pytest.raises(TaxonomyError, match="duplicate class names"):
        Taxonomy.load(write(tmp_path, data))


def test_undeclared_source_is_rejected(tmp_path):
    """Every source must be declared so its licence is on record."""
    data = {**MINIMAL, "classes": [{**MINIMAL["classes"][0], "sources": ["mystery_dataset"]}]}
    with pytest.raises(TaxonomyError, match="undeclared source"):
        Taxonomy.load(write(tmp_path, data))


def test_claiming_good_coverage_with_no_source_is_rejected(tmp_path):
    data = {**MINIMAL, "classes": [
        {**MINIMAL["classes"][0], "sources": [], "coverage": "good"}
    ]}
    with pytest.raises(TaxonomyError, match="no sources listed"):
        Taxonomy.load(write(tmp_path, data))


@pytest.mark.parametrize(("field", "value", "match"), [
    ("hazard", "quite-bad", "hazard must be one of"),
    ("coverage", "probably-fine", "coverage must be one of"),
])
def test_invalid_enum_values_list_the_options(tmp_path, field, value, match):
    data = {**MINIMAL, "classes": [{**MINIMAL["classes"][0], field: value}]}
    with pytest.raises(TaxonomyError, match=match):
        Taxonomy.load(write(tmp_path, data))


@pytest.mark.parametrize("key", ["id", "name", "hi", "group", "hazard", "coverage"])
def test_missing_required_field_names_it(tmp_path, key):
    entry = {k: v for k, v in MINIMAL["classes"][0].items() if k != key}
    data = {**MINIMAL, "classes": [entry]}
    with pytest.raises(TaxonomyError, match=key):
        Taxonomy.load(write(tmp_path, data))


def test_empty_taxonomy_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="no classes"):
        Taxonomy.load(write(tmp_path, {**MINIMAL, "classes": []}))


def test_lookup_by_id_and_by_name_agree(tmp_path):
    tax = Taxonomy.load(write(tmp_path, MINIMAL))
    assert tax[0] is tax["pole"]
    assert tax.hazard_map()[1] is Hazard.MEDIUM
