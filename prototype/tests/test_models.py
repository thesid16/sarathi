"""Tests for manifests and the model registry.

No model weights are needed: a fake adapter and small dummy files exercise the
whole path. The point of these tests is that misconfiguration fails loudly at
load time with a message that says what to do, rather than quietly producing a
model that runs and detects nonsense.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml

from sarathi.models import (
    Detector,
    Distribution,
    ManifestError,
    ModelManifest,
    ModelRegistry,
    Task,
    register_adapter,
)
from sarathi.models.manifest import FileSpec, Layout, Resize

VALID = {
    "id": "test-detector",
    "task": "detection",
    "family": "yolox",
    "version": "1.0.0",
    "license": "Apache-2.0",
    "distribution": "bundled",
    "source_url": "https://example.invalid/yolox",
    "runtime": {"prototype": "onnxruntime", "android": "litert"},
    "files": {"onnx": {"path": "test.onnx"}},
    "input": {"width": 320, "height": 320, "color": "RGB", "resize": "letterbox"},
    "output": {"decoder": "yolox", "labels": ["person", "chair"], "conf_threshold": 0.4},
}


def write_manifest(directory, data, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{data.get('id', 'model')}.yaml")
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def registry(tmp_path):
    manifests = tmp_path / "manifests"
    weights = tmp_path / "weights"
    manifests.mkdir()
    weights.mkdir()
    (weights / "test.onnx").write_bytes(b"not a real model")
    write_manifest(manifests, VALID)
    return ModelRegistry(manifests, weights)


class FakeDetector(Detector):
    def detect(self, image):
        return []


@pytest.fixture(autouse=True)
def fake_adapter():
    register_adapter(Task.DETECTION, "onnxruntime", FakeDetector)


# -- parsing -----------------------------------------------------------------


def test_valid_manifest_parses():
    manifest = ModelManifest.from_dict(VALID)
    assert manifest.id == "test-detector"
    assert manifest.task is Task.DETECTION
    assert manifest.distribution is Distribution.BUNDLED
    assert manifest.input is not None and manifest.input.width == 320
    assert manifest.input.layout is Layout.NCHW  # default
    assert manifest.input.resize is Resize.LETTERBOX
    assert manifest.output is not None and manifest.output.conf_threshold == 0.4
    assert manifest.loadable and manifest.committed


@pytest.mark.parametrize("key", ["id", "task", "license", "distribution", "files"])
def test_missing_required_key_names_the_key(key):
    data = {k: v for k, v in VALID.items() if k != key}
    with pytest.raises(ManifestError, match=key):
        ModelManifest.from_dict(data)


def test_unknown_task_lists_the_valid_options():
    with pytest.raises(ManifestError, match="detection, depth, ocr, vlm"):
        ModelManifest.from_dict({**VALID, "task": "telepathy"})


def test_unknown_distribution_lists_the_valid_options():
    with pytest.raises(ManifestError, match="bundled, user_download, excluded"):
        ModelManifest.from_dict({**VALID, "distribution": "sure-why-not"})


@pytest.mark.parametrize("section", ["input", "output"])
def test_detection_requires_input_and_output(section):
    data = {k: v for k, v in VALID.items() if k != section}
    with pytest.raises(ManifestError, match=f"detection models require an `{section}`"):
        ModelManifest.from_dict(data)


def test_non_detection_tasks_do_not_require_input_or_output():
    manifest = ModelManifest.from_dict(
        {
            "id": "vlm",
            "task": "vlm",
            "license": "Apache-2.0",
            "distribution": "user_download",
            "files": {"gguf": "vlm.gguf"},
        }
    )
    assert manifest.task is Task.VLM


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.2])
def test_out_of_range_confidence_is_rejected(value):
    data = {**VALID, "output": {**VALID["output"], "conf_threshold": value}}
    with pytest.raises(ManifestError, match="conf_threshold"):
        ModelManifest.from_dict(data)


def test_zero_std_is_rejected():
    """A zero std divides every pixel by zero - silent NaNs downstream."""
    data = {**VALID, "input": {**VALID["input"], "std": [0.229, 0, 0.225]}}
    with pytest.raises(ManifestError, match="std must not contain zero"):
        ModelManifest.from_dict(data)


def test_scalar_mean_and_std_expand_to_three_channels():
    data = {**VALID, "input": {**VALID["input"], "mean": 0.5, "std": 0.25}}
    manifest = ModelManifest.from_dict(data)
    assert manifest.input is not None
    assert manifest.input.mean == (0.5, 0.5, 0.5)
    assert manifest.input.std == (0.25, 0.25, 0.25)


def test_bad_colour_order_is_rejected():
    data = {**VALID, "input": {**VALID["input"], "color": "CMYK"}}
    with pytest.raises(ManifestError, match="color must be RGB or BGR"):
        ModelManifest.from_dict(data)


def test_unknown_output_keys_are_preserved_for_decoders():
    """Decoder-specific settings must survive parsing, not be silently dropped."""
    data = {**VALID, "output": {**VALID["output"], "strides": [8, 16, 32]}}
    manifest = ModelManifest.from_dict(data)
    assert manifest.output is not None
    assert manifest.output.extra["strides"] == [8, 16, 32]


def test_file_for_maps_runtime_through_engine_to_format():
    manifest = ModelManifest.from_dict(VALID)
    assert manifest.file_for("prototype").path == "test.onnx"


def test_file_for_missing_runtime_lists_what_is_available():
    manifest = ModelManifest.from_dict(VALID)
    with pytest.raises(ManifestError, match="available formats"):
        manifest.file_for("android")  # declares litert, but no tflite file


def test_invalid_yaml_is_reported_with_the_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("id: x\n  bad: [indent")
    with pytest.raises(ManifestError, match="invalid YAML"):
        ModelManifest.from_file(path)


# -- integrity ---------------------------------------------------------------


def test_missing_weights_file_reports_the_download_url(tmp_path):
    spec = FileSpec(path="absent.onnx", url="https://example.invalid/model.onnx")
    with pytest.raises(ManifestError, match="https://example.invalid/model.onnx"):
        spec.verify(tmp_path)


def test_checksum_mismatch_is_caught(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"content")
    spec = FileSpec(path="m.onnx", sha256="0" * 64)
    with pytest.raises(ManifestError, match="checksum mismatch"):
        spec.verify(tmp_path)


def test_correct_checksum_passes(tmp_path):
    payload = b"content"
    (tmp_path / "m.onnx").write_bytes(payload)
    spec = FileSpec(path="m.onnx", sha256=hashlib.sha256(payload).hexdigest())
    spec.verify(tmp_path)  # must not raise


# -- registry ----------------------------------------------------------------


def test_registry_finds_and_loads_a_model(registry):
    assert [m.id for m in registry.list()] == ["test-detector"]
    model = registry.load("test-detector")
    assert isinstance(model, FakeDetector)
    assert model.id == "test-detector"
    assert model.detect(np.zeros((8, 8, 3), np.uint8)) == []


def test_unknown_model_id_lists_what_is_known(registry):
    with pytest.raises(ManifestError, match="test-detector"):
        registry.load("no-such-model")


def test_excluded_model_refuses_to_load(tmp_path):
    """Licence policy is enforced by code, not by remembering to check."""
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    (weights / "test.onnx").write_bytes(b"x")
    write_manifest(
        manifests,
        {**VALID, "id": "nc-model", "license": "CC-BY-NC-4.0", "distribution": "excluded"},
    )
    registry = ModelRegistry(manifests, weights)
    with pytest.raises(ManifestError, match="distribution=excluded"):
        registry.load("nc-model")


def test_one_broken_manifest_does_not_hide_the_good_ones(tmp_path, caplog):
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    (weights / "test.onnx").write_bytes(b"x")
    write_manifest(manifests, VALID)
    (manifests / "broken.yaml").write_text("task: detection\n")  # no id
    registry = ModelRegistry(manifests, weights)
    assert [m.id for m in registry.list()] == ["test-detector"]


def test_duplicate_model_ids_are_an_error(tmp_path):
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    write_manifest(manifests, VALID, name="a.yaml")
    write_manifest(manifests, VALID, name="b.yaml")
    with pytest.raises(ManifestError, match="duplicate model id"):
        ModelRegistry(manifests, weights).list()


def test_load_fails_clearly_when_weights_are_absent(tmp_path):
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    write_manifest(manifests, VALID)
    with pytest.raises(ManifestError, match="weights file not found"):
        ModelRegistry(manifests, weights).load("test-detector")


def test_filtering_by_task(registry):
    assert len(registry.list(task="detection")) == 1
    assert registry.list(task=Task.VLM) == []


def test_missing_manifest_directory_is_not_fatal(tmp_path):
    registry = ModelRegistry(tmp_path / "nope", tmp_path)
    assert registry.list() == []


# -- labels ------------------------------------------------------------------


def test_inline_labels(registry):
    manifest = registry.get("test-detector")
    assert registry.load_labels(manifest) == ["person", "chair"]


def test_labels_from_a_file_ignore_comments_and_blanks(tmp_path):
    manifests, weights, labels = tmp_path / "m", tmp_path / "w", tmp_path / "l"
    weights.mkdir()
    labels.mkdir()
    (labels / "tiny.txt").write_text("# classes\nperson\n\nchair\n  door  \n")
    data = {**VALID, "output": {**VALID["output"], "labels": "tiny"}}
    write_manifest(manifests, data)
    registry = ModelRegistry(manifests, weights, labels_dir=labels)
    assert registry.load_labels(registry.get("test-detector")) == ["person", "chair", "door"]


def test_missing_label_set_says_where_it_looked(tmp_path):
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    data = {**VALID, "output": {**VALID["output"], "labels": "absent"}}
    write_manifest(manifests, data)
    registry = ModelRegistry(manifests, weights, labels_dir=tmp_path / "l")
    with pytest.raises(ManifestError, match="not found; looked in"):
        registry.load_labels(registry.get("test-detector"))


# -- attribution -------------------------------------------------------------


def test_attribution_is_generated_and_omits_excluded_models(tmp_path):
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    write_manifest(manifests, VALID)
    write_manifest(
        manifests,
        {**VALID, "id": "nc-model", "license": "CC-BY-NC-4.0", "distribution": "excluded"},
    )
    text = ModelRegistry(manifests, weights).attribution_text()
    assert "test-detector" in text
    assert "https://example.invalid/yolox" in text
    assert "nc-model" not in text


# -- adapters ----------------------------------------------------------------


def test_unregistered_adapter_error_lists_what_is_registered(tmp_path):
    """`onnx` resolves to a real file, so this gets past the weights check."""
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    (weights / "test.onnx").write_bytes(b"x")
    data = {**VALID, "runtime": {"prototype": "onnx"}}
    write_manifest(manifests, data)
    registry = ModelRegistry(manifests, weights)
    with pytest.raises(ManifestError, match="no adapter for task 'detection'"):
        registry.load("test-detector")


def test_engine_with_no_matching_weights_fails_before_the_adapter_lookup(tmp_path):
    """The more useful of the two errors wins: missing weights, not missing adapter."""
    manifests, weights = tmp_path / "m", tmp_path / "w"
    weights.mkdir()
    (weights / "test.onnx").write_bytes(b"x")
    write_manifest(manifests, {**VALID, "runtime": {"prototype": "tensorrt"}})
    registry = ModelRegistry(manifests, weights)
    with pytest.raises(ManifestError, match="no weights for runtime 'prototype'"):
        registry.load("test-detector")
