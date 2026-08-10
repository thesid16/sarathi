"""Checks against the manifests this project actually ships.

Everything else in the test suite builds manifests inline, which verifies the
parser but says nothing about the YAML in `models/manifests/`. That gap is
where a real bug lived: `gemma-4-e2b-vlm.yaml` declared `runtime.android:
litert-lm`, no engine-format table had an entry for it, and so the app decided
the weights were "not installed" while sitting on 1.9 GB of them. Nothing threw
- the lookup simply returned nothing, and the feature was silently absent.

These tests read the shipped files and assert that every runtime a manifest
claims to support can actually resolve weights for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sarathi.models.manifest import ManifestError, ModelManifest

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "models" / "manifests"

# The Kotlin side of the same table. Kept here as a literal rather than parsed
# out of the .kt file: the point is to notice when the two drift, and a check
# that reads one of them cannot do that.
KOTLIN_ENGINE_FORMATS = {
    "litert": "tflite",
    "tflite": "tflite",
    "litert-lm": "litertlm",
    "litertlm": "litertlm",
    "onnxruntime": "onnx",
    "onnx": "onnx",
}


def manifest_paths() -> list[Path]:
    found = sorted(MANIFEST_DIR.glob("*.yaml"))
    assert found, f"no manifests found under {MANIFEST_DIR}"
    return found


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda p: p.stem)
def test_manifest_parses(path: Path) -> None:
    ModelManifest.from_file(path)


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda p: p.stem)
def test_every_declared_runtime_resolves_weights(path: Path) -> None:
    """A manifest that names a runtime must have a file that runtime can load.

    This is the regression. Declaring `android: litert-lm` while the files
    block is keyed `litertlm` is only correct if something maps between them,
    and when nothing did, the failure surfaced three layers away as a feature
    that appeared to be uninstalled.
    """
    manifest = ModelManifest.from_file(path)
    # `vendored_weights` already means exactly this: the runtime resolves its
    # own model, so there is nothing for the manifest to point at. RapidOCR
    # ships weights inside its package and ML Kit gets them from Play Services.
    if manifest.vendored_weights:
        return
    for runtime_name in manifest.runtime:
        try:
            resolved = manifest.file_for(runtime_name)
        except ManifestError as exc:
            pytest.fail(
                f"{path.name}: runtime {runtime_name!r} resolves no weights - {exc}"
            )
        assert resolved, f"{path.name}: runtime {runtime_name!r} resolved an empty path"


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda p: p.stem)
def test_android_runtime_resolves_under_the_kotlin_table(path: Path) -> None:
    """The phone must reach the same file the prototype does.

    Two parsers reading one manifest differently defeats the entire point of
    having one manifest, and the divergence is invisible from either side
    alone - each is internally consistent.
    """
    raw = yaml.safe_load(path.read_text())
    if raw.get("vendored_weights"):
        return
    engine = (raw.get("runtime") or {}).get("android")
    if engine is None:
        return
    files = raw.get("files") or {}
    fmt = KOTLIN_ENGINE_FORMATS.get(engine, engine)
    assert fmt in files or engine in files, (
        f"{path.name}: android runtime {engine!r} maps to format {fmt!r}, "
        f"which is not in files ({sorted(files)}). Add it to ENGINE_FORMATS in "
        f"ModelManifest.kt and _ENGINE_FORMATS in manifest.py together."
    )


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda p: p.stem)
def test_licence_and_distribution_are_declared(path: Path) -> None:
    """No model ships without a licence and a distribution decision.

    Both are load-bearing: `distribution` is enforced in code, and a missing
    licence on a public AGPL project is a problem for whoever reuses this.
    """
    raw = yaml.safe_load(path.read_text())
    assert raw.get("license"), f"{path.name}: no licence declared"
    assert raw.get("distribution") in {"bundled", "user_download", "excluded"}, (
        f"{path.name}: distribution must be bundled, user_download or excluded"
    )
