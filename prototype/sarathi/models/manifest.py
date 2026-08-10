"""Model manifests - the mechanism that makes models swappable.

A model is described by a YAML file, not by code. Swapping a detector, or
adding a vision-language model that did not exist when the app shipped, means
adding a manifest and a weights file. No Python changes, no Kotlin changes, no
release.

The same manifest is read by this prototype and by the Android app. That is
what keeps a model benchmarked on a laptop meaningful once it is running on a
phone: both sides agree on input size, colour order, normalisation, decoder and
label set, because all of it comes from one file rather than being reimplemented
twice and drifting.

Why the fields that look like paranoia are not:

* **`sha256`** - a wrong or truncated model file usually still loads. It then
  produces detections that look plausible and are wrong, which is far more
  expensive to debug than a hard failure at load time.
* **`distribution`** - Sarathi is AGPL-3.0 and public. The question is not "may
  we use this model" but "does bundling it restrict the people downstream".
  A CC-BY-NC model is `excluded` and refuses to load; a use-restricted model
  like Gemma is `user_download`, so its terms stay between the user and its
  publisher.
* **`attribution`** - several datasets and models require it. Generating the
  attribution file from manifests means it cannot drift out of date.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """Raised when a manifest is malformed, incomplete or internally inconsistent."""


class Task(str, Enum):
    DETECTION = "detection"
    DEPTH = "depth"
    OCR = "ocr"
    VLM = "vlm"


class Distribution(str, Enum):
    #: Licence permits shipping the weights with the project.
    BUNDLED = "bundled"
    #: Weights are fetched by the user, who accepts the model's own terms.
    #: Never committed, never packed into the APK.
    USER_DOWNLOAD = "user_download"
    #: Licence would restrict downstream users. Refuses to load.
    EXCLUDED = "excluded"


class Layout(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"


class Resize(str, Enum):
    #: Preserve aspect ratio, pad the remainder. Correct for detection - a
    #: stretched image moves every box.
    LETTERBOX = "letterbox"
    STRETCH = "stretch"
    CENTER_CROP = "center_crop"


class PadMode(str, Enum):
    """Where letterbox padding goes. Families genuinely disagree.

    Ultralytics centres the padding; YOLOX puts the image at the top-left and
    pads right and bottom. Get it wrong and every box is offset by half the pad
    - detections that are plausible, consistently misplaced, and easy to blame
    on the model.
    """

    CENTER = "center"
    CORNER = "corner"


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ManifestError(f"{where}: missing required key {key!r}")
    return data[key]


def _enum(cls: type[Enum], value: Any, where: str, key: str) -> Any:
    try:
        return cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in cls)  # type: ignore[attr-defined]
        raise ManifestError(f"{where}: {key}={value!r} is not one of: {allowed}") from None


@dataclass(frozen=True)
class FileSpec:
    """One weights file, plus how to verify it and where to get it."""

    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: Any, where: str) -> "FileSpec":
        # A bare string is allowed for the common case of a local file with no
        # integrity check - convenient during development, flagged by `verify`.
        if isinstance(data, str):
            return cls(path=data)
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: file entry must be a string or a mapping")
        return cls(
            path=str(_require(data, "path", where)),
            sha256=(str(data["sha256"]).lower() if data.get("sha256") else None),
            size_bytes=data.get("size_bytes"),
            url=data.get("url"),
        )

    def resolve(self, weights_dir: Path) -> Path:
        p = Path(self.path).expanduser()
        return p if p.is_absolute() else weights_dir / p

    def verify(self, weights_dir: Path) -> None:
        """Raise unless the file exists and matches its declared hash."""
        full = self.resolve(weights_dir)
        if not full.exists():
            hint = f"\n  download: {self.url}" if self.url else ""
            raise ManifestError(f"weights file not found: {full}{hint}")
        if self.sha256:
            digest = hashlib.sha256()
            with full.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != self.sha256:
                raise ManifestError(
                    f"checksum mismatch for {full}\n"
                    f"  expected {self.sha256}\n"
                    f"  actual   {actual}\n"
                    "  the file is corrupt or is not the model this manifest describes"
                )


@dataclass(frozen=True)
class InputSpec:
    """Everything needed to turn a camera frame into this model's input tensor.

    Getting any of this wrong produces a model that runs happily and detects
    nothing useful, so it is declared rather than inferred.
    """

    width: int
    height: int
    layout: Layout = Layout.NCHW
    color: str = "RGB"  # RGB | BGR
    dtype: str = "float32"  # float32 | uint8 | int8
    resize: Resize = Resize.LETTERBOX
    pad_mode: PadMode = PadMode.CENTER
    pad_value: int = 114
    scale: float = 1.0  # applied after cast, before mean/std
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any], where: str) -> "InputSpec":
        where = f"{where}.input"
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: must be a mapping")
        color = str(data.get("color", "RGB")).upper()
        if color not in {"RGB", "BGR"}:
            raise ManifestError(f"{where}: color must be RGB or BGR, got {color!r}")
        dtype = str(data.get("dtype", "float32")).lower()
        if dtype not in {"float32", "uint8", "int8"}:
            raise ManifestError(f"{where}: dtype must be float32, uint8 or int8, got {dtype!r}")

        def triple(key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
            value = data.get(key, default)
            if isinstance(value, (int, float)):
                return (float(value),) * 3
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ManifestError(f"{where}: {key} must be a number or three numbers")
            return tuple(float(v) for v in value)  # type: ignore[return-value]

        std = triple("std", (1.0, 1.0, 1.0))
        if any(s == 0 for s in std):
            raise ManifestError(f"{where}: std must not contain zero")

        width = int(_require(data, "width", where))
        height = int(_require(data, "height", where))
        if width <= 0 or height <= 0:
            raise ManifestError(f"{where}: width and height must be positive")

        return cls(
            width=width,
            height=height,
            layout=_enum(Layout, str(data.get("layout", "NCHW")).upper(), where, "layout"),
            color=color,
            dtype=dtype,
            resize=_enum(Resize, str(data.get("resize", "letterbox")).lower(), where, "resize"),
            pad_mode=_enum(PadMode, str(data.get("pad_mode", "center")).lower(), where, "pad_mode"),
            pad_value=int(data.get("pad_value", 114)),
            scale=float(data.get("scale", 1.0)),
            mean=triple("mean", (0.0, 0.0, 0.0)),
            std=std,
        )


@dataclass(frozen=True)
class OutputSpec:
    """How to turn this model's raw tensors back into detections.

    `decoder` names a decode strategy rather than describing one. Every
    detector family disagrees about output layout - anchor-free grids, anchor
    boxes, DETR-style queries - and that difference is code, not configuration.
    The manifest picks which code runs.
    """

    decoder: str
    labels: str | list[str] | None = None
    conf_threshold: float = 0.35
    nms_iou: float = 0.5
    max_detections: int = 50
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], where: str) -> "OutputSpec":
        where = f"{where}.output"
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: must be a mapping")
        known = {"decoder", "labels", "conf_threshold", "nms_iou", "max_detections"}
        conf = float(data.get("conf_threshold", 0.35))
        iou = float(data.get("nms_iou", 0.5))
        if not 0.0 < conf < 1.0:
            raise ManifestError(f"{where}: conf_threshold must be between 0 and 1, got {conf}")
        if not 0.0 < iou < 1.0:
            raise ManifestError(f"{where}: nms_iou must be between 0 and 1, got {iou}")
        return cls(
            decoder=str(_require(data, "decoder", where)),
            labels=data.get("labels"),
            conf_threshold=conf,
            nms_iou=iou,
            max_detections=int(data.get("max_detections", 50)),
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass(frozen=True)
class ModelManifest:
    """A complete, validated description of one model."""

    id: str
    task: Task
    license: str
    distribution: Distribution
    files: dict[str, FileSpec]
    input: InputSpec | None = None
    output: OutputSpec | None = None
    family: str | None = None
    version: str = "0.0.0"
    runtime: dict[str, str] = field(default_factory=dict)
    delegates: list[str] = field(default_factory=list)
    source_url: str | None = None
    attribution: str | None = None
    notes: str | None = None
    perf: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path | None = None) -> "ModelManifest":
        where = str(source_path) if source_path else "<manifest>"
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: manifest must be a mapping at the top level")

        model_id = str(_require(data, "id", where))
        task = _enum(Task, str(_require(data, "task", where)).lower(), where, "task")
        distribution = _enum(
            Distribution, str(_require(data, "distribution", where)).lower(), where, "distribution"
        )

        raw_files = _require(data, "files", where)
        if not isinstance(raw_files, dict) or not raw_files:
            raise ManifestError(f"{where}: files must be a non-empty mapping of format -> file")
        files = {
            str(fmt): FileSpec.from_dict(spec, f"{where}.files.{fmt}")
            for fmt, spec in raw_files.items()
        }

        input_spec = None
        if "input" in data:
            input_spec = InputSpec.from_dict(data["input"], where)
        output_spec = None
        if "output" in data:
            output_spec = OutputSpec.from_dict(data["output"], where)

        # Detection needs both to be usable at all; failing here beats failing
        # with a shape error deep inside a decoder.
        if task is Task.DETECTION:
            if input_spec is None:
                raise ManifestError(f"{where}: detection models require an `input` section")
            if output_spec is None:
                raise ManifestError(f"{where}: detection models require an `output` section")

        runtime = data.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ManifestError(f"{where}: runtime must be a mapping of platform -> engine")

        return cls(
            id=model_id,
            task=task,
            license=str(_require(data, "license", where)),
            distribution=distribution,
            files=files,
            input=input_spec,
            output=output_spec,
            family=data.get("family"),
            version=str(data.get("version", "0.0.0")),
            runtime={str(k): str(v) for k, v in runtime.items()},
            delegates=[str(d) for d in data.get("delegates", [])],
            source_url=data.get("source_url"),
            attribution=data.get("attribution"),
            notes=data.get("notes"),
            perf=data.get("perf", {}) or {},
            source_path=source_path,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelManifest":
        p = Path(path).expanduser()
        if not p.exists():
            raise ManifestError(f"manifest not found: {p}")
        try:
            data = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            raise ManifestError(f"{p}: invalid YAML: {exc}") from exc
        return cls.from_dict(data or {}, source_path=p)

    # -- queries -----------------------------------------------------------

    @property
    def loadable(self) -> bool:
        """False for models whose licence bars them from this project."""
        return self.distribution is not Distribution.EXCLUDED

    @property
    def committed(self) -> bool:
        """Whether the weights may live in the repository."""
        return self.distribution is Distribution.BUNDLED

    def file_for(self, runtime_name: str) -> FileSpec:
        """The weights file for a runtime, resolved through `runtime`.

        `runtime: {prototype: onnxruntime}` plus `files: {onnx: ...}` means
        asking for the "prototype" runtime returns the onnx file.
        """
        engine = self.runtime.get(runtime_name, runtime_name)
        for key in (engine, _ENGINE_FORMATS.get(engine, engine)):
            if key in self.files:
                return self.files[key]
        raise ManifestError(
            f"model {self.id!r}: no weights for runtime {runtime_name!r} "
            f"(engine {engine!r}); available formats: {sorted(self.files)}"
        )

    def describe(self) -> str:
        bits = [f"{self.id}  [{self.task.value}]"]
        if self.family:
            bits.append(f"family={self.family}")
        bits.append(f"v{self.version}")
        bits.append(f"{self.license} ({self.distribution.value})")
        if self.input:
            bits.append(f"{self.input.width}x{self.input.height} {self.input.dtype}")
        return "  ".join(bits)


#: Which file format each inference engine consumes.
_ENGINE_FORMATS = {
    "onnxruntime": "onnx",
    "onnx": "onnx",
    "litert": "tflite",
    "tflite": "tflite",
    "torch": "pt",
    "coreml": "mlpackage",
}
