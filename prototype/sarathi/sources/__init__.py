"""Camera source registry.

Adding a new kind of camera means writing one `FrameSource` subclass and adding
one line to `_REGISTRY`. Nothing else in the codebase changes - which is the
point, because the product has to accept whatever camera the user happens to
own.

Sources are named in config by `kind`, or given as a bare URL/path and
inferred:

    source: {kind: webcam, index: 0}
    source: {kind: mjpeg, url: "http://192.168.4.1:81/stream"}
    source: {kind: rtsp, url: "rtsp://192.168.1.60:8554/cam"}
    source: {kind: file, path: "footage/corridor.mp4", realtime: true}
    source: "http://192.168.4.1:81/stream"      # inferred as mjpeg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .base import FrameSource, LatestFrame, SourceError
from .cv import CvSource, FileSource, RtspSource, WebcamSource
from .mjpeg import MjpegSource

__all__ = [
    "FrameSource",
    "LatestFrame",
    "SourceError",
    "CvSource",
    "WebcamSource",
    "RtspSource",
    "FileSource",
    "MjpegSource",
    "create_source",
    "register_source",
]

_REGISTRY: dict[str, Callable[..., FrameSource]] = {
    "webcam": WebcamSource,
    "mjpeg": MjpegSource,
    "rtsp": RtspSource,
    "file": FileSource,
}

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def register_source(kind: str, factory: Callable[..., FrameSource]) -> None:
    """Register a new source kind. Lets the glasses adapter drop in later."""
    _REGISTRY[kind] = factory


def infer_kind(target: str) -> str:
    """Guess the source kind from a bare string."""
    lowered = target.lower()
    if lowered.startswith("rtsp://"):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "mjpeg"
    if Path(target).suffix.lower() in _VIDEO_SUFFIXES:
        return "file"
    if target.isdigit():
        return "webcam"
    raise SourceError(f"cannot infer source kind from {target!r}; specify `kind` explicitly")


def create_source(spec: str | int | dict[str, Any], *, source_id: str | None = None) -> FrameSource:
    """Build a `FrameSource` from a config value."""
    if isinstance(spec, int):
        spec = {"kind": "webcam", "index": spec}
    elif isinstance(spec, str):
        kind = infer_kind(spec)
        key = {"webcam": "index", "file": "path"}.get(kind, "url")
        value: Any = int(spec) if kind == "webcam" else spec
        spec = {"kind": kind, key: value}

    if not isinstance(spec, dict):
        raise SourceError(f"unsupported source spec: {spec!r}")

    params = dict(spec)
    kind = params.pop("kind", None)
    if kind is None:
        raise SourceError("source spec needs a `kind` (webcam | mjpeg | rtsp | file)")
    if kind not in _REGISTRY:
        raise SourceError(f"unknown source kind {kind!r}; known: {sorted(_REGISTRY)}")

    params.setdefault("source_id", source_id or kind)
    try:
        return _REGISTRY[kind](**params)
    except TypeError as exc:
        raise SourceError(f"bad options for source kind {kind!r}: {exc}") from exc
