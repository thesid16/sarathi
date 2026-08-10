"""Model interfaces and the adapter registry.

Perception code depends on these interfaces, never on a concrete runtime. That
is what allows the same pipeline to run ONNX Runtime on a laptop and LiteRT on
a phone, and what allows a new model family to be added by writing one adapter
rather than by editing the pipeline.

An adapter is registered against a (task, engine) pair. The manifest names the
engine; the registry finds the adapter; the pipeline never learns which one it
got.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

import numpy as np

from ..types import Detection
from .manifest import ManifestError, ModelManifest, Task


class Model(abc.ABC):
    """Base class for anything loaded from a manifest."""

    def __init__(self, manifest: ModelManifest, weights: Path) -> None:
        self.manifest = manifest
        self.weights = weights

    @property
    def id(self) -> str:
        return self.manifest.id

    def warmup(self, runs: int = 2) -> None:
        """Run the model on synthetic input so first real call is not an outlier.

        Lazy graph setup, delegate compilation and memory allocation all happen
        on the first inference. Without a warmup the first frame's latency lands
        in the benchmark and skews the numbers this project publishes.
        """

    def close(self) -> None:
        """Release the underlying session. Safe to call twice."""

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Detector(Model):
    """Object detection."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR uint8 image. Boxes come back in image pixels."""


class DepthEstimator(Model):
    """Monocular depth."""

    @abc.abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Return a relative depth map. Larger values are nearer.

        Relative, not metric: these models do not produce metres, and treating
        their output as if they did is a common and dangerous mistake here.
        Metric distance comes from the geometric estimator; depth is used for
        *ordering* and for finding surfaces that have no bounding box.
        """


class TextReader(Model):
    """OCR."""

    @abc.abstractmethod
    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Return (text, confidence) in natural reading order."""


class SceneDescriber(Model):
    """Vision-language scene description. The slot a future Gemma drops into."""

    @abc.abstractmethod
    def describe(self, image: np.ndarray, prompt: str | None = None) -> str:
        """One sentence describing what is in front of the user."""


# -- adapter registry --------------------------------------------------------

AdapterFactory = Callable[[ModelManifest, Path], Model]

_ADAPTERS: dict[tuple[Task, str], AdapterFactory] = {}


def register_adapter(task: Task, engine: str, factory: AdapterFactory) -> None:
    """Register an adapter for a (task, engine) pair.

    Adding support for a new runtime or a new model family is this one call
    plus the adapter itself. Nothing in the pipeline changes.
    """
    _ADAPTERS[(task, engine)] = factory


def get_adapter(task: Task, engine: str) -> AdapterFactory:
    try:
        return _ADAPTERS[(task, engine)]
    except KeyError:
        available = sorted(f"{t.value}/{e}" for t, e in _ADAPTERS)
        raise ManifestError(
            f"no adapter for task {task.value!r} on engine {engine!r}; "
            f"registered: {available or 'none'}"
        ) from None


def registered_adapters() -> list[tuple[Task, str]]:
    return sorted(_ADAPTERS, key=lambda k: (k[0].value, k[1]))
