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
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..types import Detection
from .manifest import ManifestError, ModelManifest, Task


class Model(abc.ABC):
    """Base class for anything loaded from a manifest."""

    def __init__(
        self,
        manifest: ModelManifest,
        weights: Path,
        labels: list[str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.weights = weights
        #: Resolved class names, or None for tasks that have no label set.
        #: The registry resolves these so adapters never touch the filesystem.
        self.labels = labels

    @property
    def id(self) -> str:
        return self.manifest.id

    def label_for(self, class_id: int) -> str:
        """Class name for an index, falling back to the index itself.

        A model whose label file is one line short should mislabel one class,
        not crash the pipeline mid-walk.
        """
        if self.labels and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

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

AdapterFactory = Callable[..., Model]

_ADAPTERS: dict[tuple[Task, str], AdapterFactory] = {}


def register_adapter(task: Task, engine: str, factory: AdapterFactory) -> None:
    """Register an adapter for a (task, engine) pair.

    Adding support for a new runtime or a new model family is this one call
    plus the adapter itself. Nothing in the pipeline changes.
    """
    _ADAPTERS[(task, engine)] = factory


def _ensure_builtin_adapters() -> None:
    """Import the built-in adapters on first use.

    Deferred rather than done at package import: adapters depend on
    `sarathi.perception`, which depends back on `sarathi.models.manifest`.
    Importing them eagerly makes that a cycle, and a cycle whose failure mode
    depends on which module the caller imported first is far worse than a
    slightly lazy import.
    """
    if _ADAPTERS:
        return
    from . import adapters  # noqa: F401  - imported for registration side effects


def get_adapter(task: Task, engine: str) -> AdapterFactory:
    _ensure_builtin_adapters()
    try:
        return _ADAPTERS[(task, engine)]
    except KeyError:
        available = sorted(f"{t.value}/{e}" for t, e in _ADAPTERS)
        raise ManifestError(
            f"no adapter for task {task.value!r} on engine {engine!r}; "
            f"registered: {available or 'none'}"
        ) from None


def registered_adapters() -> list[tuple[Task, str]]:
    _ensure_builtin_adapters()
    return sorted(_ADAPTERS, key=lambda k: (k[0].value, k[1]))


@contextmanager
def adapter_override(task: Task, engine: str, factory: AdapterFactory) -> Iterator[None]:
    """Temporarily replace an adapter, restoring the previous one on exit.

    The registry is global process state. A test that swaps in a stub with a
    bare `register_adapter` leaves it swapped for everything that runs
    afterwards - which shows up as unrelated tests seeing no detections, in a
    different module, depending on collection order.
    """
    _ensure_builtin_adapters()
    key = (task, engine)
    had_previous = key in _ADAPTERS
    previous = _ADAPTERS.get(key)
    _ADAPTERS[key] = factory
    try:
        yield
    finally:
        if had_previous:
            _ADAPTERS[key] = previous  # type: ignore[assignment]
        else:
            _ADAPTERS.pop(key, None)
