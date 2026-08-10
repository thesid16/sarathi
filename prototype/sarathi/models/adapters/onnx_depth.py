"""ONNX Runtime monocular depth adapter.

Returns an inverse relative depth map: larger values are nearer, and the scale
is arbitrary and different on every frame. It is not metres, it does not become
metres by scaling, and treating it as metres is the single most likely way to
misuse this stage - it would produce confidently wrong spoken distances, which
is worse for this product than producing none.

What it is actually for is the things a bounding box cannot express: where the
floor stops, where it steps down, and how far ahead is clear. Those are
questions about *shape*, and relative depth answers them fine.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...perception.preprocess import Transform, prepare_input
from ...util.log import get_logger
from ..base import DepthEstimator, register_adapter
from ..manifest import ManifestError, ModelManifest, Task

log = get_logger(__name__)


class OnnxDepthEstimator(DepthEstimator):
    def __init__(
        self,
        manifest: ModelManifest,
        weights: Path,
        labels: list[str] | None = None,
        *,
        providers: list[str] | None = None,
    ) -> None:
        super().__init__(manifest, weights, labels)
        if manifest.input is None:
            raise ManifestError(f"{manifest.id}: depth models need an `input` section")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            available = set(ort.get_available_providers())
            preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider"]
            providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(weights), options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

        self.last_inference_ms = 0.0
        #: Maps depth-map coordinates back to frame pixels. Depth models use a
        #: stretch resize rather than a letterbox, so the two axes scale
        #: differently and the mapping is not a single ratio.
        self.last_transform: Transform | None = None

    def warmup(self, runs: int = 2) -> None:
        spec = self.manifest.input_for("prototype")
        assert spec is not None
        blank = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.estimate(blank)
        self.last_inference_ms = 0.0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        spec = self.manifest.input_for("prototype")
        assert spec is not None

        tensor, transform = prepare_input(image, spec)
        self.last_transform = transform

        started = time.perf_counter()
        raw = self.session.run(self._output_names, {self._input_name: tensor})
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        depth = np.asarray(raw[0], dtype=np.float32)
        while depth.ndim > 2:
            depth = depth[0]
        return depth

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


register_adapter(Task.DEPTH, "onnxruntime", OnnxDepthEstimator)
