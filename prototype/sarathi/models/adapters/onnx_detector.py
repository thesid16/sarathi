"""ONNX Runtime detection adapter.

The prototype's inference engine. Android uses LiteRT instead, but both read
the same manifest, so preprocessing, decoding and labels are identical on both
sides by construction rather than by discipline.

Nothing here is model-specific. Input shape, colour order, normalisation,
decoder and thresholds all come from the manifest, which is why a new detector
family needs a decoder function and a YAML file rather than a new adapter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...perception.decode import get_decoder, multiclass_nms
from ...perception.preprocess import prepare_input
from ...types import Detection
from ...util.log import get_logger
from ..base import Detector, register_adapter
from ..manifest import ManifestError, ModelManifest, Task

log = get_logger(__name__)


class OnnxDetector(Detector):
    """Runs an ONNX detection graph and returns boxes in frame coordinates."""

    def __init__(
        self,
        manifest: ModelManifest,
        weights: Path,
        labels: list[str] | None = None,
        *,
        providers: list[str] | None = None,
    ) -> None:
        super().__init__(manifest, weights, labels)
        if manifest.input is None or manifest.output is None:
            raise ManifestError(f"{manifest.id}: detection needs input and output sections")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Single-threaded is closer to the phone's budget and makes desktop
        # latency numbers less flattering, which is the point of measuring here.
        options.intra_op_num_threads = 0

        if providers is None:
            available = set(ort.get_available_providers())
            preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider"]
            providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(weights), options, providers=providers)
        self.provider = self.session.get_providers()[0]

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ManifestError(
                f"{manifest.id}: expected exactly one graph input, got {len(inputs)}"
            )
        self._input_name = inputs[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._decoder = get_decoder(manifest.output.decoder)

        self._check_input_shape(inputs[0].shape)
        #: Wall-clock of the last inference, for the benchmark harness.
        self.last_inference_ms = 0.0

    def _check_input_shape(self, graph_shape: list) -> None:
        """Warn when the manifest and the graph disagree about input size.

        Not fatal - dynamic-axis exports legitimately report strings or None -
        but a mismatch means every published latency figure is for a different
        resolution than the manifest claims, so it must not pass silently.
        """
        spec = self.manifest.input_for("prototype")
        assert spec is not None
        concrete = [d for d in graph_shape if isinstance(d, int) and d > 0]
        if spec.width in concrete and spec.height in concrete:
            return
        if len(concrete) < 2:
            log.debug("%s: graph has dynamic input axes %s", self.id, graph_shape)
            return
        log.warning(
            "%s: manifest says %dx%d but the graph input is %s. "
            "Benchmarks and detections will not mean what the manifest claims.",
            self.id,
            spec.width,
            spec.height,
            graph_shape,
        )

    def warmup(self, runs: int = 2) -> None:
        spec = self.manifest.input_for("prototype")
        assert spec is not None
        blank = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(blank)
        self.last_inference_ms = 0.0

    def detect(self, image: np.ndarray) -> list[Detection]:
        spec = self.manifest.input_for("prototype")
        out_spec = self.manifest.output
        assert spec is not None and out_spec is not None

        tensor, transform = prepare_input(image, spec)

        started = time.perf_counter()
        raw = self.session.run(self._output_names, {self._input_name: tensor})
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        boxes, scores, class_ids = self._decoder(raw, out_spec, (spec.width, spec.height))

        # Threshold before NMS: NMS is the expensive step and most anchors are
        # background. On a 320px head that is thousands of boxes discarded for
        # the cost of one comparison each.
        keep = scores >= out_spec.conf_threshold
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if boxes.size == 0:
            return []

        kept = multiclass_nms(
            boxes, scores, class_ids, out_spec.nms_iou, out_spec.max_detections
        )
        boxes = transform.to_source(boxes[kept])
        scores, class_ids = scores[kept], class_ids[kept]

        return [
            Detection(
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(s),
                class_id=int(c),
                label=self.label_for(int(c)),
            )
            for b, s, c in zip(boxes, scores, class_ids)
        ]

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


register_adapter(Task.DETECTION, "onnxruntime", OnnxDetector)
