"""On-demand text reading.

Tier 3: this never runs unprompted. Reading every sign the camera passes would
be unbearable, and the cost of OCR is only acceptable because the user asks for
it - which bounds how often it happens by how often they press the button.

What comes back is not raw OCR output. Scene text arrives in fragments and in
whatever order the detector found it, and a screen reader user cannot skim past
noise the way a sighted user skims past a cluttered sign. So the results are
filtered by confidence, ordered the way a person would read them, and merged
into lines before anything is spoken.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...util.log import get_logger
from ..base import TextReader, register_adapter
from ..manifest import ModelManifest, Task

log = get_logger(__name__)


class RapidOcrReader(TextReader):
    """RapidOCR (PaddleOCR models converted to ONNX, Apache-2.0).

    The manifest points at no weights file: RapidOCR ships its own bundled
    models and resolves them internally. The manifest still exists so the
    licence is recorded and audited alongside every other model.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        weights: Path | None = None,
        labels: list[str] | None = None,
        *,
        min_confidence: float = 0.55,
        line_tolerance: float = 0.6,
    ) -> None:
        super().__init__(manifest, weights or Path("."), labels)
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self.min_confidence = min_confidence
        #: Two fragments belong to the same line if their vertical centres are
        #: within this fraction of text height. Scene text is rarely level -
        #: a sign photographed at an angle drifts - so this cannot be exact.
        self.line_tolerance = line_tolerance
        self.last_inference_ms = 0.0

    def warmup(self, runs: int = 1) -> None:
        self.read(np.full((64, 192, 3), 255, np.uint8))
        self.last_inference_ms = 0.0

    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Return (text, confidence) per line, in reading order."""
        started = time.perf_counter()
        result, _ = self._engine(image)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if not result:
            return []

        boxes: list[tuple[float, float, float, str, float]] = []
        for entry in result:
            try:
                quad, text, score = entry[0], str(entry[1]), float(entry[2])
            except (IndexError, TypeError, ValueError):
                continue
            if score < self.min_confidence or not text.strip():
                continue
            ys = [p[1] for p in quad]
            xs = [p[0] for p in quad]
            boxes.append((
                (min(ys) + max(ys)) / 2.0,   # vertical centre
                min(xs),                      # left edge
                max(ys) - min(ys),            # text height
                text.strip(),
                score,
            ))

        if not boxes:
            return []
        return self._merge_lines(boxes)

    def _merge_lines(
        self, boxes: list[tuple[float, float, float, str, float]]
    ) -> list[tuple[str, float]]:
        """Group fragments into lines and order them the way a person reads.

        OCR returns fragments in detection order, which is close to arbitrary.
        Spoken aloud that is nonsense: "204 Lab" instead of "Lab 204". Sorting
        top-to-bottom then left-to-right recovers the intended order for the
        signage this actually encounters.
        """
        boxes.sort(key=lambda b: (b[0], b[1]))
        lines: list[list[tuple[float, float, float, str, float]]] = []
        for box in boxes:
            placed = False
            for line in lines:
                reference = line[0]
                tolerance = max(6.0, reference[2] * self.line_tolerance)
                if abs(box[0] - reference[0]) <= tolerance:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        out: list[tuple[str, float]] = []
        for line in lines:
            line.sort(key=lambda b: b[1])
            text = " ".join(b[3] for b in line)
            confidence = min(b[4] for b in line)
            out.append((text, confidence))
        return out

    def close(self) -> None:
        self._engine = None  # type: ignore[assignment]


register_adapter(Task.OCR, "rapidocr", RapidOcrReader)
