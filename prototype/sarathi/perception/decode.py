"""Detector output decoding.

Every detector family disagrees about what comes out of the graph: anchor-free
grids that still need stride multiplication, boxes already in pixels, centre
form versus corner form, a separate objectness score or none at all. That
difference is code, not configuration - so the manifest names a decoder and
this module supplies it.

Adding a family is one function plus one `register_decoder` call. The pipeline
never learns which decoder ran.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from ..models.manifest import ManifestError, OutputSpec


class DecodeResult(Protocol):
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, best score first."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        inter_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        inter_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        inter = inter_w * inter_h
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    """NMS applied per class.

    Class-agnostic NMS would suppress a person standing directly in front of a
    door, because the boxes overlap heavily - and both matter to someone
    navigating by ear.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    keep: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.nonzero(class_ids == cls)[0]
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep.extend(idx[kept].tolist())

    keep_arr = np.asarray(keep, dtype=np.int64)
    if keep_arr.size == 0:
        return keep_arr
    # Rank across classes so the cap keeps the most confident overall.
    keep_arr = keep_arr[scores[keep_arr].argsort()[::-1]]
    return keep_arr[:max_detections]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Centre form to corner form."""
    out = np.empty_like(boxes)
    half_w, half_h = boxes[:, 2] / 2.0, boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def _primary(outputs: list[np.ndarray]) -> np.ndarray:
    if not outputs:
        raise ManifestError("model produced no outputs")
    return np.asarray(outputs[0])


# -- decoders ----------------------------------------------------------------


def decode_yolo11(
    outputs: list[np.ndarray], spec: OutputSpec, input_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics YOLOv8 / YOLO11 raw head.

    Shape [1, 4 + num_classes, num_anchors]. The first four rows are cx, cy, w,
    h already in input-tensor pixels; the rest are per-class scores that have
    already been through sigmoid. There is no separate objectness term - that
    was removed in v8, and treating row 4 as objectness is a common porting bug.
    """
    raw = _primary(outputs)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2:
        raise ManifestError(f"yolo11 decoder expected a 2D head, got shape {raw.shape}")

    # [4+nc, anchors] -> [anchors, 4+nc]. Inferred from which axis is longer,
    # because anchor counts (2100 at 320px, 8400 at 640px) dwarf channel counts
    # (84 for COCO). That holds for any realistic export, but a model with a
    # very large label set and a small input could invert it, so the manifest
    # can say so explicitly with `transpose: true|false`.
    explicit = spec.extra.get("transpose")
    should_transpose = bool(explicit) if explicit is not None else raw.shape[0] < raw.shape[1]
    if should_transpose:
        raw = raw.T
    if raw.shape[1] < 5:
        raise ManifestError(
            f"yolo11 decoder expected >=5 channels, got {raw.shape[1]} from head "
            f"shape {raw.shape}. If the axes are the other way round, set "
            "`transpose` explicitly in the manifest's output section."
        )

    boxes = xywh_to_xyxy(raw[:, :4].astype(np.float32))
    class_scores = raw[:, 4:].astype(np.float32)
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    return boxes, scores, class_ids


def decode_yolox(
    outputs: list[np.ndarray], spec: OutputSpec, input_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOX raw head.

    Shape [1, num_anchors, 5 + num_classes]: grid-relative xy, log-scale wh, an
    objectness score, then class probabilities. The grid and stride
    multiplication happens here rather than in the graph, matching YOLOX's own
    ONNX demo. Set `decode_in_model: true` in the manifest for exports that
    already did it.
    """
    raw = _primary(outputs).astype(np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ManifestError(f"yolox decoder expected [anchors, 5+nc], got shape {raw.shape}")

    if not bool(spec.extra.get("decode_in_model", False)):
        strides = [int(s) for s in spec.extra.get("strides", [8, 16, 32])]
        width, height = input_wh
        grids, expanded = [], []
        for stride in strides:
            gh, gw = height // stride, width // stride
            yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32)
            grids.append(grid)
            expanded.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))
        grid_all = np.concatenate(grids, axis=0)
        stride_all = np.concatenate(expanded, axis=0)

        if grid_all.shape[0] != raw.shape[0]:
            raise ManifestError(
                f"yolox decoder: model produced {raw.shape[0]} anchors but strides "
                f"{strides} at {width}x{height} imply {grid_all.shape[0]}. "
                "The manifest's input size or strides do not match the exported graph."
            )
        raw[:, :2] = (raw[:, :2] + grid_all) * stride_all
        raw[:, 2:4] = np.exp(raw[:, 2:4]) * stride_all

    boxes = xywh_to_xyxy(raw[:, :4])
    objectness = raw[:, 4]
    class_probs = raw[:, 5:]
    class_ids = class_probs.argmax(axis=1)
    scores = objectness * class_probs[np.arange(class_probs.shape[0]), class_ids]
    return boxes, scores, class_ids


def decode_pixel_xyxy(
    outputs: list[np.ndarray], spec: OutputSpec, input_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Models whose graph already emits [N, 6] as x1, y1, x2, y2, score, class.

    Covers exports with NMS baked in, and is the format the export step in
    `training/export/` targets so quantized builds have the simplest possible
    postprocessing on the phone.
    """
    raw = _primary(outputs).astype(np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ManifestError(f"pixel_xyxy decoder expected [N, 6], got shape {raw.shape}")
    return raw[:, :4], raw[:, 4], raw[:, 5].astype(np.int64)


Decoder = Callable[
    [list[np.ndarray], OutputSpec, tuple[int, int]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]

DECODERS: dict[str, Decoder] = {
    "yolo11": decode_yolo11,
    "yolov8": decode_yolo11,  # identical head
    "yolox": decode_yolox,
    "pixel_xyxy": decode_pixel_xyxy,
}


def register_decoder(name: str, decoder: Decoder) -> None:
    DECODERS[name] = decoder


def get_decoder(name: str) -> Decoder:
    try:
        return DECODERS[name]
    except KeyError:
        raise ManifestError(
            f"unknown decoder {name!r}; registered: {sorted(DECODERS)}"
        ) from None
