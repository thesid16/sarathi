"""Frame to input tensor, and detections back to frame coordinates.

This module is small and boring and is where detection pipelines most often go
quietly wrong. Every step here has a matching inverse, and the two must agree
exactly. If preprocessing centres its letterbox padding and postprocessing
assumes corner padding, every box comes back offset by half the pad - plausible
detections, consistently in the wrong place, and easy to misdiagnose as a bad
model.

So the transform is not recomputed on the way out. `letterbox` returns a
`Transform` describing exactly what it did, and `Transform.to_source` is the
only way boxes get mapped back.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..models.manifest import InputSpec, Layout, PadMode, Resize


@dataclass(frozen=True)
class Transform:
    """What was done to a frame to make it a tensor, and how to undo it."""

    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    source_w: int
    source_h: int

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map (N, 4) xyxy boxes from network coordinates back to frame pixels."""
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale_x
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale_y
        # Objects genuinely run off the edge of the frame; clipping keeps
        # downstream geometry (bearing, ground-plane distance) well defined.
        # Assign the result back explicitly: fancy indexing returns a copy, so
        # np.clip(..., out=out[:, [0, 2]]) would write into a temporary and
        # silently do nothing.
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, self.source_w)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, self.source_h)
        return out


def letterbox(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    pad_value: int = 114,
    pad_mode: PadMode = PadMode.CENTER,
) -> tuple[np.ndarray, Transform]:
    """Resize preserving aspect ratio, pad the remainder."""
    src_h, src_w = image.shape[:2]
    ratio = min(width / src_w, height / src_h)
    new_w, new_h = int(round(src_w * ratio)), int(round(src_h * ratio))

    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_w, pad_h = width - new_w, height - new_h
    if pad_mode is PadMode.CENTER:
        left, top = pad_w // 2, pad_h // 2
    else:
        left, top = 0, 0
    right, bottom = pad_w - left, pad_h - top

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    return canvas, Transform(ratio, ratio, float(left), float(top), src_w, src_h)


def stretch(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, Transform]:
    """Resize ignoring aspect ratio. Correct for depth models, wrong for detectors."""
    src_h, src_w = image.shape[:2]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized, Transform(width / src_w, height / src_h, 0.0, 0.0, src_w, src_h)


def center_crop(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, Transform]:
    """Scale to cover, then crop the centre. Discards content at the edges."""
    src_h, src_w = image.shape[:2]
    ratio = max(width / src_w, height / src_h)
    new_w, new_h = int(round(src_w * ratio)), int(round(src_h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    cropped = resized[top : top + height, left : left + width]
    return cropped, Transform(ratio, ratio, float(-left), float(-top), src_w, src_h)


def prepare_input(image: np.ndarray, spec: InputSpec) -> tuple[np.ndarray, Transform]:
    """Turn a BGR uint8 frame into this model's input tensor.

    Order matters and is fixed: geometry, then colour order, then dtype, then
    scale, then mean/std, then layout. Applying mean/std before the 1/255 scale
    is a classic silent bug - the model runs, the numbers are wrong, and nothing
    complains.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")

    if spec.resize is Resize.LETTERBOX:
        resized, transform = letterbox(
            image, spec.width, spec.height, pad_value=spec.pad_value, pad_mode=spec.pad_mode
        )
    elif spec.resize is Resize.STRETCH:
        resized, transform = stretch(image, spec.width, spec.height)
    else:
        resized, transform = center_crop(image, spec.width, spec.height)

    # Frames arrive BGR from OpenCV; convert only if the model wants RGB.
    if spec.color == "RGB":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if spec.dtype == "float32":
        tensor = resized.astype(np.float32)
        if spec.scale != 1.0:
            tensor *= spec.scale
        if spec.mean != (0.0, 0.0, 0.0):
            tensor -= np.asarray(spec.mean, dtype=np.float32)
        if spec.std != (1.0, 1.0, 1.0):
            tensor /= np.asarray(spec.std, dtype=np.float32)
    elif spec.dtype == "uint8":
        tensor = resized.astype(np.uint8)
    else:  # int8 - quantized models taking a signed input range
        tensor = (resized.astype(np.int16) - 128).astype(np.int8)

    if spec.layout is Layout.NCHW:
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...]), transform
