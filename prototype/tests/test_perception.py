"""Tests for preprocessing, decoding and the ONNX detection adapter.

The end-to-end test builds a real ONNX graph that emits a known detection, so
the whole path - letterbox, colour order, inference, decode, NMS, unletterbox -
is exercised without downloading any weights.

The roundtrip tests matter most. A preprocessing/postprocessing mismatch does
not crash; it produces plausible boxes in consistently wrong places, which is
the hardest kind of bug to spot and the easiest to blame on the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from sarathi.models.manifest import InputSpec, Layout, ModelManifest, OutputSpec, PadMode, Resize
from sarathi.models.registry import ModelRegistry
from sarathi.perception import letterbox, nms, prepare_input, stretch
from sarathi.perception.decode import (
    decode_yolo11,
    decode_yolox,
    multiclass_nms,
    xywh_to_xyxy,
)
from sarathi.models.manifest import ManifestError


def make_image(w=800, h=450, value=200):
    img = np.full((h, w, 3), 30, np.uint8)
    img[100:200, 200:400] = value
    return img


# -- letterbox ---------------------------------------------------------------


def test_letterbox_preserves_aspect_ratio_and_fills_target():
    img = make_image(800, 450)
    out, tf = letterbox(img, 320, 320)
    assert out.shape == (320, 320, 3)
    assert tf.scale_x == pytest.approx(tf.scale_y)
    assert tf.scale_x == pytest.approx(320 / 800)


def test_center_padding_is_split_between_both_sides():
    img = make_image(800, 400)
    _, tf = letterbox(img, 320, 320, pad_mode=PadMode.CENTER)
    assert tf.pad_x == 0
    assert tf.pad_y == pytest.approx((320 - 160) / 2)


def test_corner_padding_puts_the_image_top_left():
    img = make_image(800, 400)
    _, tf = letterbox(img, 320, 320, pad_mode=PadMode.CORNER)
    assert tf.pad_x == 0 and tf.pad_y == 0


@pytest.mark.parametrize("pad_mode", [PadMode.CENTER, PadMode.CORNER])
@pytest.mark.parametrize(("w", "h"), [(800, 450), (450, 800), (640, 640), (1000, 137)])
def test_box_roundtrip_is_exact(pad_mode, w, h):
    """A box mapped into network space and back must land where it started."""
    img = make_image(w, h)
    _, tf = letterbox(img, 320, 320, pad_mode=pad_mode)

    source = np.array([[10.0, 20.0, w * 0.5, h * 0.5]], dtype=np.float32)
    # Forward transform, mirroring what a model would see.
    network = source.copy()
    network[:, [0, 2]] = network[:, [0, 2]] * tf.scale_x + tf.pad_x
    network[:, [1, 3]] = network[:, [1, 3]] * tf.scale_y + tf.pad_y

    assert np.allclose(tf.to_source(network), source, atol=1e-3)


def test_to_source_clips_to_the_frame():
    img = make_image(640, 480)
    _, tf = letterbox(img, 320, 320)
    out = tf.to_source(np.array([[-500.0, -500.0, 5000.0, 5000.0]], np.float32))
    assert out[0, 0] == 0 and out[0, 1] == 0
    assert out[0, 2] == 640 and out[0, 3] == 480


def test_stretch_uses_independent_axis_scales():
    img = make_image(800, 400)
    out, tf = stretch(img, 320, 320)
    assert out.shape == (320, 320, 3)
    assert tf.scale_x != tf.scale_y
    source = np.array([[0.0, 0.0, 800.0, 400.0]], np.float32)
    network = np.array([[0.0, 0.0, 320.0, 320.0]], np.float32)
    assert np.allclose(tf.to_source(network), source, atol=1e-3)


def test_empty_box_array_survives_the_transform():
    _, tf = letterbox(make_image(), 320, 320)
    assert tf.to_source(np.empty((0, 4), np.float32)).shape == (0, 4)


# -- prepare_input -----------------------------------------------------------


def test_nchw_float_input_shape_and_range():
    spec = InputSpec(width=320, height=320, dtype="float32", scale=1 / 255)
    tensor, _ = prepare_input(make_image(), spec)
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_nhwc_uint8_input_is_left_unscaled():
    spec = InputSpec(width=320, height=320, layout=Layout.NHWC, dtype="uint8")
    tensor, _ = prepare_input(make_image(), spec)
    assert tensor.shape == (1, 320, 320, 3)
    assert tensor.dtype == np.uint8


def test_rgb_conversion_swaps_channels_and_bgr_does_not():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :, 0] = 255  # blue in BGR
    rgb, _ = prepare_input(img, InputSpec(width=4, height=4, layout=Layout.NHWC, dtype="uint8"))
    bgr, _ = prepare_input(
        img, InputSpec(width=4, height=4, layout=Layout.NHWC, dtype="uint8", color="BGR")
    )
    assert rgb[0, 0, 0, 2] == 255  # moved to the last channel
    assert bgr[0, 0, 0, 0] == 255  # untouched


def test_scale_is_applied_before_mean_and_std():
    """Order matters: mean/std are ImageNet values in 0-1 space, not 0-255."""
    img = np.full((8, 8, 3), 255, np.uint8)
    spec = InputSpec(
        width=4, height=4, dtype="float32", scale=1 / 255, mean=(0.5,) * 3, std=(0.5,) * 3
    )
    tensor, _ = prepare_input(img, spec)
    assert tensor.max() == pytest.approx(1.0)  # (1.0 - 0.5) / 0.5


def test_non_three_channel_input_is_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        prepare_input(np.zeros((10, 10), np.uint8), InputSpec(width=4, height=4))


# -- NMS ---------------------------------------------------------------------


def test_nms_suppresses_overlapping_and_keeps_separate():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_nms_returns_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    keep = nms(boxes, np.array([0.2, 0.9], np.float32), 0.5)
    assert keep[0] == 1


def test_nms_on_empty_input():
    assert nms(np.empty((0, 4), np.float32), np.empty((0,), np.float32), 0.5).size == 0


def test_multiclass_nms_keeps_overlapping_boxes_of_different_classes():
    """A person in a doorway: heavily overlapping, both worth announcing."""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.85], np.float32)
    keep = multiclass_nms(boxes, scores, np.array([0, 1]), 0.5, 50)
    assert len(keep) == 2


def test_multiclass_nms_caps_at_max_detections():
    boxes = np.array([[i * 50, 0, i * 50 + 10, 10] for i in range(20)], np.float32)
    scores = np.linspace(0.1, 0.9, 20).astype(np.float32)
    keep = multiclass_nms(boxes, scores, np.zeros(20, int), 0.5, 5)
    assert len(keep) == 5
    assert scores[keep[0]] == pytest.approx(0.9)  # keeps the best, not the first


def test_xywh_to_xyxy():
    out = xywh_to_xyxy(np.array([[50.0, 60.0, 20.0, 10.0]], np.float32))
    assert np.allclose(out, [[40, 55, 60, 65]])


# -- decoders ----------------------------------------------------------------


def _yolo11_head(num_classes=80, num_anchors=2100):
    raw = np.zeros((1, 4 + num_classes, num_anchors), np.float32)
    raw[0, :4, 0] = [50, 60, 20, 10]  # cx, cy, w, h
    raw[0, 4 + 3, 0] = 0.9  # class 3
    return [raw]


def test_decode_yolo11_reads_centre_form_pixels_and_class_scores():
    boxes, scores, ids = decode_yolo11(_yolo11_head(), OutputSpec(decoder="yolo11"), (320, 320))
    assert boxes.shape[0] == 2100
    assert np.allclose(boxes[0], [40, 55, 60, 65])
    assert scores[0] == pytest.approx(0.9)
    assert ids[0] == 3


def test_decode_yolo11_handles_a_pre_transposed_head():
    raw = _yolo11_head()[0][0].T[None, ...]  # [1, anchors, 4+nc]
    boxes, _, _ = decode_yolo11([raw], OutputSpec(decoder="yolo11"), (320, 320))
    assert np.allclose(boxes[0], [40, 55, 60, 65])


def test_decode_yolo11_rejects_a_head_that_is_too_narrow():
    with pytest.raises(ManifestError, match=">=5 channels"):
        decode_yolo11([np.zeros((1, 3, 10), np.float32)], OutputSpec(decoder="yolo11"), (320, 320))


def test_decode_yolox_applies_grid_and_stride():
    """The first anchor of stride 8 sits at grid (0,0), so xy = raw * 8."""
    size, strides = 64, [8, 16, 32]
    anchors = sum((size // s) ** 2 for s in strides)
    raw = np.zeros((1, anchors, 6), np.float32)
    raw[0, 0, :4] = [0.5, 0.5, 0.0, 0.0]  # wh log-scale 0 -> exp(0)*8 = 8
    raw[0, 0, 4] = 1.0
    raw[0, 0, 5] = 0.8
    spec = OutputSpec(decoder="yolox", extra={"strides": strides})

    boxes, scores, ids = decode_yolox([raw], spec, (size, size))
    assert boxes[0][0] == pytest.approx(4.0 - 4.0)  # cx=4, w=8 -> x1 = 0
    assert boxes[0][2] == pytest.approx(8.0)
    assert scores[0] == pytest.approx(0.8)  # objectness * class prob
    assert ids[0] == 0


def test_decode_yolox_detects_an_input_size_mismatch():
    """Wrong imgsz in the manifest changes the anchor count - caught, not silent."""
    raw = np.zeros((1, 100, 6), np.float32)
    with pytest.raises(ManifestError, match="do not match the exported graph"):
        decode_yolox([raw], OutputSpec(decoder="yolox"), (320, 320))


def test_decode_yolox_can_skip_decoding_for_exports_that_did_it():
    raw = np.zeros((1, 3, 6), np.float32)
    raw[0, 0] = [50, 60, 20, 10, 1.0, 0.7]
    spec = OutputSpec(decoder="yolox", extra={"decode_in_model": True})
    boxes, scores, _ = decode_yolox([raw], spec, (320, 320))
    assert np.allclose(boxes[0], [40, 55, 60, 65])
    assert scores[0] == pytest.approx(0.7)


# -- end to end --------------------------------------------------------------


@pytest.fixture
def synthetic_model(tmp_path):
    """A real ONNX graph emitting one known detection at a known location."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    num_classes, anchors = 2, 2100
    head = np.zeros((1, 4 + num_classes, anchors), np.float32)
    # Centred box in a 320x320 network input: 100x100 at the middle.
    head[0, :4, 0] = [160, 160, 100, 100]
    head[0, 4 + 1, 0] = 0.95  # class 1

    const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=helper.make_tensor(
            "v", TensorProto.FLOAT, head.shape, head.flatten().tolist()
        ),
    )
    graph = helper.make_graph(
        [const],
        "synthetic-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(head.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = tmp_path / "synthetic.onnx"
    onnx.save(model, str(path))
    return path


def test_onnx_detector_end_to_end(tmp_path, synthetic_model):
    import yaml

    manifests, weights = tmp_path / "m", synthetic_model.parent
    manifests.mkdir()
    (manifests / "synthetic.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "synthetic",
                "task": "detection",
                "license": "MIT",
                "distribution": "bundled",
                "runtime": {"prototype": "onnxruntime"},
                "files": {"onnx": {"path": synthetic_model.name}},
                "input": {"width": 320, "height": 320, "scale": 1 / 255},
                "output": {
                    "decoder": "yolo11",
                    "labels": ["door", "chair"],
                    "conf_threshold": 0.3,
                },
            }
        )
    )

    registry = ModelRegistry(manifests, weights)
    model = registry.load("synthetic")

    detections = model.detect(np.zeros((320, 320, 3), np.uint8))
    assert len(detections) == 1
    det = detections[0]
    assert det.label == "chair"  # label resolved from the manifest, not the index
    assert det.class_id == 1
    assert det.score == pytest.approx(0.95)
    assert det.box == pytest.approx((110.0, 110.0, 210.0, 210.0))
    assert model.last_inference_ms > 0


def test_detections_are_mapped_back_through_the_letterbox(tmp_path, synthetic_model):
    """A non-square frame must have its padding removed from the boxes."""
    import yaml

    manifests, weights = tmp_path / "m2", synthetic_model.parent
    manifests.mkdir()
    (manifests / "s.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "synthetic2",
                "task": "detection",
                "license": "MIT",
                "distribution": "bundled",
                "runtime": {"prototype": "onnxruntime"},
                "files": {"onnx": {"path": synthetic_model.name}},
                "input": {"width": 320, "height": 320, "scale": 1 / 255},
                "output": {"decoder": "yolo11", "labels": ["a", "b"], "conf_threshold": 0.3},
            }
        )
    )
    model = ModelRegistry(manifests, weights).load("synthetic2")

    # 640x360 letterboxed into 320x320: scale 0.5, 70px of padding top and bottom.
    det = model.detect(np.zeros((360, 640, 3), np.uint8))[0]
    assert det.box[0] == pytest.approx(220.0)  # (110 - 0) / 0.5
    assert det.box[1] == pytest.approx(80.0)  # (110 - 70) / 0.5


def test_unknown_class_id_falls_back_to_the_index(tmp_path, synthetic_model):
    """A short label file must mislabel one class, not crash mid-walk."""
    import yaml

    manifests, weights = tmp_path / "m3", synthetic_model.parent
    manifests.mkdir()
    (manifests / "s.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "synthetic3",
                "task": "detection",
                "license": "MIT",
                "distribution": "bundled",
                "runtime": {"prototype": "onnxruntime"},
                "files": {"onnx": {"path": synthetic_model.name}},
                "input": {"width": 320, "height": 320, "scale": 1 / 255},
                "output": {"decoder": "yolo11", "labels": ["only-one"], "conf_threshold": 0.3},
            }
        )
    )
    model = ModelRegistry(manifests, weights).load("synthetic3")
    assert model.detect(np.zeros((320, 320, 3), np.uint8))[0].label == "1"
