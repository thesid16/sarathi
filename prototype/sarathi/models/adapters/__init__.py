"""Inference-engine adapters.

Importing this package registers every adapter, which is how the registry finds
them. Each adapter is imported for its `register_adapter` side effect.
"""

from . import gemma_vlm, onnx_depth, onnx_detector, rapid_ocr  # noqa: F401

__all__ = ["gemma_vlm", "onnx_depth", "onnx_detector", "rapid_ocr"]
