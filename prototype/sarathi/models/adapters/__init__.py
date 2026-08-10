"""Inference-engine adapters.

Importing this package registers every adapter, which is how the registry finds
them. Each adapter is imported for its `register_adapter` side effect.
"""

from . import onnx_depth, onnx_detector  # noqa: F401

__all__ = ["onnx_depth", "onnx_detector"]
