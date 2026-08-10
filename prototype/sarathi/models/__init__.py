"""Manifest-driven model loading.

Models are described by YAML manifests rather than wired into code, so a model
can be swapped - or one that did not exist when the app shipped can be added -
without touching Python or Kotlin. The same manifests are read by the Android
app, which is what keeps a model benchmarked here meaningful once it is running
on a phone.
"""

from .base import (
    Detector,
    adapter_override,
    DepthEstimator,
    Model,
    SceneDescriber,
    TextReader,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from .manifest import (
    Distribution,
    FileSpec,
    InputSpec,
    Layout,
    ManifestError,
    ModelManifest,
    OutputSpec,
    Resize,
    Task,
)
from .registry import ModelRegistry

# Adapters are NOT imported here. They depend on `sarathi.perception`, which
# depends back on `sarathi.models.manifest` - importing them at package import
# time creates a cycle whose symptom depends on which module the caller
# imported first. `get_adapter` imports them on demand instead.

__all__ = [
    "Detector",
    "adapter_override",
    "DepthEstimator",
    "Distribution",
    "FileSpec",
    "InputSpec",
    "Layout",
    "ManifestError",
    "Model",
    "ModelManifest",
    "ModelRegistry",
    "OutputSpec",
    "Resize",
    "SceneDescriber",
    "Task",
    "TextReader",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
