"""Configuration loading.

One YAML file describes a complete run: where frames come from, which models
are loaded, how talkative the guidance is, and how aggressively the scheduler
saves power. Benchmarks are then just "the same pipeline under a different
config", which is what makes the numbers in the documentation comparable.

Layering, lowest priority first:

1. `DEFAULTS` below - every key the system understands, with a sane value.
2. The YAML file, deep-merged over the defaults.
3. `--set a.b.c=value` overrides from the command line.

Because the defaults are complete, a config file only ever needs to state what
it changes, and an incomplete file can never produce a missing-key crash
halfway through a run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

DEFAULTS: dict[str, Any] = {
    "source": {"kind": "webcam", "index": 0, "width": 1280, "height": 720, "fps": 30},
    "models": {
        # Model ids resolved against models/manifests/. Null disables a stage.
        "detector": None,
        "depth": None,
        "ocr": None,
        "vlm": None,
    },
    "runtime": {
        # Ceiling on detector passes per second. The scheduler runs at or below
        # this; it never exceeds it even if the camera is faster.
        "max_inference_hz": 8.0,
        # Below this, standing still, the detector idles down to save power.
        "idle_inference_hz": 1.0,
        # Skip inference when the frame is visually unchanged. The single
        # biggest battery win when the user is stationary.
        "motion_gate": {"enabled": True, "threshold": 0.012, "downscale": 64},
        # Frames older than this when they reach inference are discarded -
        # acting on them would describe a scene the user has walked past.
        "max_frame_age_ms": 250,
        "depth_hz": 2.0,
        "thermal_governor": {"enabled": True},
    },
    "guidance": {
        "lang": "en",  # "en" | "hi"
        # Hard ceiling on how often the system speaks. Exceeding this is the
        # fastest way to make an aid unusable - users switch it off.
        "min_utterance_gap_s": 1.5,
        # Do not repeat the same subject within this window.
        "repeat_cooldown_s": 8.0,
        "max_distance_m": 6.0,
        "announce_hazards_immediately": True,
        "units": "metric",
        "earcons": True,
    },
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Turn a CLI override string into a real value."""
    lowered = text.strip().lower()
    if lowered in {"none", "null", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


class Config:
    """Dotted-path read-only view over the merged configuration."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {path}")
        return value

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, indent=2, default=str)})"


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load defaults, then a YAML file, then `a.b.c=value` overrides."""
    data = copy.deepcopy(DEFAULTS)

    if path is not None:
        p = Path(path).expanduser()
        # Bare names resolve against configs/, so `--config indoor` works.
        if not p.exists() and not p.is_absolute():
            for candidate in (CONFIG_DIR / p.name, CONFIG_DIR / f"{p.name}.yaml"):
                if candidate.exists():
                    p = candidate
                    break
        if not p.exists():
            raise FileNotFoundError(f"config not found: {path}")
        loaded = yaml.safe_load(p.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {p} must be a mapping at the top level")
        data = _deep_merge(data, loaded)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {key}: {part} is not a section")
        node[parts[-1]] = _coerce(raw)

    return Config(data)
