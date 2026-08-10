"""The model registry.

Scans a directory of manifests and loads models by id. This is the whole
"swappable models" mechanism from the outside:

    registry = ModelRegistry()
    detector = registry.load("yolox-nano-320")

Changing which detector runs means changing a string in a config file. Adding a
model that did not exist when the app was written means dropping a manifest and
a weights file into place.

The registry is also where licence policy is enforced, because a rule that
lives only in documentation is a rule that eventually gets broken.
"""

from __future__ import annotations

from pathlib import Path

from ..util.log import get_logger
from .base import Model, get_adapter
from .manifest import Distribution, ManifestError, ModelManifest, Task

log = get_logger(__name__)

#: Repository root, found relative to this file: prototype/sarathi/models -> ../../..
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = _REPO_ROOT / "models" / "manifests"
DEFAULT_WEIGHTS_DIR = _REPO_ROOT / "models" / "weights"
DEFAULT_LABELS_DIR = _REPO_ROOT / "models" / "labels"


class ModelRegistry:
    """Loads models by id from a directory of manifests."""

    def __init__(
        self,
        manifest_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        *,
        labels_dir: str | Path | None = None,
        runtime: str = "prototype",
    ) -> None:
        self.manifest_dir = Path(manifest_dir or DEFAULT_MANIFEST_DIR)
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.labels_dir = Path(labels_dir or DEFAULT_LABELS_DIR)
        self.runtime = runtime
        self._manifests: dict[str, ModelManifest] | None = None

    # -- discovery ---------------------------------------------------------

    def _scan(self) -> dict[str, ModelManifest]:
        found: dict[str, ModelManifest] = {}
        if not self.manifest_dir.exists():
            log.warning("manifest directory does not exist: %s", self.manifest_dir)
            return found

        for path in sorted(self.manifest_dir.rglob("*.y*ml")):
            try:
                manifest = ModelManifest.from_file(path)
            except ManifestError as exc:
                # One bad manifest must not hide every good one - a typo in an
                # experimental model should not stop the app from starting.
                log.error("skipping invalid manifest %s: %s", path.name, exc)
                continue
            if manifest.id in found:
                raise ManifestError(
                    f"duplicate model id {manifest.id!r} in {path} and "
                    f"{found[manifest.id].source_path}"
                )
            found[manifest.id] = manifest
        return found

    @property
    def manifests(self) -> dict[str, ModelManifest]:
        if self._manifests is None:
            self._manifests = self._scan()
        return self._manifests

    def reload(self) -> None:
        self._manifests = None

    def list(self, task: Task | str | None = None) -> list[ModelManifest]:
        items = list(self.manifests.values())
        if task is not None:
            wanted = Task(task) if isinstance(task, str) else task
            items = [m for m in items if m.task is wanted]
        return sorted(items, key=lambda m: m.id)

    def get(self, model_id: str) -> ModelManifest:
        try:
            return self.manifests[model_id]
        except KeyError:
            known = sorted(self.manifests)
            detail = ", ".join(known) if known else f"none found in {self.manifest_dir}"
            raise ManifestError(f"unknown model {model_id!r}. Known models: {detail}") from None

    # -- loading -----------------------------------------------------------

    def load(self, model_id: str, *, verify: bool = True) -> Model:
        """Load a model by id. Raises `ManifestError` with an actionable message."""
        manifest = self.get(model_id)

        if not manifest.loadable:
            raise ManifestError(
                f"model {model_id!r} is marked distribution=excluded and will not be "
                f"loaded.\n  licence: {manifest.license}\n"
                "  Sarathi is AGPL-3.0 and public; a model whose licence restricts "
                "downstream users would restrict everyone who builds on this."
            )

        engine = manifest.runtime.get(self.runtime, self.runtime)
        factory = get_adapter(manifest.task, engine)

        if manifest.vendored_weights:
            # Nothing to locate or checksum - the library owns its weights.
            weights = self.weights_dir
        else:
            file_spec = manifest.file_for(self.runtime)
            if verify:
                file_spec.verify(self.weights_dir)
            weights = file_spec.resolve(self.weights_dir)

        # Labels are resolved here rather than in the adapter, so adapters
        # never touch the filesystem and stay trivially testable.
        labels = None
        if manifest.output is not None and manifest.output.labels is not None:
            labels = self.load_labels(manifest)

        log.info("loading %s (%s, %s)", manifest.id, manifest.task.value, engine)
        model = factory(manifest, weights, labels)
        model.warmup()
        return model

    def load_labels(self, manifest: ModelManifest) -> list[str]:
        """Resolve a manifest's label set to a list of class names.

        `labels` may be an inline list, or a name resolved against
        `models/labels/<name>.txt` - one class per line, blank lines and `#`
        comments ignored.
        """
        if manifest.output is None or manifest.output.labels is None:
            raise ManifestError(f"model {manifest.id!r}: no labels declared")

        labels = manifest.output.labels
        if isinstance(labels, list):
            return [str(x) for x in labels]

        name = str(labels)
        candidates = [self.labels_dir / name, self.labels_dir / f"{name}.txt"]
        if manifest.source_path is not None:
            candidates.insert(0, manifest.source_path.parent / name)
        for candidate in candidates:
            if candidate.exists():
                lines = candidate.read_text().splitlines()
                return [
                    line.strip()
                    for line in lines
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        raise ManifestError(
            f"model {manifest.id!r}: label set {name!r} not found; looked in "
            + ", ".join(str(c) for c in candidates)
        )

    # -- licence reporting -------------------------------------------------

    def licence_table(self) -> list[tuple[str, str, str, str]]:
        """(model id, task, licence, distribution) for every known model."""
        return [
            (m.id, m.task.value, m.license, m.distribution.value)
            for m in sorted(self.manifests.values(), key=lambda m: m.id)
        ]

    def attribution_text(self) -> str:
        """Attribution notice built from manifests, so it cannot drift."""
        lines = [
            "# Model attribution",
            "",
            "Generated from model manifests. Do not edit by hand.",
            "",
        ]
        for manifest in sorted(self.manifests.values(), key=lambda m: m.id):
            if manifest.distribution is Distribution.EXCLUDED:
                continue
            lines.append(f"## {manifest.id}")
            lines.append("")
            lines.append(f"- Licence: {manifest.license}")
            lines.append(f"- Distribution: {manifest.distribution.value}")
            if manifest.source_url:
                lines.append(f"- Source: {manifest.source_url}")
            if manifest.attribution:
                lines.append(f"- Attribution: {manifest.attribution}")
            lines.append("")
        return "\n".join(lines)
