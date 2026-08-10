"""The whole thing, wired together.

    frames -> scheduler -> detector -> distance -> tracker -> saliency
           -> phrasing -> speech

Each stage is independently tested; this is where they meet. The pipeline owns
no logic of its own beyond sequencing and the label bridge - anything that
looks like a decision belongs in the stage that should be making it.

Two things it does own, because they only exist at the seam:

**Label remapping.** An off-the-shelf detector speaks COCO; the guidance layer
reasons in Sarathi classes, where the hazard and size priors live. Without the
bridge every detection would fall through to LOW hazard and the system would
say nothing at all.

**Camera construction.** The camera model needs the frame dimensions, which are
not known until the first frame arrives. Building it lazily beats making every
caller pass a resolution they may not know.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..guidance.phrasing import Phraser
from ..guidance.saliency import Ranked, SaliencyConfig, SaliencyEngine
from ..guidance.speech import NullSpeaker, VoiceOutput
from ..models import Detector, ModelRegistry
from ..perception.distance import CameraModel, SizePriors, annotate
from ..perception.tracking import Track, Tracker
from ..taxonomy import Taxonomy
from ..types import Detection, Frame, Hazard, Utterance
from ..util.log import get_logger
from .scheduler import Decision, Scheduler, SchedulerConfig

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABEL_MAP = _REPO_ROOT / "training" / "taxonomy" / "coco_to_sarathi.yaml"


class LabelBridge:
    """Maps detector class names onto taxonomy classes, with hazard priors."""

    def __init__(self, taxonomy: Taxonomy, mapping: dict[str, str] | None = None) -> None:
        self.taxonomy = taxonomy
        self.mapping = mapping or {}
        self._hazards = {c.name: c.hazard for c in taxonomy}

    @classmethod
    def load(cls, taxonomy: Taxonomy | None = None, path: str | Path | None = None) -> "LabelBridge":
        taxonomy = taxonomy or Taxonomy.load()
        p = Path(path or DEFAULT_LABEL_MAP)
        mapping: dict[str, str] = {}
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
            mapping = {str(k): str(v) for k, v in (data.get("map") or {}).items()}
        else:
            log.warning("no label map at %s; detector labels will pass through", p)
        return cls(taxonomy, mapping)

    def apply(self, detections: list[Detection]) -> list[Detection]:
        for det in detections:
            det.label = self.mapping.get(det.label, det.label)
            # Unmapped classes stay LOW: context only, never announced
            # unprompted. A detector confidently reporting "broccoli" must not
            # be able to interrupt someone crossing a road.
            det.hazard = self._hazards.get(det.label, Hazard.LOW)
        return detections


@dataclass
class PipelineResult:
    frame_seq: int
    decision: Decision
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    chosen: Ranked | None = None
    utterance: Utterance | None = None
    spoke: bool = False
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass
class PipelineConfig:
    detector: str | None = "yolo11n-coco-320"
    lang: str = "en"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    camera_hfov_deg: float = 66.0
    camera_height_m: float = 1.20
    camera_pitch_deg: float = 0.0
    speak: bool = False


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        registry: ModelRegistry | None = None,
        voice: VoiceOutput | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.registry = registry or ModelRegistry()
        self.taxonomy = Taxonomy.load()
        self.bridge = LabelBridge.load(self.taxonomy)
        self.priors = SizePriors.load()

        self.scheduler = Scheduler(self.config.scheduler)
        self.tracker = Tracker()
        self.saliency = SaliencyEngine(self.config.saliency)
        self.phraser = Phraser(lang=self.config.lang)
        self.voice = voice or VoiceOutput(NullSpeaker())

        self.detector: Detector | None = None
        if self.config.detector:
            model = self.registry.load(self.config.detector)
            assert isinstance(model, Detector)
            self.detector = model

        self._camera: CameraModel | None = None

    def camera_for(self, frame: Frame) -> CameraModel:
        """Build the camera model on first use, from the real frame size."""
        if self._camera is None or (
            self._camera.width != frame.width or self._camera.height != frame.height
        ):
            cfg = self.config
            self._camera = CameraModel(
                width=frame.width,
                height=frame.height,
                hfov_deg=cfg.camera_hfov_deg,
                mount_height_m=cfg.camera_height_m,
                pitch_deg=cfg.camera_pitch_deg,
            )
            log.info(
                "camera model: %dx%d, hfov %.0f deg, mounted %.2f m up, pitch %.0f deg",
                frame.width, frame.height, cfg.camera_hfov_deg,
                cfg.camera_height_m, cfg.camera_pitch_deg,
            )
        return self._camera

    def process(self, frame: Frame, now: float | None = None) -> PipelineResult:
        now = time.monotonic() if now is None else now
        stage: dict[str, float] = {}

        t0 = time.perf_counter()
        decision = self.scheduler.decide(frame, now)
        stage["gate"] = (time.perf_counter() - t0) * 1000.0
        if not decision.run:
            return PipelineResult(frame.seq, decision, stage_ms=stage)

        detections: list[Detection] = []
        if self.detector is not None:
            t0 = time.perf_counter()
            detections = self.detector.detect(frame.image)
            stage["detect"] = (time.perf_counter() - t0) * 1000.0
            self.scheduler.note_inference(stage["detect"])

        t0 = time.perf_counter()
        self.bridge.apply(detections)
        annotate(detections, self.camera_for(frame), self.priors, frame_height=frame.height)
        stage["geometry"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        tracks = self.tracker.update(detections, now)
        stage["track"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        chosen = self.saliency.select(tracks, now)
        stage["saliency"] = (time.perf_counter() - t0) * 1000.0

        utterance = None
        spoke = False
        if chosen is not None:
            utterance = self.phraser.utterance(chosen)
            if self.config.speak:
                spoke = self.voice.say(utterance, now=now)

        return PipelineResult(
            frame_seq=frame.seq,
            decision=decision,
            detections=detections,
            tracks=tracks,
            chosen=chosen,
            utterance=utterance,
            spoke=spoke,
            stage_ms=stage,
        )

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
        self.voice.close()
