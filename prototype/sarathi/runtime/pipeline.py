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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..guidance.phrasing import Phraser
from ..guidance.saliency import Ranked, SaliencyConfig, SaliencyEngine
from ..guidance.speech import NullSpeaker, VoiceOutput
from ..models import DepthEstimator, Detector, ModelRegistry
from ..perception.distance import CameraModel, SizePriors, annotate
from ..perception.ground import GroundReading, depth_in_box, ground_profile

#: Distance band a detection must fall in to be usable as a scale anchor.
#: Closer than this and its ground contact is at the frame edge; further and
#: the ground-distance curve is too steep for the contact row to be precise.
ANCHOR_RANGE_M = (1.5, 6.0)
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
    ground: GroundReading | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass
class PipelineConfig:
    detector: str | None = "yolo11n-coco-320"
    depth: str | None = None
    lang: str = "en"
    #: How long a ground reading stays valid. Depth runs at ~2 Hz while
    #: detection runs at up to 8 Hz, so without this the ground hazard would
    #: vanish and reappear between depth passes and the tracker would never
    #: confirm it.
    ground_validity_s: float = 1.2
    #: Independent depth passes that must agree before a drop-off is spoken.
    ground_min_votes: int = 3
    ground_vote_window_s: float = 4.0
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

        self.depth: DepthEstimator | None = None
        if self.config.depth:
            dmodel = self.registry.load(self.config.depth)
            assert isinstance(dmodel, DepthEstimator)
            self.depth = dmodel

        self._camera: CameraModel | None = None
        self._ground: GroundReading | None = None
        self._ground_at: float = -1e9
        #: (timestamp, distance) from each INDEPENDENT depth pass that saw a
        #: drop. Not from re-injected cached readings - see _ground_confirmed.
        self._ground_votes: deque[tuple[float, float]] = deque(maxlen=6)

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
        camera = self.camera_for(frame)
        annotate(detections, camera, self.priors, frame_height=frame.height)
        stage["geometry"] = (time.perf_counter() - t0) * 1000.0

        # Tier 2. Runs at its own low rate, and only while moving.
        if self.depth is not None and self.scheduler.should_run_depth(now):
            t0 = time.perf_counter()
            dmap = self.depth.estimate(frame.image)
            assert self.depth.last_transform is not None
            reading = ground_profile(
                dmap, camera, self.depth.last_transform,
                anchor=self._scale_anchor(detections, dmap, self.depth.last_transform),
            )
            stage["depth"] = (time.perf_counter() - t0) * 1000.0
            if reading.trustworthy:
                self._ground, self._ground_at = reading, now
                self._vote(reading, now)

        ground = None
        if self._ground is not None and now - self._ground_at <= self.config.ground_validity_s:
            ground = self._ground
            hazard_det = self._ground_hazard(ground, frame, camera)
            if hazard_det is not None:
                detections.append(hazard_det)

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
            # Always run the speech gate, even when muted.
            #
            # `speak=False` used to skip `voice.say` entirely, which made
            # silence a *different code path* rather than the same one with
            # the volume down: the per-object and per-class cooldowns, the
            # urgency interrupt and the drop-don't-queue rule were all
            # bypassed, so `spoke` was permanently False and `utterance`
            # reported every candidate the saliency engine picked rather than
            # the ones that would actually have been said.
            #
            # Anything reading these fields to show what the product does -
            # the desktop window, the benchmark harness - was therefore
            # reading a fiction whenever audio was off, which is the default
            # for a demo. With a NullSpeaker the gate costs nothing and
            # `spoke` now means "passed the speech policy", which is the
            # question everything downstream is actually asking.
            spoke = self.voice.say(utterance, now=now)

        return PipelineResult(
            frame_seq=frame.seq,
            decision=decision,
            detections=detections,
            tracks=tracks,
            chosen=chosen,
            utterance=utterance,
            spoke=spoke,
            ground=ground,
            stage_ms=stage,
        )

    def _scale_anchor(
        self,
        detections: list[Detection],
        depth: np.ndarray,
        transform,
    ) -> tuple[float, float] | None:
        """Pin the depth model's arbitrary scale to metres, using a detection.

        Relative depth cannot say how far below the camera a flat surface sits,
        so it cannot tell a floor from a desk - see `surface_height_m`. What it
        needs is one point whose distance is known by other means, and the
        geometric estimator already produces exactly that: metres to an object
        standing on the ground, from its ground contact and a size prior.

        The anchor is chosen to be the one most likely to be right rather than
        the nearest or the largest:

        * `grounded` only. An object whose distance came from a size prior
          while it hangs on a wall anchors the scale to a plane it is not on.
        * nothing too close or too far. Very near the camera the box bottom is
          at the frame edge and the contact row is unreliable; far away, a
          one-row error in the contact point is worth a large error in metres,
          because the ground-distance curve steepens towards the horizon.
        * the highest-confidence survivor, because a misdetection here does not
          produce a wrong distance to one object - it produces a wrong verdict
          about the entire floor.
        """
        usable = [
            det for det in detections
            if det.distance_m is not None
            # "ground" and "fused" both rest on a ground contact. "size" does
            # not - a clock measured from its size prior is on a wall, and
            # anchoring the floor's scale to it would be anchoring to a plane
            # the object is not on. "bounded" is an inequality, not a distance.
            and det.distance_source in ("ground", "fused")
            and ANCHOR_RANGE_M[0] <= det.distance_m <= ANCHOR_RANGE_M[1]
        ]
        if not usable:
            return None
        best = max(usable, key=lambda d: d.score)
        value = depth_in_box(depth, best.box, transform)
        if value is None:
            return None
        return (value, float(best.distance_m))

    def _vote(self, reading: GroundReading, now: float) -> None:
        if reading.anomaly == "step_down" and reading.anomaly_distance_m is not None:
            self._ground_votes.append((now, reading.anomaly_distance_m))
        else:
            # A pass that saw clear floor is evidence against, and it clears
            # the record rather than merely not adding to it.
            self._ground_votes.clear()

    def _ground_confirmed(self, reading: GroundReading) -> bool:
        """Require several INDEPENDENT depth passes to agree.

        This exists because of a bug that the caching introduced. A ground
        reading is valid for ~1.2 s while depth runs at ~2 Hz, so the same
        single measurement gets re-injected as a detection on every frame in
        between. The tracker then sees it repeatedly, `min_hits` is satisfied
        within two frames, and the confirmation is entirely fake - one
        measurement wearing a disguise.

        Counting distinct depth passes restores what min_hits was supposed to
        provide: agreement between independent looks at the world.
        """
        cfg = self.config
        if reading.anomaly_distance_m is None:
            return False
        recent = [
            d for ts, d in self._ground_votes
            if self._ground_at - ts <= cfg.ground_vote_window_s
        ]
        if len(recent) < cfg.ground_min_votes:
            return False
        # And they have to agree about *where*, not just that something is there.
        span = max(recent) - min(recent)
        return span <= 0.35 * max(recent)

    def _ground_hazard(
        self, reading: GroundReading, frame: Frame, camera: CameraModel
    ) -> Detection | None:
        """Turn a floor anomaly into a detection, so it flows through the
        normal tracking and saliency path rather than getting a special case.

        Only `step_down` is surfaced. A `step_up` is far more often a wall,
        a kerb the user is walking onto, or furniture - all of which the
        detector already covers - and announcing every vertical surface ahead
        would make the system unusable. The drop is the hazard nothing else
        catches, and it is the one that hurts people.

        `step_up` still shortens the free-space distance, which is what the
        on-request "how far is clear?" answer uses.
        """
        if reading.anomaly != "step_down" or reading.anomaly_distance_m is None:
            return None
        if not self._ground_confirmed(reading):
            return None
        try:
            cls = self.taxonomy["step_down"]
        except KeyError:
            return None

        # Placed in the middle of the walking corridor, which is where the
        # profile sampled it from by construction.
        cx = frame.width / 2.0
        half = frame.width * 0.08
        det = Detection(
            box=(cx - half, frame.height * 0.75, cx + half, frame.height * 0.95),
            score=min(0.99, reading.fit_quality),
            class_id=cls.id,
            label=cls.name,
            hazard=cls.hazard,
        )
        det.distance_m = reading.anomaly_distance_m
        det.distance_source = "depth"
        det.bearing_deg = 0.0
        return det

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
        if self.depth is not None:
            self.depth.close()
        self.voice.close()
