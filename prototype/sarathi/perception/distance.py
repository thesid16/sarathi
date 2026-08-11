"""Distance and bearing from a single camera, without running a depth network.

This is the always-on distance estimator. It costs microseconds, runs on every
detection, and is accurate enough for the only thing that matters here -
saying "about a metre and a half" out loud. A monocular depth network is
reserved for what geometry genuinely cannot do: hazards with no bounding box,
like a kerb or a drop-off.

Two independent estimators, deliberately:

**Ground plane.** If the camera height and pitch are known and the bottom of a
box rests on the floor, the depression angle to that bottom edge gives the
distance directly. Accurate, and it needs no idea what the object *is*.

**Size prior.** If a class has a predictable real-world height, distance falls
out of its pixel height. Works when the object's base is occluded, cut off by
the frame, or simply not on the floor.

They fail in different situations, which is the point. The ground-plane
estimate collapses when the box runs off the bottom of the frame; the size
prior collapses for classes with a wide size spread. Fusion prefers whichever
is currently trustworthy and reports which one it used, so a wrong distance can
later be traced to the estimator that produced it rather than to "the model".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..types import Detection

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIORS = _REPO_ROOT / "training" / "taxonomy" / "size_priors.yaml"

#: Below this depression angle an object is effectively at the horizon and the
#: ground-plane estimate explodes towards infinity. Refuse rather than report a
#: confident 400 metres.
_MIN_DEPRESSION_RAD = math.radians(0.6)

#: Distances outside this are not useful to speak and usually mean the estimate
#: has broken down.
MIN_DISTANCE_M = 0.3
MAX_DISTANCE_M = 60.0


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera plus how it is worn.

    Defaults describe a phone hanging on a lanyard at chest height, pointing
    straight ahead. They are a starting point, not a calibration - see
    `docs/01-architecture.md` on the open question of calibrating an arbitrary
    external camera.
    """

    width: int
    height: int
    hfov_deg: float = 66.0
    #: Height of the lens above the walking surface, in metres.
    mount_height_m: float = 1.20
    #: Downward tilt in degrees. Positive means aimed at the ground.
    pitch_deg: float = 0.0

    @property
    def fx(self) -> float:
        """Focal length in pixels, from the field of view across the long axis.

        The quoted horizontal FOV describes the sensor's wide axis, and a phone
        sensor is mounted landscape. On Android the analysis bitmap is rotated
        upright before it reaches the detector, so in portrait `width` is the
        *short* axis - dividing by it understates the focal length by 25% on a
        4:3 frame and makes every distance about 27% short.

        Focal length is a property of the optics and the pixel pitch. Rotating
        an image swaps which axis is which and changes neither, so taking the
        longer dimension is correct in both orientations.

        Must stay identical to `CameraModel.fx` in Geometry.kt.
        """
        return (max(self.width, self.height) / 2.0) / math.tan(
            math.radians(self.hfov_deg) / 2.0
        )

    @property
    def fy(self) -> float:
        # Square pixels: the vertical focal length equals the horizontal one.
        # Deriving fy from a vertical FOV instead would double-count the aspect
        # ratio and skew every distance.
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    @property
    def horizon_y(self) -> float:
        """Image row where the ground plane meets the horizon.

        Anything above this row cannot be standing on the floor in front of the
        user, so the ground-plane estimator must not be applied to it.
        """
        return self.cy - self.fy * math.tan(math.radians(self.pitch_deg))

    @property
    def nearest_visible_ground_m(self) -> float | None:
        """Distance to the closest floor the camera can actually see.

        Set by the bottom edge of the frame, and it is further out than people
        expect: a chest-mounted camera aimed nearly level cannot see the ground
        at its own feet. Anything nearer than this is a blind spot, and both the
        ground-plane estimator and the floor analysis are bounded by it.
        """
        return self.ground_distance(self.height - 1)

    def bearing_deg(self, x: float) -> float:
        """Horizontal angle from straight ahead. Negative left, positive right."""
        return math.degrees(math.atan2(x - self.cx, self.fx))

    def ground_distance(self, y_bottom: float) -> float | None:
        """Distance to a point on the floor imaged at row `y_bottom`."""
        alpha = math.atan2(y_bottom - self.cy, self.fy)
        depression = math.radians(self.pitch_deg) + alpha
        if depression <= _MIN_DEPRESSION_RAD:
            return None  # at or above the horizon
        return self.mount_height_m / math.tan(depression)

    def distance_from_height(self, pixel_height: float, real_height_m: float) -> float | None:
        """Distance from apparent size."""
        if pixel_height <= 1.0 or real_height_m <= 0.0:
            return None
        return real_height_m * self.fy / pixel_height


@dataclass(frozen=True)
class SizePrior:
    h_m: float | None = None
    spread: float = 0.30
    grounded: bool = True


@dataclass
class SizePriors:
    """Real-world size priors, loaded from YAML."""

    priors: dict[str, SizePrior] = field(default_factory=dict)
    default_spread: float = 0.30

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SizePriors":
        p = Path(path or DEFAULT_PRIORS)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) or {}
        default_spread = float((data.get("defaults") or {}).get("spread", 0.30))
        priors = {}
        for name, entry in (data.get("priors") or {}).items():
            entry = entry or {}
            priors[str(name)] = SizePrior(
                h_m=(float(entry["h_m"]) if entry.get("h_m") is not None else None),
                spread=float(entry.get("spread", default_spread)),
                grounded=bool(entry.get("grounded", True)),
            )
        return cls(priors, default_spread)

    def get(self, label: str) -> SizePrior:
        return self.priors.get(label, SizePrior(None, self.default_spread, True))


@dataclass(frozen=True)
class DistanceEstimate:
    distance_m: float | None
    source: str  # "ground" | "size" | "fused" | "none"
    #: Rough fractional uncertainty, for the phrasing layer to decide between
    #: "one and a half metres" and "a few metres".
    uncertainty: float = 1.0


#: How far below the horizon a box bottom must sit for the ground-plane
#: estimate to mean anything, as a fraction of the frame height.
#:
#: The ground-distance curve is `h / tan(depression)`, which goes to infinity
#: as the contact row approaches the horizon. A few rows either side of it are
#: the difference between "far" and "impossibly far", and no detector places a
#: box bottom to that precision.
CONTACT_MARGIN_FRACTION = 0.04


def _contact_is_usable(y2: float, camera: CameraModel, frame_h: int) -> bool:
    """Whether a box bottom is far enough below the horizon to trust.

    Found by looking at a picture instead of a log. A wall clock in the
    self-test frame was reported at **54.3 m** while a car in the same frame
    came out at 1.2 m - the clock hangs high, its box bottom lands just under
    the horizon, and there a single row of pixels is worth tens of metres.

    Nothing threw. The detection count was correct, the label was correct, and
    the number was nonsense; on a phone with the screen off it would have been
    spoken as fact. The fix is not a better prior for clocks - it is refusing
    to convert a contact row into metres where the arithmetic has no
    resolution left, whatever the object is.
    """
    return y2 >= camera.horizon_y + frame_h * CONTACT_MARGIN_FRACTION


def estimate_distance(
    detection: Detection,
    camera: CameraModel,
    priors: SizePriors,
    *,
    frame_height: int | None = None,
) -> DistanceEstimate:
    """Estimate distance to one detection."""
    x1, y1, x2, y2 = detection.box
    box_h = max(0.0, y2 - y1)
    prior = priors.get(detection.label)
    frame_h = frame_height if frame_height is not None else camera.height

    # A box whose bottom edge sits on the frame boundary is cut off: the object
    # continues below the image, so its real contact point with the floor is
    # unknown and the ground-plane estimate would place it too far away.
    truncated = y2 >= frame_h - 1.5

    ground = None
    if prior.grounded and not truncated and _contact_is_usable(y2, camera, frame_h):
        ground = camera.ground_distance(y2)

    size = None
    if prior.h_m is not None:
        size = camera.distance_from_height(box_h, prior.h_m)

    ground = _sane(ground)
    size = _sane(size)

    # A truncated box has its base below the visible frame, so the object must
    # be NEARER than the closest ground point the camera can see. That is a
    # hard geometric bound, and a size prior on a partly visible object
    # routinely violates it: a person seen from the shoulders up "measures"
    # further away than the camera can see the floor at all.
    #
    # Getting this wrong is not a rounding error. It systematically pushes the
    # nearest objects - the ones already too close to fully fit in frame -
    # further away, which is exactly backwards for a collision warning.
    bounded = False
    if truncated and prior.grounded and size is not None:
        limit = _sane(camera.ground_distance(max(1.0, frame_h - 1)))
        if limit is not None and size > limit:
            size = limit
            bounded = True

    if bounded:
        # Report it as a bound, not a measurement, and say we are unsure -
        # the true distance is somewhere between zero and this.
        return DistanceEstimate(size, "bounded", uncertainty=0.6)

    if ground is not None and size is not None:
        # Agreement is evidence both assumptions hold. Disagreement usually
        # means the object is not resting where it appears to - propped up,
        # partly occluded, or on a step - and the ground plane is the more
        # trustworthy of the two for anything actually standing on the floor.
        ratio = max(ground, size) / max(1e-6, min(ground, size))
        if ratio <= 1.35:
            fused = (ground + size) / 2.0
            return DistanceEstimate(fused, "fused", uncertainty=0.12)
        return DistanceEstimate(ground, "ground", uncertainty=0.30)

    if ground is not None:
        return DistanceEstimate(ground, "ground", uncertainty=0.20)
    if size is not None:
        return DistanceEstimate(size, "size", uncertainty=max(0.15, prior.spread))
    return DistanceEstimate(None, "none", uncertainty=1.0)


def annotate(
    detections: list[Detection],
    camera: CameraModel,
    priors: SizePriors,
    *,
    frame_height: int | None = None,
) -> list[Detection]:
    """Fill in `distance_m`, `distance_source` and `bearing_deg` in place."""
    for det in detections:
        est = estimate_distance(det, camera, priors, frame_height=frame_height)
        det.distance_m = est.distance_m
        det.distance_source = est.source if est.distance_m is not None else None
        det.bearing_deg = camera.bearing_deg(det.center[0])
    return detections


def clock_position(bearing_deg: float) -> int:
    """Bearing to a clock face, the convention blind users are trained on.

    Straight ahead is 12; each hour is 30 degrees. Only the forward half is
    meaningful for a forward-facing camera, so the result stays within 9 to 3
    o'clock.
    """
    hours = round(bearing_deg / 30.0)
    hour = 12 + hours
    if hour > 12:
        hour -= 12
    if hour < 1:
        hour += 12
    return hour


def _sane(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value < MIN_DISTANCE_M or value > MAX_DISTANCE_M:
        return None
    return value
