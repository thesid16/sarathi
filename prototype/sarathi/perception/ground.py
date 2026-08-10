"""Reading the floor out of a relative depth map.

This is what the depth tier is *for*. A detector answers "what is there"; it
cannot answer "does the floor continue", because a kerb, a step down and the
edge of a platform have no object to put a box around. Those are the hazards
that hurt people, and they are questions about surface shape.

The trick that makes relative depth usable here is that flat ground is
predictable. For a camera at height h with pitch θ, the distance to a floor
point imaged at row y is

    d(y) = h / tan(θ + atan((y - cy) / fy))

Depth models emit *inverse* relative depth - larger is nearer - on an unknown
scale, so what they report for flat floor should be an affine function of
1/d(y), which is `tan(θ + atan((y - cy) / fy))`. Fit that line against the
near field, where the floor almost certainly is floor, and then every row that
departs from it is departing from flat:

* observed **lower** than the fit - the surface is *further* than flat ground
  would be. The floor fell away: a step down, a kerb edge, a drop.
* observed **higher** than the fit - the surface is *nearer* than flat ground
  would be. Something is standing on the floor, or the floor rises: a step up,
  a ramp, an obstacle.

No metric depth is needed anywhere in that, which is what makes it safe. The
scale of the depth model cancels in the fit.

**Camera pitch is not a free parameter here.** The fit needs genuine flat floor
in the near field to calibrate against, and a near-level camera has very little
of it: aimed level-ish at chest height, the closest ground it can see is well
over two metres away, leaving almost nothing between the blind spot and the
range where hazards need detecting. Tilting the camera down trades far-field
reach for near-field floor, and the floor analysis needs the near field far
more than it needs the horizon. Around 20 degrees of downward pitch is a
reasonable starting point; it wants confirming on a real rig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .distance import CameraModel
from .preprocess import Transform


@dataclass(frozen=True)
class GroundReading:
    """What the floor looks like straight ahead."""

    #: How far the floor is continuous and flat, in metres. None when the
    #: geometry could not be read at all.
    free_distance_m: float | None
    #: "step_down" | "step_up" | None
    anomaly: str | None
    anomaly_distance_m: float | None
    #: Fraction of near-field variance the flat-ground model explains. Low
    #: values mean the near field was not floor - looking at a wall, or the
    #: camera is not pointed where the model assumes - and the reading should
    #: not be trusted or spoken.
    fit_quality: float
    samples: int

    @property
    def trustworthy(self) -> bool:
        return self.samples >= 12 and self.fit_quality >= 0.80


def _depth_row_to_frame_y(row: int, transform: Transform) -> float:
    """Map a depth-map row back to a frame row.

    Depth models use a stretch resize, not a letterbox, so the vertical scale
    differs from the horizontal one and this cannot be a single ratio.
    """
    return (row - transform.pad_y) / max(1e-9, transform.scale_y)


def ground_profile(
    depth: np.ndarray,
    camera: CameraModel,
    transform: Transform,
    *,
    corridor_frac: float = 0.34,
    fit_fraction: float = 0.5,
    sigma: float = 3.0,
    min_run: int = 3,
) -> GroundReading:
    """Fit flat ground to the near field and find where the floor stops obeying it.

    `corridor_frac` is the width of the sampled strip as a fraction of the
    frame - the walking corridor, not the whole scene. A step down two metres
    off to the left is not this user's problem.
    """
    if depth.ndim != 2 or depth.size == 0:
        return GroundReading(None, None, None, 0.0, 0)

    height, width = depth.shape
    half = max(1, int(width * corridor_frac / 2))
    centre = width // 2
    strip = depth[:, max(0, centre - half) : min(width, centre + half)]

    # Median across the corridor, so a single object in the strip does not drag
    # the row's estimate around.
    per_row = np.median(strip, axis=1)

    rows: list[int] = []
    expected: list[float] = []
    observed: list[float] = []
    distances: list[float] = []

    horizon = camera.horizon_y
    for row in range(height):
        frame_y = _depth_row_to_frame_y(row, transform)
        # Only below the horizon can there be floor at all.
        if frame_y <= horizon + 2:
            continue
        alpha = math.atan2(frame_y - camera.cy, camera.fy)
        depression = math.radians(camera.pitch_deg) + alpha
        if depression <= 1e-3:
            continue
        inverse_d = math.tan(depression)  # proportional to 1 / distance
        rows.append(row)
        expected.append(inverse_d)
        observed.append(float(per_row[row]))
        distances.append(camera.mount_height_m / inverse_d)

    if len(rows) < 12:
        return GroundReading(None, None, None, 0.0, len(rows))

    exp = np.asarray(expected, np.float64)
    obs = np.asarray(observed, np.float64)
    dist = np.asarray(distances, np.float64)

    # Fit on the near field only. Rows near the bottom of the frame are the
    # most likely to be actual floor; rows near the horizon are where walls,
    # traffic and the sky live.
    order = np.argsort(-exp)  # largest inverse depth = nearest = first

    # Bound the fit region by DISTANCE, not by row count.
    #
    # Rows are uniform in the image, not in the world: the nearest half of the
    # rows of a near-level camera reaches out past four metres. Fitting "flat
    # ground" over that region means a kerb at three metres is inside the very
    # sample the fit assumes is flat, and the line bends to accommodate the
    # hazard instead of exposing it.
    nearest = float(dist[order[0]])
    fit_limit = max(nearest * 1.6, nearest + 1.0)
    fit_idx = order[dist[order] <= fit_limit]
    if len(fit_idx) < 8:
        fit_idx = order[: max(8, int(len(order) * fit_fraction))]

    # Trim outliers out of the fit, iteratively.
    #
    # A plain least-squares fit over the near field is wrong whenever the
    # hazard is *inside* the near field - which is exactly when it matters
    # most. The anomalous rows drag the fitted line toward themselves, the
    # residuals flatten, and a real drop-off two metres ahead stops looking
    # like one. Trimming keeps the line describing the floor rather than the
    # floor plus the hole in it.
    a, b = np.polyfit(exp[fit_idx], obs[fit_idx], 1)
    for _ in range(3):
        resid = obs[fit_idx] - (a * exp[fit_idx] + b)
        spread = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
        if spread <= 1e-9:
            break
        keep = np.abs(resid - np.median(resid)) <= 2.0 * spread
        if keep.sum() < 8 or keep.all():
            break
        fit_idx = fit_idx[keep]
        a, b = np.polyfit(exp[fit_idx], obs[fit_idx], 1)

    predicted = a * exp + b
    residual = obs - predicted

    fit_residual = residual[fit_idx]
    ss_res = float(np.sum(fit_residual**2))
    ss_tot = float(np.sum((obs[fit_idx] - obs[fit_idx].mean()) ** 2))
    fit_quality = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)

    scatter = float(np.median(np.abs(fit_residual - np.median(fit_residual)))) * 1.4826
    if scatter <= 1e-9:
        scatter = float(np.std(fit_residual)) or 1e-6
    threshold = sigma * scatter

    # Walk outward from the nearest row. The first sustained departure is what
    # matters - anything beyond it is behind the hazard and irrelevant.
    #
    # `min_run` consecutive rows are required so a single noisy row cannot
    # announce a step that is not there. False "step down ahead" alerts are
    # the fastest way to make someone stop trusting the system.
    anomaly: str | None = None
    anomaly_distance: float | None = None
    free_distance: float | None = float(dist[order[0]])

    run_sign = 0
    run_len = 0
    for position, idx in enumerate(order):
        sign = 0
        if residual[idx] < -threshold:
            sign = -1  # further than flat ground: the floor fell away
        elif residual[idx] > threshold:
            sign = 1  # nearer than flat ground: something is in the way

        if sign != 0 and sign == run_sign:
            run_len += 1
        elif sign != 0:
            run_sign, run_len = sign, 1
        else:
            run_sign, run_len = 0, 0

        if run_len >= min_run:
            first = order[max(0, position - min_run + 1)]
            anomaly = "step_down" if run_sign < 0 else "step_up"
            anomaly_distance = float(dist[first])
            free_distance = anomaly_distance
            break
        if sign == 0:
            free_distance = float(dist[idx])

    return GroundReading(
        free_distance_m=free_distance,
        anomaly=anomaly,
        anomaly_distance_m=anomaly_distance,
        fit_quality=fit_quality,
        samples=len(rows),
    )


def depth_in_box(
    depth: np.ndarray,
    box: tuple[float, float, float, float],
    transform: Transform,
) -> float | None:
    """Median relative depth inside a frame-space box.

    Useful for *ordering* objects by distance when the geometric estimator is
    unreliable - the occluded case. It cannot supply metres, and the caller
    must not pretend otherwise.
    """
    if depth.ndim != 2 or depth.size == 0:
        return None
    x1, y1, x2, y2 = box
    dx1 = int(x1 * transform.scale_x + transform.pad_x)
    dx2 = int(x2 * transform.scale_x + transform.pad_x)
    dy1 = int(y1 * transform.scale_y + transform.pad_y)
    dy2 = int(y2 * transform.scale_y + transform.pad_y)

    h, w = depth.shape
    dx1, dx2 = max(0, min(w - 1, dx1)), max(1, min(w, dx2))
    dy1, dy2 = max(0, min(h - 1, dy1)), max(1, min(h, dy2))
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    return float(np.median(depth[dy1:dy2, dx1:dx2]))
