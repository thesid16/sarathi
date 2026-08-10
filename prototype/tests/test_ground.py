"""Tests for reading the floor out of a depth map.

Depth maps here are synthesised from the camera geometry rather than produced
by a model, which is the point: it means a synthetic *flat floor* is exactly
flat, a synthetic drop-off is exactly a drop-off, and the analysis is tested
against ground truth it cannot cheat on. Model output is noisy and correct
behaviour on noise is a separate question, covered by the noise test at the
end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sarathi.perception.distance import CameraModel
from sarathi.perception.ground import depth_in_box, ground_profile
from sarathi.perception.preprocess import Transform

CAM = CameraModel(width=1280, height=720, hfov_deg=66.0, mount_height_m=1.4, pitch_deg=20.0)
SIZE = 252
# Stretch resize, as depth models use: independent axis scales, no padding.
TF = Transform(SIZE / 1280, SIZE / 720, 0.0, 0.0, 1280, 720)


def frame_y_of(row: int) -> float:
    return row / TF.scale_y


def flat_floor(scale: float = 3.0, offset: float = 0.4) -> np.ndarray:
    """Inverse relative depth of a perfectly flat floor, as a model would see it.

    Larger = nearer. Above the horizon there is no floor; fill it with a small
    value, as a real model would for distant sky or wall.
    """
    depth = np.full((SIZE, SIZE), 0.05, np.float32)
    for row in range(SIZE):
        y = frame_y_of(row)
        alpha = math.atan2(y - CAM.cy, CAM.fy)
        depression = math.radians(CAM.pitch_deg) + alpha
        if depression <= 1e-3:
            continue
        depth[row, :] = scale * math.tan(depression) + offset
    return depth


def distance_to_row(metres: float) -> int:
    """Which depth-map row images flat floor at a given distance."""
    depression = math.atan2(CAM.mount_height_m, metres)
    alpha = depression - math.radians(CAM.pitch_deg)
    y = CAM.cy + CAM.fy * math.tan(alpha)
    return int(round(y * TF.scale_y))


# -- flat ground -------------------------------------------------------------


def test_flat_floor_reports_no_anomaly():
    reading = ground_profile(flat_floor(), CAM, TF)
    assert reading.anomaly is None
    assert reading.flat
    assert reading.fit_quality > 0.99
    # Flat, but not yet *trusted*: nothing has established that this plane is
    # the floor rather than any other flat thing. See the surface-height tests.
    assert not reading.trustworthy


def test_flat_floor_reports_free_space_to_the_far_field():
    reading = ground_profile(flat_floor(), CAM, TF)
    assert reading.free_distance_m is not None
    assert reading.free_distance_m > 5.0


def test_the_fit_is_scale_and_offset_invariant():
    """The depth model's arbitrary scale must cancel; if it does not, every
    reading depends on a number the model does not promise to keep stable."""
    a = ground_profile(flat_floor(scale=3.0, offset=0.4), CAM, TF)
    b = ground_profile(flat_floor(scale=17.0, offset=-2.5), CAM, TF)
    assert a.anomaly is b.anomaly is None
    assert a.free_distance_m == pytest.approx(b.free_distance_m, rel=1e-6)


# -- anomalies ---------------------------------------------------------------


def test_pitch_governs_how_much_near_floor_there_is_to_calibrate_on():
    """The blind spot is further out than people expect, and pitch is what
    controls it. A near-level camera leaves almost no flat floor between its
    blind spot and the range where hazards matter, which is why the rig is
    tilted down."""
    level = CameraModel(1280, 720, 66.0, 1.4, pitch_deg=5.0)
    assert level.nearest_visible_ground_m > 3.0        # barely any near floor
    assert CAM.nearest_visible_ground_m < 1.8          # tilted down: usable
    assert CAM.nearest_visible_ground_m == pytest.approx(1.67, abs=0.05)


def test_a_drop_off_is_detected_and_located():
    """Beyond the edge the ground is further away than flat floor predicts."""
    depth = flat_floor()
    edge = distance_to_row(4.0)
    depth[:edge, :] *= 0.55  # everything beyond the edge recedes
    reading = ground_profile(depth, CAM, TF)
    assert reading.anomaly == "step_down"
    assert reading.anomaly_distance_m == pytest.approx(4.0, abs=0.8)


def test_a_step_up_is_detected_and_located():
    """A raised surface is nearer than flat floor predicts."""
    depth = flat_floor()
    edge = distance_to_row(3.5)
    depth[: edge, :] *= 1.5
    reading = ground_profile(depth, CAM, TF)
    assert reading.anomaly == "step_up"
    assert reading.anomaly_distance_m == pytest.approx(3.5, abs=0.8)


def test_free_distance_stops_at_the_hazard():
    depth = flat_floor()
    depth[: distance_to_row(4.0), :] *= 0.5
    reading = ground_profile(depth, CAM, TF)
    assert reading.free_distance_m == pytest.approx(reading.anomaly_distance_m)
    assert reading.free_distance_m < 5.5


def test_the_nearest_hazard_wins_when_there_are_two():
    """Anything beyond the first hazard is behind it, and irrelevant."""
    depth = flat_floor()
    depth[: distance_to_row(9.0), :] *= 0.5                      # far drop
    depth[distance_to_row(9.0) : distance_to_row(3.5), :] *= 1.6  # nearer step up
    reading = ground_profile(depth, CAM, TF)
    assert reading.anomaly == "step_up"
    assert reading.anomaly_distance_m == pytest.approx(3.5, abs=1.0)


# -- robustness --------------------------------------------------------------


def test_a_single_bad_row_does_not_announce_a_step():
    """False 'step down ahead' alerts are how a system loses a user's trust."""
    depth = flat_floor()
    depth[distance_to_row(4.0), :] *= 0.3  # one wild row
    assert ground_profile(depth, CAM, TF).anomaly is None


def test_realistic_noise_does_not_invent_hazards():
    rng = np.random.default_rng(0)
    depth = flat_floor()
    depth = depth + rng.normal(0, 0.02 * float(depth.max()), depth.shape).astype(np.float32)
    reading = ground_profile(depth, CAM, TF)
    assert reading.anomaly is None
    assert reading.fit_quality > 0.9


def test_an_object_off_to_the_side_is_outside_the_corridor():
    """A step two metres to the left is not this user's problem."""
    depth = flat_floor()
    depth[:, : SIZE // 6] *= 2.0  # far left edge only
    assert ground_profile(depth, CAM, TF).anomaly is None


def test_a_scene_with_no_floor_reports_untrustworthy_rather_than_guessing():
    """Pointed at a wall, the flat-ground model does not apply and saying so
    is the only safe answer."""
    rng = np.random.default_rng(1)
    depth = rng.random((SIZE, SIZE), dtype=np.float32) * 5.0
    reading = ground_profile(depth, CAM, TF)
    assert not reading.trustworthy


def test_degenerate_input_is_handled():
    assert ground_profile(np.zeros((0, 0), np.float32), CAM, TF).free_distance_m is None
    assert ground_profile(np.zeros((5, 5), np.float32), CAM, TF).samples < 12


def test_a_camera_aimed_at_the_sky_finds_no_floor():
    up = CameraModel(1280, 720, 66.0, 1.4, pitch_deg=-40.0)
    reading = ground_profile(flat_floor(), up, TF)
    assert reading.samples < 12 or not reading.trustworthy


# -- box sampling ------------------------------------------------------------


def test_depth_in_box_reads_the_right_region():
    depth = np.zeros((SIZE, SIZE), np.float32)
    depth[100:150, 100:150] = 7.0
    # Frame-space box covering that depth-map region.
    box = (100 / TF.scale_x, 100 / TF.scale_y, 150 / TF.scale_x, 150 / TF.scale_y)
    assert depth_in_box(depth, box, TF) == pytest.approx(7.0)


def test_depth_in_box_clamps_to_the_map():
    depth = np.ones((SIZE, SIZE), np.float32)
    assert depth_in_box(depth, (-500, -500, 99999, 99999), TF) == pytest.approx(1.0)


def test_depth_in_box_rejects_an_empty_region():
    assert depth_in_box(np.ones((SIZE, SIZE), np.float32), (10, 10, 10, 10), TF) is None


# -- telling the floor from anything else flat -------------------------------
#
# `flat_floor` is parameterised by the depth model's arbitrary scale and offset
# precisely so these tests can show what the fit can and cannot see through.


def surface(height_m: float, scale: float = 3.0, offset: float = 0.4) -> np.ndarray:
    """A horizontal plane `height_m` below the camera, in inverse relative depth."""
    depth = np.full((SIZE, SIZE), 0.05, np.float32)
    for row in range(SIZE):
        y = frame_y_of(row)
        alpha = math.atan2(y - CAM.cy, CAM.fy)
        depression = math.radians(CAM.pitch_deg) + alpha
        if depression <= 1e-3:
            continue
        # observed = scale / distance + offset, distance = height / tan(dep)
        depth[row, :] = scale * math.tan(depression) / height_m + offset
    return depth


def anchor_on(height_m: float, scale: float, offset: float, distance_m: float):
    """What an object standing on that plane at `distance_m` looks like."""
    return (scale / distance_m + offset, distance_m)


def test_fit_quality_cannot_tell_a_desk_from_the_floor():
    """The central negative result, and the reason this machinery exists.

    The flat-ground fit is scale invariant - that is exactly what makes it
    usable on a depth model whose output has no units. The same invariance
    means it cannot recover how far below the camera the plane sits, because
    halving the height and halving the model's scale produce identical depth
    maps. A desk is not a worse plane than a floor. It is an equally good one.

    So this is not a threshold that needs tuning. No value of `fit_quality`
    separates these, and the measurement says so.
    """
    floor = ground_profile(surface(CAM.mount_height_m), CAM, TF)
    desk = ground_profile(surface(0.45), CAM, TF)

    assert floor.fit_quality > 0.99
    assert desk.fit_quality > 0.99
    assert abs(floor.fit_quality - desk.fit_quality) < 0.01
    assert floor.flat and desk.flat


def test_an_anchor_recovers_the_true_surface_height():
    """One known distance is enough to break the ambiguity.

    And it must work regardless of the depth model's scale, since that is the
    unknown being cancelled.
    """
    for scale, offset in ((3.0, 0.4), (17.0, -2.5), (0.5, 0.0)):
        reading = ground_profile(
            surface(1.4, scale, offset), CAM, TF,
            anchor=anchor_on(1.4, scale, offset, distance_m=3.0),
        )
        assert reading.surface_height_m == pytest.approx(1.4, rel=0.05)


def test_anchored_floor_is_trusted_and_anchored_desk_is_not():
    floor = ground_profile(
        surface(1.4), CAM, TF, anchor=anchor_on(1.4, 3.0, 0.4, distance_m=3.0)
    )
    desk = ground_profile(
        surface(0.45), CAM, TF, anchor=anchor_on(0.45, 3.0, 0.4, distance_m=3.0)
    )
    assert floor.trustworthy
    assert not desk.trustworthy
    assert desk.flat  # still a fine plane; just not the floor


@pytest.mark.parametrize("fraction", [0.85, 1.0, 1.3])
def test_the_band_absorbs_ordinary_posture(fraction: float) -> None:
    """The configured mount height is a guess about how someone stands.

    Slouching, straightening or raising the phone moves the real height by
    tens of percent without changing anything about the floor, so the band has
    to absorb that or the tier goes silent on a perfectly good floor.
    """
    height_m = CAM.mount_height_m * fraction
    reading = ground_profile(
        surface(height_m), CAM, TF, anchor=anchor_on(height_m, 3.0, 0.4, 3.0)
    )
    assert reading.trustworthy


def test_a_phone_held_far_lower_than_configured_goes_silent() -> None:
    """A deliberate false negative, recorded so it is not mistaken for a bug.

    The band cannot be widened to cover a phone carried at hip height without
    also admitting a counter or a desk, because at that point the two are the
    same measurement. The two errors do not cost the same: a rejected floor is
    silence, which is what this tier does today anyway, while an accepted desk
    is a confident "step down ahead" that is not there.

    So the tier declines rather than guesses, and the fix is for the user's
    configured mount height to match how they actually carry the phone.
    """
    height_m = CAM.mount_height_m * 0.6
    reading = ground_profile(
        surface(height_m), CAM, TF, anchor=anchor_on(height_m, 3.0, 0.4, 3.0)
    )
    assert reading.flat
    assert reading.surface_height_m == pytest.approx(height_m, rel=0.05)
    assert not reading.trustworthy


@pytest.mark.parametrize("height_m", [0.35, 0.45, 0.7, 0.9])
def test_surfaces_well_above_the_floor_are_refused(height_m: float):
    """Desks, counters, low walls and platform edges.

    Accepting one of these produces a confident "step down ahead" at its far
    edge, which is the single fastest way to make someone stop trusting the
    system.
    """
    reading = ground_profile(
        surface(height_m), CAM, TF, anchor=anchor_on(height_m, 3.0, 0.4, 3.0)
    )
    assert not reading.trustworthy


def test_without_an_anchor_nothing_is_trusted():
    """The honest default.

    No detection standing on the ground means no metric scale, which means the
    plane's identity is unknown. Unknown must not read as floor.
    """
    reading = ground_profile(surface(CAM.mount_height_m), CAM, TF)
    assert reading.flat
    assert reading.surface_height_m is None
    assert not reading.trustworthy
