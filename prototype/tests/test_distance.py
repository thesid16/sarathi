"""Tests for geometric distance and bearing.

Numbers here are worked out by hand from the pinhole model rather than copied
from the implementation, so a change in the geometry fails the test instead of
quietly redefining what "two metres" means.
"""

from __future__ import annotations

import math

import pytest

from sarathi.perception.distance import (
    CameraModel,
    SizePrior,
    SizePriors,
    annotate,
    clock_position,
    estimate_distance,
)
from sarathi.types import Detection

# 1280x720, 66 degree horizontal FOV, lens 1.2 m up, aimed level.
#   fx = 640 / tan(33 deg) = 985.5 px
CAM = CameraModel(width=1280, height=720, hfov_deg=66.0, mount_height_m=1.2, pitch_deg=0.0)
PRIORS = SizePriors.load()


def det(x1, y1, x2, y2, label="person"):
    return Detection(box=(x1, y1, x2, y2), score=0.9, class_id=0, label=label)


# -- camera model ------------------------------------------------------------


def test_focal_length_from_field_of_view():
    assert CAM.fx == pytest.approx(640 / math.tan(math.radians(33)), rel=1e-6)
    assert CAM.fy == CAM.fx  # square pixels


def test_vertical_fov_follows_from_the_horizontal_one():
    # 720/1280 aspect at 66 deg horizontal gives roughly 40 deg vertical.
    assert CAM.vfov_deg == pytest.approx(40.0, abs=1.0)


def test_horizon_sits_at_image_centre_when_level():
    assert CAM.horizon_y == pytest.approx(CAM.cy)


def test_tilting_down_moves_the_horizon_up_the_image():
    tilted = CameraModel(1280, 720, 66.0, 1.2, pitch_deg=10.0)
    assert tilted.horizon_y < tilted.cy


# -- bearing -----------------------------------------------------------------


def test_bearing_is_zero_dead_ahead_and_signed_left_negative():
    assert CAM.bearing_deg(640) == pytest.approx(0.0)
    assert CAM.bearing_deg(1000) > 0  # right
    assert CAM.bearing_deg(280) < 0  # left


def test_bearing_at_the_frame_edge_is_half_the_fov():
    assert CAM.bearing_deg(1280) == pytest.approx(33.0, abs=0.1)


@pytest.mark.parametrize(("bearing", "hour"), [
    (0, 12), (30, 1), (60, 2), (-30, 11), (-60, 10), (14, 12), (16, 1),
])
def test_clock_position(bearing, hour):
    assert clock_position(bearing) == hour


# -- ground plane ------------------------------------------------------------


def test_ground_distance_matches_hand_computed_geometry():
    # y = 560 -> alpha = atan(200 / 985.5) = 11.47 deg
    #        -> d = 1.2 / tan(11.47 deg) = 5.91 m
    assert CAM.ground_distance(560) == pytest.approx(5.91, abs=0.02)
    # y = 700 -> alpha = atan(340 / 985.5) = 19.03 deg -> d = 3.48 m
    assert CAM.ground_distance(700) == pytest.approx(3.48, abs=0.02)


def test_lower_in_the_frame_means_nearer():
    assert CAM.ground_distance(700) < CAM.ground_distance(600) < CAM.ground_distance(500)


def test_at_or_above_the_horizon_there_is_no_ground_distance():
    assert CAM.ground_distance(CAM.cy) is None
    assert CAM.ground_distance(100) is None


def test_size_estimate_matches_hand_computed_geometry():
    # 1.65 m person occupying 200 px -> 1.65 * 985.5 / 200 = 8.13 m
    assert CAM.distance_from_height(200, 1.65) == pytest.approx(8.13, abs=0.02)


def test_a_degenerate_pixel_height_yields_nothing():
    assert CAM.distance_from_height(0, 1.65) is None
    assert CAM.distance_from_height(1, 0) is None


# -- fusion ------------------------------------------------------------------


def test_agreeing_estimators_are_fused():
    """A 1.65 m person at 5 m: bottom at y=596.5, 325 px tall. Both agree."""
    est = estimate_distance(det(600, 271.3, 700, 596.5, "person"), CAM, PRIORS)
    assert est.source == "fused"
    assert est.distance_m == pytest.approx(5.0, abs=0.15)
    assert est.uncertainty < 0.2


def test_disagreement_prefers_the_ground_plane():
    """Both estimates plausible, but far apart: geometry wins, and says so.

    Bottom at y=600 puts the floor contact at 4.93 m; a 203 px tall person
    implies 8.0 m. A 1.6x gap means the object is not standing where it looks
    like it is, and the ground plane is the more trustworthy of the two.
    """
    est = estimate_distance(det(600, 397, 700, 600, "person"), CAM, PRIORS)
    assert est.source == "ground"
    assert est.distance_m == pytest.approx(4.93, abs=0.05)
    assert est.uncertainty > 0.2  # and reports lower confidence


def test_an_implausible_size_estimate_is_dropped_before_fusion():
    """A 10 px person implies 162 m - discarded, leaving geometry alone."""
    est = estimate_distance(det(600, 690, 620, 700, "person"), CAM, PRIORS)
    assert est.source == "ground"
    assert est.distance_m == pytest.approx(CAM.ground_distance(700), abs=0.01)


def test_a_box_running_off_the_bottom_of_frame_ignores_the_ground_plane():
    """The object continues below the image, so its floor contact is unknown."""
    est = estimate_distance(det(600, 300, 700, 720, "person"), CAM, PRIORS)
    assert est.source == "size"


def test_elevated_objects_never_use_the_ground_plane():
    """A traffic light's box bottom is not on the road."""
    est = estimate_distance(det(600, 200, 640, 290, "traffic_light_red"), CAM, PRIORS)
    assert est.source == "size"
    # 0.9 m over 90 px -> 9.9 m, not the hundreds of metres the ground plane
    # would report for something near the horizon.
    assert est.distance_m == pytest.approx(9.86, abs=0.2)


def test_ground_surface_classes_use_geometry_alone():
    """An open manhole has no meaningful height, but geometry locates it well."""
    est = estimate_distance(det(500, 640, 700, 690, "open_manhole"), CAM, PRIORS)
    assert est.source == "ground"
    assert est.distance_m is not None


def test_unknown_labels_still_get_a_ground_estimate():
    est = estimate_distance(det(600, 500, 700, 600, "something_new"), CAM, PRIORS)
    assert est.source == "ground"


def test_an_object_above_the_horizon_with_no_prior_yields_nothing():
    est = estimate_distance(det(600, 100, 700, 200, "something_new"), CAM, PRIORS)
    assert est.distance_m is None
    assert est.source == "none"


def test_implausible_distances_are_refused_not_reported():
    """A person 1 px tall would be kilometres away - say nothing instead."""
    priors = SizePriors({"tiny": SizePrior(h_m=1.65, spread=0.1, grounded=False)})
    est = estimate_distance(det(600, 300, 601, 302, "tiny"), CAM, priors)
    assert est.distance_m is None


# -- annotate ----------------------------------------------------------------


def test_annotate_fills_distance_bearing_and_source():
    dets = [det(600, 271.3, 700, 596.5, "person"), det(100, 500, 200, 600, "chair")]
    annotate(dets, CAM, PRIORS)
    for d in dets:
        assert d.distance_m is not None
        assert d.bearing_deg is not None
        assert d.distance_source in {"ground", "size", "fused"}
    assert dets[1].bearing_deg < 0  # chair is left of centre


def test_annotate_leaves_source_none_when_distance_is_unknown():
    dets = [det(600, 100, 700, 200, "something_new")]
    annotate(dets, CAM, PRIORS)
    assert dets[0].distance_m is None and dets[0].distance_source is None
    assert dets[0].bearing_deg is not None  # bearing is always knowable


def test_priors_file_loads_and_covers_the_common_classes():
    assert PRIORS.get("person").h_m == pytest.approx(1.65)
    assert PRIORS.get("traffic_light_red").grounded is False
    assert PRIORS.get("open_manhole").h_m is None
    assert PRIORS.get("not_a_real_class").h_m is None
