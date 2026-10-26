import math

import pytest

from distance import DistanceEstimator, calibrate_focal, focal_from_fov


@pytest.fixture
def est():
    return DistanceEstimator.from_config()


def test_focal_from_fov():
    # f = (W/2) / tan(HFOV/2)
    assert focal_from_fov(1280, 65.0) == pytest.approx(640 / math.tan(math.radians(32.5)))
    assert focal_from_fov(1280, 90.0) == pytest.approx(640.0)


def test_cone_distance_matches_pinhole_formula(est):
    f = est.focal_for(1280)
    assert est.estimate("cone", (100, 200, 140, 270), 1280) == pytest.approx(f * 0.70 / 70)


def test_distance_is_resolution_independent(est):
    full = est.estimate("barrier", (200, 300, 300, 400), 1280)
    half = est.estimate("barrier", (100, 150, 150, 200), 640)
    assert full == pytest.approx(half)


def test_stop_sign_uses_larger_side(est):
    # bottom of the sign cut off by the frame: box is wider than tall
    cut = est.estimate("stop_sign", (0, 0, 80, 40), 1280)
    full = est.estimate("stop_sign", (0, 0, 80, 80), 1280)
    assert cut == pytest.approx(full)


def test_unknown_class_or_degenerate_box(est):
    assert est.estimate("car", (0, 0, 10, 10), 1280) is None
    assert est.estimate("cone", (0, 50, 10, 50), 1280) is None


def test_calibrate_focal_round_trip(est):
    f = calibrate_focal(distance_m=10.0, h_px=70.0, height_m=0.70)
    assert f == pytest.approx(1000.0)
