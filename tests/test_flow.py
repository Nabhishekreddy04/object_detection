import cv2
import numpy as np
import pytest

from flow import expansion, iou, shift_scale, track_points


def zoom(img, k, cx, cy):
    m = np.float32([[k, 0, cx - k * cx], [0, k, cy - k * cy]])
    return cv2.warpAffine(img, m, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@pytest.fixture
def textured():
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(0, 255, (360, 640), dtype=np.uint8), (5, 5), 0)


def test_expansion_of_scaled_points():
    rng = np.random.default_rng(1)
    p0 = rng.uniform(0, 100, (20, 2))
    p1 = (p0 - 50) * 1.08 + [53, 47]  # scaled about (50, 50), then shifted
    assert expansion(p0, p1) == pytest.approx(1.08)


def test_expansion_needs_enough_spread_points():
    p = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert expansion(p, p) is None


def test_lucas_kanade_recovers_known_zoom(textured):
    k = 1.03  # 3% expansion per frame, e.g. closing at 10 m/s from 11 m at 30 fps
    nxt = zoom(textured, k, 320, 180)
    pts = track_points(textured, nxt, [260, 120, 380, 240])
    assert pts is not None
    assert expansion(*pts) == pytest.approx(k, abs=0.005)


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_shift_scale():
    assert shift_scale([0, 0, 10, 20], (5, -5), 2.0) == pytest.approx([0, -15, 20, 25])
