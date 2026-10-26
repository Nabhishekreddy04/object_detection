import cv2
import numpy as np
import pytest

from bev import BevGrid, GroundPlane, render_bev


@pytest.fixture
def gp():
    return GroundPlane(focal=1000.0, cx=640.0, v0=380.0, height_m=1.3)


@pytest.mark.parametrize("x, z", [(0.0, 5.0), (-3.0, 12.0), (4.5, 30.0)])
def test_ground_image_round_trip(gp, x, z):
    u, v = gp.to_image(x, z)
    assert gp.to_ground(u, v) == pytest.approx((x, z))


def test_closer_points_are_lower_in_the_image(gp):
    assert gp.to_image(0, 5)[1] > gp.to_image(0, 20)[1] > gp.v0


def test_no_ground_point_at_or_above_horizon(gp):
    assert gp.to_ground(640, gp.v0) is None
    assert gp.to_ground(640, gp.v0 - 50) is None


def test_homography_maps_any_road_point_to_its_grid_cell(gp):
    grid = BevGrid()
    h = grid.homography(gp)
    for x, z in [(1.5, 8.0), (-6.0, 25.0)]:  # not the 4 corners used to build it
        img_pt = np.float32([[gp.to_image(x, z)]])
        bev_pt = cv2.perspectiveTransform(img_pt, h)[0, 0]
        assert bev_pt == pytest.approx(grid.to_px(x, z), abs=1.0)


def test_render_bev_shape_and_markers(gp):
    grid = BevGrid()
    frame = np.full((720, 1280, 3), 80, np.uint8)
    empty = render_bev(frame, [], gp, grid)
    det = {"class": "cone", "box": [620, 400, 660, 470], "distance_m": 10.0}
    marked = render_bev(frame, [det], gp, grid)
    assert empty.shape == marked.shape == (grid.size[1], grid.size[0], 3)
    assert not np.array_equal(empty, marked)
