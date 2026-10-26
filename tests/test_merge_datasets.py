import pytest

from data.merge_datasets import parse_label, take


def write(tmp_path, text):
    p = tmp_path / "img.txt"
    p.write_text(text)
    return p


def test_box_labels_are_remapped_and_unmapped_classes_dropped(tmp_path):
    lbl = write(tmp_path, "0 0.5 0.5 0.2 0.2\n5 0.3 0.3 0.1 0.1\n3 0.1 0.1 0.05 0.05\n")
    assert parse_label(lbl, {0: 1, 3: 1}) == [
        "1 0.500000 0.500000 0.200000 0.200000",
        "1 0.100000 0.100000 0.050000 0.050000",
    ]


def test_polygon_becomes_clipped_bounding_box(tmp_path):
    # triangle sticking out of the image on the left
    lbl = write(tmp_path, "0 -0.1 0.2 0.4 0.2 0.3 0.6\n")
    cls, cx, cy, w, h = parse_label(lbl, {0: 2})[0].split()
    assert cls == "2"
    assert [float(v) for v in (cx, cy, w, h)] == pytest.approx([0.2, 0.4, 0.4, 0.4])


def test_tiny_boxes_are_dropped(tmp_path):
    lbl = write(tmp_path, "0 0.5 0.5 0.001 0.2\n")
    assert parse_label(lbl, {0: 0}) == []


def test_take_caps_by_images_not_groups():
    groups = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1]}
    assert take(["a", "b", "c"], groups, 4) == ["a", "b"]
    assert take(["c", "a", "b"], groups, 1) == ["c"]


def test_near_duplicate_boxes_are_removed_only_when_asked(tmp_path):
    # second box is the first shifted by 0.2% of the image; third is another class at the same place
    lbl = write(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.502 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    assert len(parse_label(lbl, {0: 0, 1: 1})) == 3
    kept = parse_label(lbl, {0: 0, 1: 1}, dedup_iou=0.9)
    assert [line.split()[0] for line in kept] == ["0", "1"]
