from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from serve import app

SAMPLE = Path(__file__).resolve().parent.parent / "assets" / "samples" / "bca36352-acc82f86.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["classes"] == ["cone", "barrier", "stop_sign"]


def test_detect_returns_barriers_with_distances(client):
    r = client.post("/detect", files={"file": SAMPLE.open("rb")})
    assert r.status_code == 200
    body = r.json()
    assert body["image_size"] == {"width": 1280, "height": 720}
    barriers = [d for d in body["detections"] if d["class"] == "barrier"]
    assert len(barriers) == 3
    assert all(3 < d["distance_m"] < 12 for d in barriers)


def test_detect_image_returns_annotated_jpeg(client):
    r = client.post("/detect/image", files={"file": SAMPLE.open("rb")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (720, 1280, 3)


def test_rejects_non_images(client):
    r = client.post("/detect", files={"file": ("notes.txt", b"not an image")})
    assert r.status_code == 400
