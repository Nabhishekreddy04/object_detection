"""REST inference service: upload an image, get detections with distances (JSON) or an annotated image.

Endpoints:
    GET  /health          model and config in use
    POST /detect          multipart image -> {"detections": [{class, confidence, box, distance_m}], ...}
    POST /detect/image    multipart image -> annotated JPEG ("Cone, 4.2m")

Configuration (environment variables):
    MODEL_PATH   weights to serve (default models/yolo26n_nav_v3.pt; the INT8 .onnx also works)
    CAMERA_CFG   camera / object-size config (default configs/camera.yaml)

Usage:
    uvicorn serve:app --app-dir src --host 0.0.0.0 --port 8000
    curl -F file=@assets/samples/bca36352-acc82f86.jpg localhost:8000/detect
"""

import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO

from detect import CLASS_CONF, annotate
from distance import DEFAULT_CONFIG, DistanceEstimator

MODEL_PATH = os.environ.get("MODEL_PATH", "models/yolo26n_nav_v3.pt")
CAMERA_CFG = os.environ.get("CAMERA_CFG", str(DEFAULT_CONFIG))


@lru_cache(maxsize=1)
def model() -> YOLO:
    return YOLO(MODEL_PATH, task="detect")


@lru_cache(maxsize=1)
def estimator() -> DistanceEstimator:
    return DistanceEstimator.from_config(CAMERA_CFG)


def decode(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="file is not a decodable image")
    return frame


def run(frame: np.ndarray, conf: float) -> tuple[list[dict], float]:
    """Detect + estimate distance; draws on frame in place. Returns (detections, inference ms)."""
    start = time.perf_counter()
    result = model()(frame, conf=conf, verbose=False)[0]
    dets = annotate(frame, result, estimator())
    return dets, (time.perf_counter() - start) * 1000


@asynccontextmanager
async def lifespan(_: FastAPI):
    model()(np.zeros((640, 640, 3), np.uint8), verbose=False)  # warm-up: first call loads / compiles the model
    yield


app = FastAPI(title="Navigation object detection", version="1.0.0", lifespan=lifespan)
ImageFile = Annotated[UploadFile, File(description="image (jpg / png)")]
Conf = Annotated[float, Query(ge=0.0, le=1.0, description="confidence threshold")]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "classes": list(model().names.values()), "class_conf": CLASS_CONF}


@app.post("/detect")
async def detect(file: ImageFile, conf: Conf = 0.35) -> dict:
    frame = decode(await file.read())
    dets, ms = run(frame, conf)
    return {
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "inference_ms": round(ms, 1),
        "detections": [
            {
                "class": d["class"],
                "confidence": round(d["conf"], 3),
                "box_xyxy": [round(v, 1) for v in d["box"]],
                "distance_m": None if d["distance_m"] is None else round(d["distance_m"], 2),
            }
            for d in dets
        ],
    }


@app.post("/detect/image")
async def detect_image(file: ImageFile, conf: Conf = 0.35) -> Response:
    frame = decode(await file.read())
    run(frame, conf)
    _, jpg = cv2.imencode(".jpg", frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")
