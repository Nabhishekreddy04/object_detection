"""Optical flow on video: track each detected object and estimate its time-to-contact (TTC).

For every detection, Shi-Tomasi corners inside its box are tracked to the next frame with
pyramidal Lucas-Kanade flow (forward-backward checked). As the camera approaches an object
its image expands; the expansion rate s (median ratio of pairwise point distances between
frames) gives
    TTC = dt / (s - 1)              seconds until the camera reaches the object's plane
    closing speed = distance / TTC  using the size-based distance from src/distance.py
TTC needs no camera calibration, so it is an independent check on the distance estimate.
Per-track expansion is smoothed with an exponential moving average. The flow also predicts
each box's next position and size, which is used to match detections to tracks and lets a
track coast for up to MAX_MISSED frames when the detector misses the object.

Usage:
    python src/flow.py --source video.mp4
Writes runs/flow/<video>_flow.mp4 and runs/flow/<video>_tracks.csv
"""

import argparse
import csv
from itertools import count
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detect import COLORS, annotate
from distance import DEFAULT_CONFIG, DistanceEstimator

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
MIN_POINTS = 6
EMA = 0.5  # weight of the newest expansion measurement (lower = smoother but lags more)
MAX_MISSED = 10  # frames a track may coast on optical flow without a detection


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def track_points(prev_gray, gray, box):
    """LK-track corners inside box; return (p0, p1) of points passing the forward-backward check."""
    x1, y1, x2, y2 = (int(v) for v in box)
    mask = np.zeros_like(prev_gray)
    mask[max(y1, 0) : y2, max(x1, 0) : x2] = 255
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=60, qualityLevel=0.01, minDistance=3, mask=mask)
    if p0 is None or len(p0) < MIN_POINTS:
        return None
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
    good = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - back, axis=2)[:, 0] < 1.0)
    return (p0[good, 0], p1[good, 0]) if good.sum() >= MIN_POINTS else None


def expansion(p0, p1) -> float | None:
    """Median ratio of pairwise point distances (frame t / frame t-1); > 1 means approaching."""
    i, j = np.triu_indices(len(p0), 1)
    d0 = np.linalg.norm(p0[i] - p0[j], axis=1)
    d1 = np.linalg.norm(p1[i] - p1[j], axis=1)
    ok = d0 > 5
    return float(np.median(d1[ok] / d0[ok])) if ok.sum() >= 3 else None


def shift_scale(box, shift, s: float):
    """Move a box by the median flow and scale it about its centre by the expansion."""
    cx, cy = (box[0] + box[2]) / 2 + shift[0], (box[1] + box[3]) / 2 + shift[1]
    hw, hh = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def report(frame, rows: list, frame_idx: int, tid: int, t: dict, pts, fps: float) -> None:
    """Draw flow arrows and the TTC / closing-speed label for one track; append a CSV row."""
    if pts is not None:
        for a, b in zip(*pts, strict=True):
            cv2.arrowedLine(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 0), 1, tipLength=0.4)
    ema, dist = t["exp"], t["distance_m"]
    ttc = 1 / (fps * (ema - 1)) if ema is not None and ema > 1.0005 else None
    speed = dist / ttc if ttc and dist else None
    rows.append(
        {
            "frame": frame_idx,
            "track": tid,
            "class": t["class"],
            "distance_m": dist,
            "expansion": ema,
            "ttc_s": ttc,
            "closing_speed_mps": speed,
            "source": "flow" if t["missed"] else "detector",
        }
    )
    text = f"#{tid} TTC {ttc:.1f}s" + (f", {speed:.1f}m/s" if speed else "") if ttc else f"#{tid} not approaching"
    if t["missed"]:
        text += f" (flow, {dist:.1f}m)" if dist else " (flow)"
    x1, _, _, y2 = (int(v) for v in t["box"])
    cv2.putText(frame, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="video file")
    parser.add_argument("--weights", default="models/yolo26n_nav_v3.pt")
    parser.add_argument("--camera", default=str(DEFAULT_CONFIG))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="runs/flow")
    args = parser.parse_args()

    model = YOLO(args.weights)
    estimator = DistanceEstimator.from_config(args.camera)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out / f"{src.stem}_flow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rows, tracks, next_id, prev_gray = [], {}, count(), None  # tracks: id -> {box, class, exp, missed, distance_m, pts}

    for frame_idx in count():
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = annotate(frame, model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0], estimator)

        # 1. propagate every track with optical flow: update its expansion and predict its box
        for t in tracks.values():
            t["pts"] = track_points(prev_gray, gray, t["box"])
            if t["pts"] is None:
                continue
            p0, p1 = t["pts"]
            if (s := expansion(p0, p1)) is not None:
                t["exp"] = s if t["exp"] is None else EMA * s + (1 - EMA) * t["exp"]
            t["box"] = shift_scale(t["box"], np.median(p1 - p0, axis=0), s or 1.0)

        # 2. match detections to the predicted boxes (same class, highest IoU)
        new_tracks = {}
        for d in dets:
            match = max(
                (
                    (tid, iou(t["box"], d["box"]))
                    for tid, t in tracks.items()
                    if t["class"] == d["class"] and tid not in new_tracks
                ),
                key=lambda m: m[1],
                default=(None, 0),
            )
            tid = match[0] if match[1] > 0.3 else next(next_id)
            old = tracks.get(tid, {"exp": None, "pts": None})
            new_tracks[tid] = {
                "box": d["box"],
                "class": d["class"],
                "exp": old["exp"],
                "missed": 0,
                "distance_m": d["distance_m"],
            }
            report(frame, rows, frame_idx, tid, new_tracks[tid], old["pts"], fps)

        # 3. a track the detector missed this frame coasts on the flow prediction for a few frames
        for tid, t in tracks.items():
            if tid in new_tracks or t["pts"] is None or t["missed"] >= MAX_MISSED:
                continue
            new_tracks[tid] = {
                **t,
                "missed": t["missed"] + 1,
                "distance_m": estimator.estimate(t["class"], t["box"], frame.shape[1]),
            }
            x1, y1, x2, y2 = (int(v) for v in t["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS.get(t["class"], (0, 255, 0)), 1)
            report(frame, rows, frame_idx, tid, new_tracks[tid], t["pts"], fps)

        tracks, prev_gray = new_tracks, gray
        writer.write(frame)

    cap.release()
    writer.release()
    with open(out / f"{src.stem}_tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["frame"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out / f'{src.stem}_flow.mp4'} and {out / f'{src.stem}_tracks.csv'} ({len(rows)} detections)")


if __name__ == "__main__":
    main()
