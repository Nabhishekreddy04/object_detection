"""Run the detector on images / a folder / a video and annotate each box as "Cone, 1.5m".

Usage:
    python src/detect.py --source data/raw/bdd100k/val/images --limit 20
    python src/detect.py --source 0 --device mps                 # live camera (e.g. iPhone via Continuity Camera)
    python src/detect.py --source http://192.168.1.20:8081/video  # phone camera streamed over Wi-Fi
    python src/detect.py --list-cameras
    python src/detect.py --source video.mp4 --weights models/yolo26n_nav_v3.pt
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from distance import DEFAULT_CONFIG, DistanceEstimator, focal_from_fov

LABELS = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}
COLORS = {"cone": (0, 140, 255), "barrier": (0, 220, 220), "stop_sign": (40, 40, 230)}  # BGR
STREAM_PREFIXES = ("http://", "https://", "rtsp://")  # network camera streams, e.g. a phone camera app
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Stricter per-class thresholds. stop_sign at 0.7 (v3 val): false positives fall 27 -> 7
# while recall drops 0.81 -> 0.71; other red/diamond signs are the main confusion on BDD100K.
CLASS_CONF = {"stop_sign": 0.7}


def annotate(frame, result, estimator: DistanceEstimator) -> list[dict]:
    """Draw boxes + "Class, X.Xm" labels on frame in place; return the detections."""
    detections = []
    width = frame.shape[1]
    for box, cls_id, conf in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
    ):
        name = result.names[int(cls_id)]
        if conf < CLASS_CONF.get(name, 0.0):
            continue
        dist = estimator.estimate(name, box, width)
        label = LABELS.get(name, name) + (f", {dist:.1f}m" if dist is not None else "")
        detections.append({"class": name, "conf": conf, "box": box, "distance_m": dist})

        color = COLORS.get(name, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + base + 2)
        cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return detections


def list_cameras(max_index: int = 5) -> None:
    """Print the camera indices OpenCV can open, with their resolution."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            print(f"camera {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()


def run_live(model, estimator: DistanceEstimator, args) -> None:
    """Annotate a live camera feed in a window (q / Esc to quit); optionally record it."""
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(
            f"Camera {args.source} not available (try --list-cameras; allow camera access for your terminal)"
        )
    writer, fps, t_prev = None, 0.0, time.perf_counter()
    while (ok := cap.read())[0]:
        frame = ok[1]
        result = model(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        annotate(frame, result, estimator)
        now = time.perf_counter()
        fps = 1 / (now - t_prev) if not fps else 0.9 * fps + 0.1 / (now - t_prev)
        t_prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if args.save:
            if writer is None:
                path = Path(args.out) / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, frame.shape[1::-1])
                print(f"Recording to {path}")
            writer.write(frame)
        cv2.imshow("detect - q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", help="image, folder, video file, camera index (0, 1, ...) or stream URL (http/rtsp)"
    )
    parser.add_argument("--list-cameras", action="store_true", help="print available camera indices and exit")
    parser.add_argument("--device", default=None, help="cpu, mps or 0 (CUDA); default: Ultralytics' choice")
    parser.add_argument("--hfov", type=float, default=None, help="camera horizontal FOV in degrees (overrides config)")
    parser.add_argument("--save", action="store_true", help="live camera: record the annotated stream to --out")
    parser.add_argument("--weights", default="models/yolo26n_nav_v3.pt")
    parser.add_argument("--camera", default=str(DEFAULT_CONFIG))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="max images from a folder (0 = all)")
    parser.add_argument("--skip-empty", action="store_true", help="don't save images with no detections")
    parser.add_argument("--out", default="runs/detect/annotated")
    args = parser.parse_args()
    if args.list_cameras:
        list_cameras()
        return
    if not args.source:
        parser.error("--source is required")

    model = YOLO(args.weights)
    estimator = DistanceEstimator.from_config(args.camera)
    if args.hfov:
        estimator.focal_px = focal_from_fov(estimator.ref_width, args.hfov)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.source.isdigit() or args.source.startswith(STREAM_PREFIXES):
        run_live(model, estimator, args)
        return
    src = Path(args.source)

    if src.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(out / f"{src.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        while (ok := cap.read())[0]:
            frame = ok[1]
            annotate(
                frame, model(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0], estimator
            )
            writer.write(frame)
        cap.release()
        writer.release()
        print(f"Saved {out / f'{src.stem}_annotated.mp4'}")
        return

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    saved = 0
    for path in paths:
        frame = cv2.imread(str(path))
        dets = annotate(frame, model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0], estimator)
        if args.skip_empty and not dets:
            continue
        cv2.imwrite(str(out / path.name), frame)
        saved += 1
        print(f"{path.name}: " + ", ".join(f"{d['class']} {d['distance_m']:.1f}m" for d in dets if d["distance_m"]))
        if args.limit and saved >= args.limit:
            break
    print(f"Saved {saved} annotated images to {out}")


if __name__ == "__main__":
    main()
