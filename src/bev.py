"""Bird's-eye view (BEV): map the road and the detected objects onto a top-down metric grid.

Flat-road, level-camera model (camera height h, horizon row v0, focal f, centre column cx):
    ground point (X right, Z forward) -> pixel   u = cx + f * X / Z,   v = v0 + f * h / Z
This is a plane-to-plane homography, so the image is warped to the top-down view with
cv2.getPerspectiveTransform from 4 ground points. Objects are placed at their size-based
distance Z (src/distance.py) and lateral offset X = (u - cx) * Z / f. For cones and barriers,
which stand on the road, the box's bottom edge gives a second, independent distance
(ground contact) that is printed as a cross-check. The horizon row depends on how the camera
is mounted and tilted, so set --horizon-frac per camera (where lane lines meet).

Usage:
    python src/bev.py --source data/raw/bdd100k/val/images --limit 10 --skip-empty
Writes side-by-side images (annotated frame | BEV) to runs/bev/
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from detect import COLORS, IMG_EXTS, LABELS, annotate
from distance import DEFAULT_CONFIG, DistanceEstimator

GROUND_CLASSES = {"cone", "barrier"}  # objects whose box bottom touches the road


@dataclass
class GroundPlane:
    focal: float
    cx: float
    v0: float  # horizon row
    height_m: float

    @classmethod
    def for_image(
        cls, estimator: DistanceEstimator, config: Path, width: int, height: int, horizon_frac: float | None = None
    ) -> "GroundPlane":
        cam = yaml.safe_load(Path(config).read_text())["camera"]
        return cls(
            estimator.focal_for(width), width / 2, (horizon_frac or cam["horizon_frac"]) * height, cam["height_m"]
        )

    def to_image(self, x: float, z: float) -> tuple[float, float]:
        return self.cx + self.focal * x / z, self.v0 + self.focal * self.height_m / z

    def to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Road point under pixel (u, v); None at or above the horizon."""
        if v <= self.v0 + 1:
            return None
        z = self.focal * self.height_m / (v - self.v0)
        return (u - self.cx) * z / self.focal, z


@dataclass
class BevGrid:
    x_half: float = 10.0  # metres left/right of the camera
    z_min: float = 3.0
    z_max: float = 40.0
    ppm: float = 15.0  # BEV pixels per metre

    @property
    def size(self) -> tuple[int, int]:
        return int(2 * self.x_half * self.ppm), int((self.z_max - self.z_min) * self.ppm)

    def to_px(self, x: float, z: float) -> tuple[int, int]:
        return int((x + self.x_half) * self.ppm), int((self.z_max - z) * self.ppm)

    def homography(self, gp: GroundPlane) -> np.ndarray:
        """Image pixel -> BEV pixel, from the 4 corners of the grid's road rectangle."""
        corners = [
            (-self.x_half, self.z_min),
            (self.x_half, self.z_min),
            (self.x_half, self.z_max),
            (-self.x_half, self.z_max),
        ]
        src = np.float32([gp.to_image(x, z) for x, z in corners])
        dst = np.float32([self.to_px(x, z) for x, z in corners])
        return cv2.getPerspectiveTransform(src, dst)


def render_bev(frame, detections: list[dict], gp: GroundPlane, grid: BevGrid):
    """Top-down road view with range rings, the camera, and one marker per detection."""
    w, h = grid.size
    bev = cv2.warpPerspective(frame, grid.homography(gp), (w, h))
    bev = cv2.addWeighted(bev, 0.6, np.zeros_like(bev), 0.4, 0)  # dim so markers stand out
    ego = grid.to_px(0, grid.z_min)
    for r in range(10, int(grid.z_max) + 1, 10):
        cv2.ellipse(bev, ego, (int(r * grid.ppm),) * 2, 0, 180, 360, (200, 200, 200), 1)
        cv2.putText(
            bev, f"{r}m", (ego[0] + 4, grid.to_px(0, r)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )
    cv2.drawMarker(bev, (ego[0], h - 6), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)

    for d in detections:
        z = d["distance_m"]
        if z is None or not grid.z_min <= z <= grid.z_max:
            continue
        u = (d["box"][0] + d["box"][2]) / 2
        x = (u - gp.cx) * z / gp.focal
        if abs(x) > grid.x_half:
            continue
        px = grid.to_px(x, z)
        cv2.circle(bev, px, 7, COLORS.get(d["class"], (0, 255, 0)), -1)
        cv2.circle(bev, px, 7, (0, 0, 0), 1)
        text = f"{LABELS.get(d['class'], d['class'])} {z:.1f}m"
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        tx = px[0] + 9 if px[0] + 9 + tw < w else px[0] - 9 - tw  # keep the label inside the view
        cv2.putText(bev, text, (tx, px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bev


def process(frame, model, estimator, config, grid, conf, imgsz, horizon_frac=None):
    """Return (side-by-side image, detections) for one BGR frame."""
    h, w = frame.shape[:2]
    gp = GroundPlane.for_image(estimator, config, w, h, horizon_frac)
    raw = frame.copy()
    dets = annotate(frame, model(frame, conf=conf, imgsz=imgsz, verbose=False)[0], estimator)
    for d in dets:
        x1, _, x2, y2 = d["box"]
        g = gp.to_ground((x1 + x2) / 2, y2) if d["class"] in GROUND_CLASSES else None
        d["ground_distance_m"] = g[1] if g else None
    bev = render_bev(raw, dets, gp, grid)
    scale = bev.shape[0] / h
    left = cv2.resize(frame, (int(w * scale), bev.shape[0]))
    return np.hstack([left, bev]), dets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="image, folder of images, or video file")
    parser.add_argument("--weights", default="models/yolo26n_nav_v3.pt")
    parser.add_argument("--camera", default=str(DEFAULT_CONFIG))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="max images from a folder (0 = all)")
    parser.add_argument("--skip-empty", action="store_true", help="don't save images with no detections")
    parser.add_argument(
        "--horizon-frac",
        type=float,
        default=None,
        help="horizon row / image height for this camera (overrides configs/camera.yaml)",
    )
    parser.add_argument("--out", default="runs/bev")
    args = parser.parse_args()

    model = YOLO(args.weights)
    estimator = DistanceEstimator.from_config(args.camera)
    grid = BevGrid()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)

    if src.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        while (ok := cap.read())[0]:
            img, _ = process(ok[1], model, estimator, args.camera, grid, args.conf, args.imgsz, args.horizon_frac)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / f"{src.stem}_bev.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1]
                )
            writer.write(img)
        cap.release()
        if writer:
            writer.release()
        print(f"Saved {out / f'{src.stem}_bev.mp4'}")
        return

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    saved = 0
    for path in paths:
        img, dets = process(
            cv2.imread(str(path)), model, estimator, args.camera, grid, args.conf, args.imgsz, args.horizon_frac
        )
        if args.skip_empty and not dets:
            continue
        cv2.imwrite(str(out / path.name), img)
        saved += 1
        print(
            f"{path.name}: "
            + ", ".join(
                f"{d['class']} size {d['distance_m']:.1f}m"
                + (f" / ground {d['ground_distance_m']:.1f}m" if d["ground_distance_m"] else "")
                for d in dets
                if d["distance_m"]
            )
        )
        if args.limit and saved >= args.limit:
            break
    print(f"Saved {saved} BEV images to {out}")


if __name__ == "__main__":
    main()
