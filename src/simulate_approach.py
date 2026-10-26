"""Make a synthetic "driving towards an object" video from one still image, with ground-truth TTC.

Approaching an object at constant speed v from distance Z0 magnifies its image by
k(t) = Z0 / (Z0 - v t), so zooming the still image about the object's centre by k(t)
reproduces the object's apparent expansion exactly (the background is not physically
correct, but src/flow.py only measures points inside the object's box).
Ground truth: TTC(t) = (Z0 - v t) / v.

Usage:
    python src/simulate_approach.py --image scene.jpg --center 640 400 --z0 20 --speed 8
Writes <out>.mp4 and <out>_gt.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--center", type=float, nargs=2, required=True, help="object centre (x y) in pixels")
    parser.add_argument("--z0", type=float, default=20.0, help="starting distance (m)")
    parser.add_argument("--speed", type=float, default=8.0, help="approach speed (m/s)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("runs/flow/simulated_approach"))
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    h, w = img.shape[:2]
    cx, cy = args.center
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    rows = []
    for i in range(int(args.seconds * args.fps)):
        z = args.z0 - args.speed * i / args.fps
        k = args.z0 / z
        m = np.float32([[k, 0, cx - k * cx], [0, k, cy - k * cy]])  # zoom by k about (cx, cy)
        writer.write(cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        rows.append({"frame": i, "true_distance_m": round(z, 3), "true_ttc_s": round(z / args.speed, 3)})
    writer.release()
    with open(f"{args.out}_gt.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {args.out.with_suffix('.mp4')} ({len(rows)} frames) and {args.out}_gt.csv")


if __name__ == "__main__":
    main()
