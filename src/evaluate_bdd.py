"""Compare models on unseen BDD100K driving scenes (no cone/barrier labels exist there).

For each model: detection counts per class, plus a contact sheet with a zoomed crop of
every detection, so true and false positives can be judged by eye.

Usage:
    python src/evaluate_bdd.py --models v1=runs/train/yolo26n_nav_v1_1epoch/weights/best.pt \
        v2=runs/train/yolo26n_nav_v2/weights/best.pt v3=runs/train/yolo26n_nav_v3/weights/best.pt
Writes results/bdd_eval/<tag>_crops.jpg and results/bdd_eval/summary.md
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detect import CLASS_CONF

TILE, COLS = 200, 8


def crop_tile(img, box, text: str):
    """Square crop around a box (with context), box drawn, caption on top."""
    x1, y1, x2, y2 = box
    pad = max(x2 - x1, y2 - y1) * 0.6 + 20
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(max(cx - pad, 0)), int(max(cy - pad, 0))
    X2, Y2 = int(min(cx + pad, img.shape[1])), int(min(cy + pad, img.shape[0]))
    crop = img[Y1:Y2, X1:X2].copy()
    cv2.rectangle(crop, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 255, 0), 2)
    tile = cv2.resize(crop, (TILE, TILE))
    cv2.rectangle(tile, (0, 0), (TILE, 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, help="tag=path/to/weights.pt")
    parser.add_argument("--images", type=Path, default=Path("data/raw/bdd100k/val/images"))
    parser.add_argument("--n", type=int, default=500, help="evenly spaced scenes to test")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=Path("results/bdd_eval"))
    args = parser.parse_args()

    all_imgs = sorted(args.images.glob("*.jpg"))
    imgs = all_imgs[:: max(1, len(all_imgs) // args.n)][: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, weights in (m.split("=", 1) for m in args.models):
        model = YOLO(weights)
        counts, tiles, with_det = Counter(), [], 0
        for path in imgs:
            img = cv2.imread(str(path))
            r = model.predict(img, conf=args.conf, device="cpu", verbose=False)[0]  # CPU: MPS breaks on long runs
            found = False
            for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=True):
                name = r.names[int(c)]
                if s < CLASS_CONF.get(name, 0.0):
                    continue
                counts[name] += 1
                found = True
                tiles.append(crop_tile(img, box, f"{len(tiles)} {name} {s:.2f}"))
            with_det += found
        if tiles:
            tiles += [np.zeros((TILE, TILE, 3), np.uint8)] * (-len(tiles) % COLS)
            sheet = np.vstack([np.hstack(tiles[i : i + COLS]) for i in range(0, len(tiles), COLS)])
            cv2.imwrite(str(args.out / f"{tag}_crops.jpg"), sheet)
        rows.append((tag, sum(counts.values()), with_det, counts))
        print(f"{tag}: {sum(counts.values())} detections in {with_det}/{len(imgs)} scenes {dict(counts)}")

    lines = [
        f"BDD100K val, {len(imgs)} evenly spaced scenes, "
        f"conf >= {args.conf} (stop_sign >= {CLASS_CONF.get('stop_sign')})\n",
        "| model | detections | scenes with a detection | cone | barrier | stop_sign |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {t} | {n} | {s} | {c['cone']} | {c['barrier']} | {c['stop_sign']} |" for t, n, s, c in rows]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
