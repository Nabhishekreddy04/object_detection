"""Merge the raw YOLO-format datasets into one 3-class dataset.

Final classes: 0 = cone, 1 = barrier, 2 = stop_sign

Each source's own splits are pooled and re-split (train/val) per source, grouped
by original image so Roboflow's augmented copies never leak across splits.
Polygon labels are converted to bounding boxes. Images are hard-linked (no
extra disk space) into:
    data/merged/{train,val}/{images,labels}
    data/merged/data.yaml

Usage:
    python src/data/merge_datasets.py                                        # v1
    python src/data/merge_datasets.py --backgrounds --out data/merged_v2     # v2: + BDD100K backgrounds
    python src/data/merge_datasets.py --backgrounds bdd_backgrounds_v3 --out data/merged_v3  # v3: filtered backgrounds
    python src/data/merge_datasets.py --backgrounds bdd_backgrounds_v3 --test-frac 0.15 --dedup-iou 0.9 \
        --out data/merged_v4                                   # v4: + held-out test split, de-duplicated boxes
"""

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["cone", "barrier", "stop_sign"]

# source folder -> ({source class id: merged class id}, max images or None)
# Classes missing from a mapping are dropped.
SOURCES = {
    "coco_stop": ({0: 2}, None),
    "roboflow_cones": ({0: 0, 1: 0, 2: 0}, 3000),  # all three are cones
    "roboflow_barriers": ({0: 1, 3: 1, 4: 0}, None),  # 2 = unclear, 5 = car
}
# Unlabelled driving scenes (from select_bdd_backgrounds.py), added with --backgrounds
BACKGROUND_SOURCE = "bdd_backgrounds"  # default folder; override with --backgrounds <folder>
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = 0.002  # drop boxes narrower/shorter than this (normalized)


def box_iou(a: tuple, b: tuple) -> float:
    """IoU of two (cx, cy, w, h) normalized boxes."""
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def parse_label(path: Path, mapping: dict, dedup_iou: float | None = None) -> list[str]:
    """Read a YOLO box/polygon label file and return remapped box lines.

    With dedup_iou, a box overlapping an already kept box of the same class by more than
    that IoU is dropped (near-duplicate annotations found in the EDA).
    """
    out, kept = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) not in mapping:
            continue
        cls, coords = mapping[int(parts[0])], [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
        else:  # polygon: x1 y1 x2 y2 ...
            xs, ys = coords[0::2], coords[1::2]
            x0, x1 = max(min(xs), 0.0), min(max(xs), 1.0)
            y0, y1 = max(min(ys), 0.0), min(max(ys), 1.0)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        if dedup_iou and any(c == cls and box_iou((cx, cy, w, h), k) > dedup_iou for c, k in kept):
            continue
        kept.append((cls, (cx, cy, w, h)))
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def collect(
    src: Path, mapping: dict, keep_empty: bool = False, dedup_iou: float | None = None
) -> dict[str, list[tuple[Path, list[str]]]]:
    """Return {group key: [(image, label lines), ...]}; unlabelled images only if keep_empty."""
    groups = defaultdict(list)
    for img in sorted(src.glob("*/images/*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        lines = parse_label(lbl, mapping, dedup_iou) if lbl.exists() else []
        if lines or keep_empty:
            groups[img.name.split(".rf.")[0]].append((img, lines))
    return groups


def take(keys: list[str], groups: dict, limit: int) -> list[str]:
    """Take whole groups in order until about `limit` images are selected."""
    out, n = [], 0
    for key in keys:
        if n >= limit:
            break
        out.append(key)
        n += len(groups[key])
    return out


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/merged"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.0, help="held-out test split (0 = none, as in v1-v3)")
    parser.add_argument(
        "--dedup-iou", type=float, default=None, help="drop same-class boxes overlapping above this IoU"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backgrounds",
        nargs="?",
        const=BACKGROUND_SOURCE,
        default=None,
        help=f"add unlabelled images from data/raw/<folder> (default folder: {BACKGROUND_SOURCE})",
    )
    args = parser.parse_args()

    sources = dict(SOURCES)
    if args.backgrounds:
        sources[args.backgrounds] = ({}, None)

    if args.out.exists():
        shutil.rmtree(args.out)
    split_names = ("train", "val", "test") if args.test_frac else ("train", "val")
    for split in split_names:
        for kind in ("images", "labels"):
            (args.out / split / kind).mkdir(parents=True)

    rng = random.Random(args.seed)
    stats = defaultdict(Counter)
    for name, (mapping, cap) in sources.items():
        groups = collect(args.raw / name, mapping, keep_empty=name == args.backgrounds, dedup_iou=args.dedup_iou)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_frac))
        n_test = round(len(keys) * args.test_frac)
        splits = {"val": keys[:n_val], "train": keys[n_val + n_test :]}
        fracs = {"val": args.val_frac, "train": 1 - args.val_frac - args.test_frac}
        if n_test:
            splits["test"], fracs["test"] = keys[n_val : n_val + n_test], args.test_frac
        if cap:  # cap total images, keeping the same split ratios
            splits = {s: take(k, groups, round(cap * fracs[s])) for s, k in splits.items()}

        for split, split_keys in splits.items():
            for key in split_keys:
                for img, lines in groups[key]:
                    stem = f"{name}__{img.stem}"
                    link(img, args.out / split / "images" / (stem + img.suffix.lower()))
                    (args.out / split / "labels" / (stem + ".txt")).write_text("".join(f"{line}\n" for line in lines))
                    stats[split][f"{name} images"] += 1
                    for line in lines:
                        stats[split][CLASSES[int(line.split()[0])]] += 1

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\n"
        + ("test: test/images\n" if args.test_frac else "")
        + f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    for split in split_names:
        print(f"[{split}]")
        for k, v in sorted(stats[split].items()):
            print(f"  {k:28s} {v}")
    print(f"Done -> {args.out}/data.yaml")


if __name__ == "__main__":
    main()
