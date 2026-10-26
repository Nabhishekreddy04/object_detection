"""Select BDD100K driving scenes to use as background (negative) training images.

Background images have no labels: they teach the detector that cars, poles and
other street clutter are NOT cones / barriers / stop signs.

Every image must contain at least one vehicle, so they are "hard" negatives.
Two pools:
  * no-sign scenes:   no BDD100K `traffic sign` label at all
  * sign scenes:      contain traffic signs (--sign-frac of the total); teach that
                      other (often red) signs are not stop signs

BDD100K does not label cones/barriers and lumps stop signs into `traffic sign`, so
some scenes contain unlabelled targets. With --filter-models, every candidate is run
through those detectors and rejected if any of them sees a cone / barrier / stop
sign at >= --filter-conf (deliberately low, to reject anything doubtful).

Output (hard links, no extra disk space):
    <out>/train/images/*.jpg

Usage:
    python src/data/select_bdd_backgrounds.py                    # v2 selection
    python src/data/select_bdd_backgrounds.py --out data/raw/bdd_backgrounds_v3 --sign-frac 0.5 \
        --filter-models runs/train/yolo26n_nav_v1_1epoch/weights/best.pt \
        runs/train/yolo26n_nav_v2/weights/best.pt models/yolo26n.pt
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

VEHICLES = {"car", "truck", "bus"}
TARGETS = {"cone", "barrier", "stop_sign", "stop sign"}  # "stop sign" = COCO class name


def make_filter(model_paths: list[str], conf: float):
    """Return is_clean(image_path) -> True if no model detects a target class."""
    if not model_paths:
        return lambda path: True
    from ultralytics import YOLO

    models = [YOLO(p) for p in model_paths]

    def is_clean(path: Path) -> bool:
        for m in models:
            r = m.predict(str(path), conf=conf, device="cpu", verbose=False)[0]
            if any(r.names[int(c)] in TARGETS for c in r.boxes.cls.tolist()):
                return False
        return True

    return is_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bdd", type=Path, default=Path("data/raw/bdd100k/train"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/bdd_backgrounds"))
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--sign-frac", type=float, default=0.0, help="fraction of scenes that contain traffic signs")
    parser.add_argument("--filter-models", nargs="*", default=[])
    parser.add_argument("--filter-conf", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ann = next((args.bdd / "annotations").glob("*.json"))
    print(f"Reading {ann} ...")
    pools = {"no_sign": [], "sign": []}
    for img in json.loads(ann.read_text()):
        cats = {lbl["category"] for lbl in img.get("labels") or []}
        if cats & VEHICLES:
            pools["sign" if "traffic sign" in cats else "no_sign"].append(img)

    is_clean = make_filter(args.filter_models, args.filter_conf)
    n_sign = round(args.n * args.sign_frac)
    chosen, rejected = [], Counter()
    for pool, quota in (("no_sign", args.n - n_sign), ("sign", n_sign)):
        candidates = pools[pool]
        random.Random(args.seed).shuffle(candidates)
        taken = 0
        for img in candidates:
            if taken == quota:
                break
            if is_clean(args.bdd / "images" / img["name"]):
                chosen.append(img)
                taken += 1
            else:
                rejected[pool] += 1
        print(f"{pool}: {len(candidates)} candidates, selected {taken}, rejected {rejected[pool]} by filter")

    img_dir = args.out / "train" / "images"
    if args.out.exists():
        shutil.rmtree(args.out)
    img_dir.mkdir(parents=True)
    for img in chosen:
        src = args.bdd / "images" / img["name"]
        try:
            os.link(src, img_dir / img["name"])
        except OSError:
            shutil.copy2(src, img_dir / img["name"])

    print(f"selected {len(chosen)} images")
    print("time of day:", dict(Counter(i["attributes"]["timeofday"] for i in chosen)))
    print("scene:", dict(Counter(i["attributes"]["scene"] for i in chosen)))


if __name__ == "__main__":
    main()
