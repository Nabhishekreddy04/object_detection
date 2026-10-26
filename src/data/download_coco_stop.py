"""Download only the COCO 2017 images that contain stop signs, in YOLO format.

Output layout (same as a Roboflow YOLOv8 export):
    data/raw/coco_stop/
        train/images, train/labels
        valid/images, valid/labels
        data.yaml

Usage:
    python src/data/download_coco_stop.py [--out data/raw/coco_stop] [--workers 16]
"""

import argparse
import json
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}2017/{file_name}"
SPLITS = {"train": "train", "val": "valid"}  # COCO split -> output folder
CATEGORY = "stop sign"


def download(url: str, dest: Path, retries: int = 10, timeout: int = 30) -> None:
    """Download with a socket timeout, resuming partial files on retry."""
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "ab" if done else "wb") as f:
                if done and resp.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except OSError as e:
            print(f"  retry {attempt}/{retries} for {Path(url).name}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {url}")


def load_annotations(cache: Path, split: str) -> dict:
    zip_path = cache / "annotations_trainval2017.zip"
    member = f"annotations/instances_{split}2017.json"
    json_path = cache / Path(member).name
    if not json_path.exists():
        print("Downloading COCO annotations (~240 MB) ...")
        download(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in (f"annotations/instances_{s}2017.json" for s in SPLITS):
                (cache / Path(name).name).write_bytes(zf.read(name))
    return json.loads(json_path.read_text())


def to_yolo(bbox: list, w: int, h: int) -> str:
    x, y, bw, bh = bbox  # COCO: top-left x, y, width, height in pixels
    return f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def export_split(coco: dict, split: str, out_dir: Path, workers: int) -> int:
    cat_id = next(c["id"] for c in coco["categories"] if c["name"] == CATEGORY)
    boxes = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == cat_id and not ann.get("iscrowd", 0):
            boxes[ann["image_id"]].append(ann["bbox"])

    images = {img["id"]: img for img in coco["images"] if img["id"] in boxes}
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    def fetch(img: dict) -> None:
        name = img["file_name"]
        lines = [to_yolo(b, img["width"], img["height"]) for b in boxes[img["id"]]]
        (lbl_dir / Path(name).with_suffix(".txt")).write_text("\n".join(lines) + "\n")
        download(IMG_URL.format(split=split, file_name=name), img_dir / name)

    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(fetch, images.values()), 1):
            if i % 200 == 0 or i == len(images):
                print(f"  {split}: {i}/{len(images)} images")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw/coco_stop"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cache = args.out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for split, folder in SPLITS.items():
        n = export_split(load_annotations(cache, split), split, args.out / folder, args.workers)
        print(f"{folder}: {n} images with stop signs")

    (args.out / "data.yaml").write_text("train: train/images\nval: valid/images\nnc: 1\nnames: ['stop_sign']\n")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
