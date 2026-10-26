"""Fine-tune a COCO-pretrained YOLO model on the merged cone/barrier/stop_sign dataset.

Usage:
    python src/train.py                         # full run (v3 dataset), on CUDA / Apple GPU / CPU
    python src/train.py --epochs 1 --name test  # quick smoke test
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/yolo26n.pt", help="pretrained weights to start from")
    parser.add_argument("--data", default="data/merged_v3/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)  # 8 GB RAM: keep <= 16
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15, help="early-stop after N epochs without improvement")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--name", default="yolo26n_nav_v3")
    args = parser.parse_args()

    model = YOLO(args.model)  # COCO-pretrained backbone + head -> transfer learning
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        project=str(Path("runs/train").resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
    )


if __name__ == "__main__":
    main()
