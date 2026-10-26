"""Benchmark model variants: accuracy (mAP), speed (FPS) on CPU / GPU, and size.

FPS is measured end-to-end per image (preprocess + inference + postprocess,
batch 1), which is what a robot running frame by frame actually experiences.

Usage:
    python src/benchmark.py --name yolo26n_nav --weights models/yolo26n_nav_v3.pt
Writes results/benchmark_<tag>.csv and .md
"""

import argparse
import csv
import gzip
import platform
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def variants(weights: Path, opt_dir: Path, name: str) -> list[dict]:
    """(label, file, device, half, imgsz) for every variant that exists on this machine."""
    gpus = [d for d, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]
    rows = [dict(label="PyTorch FP32 (baseline)", path=weights, device="cpu")]
    for gpu in gpus:
        rows.append(dict(label="PyTorch FP32", path=weights, device=gpu))
        rows.append(dict(label="PyTorch FP16", path=weights, device=gpu, half=True))
    rows += [
        dict(label="ONNX Runtime FP32", path=opt_dir / f"{name}_fp32.onnx", device="cpu"),
        dict(label="ONNX Runtime INT8 (static PTQ)", path=opt_dir / f"{name}_int8.onnx", device="cpu"),
        dict(label="ONNX Runtime FP32 @320px", path=opt_dir / f"{name}_fp32_320.onnx", device="cpu", imgsz=320),
    ]
    rows += [
        dict(label=f"PyTorch pruned {p.stem.rsplit('pruned', 1)[1]}%", path=p, device="cpu")
        for p in sorted(opt_dir.glob(f"{name}_pruned*.pt"), key=lambda p: int(p.stem.rsplit("pruned", 1)[1]))
    ]
    return [dict(half=False, imgsz=640) | r for r in rows if Path(r["path"]).exists()]


def measure_fps(model: YOLO, images: list[Path], device: str, half: bool, imgsz: int, warmup: int = 10) -> float:
    frames = [cv2.imread(str(p)) for p in images]
    kw = dict(device=device, half=half, imgsz=imgsz, verbose=False)
    for f in frames[:warmup]:
        model.predict(f, **kw)
    start = time.perf_counter()
    for f in frames:
        model.predict(f, **kw)  # results are copied to CPU, so GPU work is finished when this returns
    return len(frames) / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("models/yolo26n_nav_v3.pt"))
    parser.add_argument("--name", default="yolo26n_nav")
    parser.add_argument("--opt-dir", type=Path, default=Path("models/optimized"))
    parser.add_argument("--data", default="data/merged_v3/data.yaml")
    parser.add_argument("--fps-images", type=int, default=200)
    parser.add_argument("--tag", default=platform.machine())
    args = parser.parse_args()

    val_dir = Path(args.data).parent / "val" / "images"
    images = sorted(val_dir.iterdir())[:: max(1, len(list(val_dir.iterdir())) // args.fps_images)][: args.fps_images]

    results = []
    for v in variants(args.weights, args.opt_dir, args.name):
        print(f"\n=== {v['label']} on {v['device']} ===")
        model = YOLO(v["path"], task="detect")
        onnx = str(v["path"]).endswith(".onnx")
        metrics = model.val(
            data=args.data,
            imgsz=v["imgsz"],
            device=v["device"],
            half=v["half"],
            batch=1 if onnx else 16,
            rect=False,
            plots=False,
            verbose=False,
        )
        fps = measure_fps(YOLO(v["path"], task="detect"), images, v["device"], v["half"], v["imgsz"])
        raw = Path(v["path"]).read_bytes()
        results.append(
            {
                "variant": v["label"],
                "device": v["device"].upper(),
                "imgsz": v["imgsz"],
                "mAP50": round(metrics.box.map50, 3),
                "mAP50-95": round(metrics.box.map, 3),
                "FPS": round(fps, 1),
                "latency_ms": round(1000 / fps, 1),
                "size_MB": round(len(raw) / 1e6, 2),
                "gzip_MB": round(len(gzip.compress(raw)) / 1e6, 2),
            }
        )
        print(results[-1])

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / f"benchmark_{args.tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    header = "| " + " | ".join(results[0]) + " |\n|" + "---|" * len(results[0]) + "\n"
    table = header + "".join("| " + " | ".join(str(x) for x in r.values()) + " |\n" for r in results)
    hw = f"{platform.processor() or platform.machine()}, torch {torch.__version__}"
    (out / f"benchmark_{args.tag}.md").write_text(f"Hardware: {args.tag} ({hw})\n\n{table}")
    print("\n" + table)


if __name__ == "__main__":
    main()
