"""Produce edge-optimized variants of a trained detector.

Variants written to --out (default models/optimized/):
    <name>_fp32.onnx           ONNX export (portable runtime: ONNX Runtime on CPU / ARM boards)
    <name>_int8.onnx           post-training static INT8 quantization of the ONNX model
    <name>_fp32_320.onnx       same model at 320x320 input (4x fewer pixels)
    <name>_pruned{P}.pt        L1 unstructured pruning of P% of every conv layer's weights

FP16 needs no separate file: it is a runtime flag (half=True) on GPU.

Usage:
    python src/optimize.py --weights models/yolo26n_nav_v3.pt
"""

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.nn.utils import prune
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def export_onnx(weights: Path, imgsz: int, dst: Path) -> Path:
    """Export with Ultralytics (graph simplified, metadata such as class names embedded)."""
    path = Path(YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True))
    shutil.move(path, dst)
    return dst


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds real training images, preprocessed exactly like inference, to the quantizer."""

    def __init__(self, images: list[Path], input_name: str, imgsz: int):
        self.images, self.input_name = iter(images), input_name
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False)

    def get_next(self):
        path = next(self.images, None)
        if path is None:
            return None
        img = self.letterbox(image=cv2.imread(str(path)))
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        return {self.input_name: np.ascontiguousarray(x)}


def quantize_int8(fp32: Path, dst: Path, calib_dir: Path, n_calib: int, imgsz: int, keep_head_fp32: bool) -> Path:
    """Static INT8 quantization: weights per-channel int8, activations uint8 (QDQ format).

    Activation ranges come from running real images through the model (calibration).
    The detection head can be kept in FP32: its box-regression outputs are very
    sensitive to rounding, and it is a small share of the compute.
    """
    prep = dst.with_name(dst.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prep))

    model = onnx.load(str(prep))
    input_name = model.graph.input[0].name
    head = max(int(m[1]) for n in model.graph.node if (m := re.match(r"/model\.(\d+)/", n.name)))
    exclude = [n.name for n in model.graph.node if keep_head_fp32 and n.name.startswith(f"/model.{head}/")]

    images = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    images = images[:: max(1, len(images) // n_calib)][:n_calib]  # evenly spread across sources
    quantize_static(
        str(prep),
        str(dst),
        ImageCalibrationReader(images, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
    )
    prep.unlink()

    # Copy Ultralytics metadata (class names, stride, imgsz) so YOLO() can load the INT8 model
    q = onnx.load(str(dst))
    q.metadata_props.extend(onnx.load(str(fp32)).metadata_props)
    onnx.save(q, str(dst))
    print(f"INT8: {len(exclude)} head nodes kept in FP32, calibrated on {len(images)} images")
    return dst


def prune_model(weights: Path, amount: float, dst: Path) -> Path:
    """Zero the `amount` fraction of smallest-magnitude weights in every Conv2d (L1 unstructured)."""
    yolo = YOLO(weights)
    convs = [m for m in yolo.model.modules() if isinstance(m, torch.nn.Conv2d)]
    for conv in convs:
        prune.l1_unstructured(conv, name="weight", amount=amount)
        prune.remove(conv, "weight")  # make the zeros permanent
    zeros = sum((c.weight == 0).sum().item() for c in convs)
    total = sum(c.weight.numel() for c in convs)
    yolo.save(dst)
    print(f"Pruned {amount:.0%}: {zeros / total:.1%} of conv weights are zero")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("models/yolo26n_nav_v3.pt"))
    parser.add_argument("--name", default="yolo26n_nav")
    parser.add_argument("--out", type=Path, default=Path("models/optimized"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/merged_v3/train/images"))
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--quantize-head", action="store_true", help="also quantize the detection head to INT8")
    parser.add_argument("--prune", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = export_onnx(args.weights, 640, args.out / f"{args.name}_fp32.onnx")
    export_onnx(args.weights, 320, args.out / f"{args.name}_fp32_320.onnx")
    quantize_int8(fp32, args.out / f"{args.name}_int8.onnx", args.calib_dir, args.n_calib, 640, not args.quantize_head)
    for amount in args.prune:
        prune_model(args.weights, amount, args.out / f"{args.name}_pruned{round(amount * 100)}.pt")

    for f in sorted(args.out.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
