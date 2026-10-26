# Project Log — Object Detection + Distance Estimation for Robotics Navigation

This file records every step taken in the project, in order, with the reason
behind each decision. It is updated as the project progresses.

## Project goals (from the assignment)

1. **Object detection** — detect cones, barriers and stop signs using transfer
   learning on a pretrained model (BDD100K dataset suggested; other datasets allowed).
2. **Distance estimation** — estimate each object's distance from the camera
   using geometry (focal length, real object size, pixel size) and annotate
   boxes as e.g. `Cone, 1.5m`.
3. **Edge optimization** — try quantization / pruning / lightweight backbones,
   record FPS on CPU and GPU, and show before-vs-after results.
4. *(Optional)* Epipolar geometry, homography (bird's-eye view), optical flow.

---

## Step 1 — Choosing the datasets

**Problem:** BDD100K's detection labels are `pedestrian, rider, car, truck, bus,
train, motorcycle, bicycle, traffic light, traffic sign`. It has **no cone or
barrier class**, and stop signs are mixed into the generic `traffic sign` class.
So BDD100K alone cannot train the three required classes.

**Decision:** combine several datasets, one per class:

| Class | Source | Why |
|---|---|---|
| Stop sign | COCO 2017 (`stop sign` category) | Clean, well-labelled, free, no login; ~1.8k images |
| Cone | Roboflow Universe — [cones-mapit v2](https://universe.roboflow.com/arrows-d3g7d/cones-mapit/dataset/2) (CC BY 4.0) | Large, already in YOLO format |
| Barrier | Roboflow Universe — [barriers-ctzjn v2](https://universe.roboflow.com/evoiz/barriers-ctzjn/dataset/2) (CC BY 4.0) | Road barriers/barricades, already in YOLO format |
| Demo / test scenes | BDD100K | Real driving footage to run the final model on (assignment requirement) |
| Camera intrinsics (optional) | nuImages mini | Real calibrated focal lengths to justify the distance formula |

All raw data lives in `data/raw/<source>/` (not committed to git).

## Step 2 — Choosing the label format and model family

**Decision:** YOLO label format (one `.txt` per image: `class cx cy w h`,
normalized 0–1), exported from Roboflow as "YOLOv8".

**Why:** YOLOv8, YOLO11 and YOLO26 all use the identical label format, so the
data does not lock us into a model version.

**Model choice:** **YOLO26n** as the main model.
- Pretrained on COCO → transfer learning (COCO already contains `stop sign`).
- Designed for edge devices (NMS-free, faster CPU inference) → fits Part 3.
- Nano (`n`) size = smallest/fastest, appropriate for limited hardware.
- YOLOv8n / YOLO11n kept as comparison baselines for the optimization table.

## Step 3 — Downloading COCO stop signs (`src/data/download_coco_stop.py`)

**What it does:** downloads the COCO 2017 annotation file, finds every image
containing a `stop sign`, downloads **only those images**, and writes YOLO labels.

**Why this approach:**
- Full COCO is ~20 GB; we only need ~1.8k images (~290 MB).
- Uses only the Python standard library — the common alternative (`fiftyone`)
  is heavy and needs MongoDB.
- Crowd annotations (`iscrowd=1`) are skipped since they are not single objects.

**Problem hit:** the first run hung forever at 97 MB because Python's
`urlretrieve` has no timeout. **Fix:** a 30 s socket timeout plus up to 10
retries that resume partial downloads via HTTP `Range` requests.

**Result:** train 1,734 images / 1,983 boxes; val 69 images / 75 boxes.

```bash
python3 src/data/download_coco_stop.py
```

## Step 4 — Inspecting the Roboflow datasets

Roboflow datasets often have messy class names, so each was checked before merging.

- **Cones:** class names were leftover version strings
  (`'cone - v2 2024-06-17 ...'` etc.) but all three classes are cones →
  all mapped to `cone`.
- **Barriers:** classes were named `'0'`–`'5'` with no meaning given. Sample
  images were viewed to identify them:
  - `0` → barrier (vast majority of labels)
  - `3` → barrier (road-closed barricade)
  - `4` → cone
  - `5` → car → **dropped**
  - `2` → a single unclear object → **dropped**
- Both datasets mix **bounding boxes and polygon (segmentation) labels** →
  polygons must be converted to boxes.
- Some label files have no trailing newline, so labels must be parsed per
  file (not by concatenating files).

## Step 5 — Merging into one 3-class dataset (`src/data/merge_datasets.py`)

**Final classes:** `0: cone`, `1: barrier`, `2: stop_sign`

**What it does and why:**

| Action | Reason |
|---|---|
| Remap each source's classes to the 3 final classes; drop the rest | One consistent label set for training |
| Convert polygons → bounding boxes (min/max of points, clipped to the image) | Detection needs boxes |
| Drop boxes smaller than 0.2% of the image side | Removes degenerate/invalid labels |
| Pool each source's train/valid/test and **re-split 85/15** | COCO's val had only 69 stop-sign images and the cone valid split had only 30 — too few to evaluate reliably |
| Keep all augmented copies of one original photo in the **same split** (grouped by Roboflow's `.rf.` filename prefix) | Prevents near-duplicate images leaking from train into val, which would inflate accuracy |
| Cap cones at ~3,000 images | Cones (~8.7k images) would otherwise dominate training |
| Hard-link images instead of copying | No extra disk space |
| Fixed random seed | Re-running gives the identical split |

**Bug fixed:** the first run kept 7,370 cone images because the cap counted
photo *groups* (each ≈3 augmented images) instead of images. Fixed to count images.

**Result** (`data/merged/data.yaml`):

| | Train | Val |
|---|---|---|
| Images | 4,988 | 877 |
| Cone boxes | 6,157 | 1,272 |
| Barrier boxes | 1,240 | 234 |
| Stop-sign boxes | 1,746 | 312 |

Verified: every image has a label file, and no original photo appears in both splits.

**Known limitation:** barrier is the smallest class (~1k source images), so it
is expected to have the lowest mAP. Adding another barrier dataset would help.

```bash
python3 src/data/merge_datasets.py
```

## Step 6 — Environment setup and 1-epoch smoke test (`src/train.py`)

**Environment:** `.venv` with Python 3.10.13 (system Python 3.14 is too new for
reliable PyTorch/Ultralytics support). Versions pinned in `requirements.txt`
(torch 2.14.1, ultralytics 8.4.188). Hardware: Apple M3, 8 GB RAM; training
runs on the M3 GPU through PyTorch's MPS backend.

```bash
~/.pyenv/versions/3.10.13/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Training script choices:**

| Setting | Value | Reason |
|---|---|---|
| Start weights | `models/yolo26n.pt` (COCO-pretrained) | Transfer learning; COCO already knows `stop sign` |
| Image size | 640 | YOLO default, keeps small/distant objects visible |
| Batch | 16 | Fits in 8 GB unified memory (~5.2 GB used) |
| Early stopping | patience 15 | Stops automatically when val mAP stops improving |
| Seed | 0 | Reproducible runs |

**Smoke test (1 epoch, before any tuning):** confirmed the pipeline works end-to-end.

| Class | mAP@50 | mAP@50-95 |
|---|---|---|
| all | 0.472 | 0.263 |
| cone | 0.456 | 0.202 |
| barrier | 0.298 | 0.133 |
| stop_sign | 0.662 | 0.453 |

Stop sign is already strongest after one epoch because the pretrained COCO
model knew that class; barrier is weakest, as expected from Step 5.

**Finding:** one epoch took **~20 minutes** on the M3 (3.9 s per batch), so
50 epochs would take ~17 hours — too long for the deadline.

**Bug fixed:** Ultralytics nested a relative `project` path under
`runs/detect/`; the script now passes an absolute path so outputs land in
`runs/train/<name>/`.

## Step 7 — Moving full training to Google Colab (`notebooks/train_colab.ipynb`)

**Decision:** train the full run on a free Colab NVIDIA T4 GPU instead of the M3.

**Why:** the M3 needed ~17 hours for 50 epochs. A T4 is expected to be several
times faster (to be confirmed from the run log), and it fits a larger batch (32)
in its 15 GB of memory.
It also gives CUDA numbers for the CPU/GPU FPS comparison in Part 3.

**How:**
- The merged dataset is zipped (`colab_upload/merged.zip`, 453 MB, 5,865 images)
  and uploaded with `train.py` to `MyDrive/object_detection/` on Google Drive.
- The notebook installs the same Ultralytics version (8.4.188), unzips the data
  to Colab's local disk (faster than reading from Drive), and rewrites the
  `path:` line of `data.yaml` for the new location.
- Training outputs are written directly to Google Drive, and a resume cell is
  provided, so a Colab disconnect does not lose progress.

## Step 8 — Distance estimation (`src/distance.py`, `src/detect.py`, `configs/camera.yaml`)

**Method — pinhole camera model (similar triangles):**

```
Z = f_px × H_real / h_px
```

An object of real height `H_real` at distance `Z` appears `h_px = f_px × H_real / Z`
pixels tall, so measuring the box height gives the distance.

**Focal length.** BDD100K does not publish per-image camera intrinsics, so the
focal length is derived from an assumed dash-cam horizontal field of view:

```
f_px = (W / 2) / tan(HFOV / 2) = 640 / tan(32.5°) ≈ 1005 px   (W = 1280, HFOV = 65°)
```

`f_px` is scaled with image width, so the estimator works at any resolution.
`calibrate_focal()` recovers `f_px` from one object at a measured distance
(`f = Z × h / H`) if a calibrated value becomes available.

**Real object heights** (`configs/camera.yaml`):

| Class | Height | Basis |
|---|---|---|
| Cone | 0.70 m | standard 28-inch traffic cone |
| Barrier | 1.00 m | A-frame / Type I–II barricade (real barriers vary 0.8–1.5 m → least reliable) |
| Stop sign | 0.75 m | standard 30-inch octagon; uses the **larger** box side because the sign is as wide as it is tall, so a partly hidden sign still gives a sensible size |

**Sanity check:** a 70 px tall cone → 1005 × 0.70 / 70 = **10.05 m** ✔.

**Annotation:** `src/detect.py` runs the model on an image, folder or video and
draws each box with a label in the required format, e.g. `Cone, 2.6m`.

```bash
python src/detect.py --source data/raw/bdd100k/val/images --skip-empty --limit 20
```

**Known limitations:** the error grows with distance (a 1-pixel box error matters
more for small boxes); assumed sizes vary between real objects; the camera's
focal length is an assumption rather than calibrated.

**Finding on BDD100K (1-epoch smoke-test model):** the labelling pipeline works,
but detections on real driving scenes included clear false positives (a car
labelled "cone", a red post labelled "stop sign"). Besides the model being
barely trained, the merged dataset contains **0 background images** — no road
scenes without the target objects — so the model never learns that cars and
other street clutter are *not* cones/barriers.

## Step 9 — Dataset v2: adding BDD100K background images (`src/data/select_bdd_backgrounds.py`)

**Why:** fixes the false positives found in Step 8. Background (negative) images
are images with an empty label file; the model is penalised for any box it predicts
on them, so it learns that ordinary street scenes contain no targets. It also puts
BDD100K driving scenes into training, which reduces the domain gap with the
demo images.

**How the 900 images were chosen** (from BDD100K **train**; BDD100K **val** stays
reserved for demo/evaluation, so there is no overlap):

| Rule | Reason |
|---|---|
| No `traffic sign` label | BDD100K's `traffic sign` includes stop signs, which would otherwise be unlabelled positives (12,668 of 69,863 images qualify) |
| At least one car / truck / bus | "Hard" negatives: vehicles were the main source of false cones/barriers |
| Random sample with a fixed seed | Natural mix: 435 day / 382 night / 73 dawn-dusk; city street, highway, residential |

**Known risk:** BDD100K does not label cones or barriers, so a few selected images
may contain an unlabelled cone. This is expected to be rare in scenes without traffic
signs and is accepted as label noise.

The merge script gained a `--backgrounds` flag; v1 is unchanged and reproducible.

```bash
python src/data/select_bdd_backgrounds.py            # -> data/raw/bdd_backgrounds
python src/data/merge_datasets.py --backgrounds --out data/merged_v2
```

**Result** (`data/merged_v2/`): train 5,753 images (765 backgrounds = 13%),
val 1,012 images (135 backgrounds). Box counts per class are identical to v1, so
v1 and v2 differ only in the background images.

The Colab notebook now has a `DATASET` setting (`'merged'` or `'merged_v2'`)
so the same notebook trains either version (`yolo26n_nav` / `yolo26n_nav_v2`).

**Training order changed:** the v1 Colab run was stopped during its first epoch
(no checkpoint existed yet) so the limited free Colab GPU time goes to v2 — the
model actually used for the demo, distance estimation and optimization. v1 will be
retrained afterwards if GPU time allows; otherwise the 1-epoch smoke-test model is
the "before" reference for false positives.

**Lesson:** files written to a mounted Google Drive upload in the background, so the
notebook now ends with a `drive.flush_and_unmount()` cell to run before stopping or
disconnecting, so recent checkpoints are not lost.

## Step 10 — Edge optimization pipeline (`src/optimize.py`, `src/benchmark.py`)

Built and tested on the 1-epoch smoke-test model while v2 trains on Colab. **Speed
and size depend only on the architecture, not on training**, so these numbers carry
over to the final model; accuracy numbers will be re-measured on v2.

**Variants and why each was chosen:**

| Technique | How | Why |
|---|---|---|
| GPU vs CPU | PyTorch on M3 CPU vs M3 GPU (MPS); CUDA rows are added automatically on Colab | Required CPU/GPU FPS comparison |
| FP16 | `half=True` at runtime on GPU | Halves memory traffic; GPUs have fast FP16 units |
| ONNX export | Ultralytics → ONNX (opset 18, simplified) run with ONNX Runtime | Portable runtime for ARM boards (Raspberry Pi, Jetson) — no PyTorch needed |
| INT8 quantization | ONNX Runtime **static** post-training quantization, QDQ format, per-channel int8 weights, uint8 activations, calibrated on 200 real training images | 4× fewer bits per weight; int8 math is faster on CPUs. Static (calibrated) quantization is more accurate than dynamic for conv nets |
| Keep detection head in FP32 | 79 nodes of the head (`/model.23/`) excluded from quantization | Box-coordinate outputs are very sensitive to rounding; the head is a small share of compute |
| Smaller input | Export at 320×320 instead of 640×640 | 4× fewer pixels → much faster; costs accuracy on small/distant objects |
| Pruning | L1 unstructured pruning of 10/30/50% of every Conv2d's weights | Required technique to try; shows the accuracy cost of sparsity |

**FPS method:** end-to-end per frame (preprocess + inference + postprocess, batch 1)
on 100 validation images after 10 warm-up frames, i.e. what a robot processing a
camera stream experiences. mAP on the v2 validation set (1,012 images).

**Results — smoke-test model, Apple M3 (8 GB)** (pipeline check only; the file was removed in Step 24, superseded by `results/benchmark_m3.md`):

| Variant | Device | mAP50 | FPS | Size (MB) |
|---|---|---|---|---|
| PyTorch FP32 (baseline) | CPU | 0.447 | 12.5 | 5.44 |
| PyTorch FP32 | GPU (MPS) | 0.447 | 24.6 | 5.44 |
| PyTorch FP16 | GPU (MPS) | 0.447 | **38.0** | 5.44 |
| ONNX Runtime FP32 | CPU | 0.447 | 11.9 | 9.85 |
| ONNX Runtime INT8 | CPU | 0.455 | 15.3 | **3.44** |
| ONNX Runtime FP32 @320px | CPU | 0.425 | **38.5** | 9.73 |
| Pruned 10% | CPU | 0.448 | 11.3 | 5.47 |
| Pruned 30% | CPU | 0.396 | 10.3 | 5.47 |
| Pruned 50% | CPU | 0.183 | 10.4 | 5.47 |

**Observations:**
- **GPU:** 2× faster than CPU in FP32, 3× with FP16, with no accuracy change.
- **INT8:** 2.9× smaller than the FP32 ONNX model and +29% CPU FPS vs ONNX FP32,
  with no accuracy loss (+0.008 mAP50 is within noise) — the best CPU trade-off.
- **320 px input:** 3× faster on CPU for a small mAP drop; the drop will matter most
  for small, distant cones.
- **Pruning gives no speed-up:** unstructured pruning only sets weights to zero, and
  dense CPU/GPU kernels still multiply by those zeros. The benefit is storage (gzip size
  4.84 → 3.26 MB at 50%). Accuracy collapses at 50% without fine-tuning. A real speed-up
  would need *structured* pruning (removing whole channels) plus re-training.
- The `.pt` file is smaller than the FP32 ONNX file because Ultralytics stores
  `.pt` weights in FP16.

```bash
python src/optimize.py  --weights runs/train/yolo26n_nav_v2/weights/best.pt
python src/benchmark.py --weights runs/train/yolo26n_nav_v2/weights/best.pt --tag m3
```

## Step 11 — Training YOLO26n on dataset v2 (Colab T4)

**Setup:** COCO-pretrained YOLO26n (2.5M parameters), 50 epochs, 640 px, batch 32,
AdamW (auto-selected, lr 0.00143), automatic mixed precision, default Ultralytics
augmentation (mosaic, HSV jitter, flips, scale/translate); mosaic switched off for
the last 10 epochs. 696/708 pretrained tensors transferred (only the class-specific
output layers were re-initialised for 3 classes instead of 80).

**Time:** 1.2 hours on a Tesla T4 (~1.3 min/epoch + ~9 s validation), vs ~20 min
per epoch on the M3 (≈15× faster).

**Training curve (validation mAP50):** 0.70 after epoch 1 → 0.79 (ep 10) → 0.84
(ep 20) → 0.87 (ep 35) → flat ~0.87 for the last 10 epochs. Losses fell steadily
and validation mAP never dropped, so there was no overfitting; early stopping did
not trigger.

**Final results** (`best.pt`, v2 validation set: 1,012 images incl. 135 backgrounds):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.902 | 0.793 | **0.874** | **0.657** |
| cone | 0.937 | 0.740 | 0.839 | 0.535 |
| barrier | 0.874 | 0.831 | 0.904 | 0.697 |
| stop_sign | 0.894 | 0.809 | 0.878 | 0.739 |

Inference on T4: 6.5 ms per image (model only).

**Interpretation:**
- **Cone has the lowest recall (0.74):** cone images contain many small, distant cones
  in clusters, which are the hardest objects for any detector. Precision stays high
  (0.94), so the model misses far cones rather than inventing them. mAP50-95 (0.535) is
  also lowest, meaning box edges on small cones are less tight.
- **Barrier scored best**, despite being the smallest class. Barrier images are mostly
  close-up, large objects, and its validation set is small (234 boxes), so this number
  is less certain than the others and is likely optimistic for real driving scenes.
- **Stop sign has the tightest boxes** (mAP50-95 0.739): a distinctive, rigid shape,
  and the class was already known to the COCO-pretrained model.

**Note — "duplicate labels removed" warnings:** some Roboflow cone images contain the
same box twice. Ultralytics drops exact duplicates automatically; no effect on training.

## Step 12 — Before vs. after on real BDD100K driving scenes

Validation mAP only measures the training-style images, so both models were run on
the **same 500 BDD100K val scenes** (every 20th image; never used in training), and
every detection was inspected visually as a zoomed crop.

| | 1-epoch model (v1 data) | v2 model (50 epochs + backgrounds) |
|---|---|---|
| Detections (conf ≥ 0.35) | 32 in 26 images | 17 in 16 images |
| Cones | 12 (mostly cars, flowers, road markings; 3 real cones) | 1 (false) |
| Stop signs | 18 | 14 |
| Barriers | 2 (false) | 2 (both real orange road barriers) |

**What improved:** the obvious false positives (cars, flowers, lane markings, building
corners labelled as cones or barriers) are gone, and v2 found real road barriers.

**Problem 1 — other red signs labelled "stop sign":** DO NOT ENTER, a CVS logo and a
pedestrian "don't walk" hand were detected as stop signs. Cause: the v2 background
images deliberately excluded every BDD100K scene with a traffic sign (to avoid
unlabelled stop signs), so the model never saw other red signs as negatives.
**Mitigation:** a stricter stop-sign threshold. Measured on the v2 val set (IoU ≥ 0.5):

| stop_sign threshold | Precision | Recall | False positives |
|---|---|---|---|
| 0.35 | 0.900 | 0.808 | 28 |
| 0.50 | 0.935 | 0.779 | 17 |
| **0.60** | **0.948** | **0.756** | **13** |
| 0.70 | 0.964 | 0.696 | 8 |

0.6 halves false stop signs for a 5-point recall loss and is now the default in
`src/detect.py` (`CLASS_CONF`).

**Problem 2 — v2 misses real cones in BDD100K scenes:** on the images where the 1-epoch
model found real cones, v2 gave no cone prediction even at 5% confidence. Cause found:
BDD100K does not label cones, and **38 of the 900 "background" images actually contain
cones, traffic drums or barriers** (confirmed visually; likely more that the weak
1-epoch model missed). Training told the model these were "nothing", teaching it to
ignore cones in exactly the driving scenes that matter. The risk was noted in Step 9;
it turned out to be significant.

**Side finding — Apple MPS:** passing a long list of images to `predict()` on MPS
crashed or returned no boxes; single-frame prediction on MPS matches CPU exactly
(verified box-for-box), so the per-frame MPS FPS benchmark is valid.

## Step 13 — Dataset v3: filtered background images

**Goal (restated by the user):** whenever a scene contains a cone, barrier or stop sign,
the model must detect it and label it with class and distance. v2 fails this for cones
in driving scenes, so the background images were rebuilt.

**Changes to `select_bdd_backgrounds.py`** (defaults still reproduce the v2 selection):

| Change | Reason |
|---|---|
| Every candidate is run through 3 detectors — the 1-epoch model, v2, and the original COCO-pretrained YOLO26n (knows `stop sign`) — and **rejected if any sees a cone / barrier / stop sign at ≥ 10% confidence** | Removes backgrounds with unlabelled targets (Problem 2). The threshold is deliberately low: a wrongly rejected clean image costs nothing, a kept dirty one teaches the model to ignore real objects |
| Half of the 900 scenes now **contain traffic signs** (stop signs screened out by the same filter) | Teaches that other red signs (DO NOT ENTER, shop logos) are not stop signs (Problem 1) |

**Selection result:** the filter rejected 262 of 712 no-sign candidates (37%) and 356 of
806 sign candidates (44%) before 450 clean images per pool were found — confirming how
often BDD100K scenes hold unlabelled targets. Mix: 464 day / 345 night / 84 dawn-dusk.

**Spot check:** 30 random selected images viewed; 29 clean, 1 contains orange traffic drums
in a dark night scene that all three filters missed → estimated ~3% residual noise
(vs. 38+ confirmed dirty images in v2). Accepted; a full manual review of 900 images was
not worth the time.

**Dataset** (`data/merged_v3/`, `colab_upload/merged_v3.zip`): identical positive
images and splits to v1/v2; only the 900 background images differ (765 train / 135 val).
Because the validation backgrounds differ, v2 and v3 val mAP are close but not strictly
comparable — the BDD100K scene comparison (Step 12) is the decisive test.

```bash
python src/data/select_bdd_backgrounds.py --out data/raw/bdd_backgrounds_v3 --sign-frac 0.5 \
    --filter-models runs/train/smoke_test/weights/best.pt runs/train/yolo26n_nav_v2/weights/best.pt models/yolo26n.pt
python src/data/merge_datasets.py --backgrounds bdd_backgrounds_v3 --out data/merged_v3
```

## Step 14 — v3 results: final model, real-scene check, optimization benchmark

**Training (Colab T4, 50 epochs, same settings as v2).** v3 is now the final model and the
default in `detect.py`, `optimize.py`, `benchmark.py`, `bev.py` and `flow.py`.

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| cone | 0.940 | 0.737 | 0.843 | 0.536 |
| barrier | 0.880 | 0.814 | 0.898 | 0.706 |
| stop_sign | 0.900 | 0.824 | 0.889 | 0.748 |
| all | 0.906 | 0.792 | 0.877 | 0.663 |

Val mAP is about the same as v2 (0.874 / 0.657). That was expected: the positive images are
identical, so the real test is on driving scenes.

**Stop-sign threshold raised 0.6 → 0.7.** At 0.6, 6 of v3's 10 BDD stop-sign detections were
other signs (DO NOT ENTER, speed limit, diamond road-work signs), all scoring 0.62–0.75.
Measured on the v3 val set (`IoU ≥ 0.5`):

| stop_sign threshold | Precision | Recall | False positives |
|---|---|---|---|
| 0.35 | 0.904 | 0.814 | 27 |
| 0.60 | 0.948 | 0.756 | 13 |
| **0.70** | **0.969** | **0.705** | **7** |
| 0.75 | 0.972 | 0.673 | 6 |

0.7 drops 5 of the 6 false BDD stop signs and keeps all 3 real ones, for a 5-point val recall
loss. For a robot, a missed stop sign is worse than a false one, so this is a judgement
call. It is a single constant (`CLASS_CONF` in `src/detect.py`).

**Real-scene check (`src/evaluate_bdd.py`, same 500 BDD100K val scenes, every detection
inspected visually):**

| Model | Detections | Correct | Real cones | Notes |
|---|---|---|---|---|
| v1 | 16 | mostly false | ≈ 3 of 12 | cars, flowers, markings as cones |
| v2 | 5 | 4 | 0 | clean but blind to real cones |
| **v3** | 11 | 8 | 2 (cone + orange road post) | 3 barriers, 3 stop signs; errors: DO NOT ENTER, hand signal, brake lights |

**v3 fixes the v2 problem:** it detects real cones in driving scenes again while staying
mostly clean. **Remaining weakness:** small yellow-green cones are missed (the cone dataset is
mostly orange), and other red signs still cause some false stop signs.

**Optimization benchmark (M3, v3 model; `results/benchmark_m3.md`):**

| Variant | Device | mAP50 | FPS | Size MB |
|---|---|---|---|---|
| PyTorch FP32 (before) | CPU | 0.867 | 31.6 | 5.44 |
| PyTorch FP32 | MPS | 0.867 | 50.4 | 5.44 |
| PyTorch FP16 | MPS | 0.866 | 52.3 | 5.44 |
| ONNX FP32 | CPU | 0.867 | 34.5 | 9.85 |
| **ONNX INT8** | CPU | 0.844 | **54.1** | **3.44** |
| ONNX FP32 @320 | CPU | 0.744 | 117.9 | 9.73 |
| Pruned 10 / 30 / 50% | CPU | 0.864 / 0.713 / 0.183 | 29–31 | 5.48 |

**Conclusions:**
- INT8 is the best CPU deployment: 1.7× faster, 37% smaller, −2.3 mAP50.
- 320 px input gives 3.7× the speed, but loses distant objects.
- Unstructured pruning gives no speed (dense kernels still multiply zeros) and destroys
  accuracy beyond 10%.

FPS is higher than the smoke-test benchmark (Step 10) because the M3 was otherwise idle this
time. Every variant was measured in the same run, so the comparison is fair.

Annotated examples (`results/annotated/`) were produced with
`make detect` (`src/detect.py --source assets/samples`), where `assets/samples` holds 4 BDD100K val frames
containing detections.

## Step 15 — Extra: bird's-eye view with homography (`src/bev.py`)

**Method:** assume a flat road and a level camera at height h = 1.3 m (typical dash-cam). A
road point (X right, Z forward) appears at pixel `u = cx + f·X/Z`, `v = v0 + f·h/Z`, where
v0 is the horizon row. This plane-to-plane mapping is a homography.
`cv2.getPerspectiveTransform` builds it from 4 ground points, and the frame is warped onto
a top-down grid (20 m wide, 3–40 m ahead, 10 m range rings). Objects are plotted at their
size-based distance with lateral offset `X = (u − cx)·Z/f`.

**Second distance estimate:** cones and barriers stand on the road, so the bottom edge of
the box gives a ground-contact distance that does not depend on the object's real size.

| Object | Size-based | Road contact |
|---|---|---|
| Barriers | 5.1 / 7.3 / 8.1 m | 5.6 / 8.3 / 9.3 m (8–15% apart) |
| Cone | 18.1 m | 20.7 m (13%) |
| Orange road post ("cone") | 3.2 m | 5.9 m |

The post is about 1.2 m tall, but the size method assumes a 0.7 m cone, so it reads too
close. This shows why a second, size-free estimate is useful.

**Checks and decisions:**
- **Horizon:** candidate horizon lines were drawn on BDD frames, giving 0.53 × image
  height as the default (`configs/camera.yaml`). A straight-road frame warped with it shows
  parallel lane lines about 3.5 m apart, which is correct.
- **Horizon varies by camera:** in the cone frame the lane lines meet at 0.42, not 0.53,
  because BDD cars mount their cameras differently. With 0.53 that cone's base sat above the
  horizon, giving no ground estimate. Added `--horizon-frac` to set it per camera/video.
- **Label clipping:** BEV labels near the right edge are now drawn to the left of the marker.

## Step 16 — Extra: optical flow tracking and time-to-contact (`src/flow.py`)

**Why TTC:** for navigation, "how many seconds until I reach it" matters as much as
distance. TTC comes from how fast the object grows in the image, `TTC = dt / (s − 1)` with
s = per-frame expansion. It needs **no camera calibration**, so it is independent of the
pinhole assumptions. Closing speed = distance ÷ TTC.

**Method:**
1. Shi-Tomasi corners are taken inside each box.
2. They are tracked with pyramidal Lucas-Kanade; points failing a forward-backward check
   (more than 1 px error) are dropped.
3. Expansion s is the median ratio of pairwise point distances, which is robust to outliers.
4. s is smoothed with an EMA.

**No video data:** BDD100K's image set is single frames and nuImages wasn't downloaded.
`src/simulate_approach.py` therefore zooms a real BDD frame about a detected cone by
`k(t) = Z0 / (Z0 − v·t)`. That is exactly how a real approach magnifies the object (the
background is not physically correct, but only points inside the box are used). This gives
exact ground truth: 18 m → 6 m at 8 m/s, 30 fps, 45 frames.

**Iterations on the simulated clip:**

| Version | Frames tracked | Track IDs | Median TTC error |
|---|---|---|---|
| Detector-only tracking, EMA 0.3 | 28 / 45 | 4 | 11.2% (always high: smoothing lag) |
| + flow-predicted boxes, coast 5 frames, EMA 0.5 | 38 / 45 | 4 | 8.5% |
| + coast up to 10 frames | **44 / 45** | 3 | **8.6%** (max 23%) |

- Flow-predicted boxes (shifted by the median flow, scaled by s) make matching robust.
- Coasting bridges the frames where the 3×-upsampled, blurry cone fools the detector.
- One extra track is a yellow-green cone the detector picked up as it grew.
- Closing speed came out at 6.5–8.8 m/s (true 8).

**Limitation:** the test is synthetic. A real phone video walking toward a cone would be the
proper validation; the script works on any video.

## Step 17 — Making the repository reviewable and reproducible

**Goal:** someone cloning the repo can run the demos, check the claims, and rebuild every
result without asking questions.

| Change | Why |
|---|---|
| Final weights shipped in `models/` (FP32 `.pt` 5.4 MB + INT8 `.onnx` 3.4 MB); all inference scripts default to them | `runs/` is not committed, so the old defaults pointed at files a reviewer wouldn't have |
| `assets/samples/` (4 BDD100K val frames) | Demo inputs that exist in a fresh clone (`data/` is git-ignored) |
| `Makefile` (`make detect / bev / flow-demo / test / optimize / benchmark / eval-bdd / data / train`) | One command per result, so the README commands can't drift from what was run |
| 22 `pytest` unit tests (`tests/`) | Check the maths behind the numbers, with no datasets, in about 1 s: pinhole formula and resolution independence; ground ↔ image round trip and homography on non-corner points; LK flow recovering a known 3% zoom; IoU; polygon → box clipping; image-count capping |
| `ruff` lint + format, config in `pyproject.toml`; `zip(..., strict=True)` | Consistent style; strict zips fail loudly if box / class / score arrays ever differ in length |
| GitHub Actions CI (lint + tests, CPU PyTorch) | Every push is checked automatically |
| `docs/MODEL_CARD.md` | Standard summary of intended use, training data and licenses, metrics, known failure modes, safety notes |
| README: key results, quick start, reproduce-from-scratch, licenses and credits | A reviewer sees the outcome first and the details after |
| `train.py` defaults to `merged_v3` / `yolo26n_nav_v3` | Defaults now match the final model |

**Checks:** after reformatting, `make detect bev flow-demo` reproduced exactly the same
detections, distances and track count (55) as before.

**Correction found while writing the model card:** Ultralytics `optimizer=auto` chose
**AdamW** (not SGD) for this run length, as recorded in the checkpoint's `train_args`.

## Step 18 — Live camera mode (iPhone via Continuity Camera)

`src/detect.py --source <camera index>` opens a live window with "Class, X.Xm" boxes and an
FPS counter (`q` quits). Extra options:
- `--save` records the annotated stream.
- `--list-cameras` prints the available indices.
- `--device mps` runs on the M3 GPU.
- `--hfov` sets the camera's field of view for distance.

**Why:** the model can be tested on real objects with a phone, with no app development:
macOS Continuity Camera exposes an iPhone as a normal webcam.

**Finding: OpenCV does not list Continuity Camera.** macOS sees the iPhone
(`system_profiler SPCameraDataType`), but OpenCV's AVFoundation backend uses the older
device list, which excludes Continuity Camera unless the app opts in. So `detect.py`
also accepts **stream URLs** (`http://`, `rtsp://`): an iPhone IP-camera app streams over
Wi-Fi and OpenCV reads it through its FFmpeg backend. Recording on the phone and running
`detect.py` / `flow.py` on the `.MOV` file always works as a fallback.

**Distance with a different camera:** the focal length in `configs/camera.yaml` assumes a 65°
dash-cam. For a new camera, place a cone at a known distance and read the reported distance.
Then `focal_px_new = focal_px × true / reported`, set as `focal_px` in the config. The
default 1005 px is at 1280 px width.

## Step 19 — Completing the ML lifecycle: EDA, a held-out test split, deployment

The project was mapped onto the standard lifecycle (collection → preprocessing → EDA →
training → validation → testing → deployment). Three gaps were found and filled.

### EDA (`src/data/eda.py` → `results/eda/`)
EDA was missing, which is why some data problems were only found later (Step 12) by error
analysis. Findings on v3:

| Finding | Action |
|---|---|
| Box size implies training objects are **close**: barriers 0.9–3.4 m, cones 0.7–12.6 m (p10–p90) | Main cause of low recall on distant BDD objects (5–40 m). Recorded as the next data gap (far-range data), not fixed now |
| **1 of 6765** images has two different classes | Mixed scenes never appear in training; covered by the BDD scene check |
| **90 near-duplicate cone boxes** (IoU > 0.9; Ultralytics also warned about duplicate labels) | `--dedup-iou 0.9` in `merge_datasets.py` → 0 in v4 |
| 18% of cone boxes touch the image border | Distance limitation (cut box → reads too far) |
| Sample grid: a COCO "stop sign" box on a round no-parking sign; catalogue photos of barriers on white backgrounds | Label noise and domain gap, documented in the model card |
| 307 distinct resolutions | Handled by letterboxing; distance rescales focal length by width |

### Held-out test split (dataset v4)
Before this step, the val split did three jobs: it chose the best epoch, tuned the stop-sign
threshold, and reported the final metrics. A score on data used for decisions is optimistic,
so `merge_datasets.py` gained `--test-frac`:
- v4 is a 70/15/15 split, grouped like before, with near-duplicates removed.
- Train 4737 / val 1012 / test 1016 images.
- **The val split is the same as v3's** (same seed and order), so v3 and v4 val scores
  remain comparable.
- With default arguments the script still rebuilds v3 byte-for-byte (verified by diff).

**v3 cannot be scored on the v4 test split:** v3 was trained on those images. A v4 model is
needed, trained on Colab (notebook updated: `DATASET = 'merged_v4'`, plus a test-set cell
to run once at the very end).

### Deployment
- `src/serve.py` is a FastAPI service with `GET /health`, `POST /detect` (JSON with
  distances) and `POST /detect/image` (annotated JPEG). The model is chosen with
  `MODEL_PATH`, so the INT8 ONNX file can be served too.
- Tested with `TestClient` (4 tests) and with a live `uvicorn` + `curl` run: cone at 18.12 m,
  34.6 ms.
- `Dockerfile`: python:3.10-slim, CPU PyTorch, non-root user, health check.
- Docker is not installed on the dev Mac, so **CI builds the image and sends a real request
  to the running container** on every push.

---

## Step 20 — v4 training and the held-out test score

v4 (YOLO26n, same recipe as v3, trained on `merged_v4`, Colab T4) was scored on val, then
on the test split **once**, with no changes made afterwards.

| Model / split | P | R | mAP50 | mAP50-95 | cone | barrier | stop sign |
|---|---|---|---|---|---|---|---|
| v3 / val (5753 train imgs) | 0.906 | 0.792 | 0.877 | 0.663 | 0.843 | 0.898 | 0.889 |
| v4 / val (4737 train imgs) | 0.898 | 0.769 | 0.854 | 0.648 | 0.806 | 0.880 | 0.877 |
| **v4 / test (held out)** | 0.894 | 0.823 | **0.883** | **0.657** | 0.896 | 0.897 | 0.855 |

Per-class columns are mAP50.

**Reading the results**
- **v4 val is 0.023 mAP50 below v3.** This was expected: v4 trained on 18% fewer images
  (the test images were taken out of train). Cone drops the most (0.843 → 0.806); it is the
  class with the most varied images, so it gains the most from extra data.
- **v4 test (0.883) ≥ v4 val (0.854).** The val split was used to pick the best epoch and
  the stop-sign threshold. If that had overfitted val, test would score lower. It doesn't,
  so the val numbers reported for the project are **not optimistic**.
- Test vs val differ mostly on cones (0.896 vs 0.806); cone-heavy source groups landed
  differently in each split. With ~1000 images per split, ±0.03 between splits is normal
  sampling noise.

**Test set at the deployed thresholds.** The default Ultralytics confusion matrix is drawn
at a near-zero confidence, so it shows thousands of "background → cone" boxes the detector
would never output. The test split was re-scored at the operating point `detect.py` uses
(0.35, and 0.7 for stop signs) to get a confusion matrix that matches deployment. Outputs
are in `runs/test/yolo26n_nav_v4_conf{0.35,0.7}/`.

| Class (threshold) | Correct | Missed | False positives | Recall | Precision |
|---|---|---|---|---|---|
| cone (0.35) | 825 / 1066 | 241 | 39 | 0.77 | 0.95 |
| barrier (0.35) | 197 / 224 | 27 | 19 | 0.88 | 0.91 |
| stop sign (0.7) | 227 / 311 | 84 | 6 | 0.73 | 0.97 |

- **Classes are almost never confused with each other:** 1 cone/stop-sign mix-up in the
  whole test set. Errors are missed objects, mostly small or partly hidden cones, which fits
  the EDA finding that training cones are mostly close-range.
- At this operating point the detector is precision-first: it rarely invents an obstacle but
  misses about 1 in 4 cones. For navigation, the optical-flow tracker in Step 16 helps here,
  because it keeps a track alive through missed frames.
- The training curve peaked at epoch 47 of 50 (mAP50-95 0.649) and stayed flat, so the run
  did not overfit and did not need more epochs.

**Decision: keep v3 as the deployed model**, and report v4's test score as the unbiased
estimate of the recipe.
- v3 is the same recipe with more training data, and it scores higher on the shared val split.
- v4's test score is a fair (slightly conservative) estimate for v3, because v3 had more data.
- This is the standard "estimate with a held-out split, then ship the model trained on more
  data" practice.
- v3 cannot be scored on the v4 test split itself, because it trained on those images.

## Step 21 — Live webcam test and submission clean-up

**Live test:** the v3 model was run on the MacBook's built-in camera
(`python src/detect.py --source 0 --device mps`). Detection worked well live. The README
now has a step-by-step webcam guide covering environment activation, the camera index,
macOS camera permission, and why webcam distances are only approximate (the camera is not
calibrated, and phone-screen or toy objects are not real size).

**Clean-up before submission**
- Removed tool caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.DS_Store`), 18 empty
  scratch `runs/detect/val-*` folders, and `models/optimized_smoke/`.
- Kept everything the docs or Makefile refer to at the time (`runs/train/smoke_test`, the v1
  model, and `results/benchmark_smoke_m3.md`). Both were revisited in Step 24.
- **Bug found while checking what a commit would include:** the `.gitignore` rule `data/`
  matched *any* folder named `data`, so `src/data/` (collection, merging and EDA code) would
  have been silently left out of the repo. The rules are now anchored to the root (`/data/`,
  `/runs/`, `/colab_upload/`).
- Verified the final file list: 64 files, about 13 MB including both shipped model files.
  It contains no datasets and no run folders.

**README:** added an "Assignment objectives and how each is met" table that maps each
requirement to the method and the file that implements it. Also added the held-out test
results.

## Step 22 — README: step-by-step run guide, EDA and evaluation figures

**Why:** a reviewer should be able to run the model and see the evidence without reading
code or opening run folders. `runs/` is not committed, so its plots would not be visible on
GitHub.

- **"How to run: step by step"** replaces the short Quick start. It has 8 numbered steps:
  clone, set up the environment, run tests, sample images, your own images or video, live
  webcam, extras and the API, then EDA and evaluation. Each step has exact commands, and
  every command was checked against the Makefile and the CLIs.
- **EDA section:** added the class-balance table and the box-centre and sample-grid figures.
- **New "Evaluation metrics" section.** Key plots were copied into `results/metrics/`, which
  is committed:
  - v3 training curves
  - PR curve and F1 curve
  - val predictions next to the ground truth
  - the test confusion matrix at the deployed threshold
- The F1 curve was checked rather than assumed: F1 over all classes peaks at 0.84 at a
  confidence of 0.39. That supports the 0.35 threshold. Stop-sign F1 stays flat up to
  about 0.7, which supports its stricter threshold.
- The default Ultralytics confusion matrix (drawn at confidence 0.001) was **left out on
  purpose**. Its background column, for example "0.64 of background predictions are cones",
  describes boxes the deployed detector never outputs, so it would mislead a reader.

## Step 23 — README: how distance is calculated and how to change object heights

**Why:** the distance is only as good as its assumptions, the real object height and the
focal length. A user with different objects or a different camera needs to see those
assumptions and know how to change them without reading the code.

The README's distance section was rewritten to cover:
- **How it works:** a similar-triangles diagram and the four steps for each box.
- **The standard heights and where they come from:**
  - cone 0.70 m (28-inch cone)
  - barrier 1.00 m (Type I–II barricade)
  - stop sign 0.75 m (30-inch face)
- **Why the heights matter:** distance is proportional to the height, so a half-size object
  reads 2× too far.
- **How to change the heights:** edit `objects:` in `configs/camera.yaml`, or pass a separate
  config with `--camera` (or `CAMERA_CFG` for the API). No retraining is needed.
- **How to calibrate the focal length:** with `--hfov`, or once with `calibrate_focal()`.
  The example command was run to check its output (1085.7).

All the flags mentioned were checked against the CLIs: `--camera` exists in `detect.py`,
`bev.py` and `flow.py`, and `--hfov` in `detect.py`.

## Step 24 — Final repository clean-up

**Why:** the repository should contain only what a reviewer needs: code, tests, configs,
docs, final weights and real results. Names should say exactly what each thing is.

| Change | Reason |
|---|---|
| Deleted `results/benchmark_smoke_m3.{md,csv}` | It benchmarked a 1-epoch throwaway model to test the pipeline. The real benchmark is `results/benchmark_m3.md`. |
| Renamed `runs/train/smoke_test` → `runs/train/yolo26n_nav_v1_1epoch`, and updated `Makefile`, `evaluate_bdd.py` and `select_bdd_backgrounds.py` | This run is the "v1" model in the BDD comparison and the background filter. Checking `args.yaml` showed it was trained for **1 epoch**, so the name says so rather than suggesting a full v1 run. |
| README BDD table: "v1 (no backgrounds)" → "v1 (1 epoch, no backgrounds)" | The README did not mention that v1 was a 1-epoch model, which matters when comparing it with v2/v3 (50 epochs). |
| Deleted the local scratch outputs `runs/detect/` and `runs/benchmark_smoke.log`, and the superseded Colab zips `merged.zip`, `merged_v2.zip`, `merged_v3.zip` (1.5 GB) | All are git-ignored, rebuildable, and no longer used. `merged_v4.zip` is kept. |

Kept on purpose:
- `data/` (raw sources and dataset versions), needed to reproduce EDA and evaluation.
- The `runs/train/*` run folders, which hold the weights and logs behind every reported number.
- `models/yolo26n.pt`, the COCO starting point for training.

All of these are git-ignored.

## Step 25 — README rewrite for readability

The README had grown section by section, and the same numbers appeared in three different
tables. It was rewritten as one continuous write-up:
- First-person prose wherever a table wasn't needed (EDA findings, dataset versions,
  optimization takeaways, API endpoints).
- The "Key results" table and the "ML lifecycle" table were merged into a short
  "Results at a glance" list.
- Less bold text.

All numbers, commands, file paths and links are unchanged, and all internal anchors were
re-checked.

## Next steps

- [x] Set up a Python 3.10 environment (PyTorch + Ultralytics; system Python 3.14 is too new for reliable support)
- [x] Fine-tune YOLO26n (final: v3 dataset, Colab T4) — mAP50 0.877, mAP50-95 0.663
- [x] Distance estimation with the pinhole camera model
- [x] Run on BDD100K scenes and save annotated results (`results/annotated`, `results/bdd_eval`)
- [x] Optimization: ONNX export, INT8 quantization, pruning; CPU/GPU FPS benchmarks (`results/benchmark_m3.md`)
- [x] Optional extras: bird's-eye view (homography), optical flow TTC
- [ ] Optional: CUDA (Colab T4) FPS numbers
- [x] Repo hardening: tests, lint, CI, Makefile, model card, shipped weights
- [x] EDA, test split (v4 dataset), REST API + Docker
- [x] Train v4 on Colab and score the test split once (Step 20: test mAP50 0.883; v3 kept)
- [ ] Epipolar geometry / stereo depth (needs two sideways iPhone photos)
- [ ] Real iPhone video for optical flow
- [x] Commit and push to GitHub (github.com/Nabhishekreddy04/object_detection, branch main)
