# Model Card — YOLO26n-Nav v3

## Model details

| | |
|---|---|
| Architecture | YOLO26n (Ultralytics 8.4.188), NMS-free, 2.5 M parameters |
| Initialisation | COCO-pretrained `yolo26n.pt` (transfer learning, all layers fine-tuned) |
| Classes | `0 cone`, `1 barrier`, `2 stop_sign` |
| Input | RGB image, letterboxed to 640 px |
| Training | 50 epochs, batch 32, 640 px, AdamW (Ultralytics `optimizer=auto`), seed 0, Colab T4, ~1.2 h |
| Files | `models/yolo26n_nav_v3.pt` (PyTorch, 5.4 MB), `models/yolo26n_nav_v3_int8.onnx` (ONNX Runtime INT8, 3.4 MB) |
| Post-processing | conf ≥ 0.35; stop_sign ≥ 0.7 (`CLASS_CONF` in `src/detect.py`) |

## Intended use

- Perception for a ground robot or vehicle prototype: find cones, barriers and stop signs in
  a forward-facing camera and estimate their distance (`src/distance.py`) for navigation.
- Coursework / research. **Not validated for safety-critical use**: it must not be the only
  source for braking or right-of-way decisions.

## Training data

| Source | Classes | License |
|---|---|---|
| COCO 2017 (`stop sign` images) | stop_sign | CC BY 4.0 (annotations); images per Flickr terms |
| Roboflow Universe *cones-mapit v2* (capped at 3000 images) | cone | CC BY 4.0 |
| Roboflow Universe *barriers-ctzjn v2* | barrier (cones in it mapped to cone) | CC BY 4.0 |
| BDD100K train (900 scenes, no labels) | background negatives | BDD100K license (non-commercial research) |

- Total: 5753 train / 1012 val images (765 / 135 backgrounds).
- The split is grouped by source image, so augmented copies never cross train/val.
- Background scenes were screened by three detectors to remove unlabelled targets
  (about 3% residual noise in a 30-image spot check).

## Evaluation

**Held-out val set (1012 images):**

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| cone | 0.940 | 0.737 | 0.843 | 0.536 |
| barrier | 0.880 | 0.814 | 0.898 | 0.706 |
| stop_sign | 0.900 | 0.824 | 0.889 | 0.748 |
| all | 0.906 | 0.792 | 0.877 | 0.663 |

**Held-out test set (1016 images, scored once).** The val split above was used to choose the
epoch and the stop-sign threshold, so it is not a fully unbiased score. To get one, the same
recipe was retrained on a 70/15/15 split (v4, 4737 train images) and scored once on its test
split:

| Model / split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v4 / val | 0.898 | 0.769 | 0.854 | 0.648 |
| v4 / test | 0.894 | 0.823 | 0.883 | 0.657 |

- Test is not lower than val, so model selection did not overfit the val split.
- The shipped model is v3: the same recipe trained on ~18% more data, scoring 0.877 vs 0.854
  on the shared val split.
- v4's test score is therefore a conservative estimate for v3.

At the deployed thresholds (0.35; 0.7 for stop signs), v4 on the test split:

| Class | Recall | Precision |
|---|---|---|
| cone | 0.77 | 0.95 |
| barrier | 0.88 | 0.91 |
| stop sign | 0.73 | 0.97 |

Only 1 test object was given the wrong class. Almost all errors are missed objects, mostly
small cones, not false alarms.

**Out-of-distribution check:** 500 unseen BDD100K val driving scenes, every detection
inspected visually. There are 11 detections, of which 8 are correct (3 barriers, 3 stop signs,
1 cone, 1 cone-like road post). BDD100K has no cone/barrier labels, so recall there cannot be
measured.

**Deployment variants (Apple M3, single frame, end-to-end):**

| Variant | mAP50 | FPS |
|---|---|---|
| PyTorch FP32, CPU | 0.867 | 31.6 |
| PyTorch FP32, MPS GPU | 0.867 | 50.4 |
| ONNX Runtime INT8, CPU | 0.844 | 54.1 |

## Known limitations

- **Cone colour:** the cone data is mostly orange; small yellow-green cones are often missed.
- **Other red signs** (DO NOT ENTER, hand signals) can still be called stop signs. Raising
  the threshold from 0.6 to 0.7 cut val false positives from 13 to 7, but cost 5 points of recall.
- **Night and rain:** under-represented in the positive data (COCO and Roboflow are mostly
  daytime). Brake lights through a wet windscreen produced one false cone.
- **Distance:**
  - The pinhole estimate assumes standard object heights (cone 0.70 m, barrier 1.00 m, stop
    sign 0.75 m) and a 65° FOV camera. Taller objects, such as 1.2 m road posts, read too
    close.
  - Boxes cut off by the frame edge read too far.
  - Accuracy depends on calibrating `focal_px` for the actual camera.
- **INT8:** −2.3 mAP50. On one sample frame it kept 1 of the 3 barriers the FP32 model found.
  Use FP32/FP16 where accuracy matters more than CPU speed.

## Ethical and safety considerations

- The model detects objects, not people, and was not trained or evaluated for pedestrian
  safety.
- Missed detections, especially at night and of non-orange cones, are the main hazard. A
  deployed system needs redundant sensing (e.g. depth or LiDAR) and conservative fallbacks.
- BDD100K frames contain incidental faces and licence plates. Only 4 frames are redistributed
  in `assets/samples/` for demonstration.
