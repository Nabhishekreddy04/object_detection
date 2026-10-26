PY ?= .venv/bin/python
SAMPLES ?= assets/samples
.DEFAULT_GOAL := help

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv (Python 3.10) and install pinned dependencies
	python3.10 -m venv .venv
	$(PY) -m pip install -r requirements.txt -r requirements-serve.txt -r requirements-dev.txt

lint:  ## ruff lint + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

test:  ## unit tests (no datasets needed)
	$(PY) -m pytest

data:  ## download COCO stop signs, select filtered BDD backgrounds, build data/merged_v3
	$(PY) src/data/download_coco_stop.py
	$(PY) src/data/select_bdd_backgrounds.py --out data/raw/bdd_backgrounds_v3 --sign-frac 0.5 \
		--filter-models runs/train/yolo26n_nav_v1_1epoch/weights/best.pt runs/train/yolo26n_nav_v2/weights/best.pt models/yolo26n.pt
	$(PY) src/data/merge_datasets.py --backgrounds bdd_backgrounds_v3 --out data/merged_v3

data-v4:  ## v4 = v3 data + held-out test split (15%) + near-duplicate boxes removed
	$(PY) src/data/merge_datasets.py --backgrounds bdd_backgrounds_v3 --test-frac 0.15 --dedup-iou 0.9 --out data/merged_v4

eda:  ## dataset analysis (balance, box geometry, label quality) -> results/eda
	$(PY) src/data/eda.py --data data/merged_v4

train:  ## fine-tune YOLO26n on data/merged_v3 (GPU recommended; see notebooks/train_colab.ipynb)
	$(PY) src/train.py

detect:  ## annotate $(SAMPLES) with "Class, X.Xm" -> results/annotated
	$(PY) src/detect.py --source $(SAMPLES) --out results/annotated

CAM ?= 0
camera:  ## live detection from camera $(CAM) (e.g. iPhone via Continuity Camera); q to quit
	$(PY) src/detect.py --source $(CAM) --device mps --out results/camera

bev:  ## bird's-eye view of $(SAMPLES) -> results/bev
	$(PY) src/bev.py --source $(SAMPLES) --out results/bev
	$(PY) src/bev.py --source $(SAMPLES)/c6f14304-e9ad93b9.jpg --horizon-frac 0.42 --out results/bev  # camera tilted up

flow-demo:  ## simulated approach + optical-flow TTC -> results/flow
	$(PY) src/simulate_approach.py --image $(SAMPLES)/c6f14304-e9ad93b9.jpg --center 1140 345 --z0 18 --speed 8 \
		--out results/flow/simulated_approach
	$(PY) src/flow.py --source results/flow/simulated_approach.mp4 --out results/flow

optimize:  ## ONNX FP32/320 px, INT8, pruned variants -> models/optimized
	$(PY) src/optimize.py

benchmark:  ## mAP + FPS of every variant -> results/benchmark_<tag>.md
	$(PY) src/benchmark.py --tag $(shell uname -m)

serve:  ## REST API on localhost:8000 (docs at /docs)
	$(PY) -m uvicorn serve:app --app-dir src --host 0.0.0.0 --port 8000

docker:  ## build and run the API container
	docker build -t nav-detector . && docker run --rm -p 8000:8000 nav-detector

eval-bdd:  ## v1/v2/v3 comparison on 500 BDD100K val scenes -> results/bdd_eval
	$(PY) src/evaluate_bdd.py --models v1=runs/train/yolo26n_nav_v1_1epoch/weights/best.pt \
		v2=runs/train/yolo26n_nav_v2/weights/best.pt v3=models/yolo26n_nav_v3.pt

.PHONY: help setup lint test data data-v4 eda train detect camera bev flow-demo optimize benchmark serve docker eval-bdd
