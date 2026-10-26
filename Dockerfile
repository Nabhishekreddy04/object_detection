# CPU inference service: FastAPI + YOLO26n-Nav (see src/serve.py)
#   docker build -t nav-detector .
#   docker run -p 8000:8000 nav-detector
#   docker run -p 8000:8000 -e MODEL_PATH=models/yolo26n_nav_v3_int8.onnx nav-detector   # INT8, faster on CPU
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt -r requirements-serve.txt

COPY configs/ configs/
COPY models/yolo26n_nav_v3.pt models/yolo26n_nav_v3_int8.onnx models/
COPY src/ src/

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs localhost:8000/health || exit 1
CMD ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
