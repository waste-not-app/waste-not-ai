FROM python:3.15-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    AI_BIND_HOST=0.0.0.0
RUN useradd --system --uid 10001 --user-group --no-create-home wastenot
WORKDIR /app
LABEL org.opencontainers.image.source="https://github.com/waste-not-app/waste-not-ai"

FROM base AS recommendation
COPY requirements-runtime.lock ./
RUN pip install --require-hashes -r requirements-runtime.lock
COPY serve.py model.json ./
COPY src ./src
COPY guides ./guides
LABEL org.opencontainers.image.title="wastenot-ai-recommendation"
USER wastenot
EXPOSE 8093
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,socket;socket.create_connection(('127.0.0.1',int(os.getenv('AI_PORT','8093'))),3)"]
CMD ["python", "serve.py", "recommendation"]

FROM base AS detection
COPY requirements-cpu.lock ./
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock
COPY serve.py model.json ./
COPY src ./src
ENV AI_MODEL_DIRECTORY=/models \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
LABEL org.opencontainers.image.title="wastenot-ai-detection"
USER wastenot
EXPOSE 8091
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os,socket;socket.create_connection(('127.0.0.1',int(os.getenv('AI_PORT','8091'))),3)"]
CMD ["python", "serve.py", "detection"]
