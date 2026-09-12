# Local replica of the Alpha Connectome scoring container.
# Mirrors the Dockerfile fragment in wunder_docs/03_submission_guide.md, with
# the pinned deps shipped in the starter pack's requirements.txt.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY wnn_connectome_starterpack/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary -r /tmp/requirements.txt \
 && python -m pip check && pip cache purge

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba

WORKDIR /app
COPY wnn_connectome_starterpack/utils.py /app/utils.py
COPY docker/run_scorer.py /app/run_scorer.py

# src/ (solution.py + *.onnx) and datasets/ are bind-mounted at run time.
CMD ["python", "run_scorer.py"]
