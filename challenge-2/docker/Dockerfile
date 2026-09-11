# Wunder Predictorium — Scorer Environment
# Matches the competition scoring container as closely as possible.
# See: wunder_docs/03_submission_guide.md

FROM python:3.11-slim-bookworm

# System deps (matches competition Dockerfile)
RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps (matches competition requirements.txt)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime (matches competition env)
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

# Working directory
WORKDIR /app

# Copy utils and datasets
COPY competition_package/utils.py /app/utils.py
COPY competition_package/datasets /app/datasets

# Copy solution (mounted or copied at test time)
# COPY src/solution.py /app/solution.py

# Default: run the scorer
CMD ["python", "run_scorer.py"]
