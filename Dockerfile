FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /workspace

# System deps: python, git, build tools, tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git build-essential \
    tini ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install CUDA 12.1 wheels of torch separately (Option A)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

# Copy requirements and install (no torch here)
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Create cache/results dirs (can be mounted)
RUN mkdir -p /workspace/scripts /workspace/data /workspace/results /workspace/.cache

# Copy scripts (bind-mounts in compose will override during dev)
COPY scripts/ /workspace/scripts/

# Default entry via tini into bash
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]
