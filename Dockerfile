# CUDA runtime only (no toolkit). Works on 3080/4090.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates python3 python3-pip python3-venv build-essential \
 && rm -rf /var/lib/apt/lists/*

# Make python/pip the defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---- PYTHON PACKAGES ----
# Pins that are compatible (no flash-attn, NumPy < 2 to avoid extension ABI issues)
# torch 2.1.2 + cu121, transformers 4.56.1, tokenizers 0.20.x, bitsandbytes 0.43.1
RUN pip install --no-cache-dir \
    torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    "transformers==4.40.0" \
    "tokenizers==0.19.1" \
    "peft==0.10.0" \
    "accelerate==0.29.3" \
    "sentencepiece==0.1.99" \
    "datasets==2.20.0" \
    "evaluate==0.4.2" \
    "numpy<2" "scipy" "tqdm" "psutil" \
    "bitsandbytes==0.43.1" "hf_transfer==0.1.6"

# Create workspace and mount points
WORKDIR /workspace
RUN mkdir -p /workspace/scripts /workspace/results /workspace/logs /workspace/ext_mplus /workspace/hf_cache /workspace/data


# Copy your scripts (only what we need)
# If you keep them beside this Dockerfile:
#   - run_knowledge_retention.py
#   - run_longbench.py (optional)
#   - sanity_check.py (optional)

# Pull M+ helper modules (straight from the repo via curl)
RUN curl -L -o /workspace/ext_mplus/modeling_mplus.py https://raw.githubusercontent.com/wangyu-ustc/MemoryLLM/main/modeling_mplus.py && \
    curl -L -o /workspace/ext_mplus/configuration_memoryllm.py https://raw.githubusercontent.com/wangyu-ustc/MemoryLLM/main/configuration_memoryllm.py

# Keep stdout unbuffered for logs
ENV PYTHONUNBUFFERED=1
# Hugging Face cache inside container (will be volume-mapped)
ENV HF_HOME=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache \
    HF_DATASETS_CACHE=/workspace/hf_cache/datasets \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Default command is bash; compose will override
CMD ["/bin/bash"]
