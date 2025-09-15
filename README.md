# M+ Bench Reproduction

This repo reproduces two benchmarks from the M+ paper:

1. **Knowledge Retention** (SQuAD + NaturalQA with distractor passages)
2. **LongBench Subset** (HotpotQA + MuSiQue at 8k/16k tokens)

## Setup

### Requirements
- Docker
- NVIDIA drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Build environment
```bash
docker compose build

docker load -i mplus-bench-cu121.tar

docker compose run --rm bench bash

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  mplus-bench:cu121 \
  bash