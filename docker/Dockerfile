# Linux environment with NeMo installed the way the Nemotron 3.5 ASR model card asks (main branch).
# The repo is mounted at run time, not copied in. Run from the repo root.
#
# CPU:
#   docker build -t asr-cpu docker
#   docker run --rm -v "$PWD:/work" -v hf-cache:/root/.cache/huggingface asr-cpu \
#     python scripts/run_batch_eval.py --manifest data/fleurs/el_gr_test_first10.jsonl --target-lang el --output experiments/baseline/el_results_smoke.csv
#
# NVIDIA GPU (needs the NVIDIA driver and Docker GPU support on the host):
#   docker build -t asr-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 docker
#   docker run --rm --gpus all -v "$PWD:/work" -v hf-cache:/root/.cache/huggingface asr-gpu python scripts/run_batch_eval.py ...
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchaudio --index-url "$TORCH_INDEX"

RUN pip install --no-cache-dir Cython packaging \
    && pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

RUN pip install --no-cache-dir jiwer soundfile numpy huggingface_hub

# The Xet download backend hung indefinitely on the model download inside Docker; plain HTTP works.
ENV HF_HUB_DISABLE_XET=1

WORKDIR /work
