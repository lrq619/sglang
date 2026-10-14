# Source before launching SGLang on this H100 box:  source controller/queuing_predictor/env_h100.sh
# The venv ships CUDA 13.0 runtime headers (torch cu130); nvcc/nvvm/crt are pinned to
# 13.0 to match, and cuda13-home is a symlink-only CUDA_HOME giving the pip wheel the
# lib64/ + libcudart.so layout FlashInfer's JIT links against. Caches stay off /home.
export SGLANG_VENV=/mnt/vdb/lrq/venvs/sglang
export CUDA_HOME=/mnt/vdb/lrq/cuda13-home
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_CACHE_DIR=/mnt/vdb/lrq/sglang_cache
export TMPDIR=/mnt/vdb/lrq/tmp
export HF_HOME=/mnt/vdb/hf_home
export UV_CACHE_DIR=/mnt/vdb/lrq/uv_cache
mkdir -p "$TMPDIR"
