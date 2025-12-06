#!/usr/bin/env bash
set -euo pipefail

###############################################
# DEBUG / ENV
###############################################

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="./nccl_debug_log_%p.txt"

# Do NOT enable SGLang’s torch profiler for now (it was causing NCCL aborts)
# export SGLANG_TORCH_PROFILER_DIR="./torch_profil_dir"

# Nsight Systems in your home directory

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

# Local path to the HF-converted model
MODEL_PATH="Qwen/Qwen3-32B"

# Server listen address / port
HOST="0.0.0.0"
PORT="30000"

# GPUs on the node to use
CUDA_VISIBLE_DEVICES="0,1,2,3"

# Parallelism
TP_SIZE=4          # tensor parallel size
PP_SIZE=1          # pipeline parallel size

# Nsight Systems config
NSYS_SESSION="qwen_tp${TP_SIZE}_pp${PP_SIZE}"
NSYS_OUTPUT="./${NSYS_SESSION}"
NSYS_TRACE="cuda,nvtx"

###############################################
# INTERNAL
###############################################

export CUDA_VISIBLE_DEVICES

echo "======================================="
echo " Starting SGLang server"
echo "  MODEL_PATH     = ${MODEL_PATH}"
echo "  HOST:PORT      = ${HOST}:${PORT}"
echo "  CUDA_VISIBLE   = ${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE        = ${TP_SIZE}"
echo "  PP_SIZE        = ${PP_SIZE}"
echo "  NSYS_OUTPUT    = ${NSYS_OUTPUT}.nsys-rep"
echo "======================================="

# Run SGLang server under Nsight Systems
nsys profile \
  -o "${NSYS_OUTPUT}" \
  --trace="${NSYS_TRACE}" \
  --sample=none \
  --force-overwrite=true \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --attention-backend torch_native \
    --disable-cuda-graph \
  > >(tee server_stdout.log) 2> >(tee server_stderr.log >&2)
