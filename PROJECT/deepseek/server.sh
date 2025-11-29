#!/usr/bin/env bash
set -euo pipefail

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

MODEL_PATH="deepseek-ai/DeepSeek-V2-Lite"

HOST="0.0.0.0"
PORT="30000"
CUDA_VISIBLE_DEVICES="0,1,2,3"

TP_SIZE=4
EP_SIZE=4



# Nsight Systems session name (must match benchmark.sh)
NSYS_SESSION="sglang"
NSYS_TRACE="cuda,cudnn,cublas,nvtx,osrt,mpi"

###############################################
# INTERNAL
###############################################

export CUDA_VISIBLE_DEVICES


echo "======== SGLang SERVER (nsys launch interactive) ========"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "HOST:PORT         = ${HOST}:${PORT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "TP_SIZE           = ${TP_SIZE}"
echo "EP_SIZE           = ${EP_SIZE}"
echo "NSYS_SESSION      = ${NSYS_SESSION}"
echo "========================================================"
echo ">>> This terminal: SGLang server under 'nsys launch'."
echo ">>> Other terminal: run benchmark.sh (it will nsys start/stop)."
echo

###############################################
# NSYS LAUNCH AROUND SERVER (INTERACTIVE)
###############################################

nsys launch \
  --session="${NSYS_SESSION}" \
  --trace="${NSYS_TRACE}" \
  --trace-fork-before-exec=true \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --expert-parallel-size "${EP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
