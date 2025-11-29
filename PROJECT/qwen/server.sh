#!/usr/bin/env bash
set -euo pipefail

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

MODEL_PATH="Qwen/Qwen3-32B"

HOST="0.0.0.0"
PORT="30000"
CUDA_VISIBLE_DEVICES="0,1,2,3"

TP_SIZE=4
PP_SIZE=1

NNODES=1
NODE_RANK=0

MASTER_ADDR="10.0.0.1"   # used only if NNODES > 1
MASTER_PORT="50000"

# Nsight Systems session name (must match benchmark.sh)
NSYS_SESSION="sglang"
NSYS_TRACE="cuda,cudnn,cublas,nvtx,osrt,mpi"

###############################################
# INTERNAL
###############################################

export CUDA_VISIBLE_DEVICES

DIST_FLAGS=()
if [[ ${NNODES} -gt 1 ]]; then
  export MASTER_ADDR MASTER_PORT
  DIST_FLAGS+=(
    "--nnodes" "${NNODES}"
    "--node-rank" "${NODE_RANK}"
    "--dist-init-addr" "${MASTER_ADDR}:${MASTER_PORT}"
  )
fi

echo "======== SGLang SERVER (nsys launch interactive) ========"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "HOST:PORT         = ${HOST}:${PORT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "TP_SIZE           = ${TP_SIZE}"
echo "PP_SIZE           = ${PP_SIZE}"
echo "NNODES            = ${NNODES}"
echo "NODE_RANK         = ${NODE_RANK}"
echo "MASTER_ADDR       = ${MASTER_ADDR}"
echo "MASTER_PORT       = ${MASTER_PORT}"
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
  --cuda-graph-trace=node \
  --wait primary \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${DIST_FLAGS[@]}"
