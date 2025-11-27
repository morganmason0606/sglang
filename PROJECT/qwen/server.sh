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
PP_SIZE=1          # for 2-node set to 2

NNODES=1           # 1 for single-node, 2 for two-node
NODE_RANK=0

MASTER_ADDR="10.0.0.1"   # used only if NNODES > 1
MASTER_PORT="50000"

# Nsight Systems config
NSYS_TRACE="cuda"         # you can change to "cuda,nvtx,osrt" if stable
NSYS_OUT_PREFIX="qwen32b"

###############################################
# INTERNAL
###############################################

export CUDA_VISIBLE_DEVICES

EXTRA_FLAGS=(
  "--disable-cuda-graph"
  # leave out --enable-layerwise-nvtx-marker for now
)

DIST_FLAGS=()
if [[ ${NNODES} -gt 1 ]]; then
  export MASTER_ADDR MASTER_PORT
  DIST_FLAGS+=(
    "--nnodes" "${NNODES}"
    "--node-rank" "${NODE_RANK}"
    "--dist-init-addr" "${MASTER_ADDR}:${MASTER_PORT}"
  )
fi

OUT_NAME="${NSYS_OUT_PREFIX}_n${NNODES}_tp${TP_SIZE}_pp${PP_SIZE}_rank${NODE_RANK}"

echo "======== SGLang SERVER (nsys profile) ========"
echo "MODEL_PATH      = ${MODEL_PATH}"
echo "HOST:PORT       = ${HOST}:${PORT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "TP_SIZE         = ${TP_SIZE}"
echo "PP_SIZE         = ${PP_SIZE}"
echo "NNODES          = ${NNODES}"
echo "NODE_RANK       = ${NODE_RANK}"
echo "MASTER_ADDR     = ${MASTER_ADDR}"
echo "MASTER_PORT     = ${MASTER_PORT}"
echo "NSYS_OUT        = ${OUT_NAME}.nsys-rep"
echo "NSYS_TRACE      = ${NSYS_TRACE}"
echo "=============================================="
echo ">>> Run your benchmark while this server is up."
echo ">>> When finished, Ctrl-C here to end profiling."
echo

###############################################
# NSYS PROFILE AROUND SERVER
###############################################

nsys profile \
  --trace="${NSYS_TRACE}" \
  --sample=none \
  --force-overwrite=true \
  -o "${OUT_NAME}" \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${DIST_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}"
