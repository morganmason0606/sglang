#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="./nccl_debug_log_%p.txt"

export SGLANG_TORCH_PROFILER_DIR="./torch_profil_dir"
export PATH="$HOME/nsight-systems-2025.5.1/bin:$PATH"
###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

MODEL_PATH="/pscratch/sd/f/fsv5/models/Llama-3.1-8B"

HOST="0.0.0.0"
PORT="30000"
CUDA_VISIBLE_DEVICES="0,1,2,3"

TP_SIZE=4
PP_SIZE=1



# Nsight Systems session name (must match benchmark.sh)
NSYS_SESSION="sglang"
NSYS_TRACE="cuda,nvtx,osrt,mpi,ucx"

###############################################
# INTERNAL
###############################################

export CUDA_VISIBLE_DEVICES


echo "======== SGLang SERVER (nsys launch interactive) ========"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "HOST:PORT         = ${HOST}:${PORT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "TP_SIZE           = ${TP_SIZE}"
echo "PP_SIZE           = ${PP_SIZE}"
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
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --enable-profile-cuda-graph \
 > >(tee server_stdout.log) 2> >(tee server_stderr.log >&2)