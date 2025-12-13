#!/usr/bin/env bash
set -euo pipefail

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

SERVER_HOST="127.0.0.1"
SERVER_PORT="30000"
export PATH="$HOME/nsight-systems-2025.5.1/bin:$PATH"

MODEL_ID="/pscratch/sd/f/fsv5/models/Llama-3.1-8B"

NUM_PROMPTS=32
INPUT_LEN=1024
OUTPUT_LEN=256
RANGE_RATIO=0.5   # fine to leave as-is if you want variable lengths

MAX_CONCURRENCY=4
OUT_PREFIX="llama"

# You can keep these vars, but we won't actually use nsys here
NSYS_SESSION="sglang"        
NSYS_REPORT="$(pwd)/${OUT_PREFIX}_c${MAX_CONCURRENCY}.nsys-rep"

# IMPORTANT: disable SGLang torch profiler on the client
# (we're not using it for this benchmark run)
# export SGLANG_TORCH_PROFILER_DIR="./torch_profil_dir"

###############################################
# INTERNAL
###############################################

OUT_FILE="${OUT_PREFIX}_c${MAX_CONCURRENCY}.jsonl"

echo "============== Benchmark =============="
echo "SERVER_HOST       = ${SERVER_HOST}"
echo "SERVER_PORT       = ${SERVER_PORT}"
echo "MAX_CONCURRENCY   = ${MAX_CONCURRENCY}"
echo "BENCH JSONL FILE  = ${OUT_FILE}"
echo "======================================="
echo

###############################################
# RUN BENCHMARK (NO CLIENT-SIDE NSIGHT)
###############################################

python -m sglang.bench_serving \
  --backend sglang \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --model "${MODEL_ID}" \
  --dataset-name random \
  --num-prompts "${NUM_PROMPTS}" \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --random-range-ratio "${RANGE_RATIO}" \
  --request-rate inf \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --output-file "${OUT_FILE}" \
  --output-details \
  > >(tee benchmark_stdout.log) 2> >(tee benchmark_stderr.log >&2)
