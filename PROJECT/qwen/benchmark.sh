#!/usr/bin/env bash
set -euo pipefail

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

SERVER_HOST="127.0.0.1"      # for 2-node: node0 IP
SERVER_PORT="30000"

MODEL_ID="Qwen/Qwen3-32B"

NUM_PROMPTS=256
INPUT_LEN=1024
OUTPUT_LEN=256
RANGE_RATIO=0.5

MAX_CONCURRENCY=4            # set to 4 or 16
OUT_PREFIX="qwen32b_test"

###############################################
# INTERNAL
###############################################

OUT_FILE="${OUT_PREFIX}_c${MAX_CONCURRENCY}.jsonl"

echo "============== Benchmark =============="
echo "SERVER_HOST       = ${SERVER_HOST}"
echo "SERVER_PORT       = ${SERVER_PORT}"
echo "MAX_CONCURRENCY   = ${MAX_CONCURRENCY}"
echo "BENCH JSONL FILE  = ${OUT_FILE}"
echo "========================================"
echo

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
  --output-details
