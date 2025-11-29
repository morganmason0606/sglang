#!/usr/bin/env bash
set -euo pipefail

###############################################
# HARD-CODED CONFIG (EDIT THESE)
###############################################

SERVER_HOST="127.0.0.1"
SERVER_PORT="30000"

MODEL_ID="deepseek-ai/DeepSeek-V2-Lite"

NUM_PROMPTS=256
INPUT_LEN=1024
OUTPUT_LEN=256
RANGE_RATIO=0.5

MAX_CONCURRENCY=4
OUT_PREFIX="deepseek"

# Nsight Systems config
NSYS_SESSION="sglang"        # must match server.sh
NSYS_REPORT="${OUT_PREFIX}_c${MAX_CONCURRENCY}.nsys-rep"

###############################################
# INTERNAL
###############################################

OUT_FILE="${OUT_PREFIX}_c${MAX_CONCURRENCY}.jsonl"

echo "============== Benchmark =============="
echo "SERVER_HOST       = ${SERVER_HOST}"
echo "SERVER_PORT       = ${SERVER_PORT}"
echo "MAX_CONCURRENCY   = ${MAX_CONCURRENCY}"
echo "BENCH JSONL FILE  = ${OUT_FILE}"
echo "NSYS REPORT       = ${NSYS_REPORT}"
echo "NSYS_SESSION      = ${NSYS_SESSION}"
echo "======================================="
echo

###############################################
# NSIGHT SYSTEMS: START / STOP AROUND BENCH
###############################################

# Start interactive collection for the server launched via `nsys launch`
nsys start \
  --session="${NSYS_SESSION}" \
  --sample=none \
  -o "${NSYS_REPORT}" \
  --force-overwrite=true

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

echo "stopping"
# Stop collection once the benchmark is done
nsys stop --session="${NSYS_SESSION}"
