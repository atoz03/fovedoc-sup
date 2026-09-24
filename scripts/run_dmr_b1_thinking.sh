#!/usr/bin/env bash
# Lever B1: does an off-the-shelf REASONING reader (Qwen3-VL-4B-Thinking) close the
# cross-page JOIN residual? Naive-lexical-memory arm on V-MQAR, full 1173, matched to the
# stock-4B-Instruct naive baseline (0.659). Thinking traces -> big --max-new-tokens + --strip-think.
# Sharded across 8 GPUs; merge + judge after.
# Usage: bash scripts/run_dmr_b1_thinking.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="configs/dmr_eval_stock_qwen3vl_4b_thinking.yaml"
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
OUT="code/runs/20260630/dmr_levers/B1_4b_thinking"
mkdir -p "$OUT"
COMMON=(--eval-config "$CFG" --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved --retrieved-pages-jsonl "$RET" \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 \
  --max-new-tokens 1536 --strip-think --datasets V-MQAR --modes pages_memory)

echo "== DMR lever B1 (Qwen3-VL-4B-Thinking, V-MQAR naive memory, full 1173) =="
for s in 0 1 2 3 4 5 6 7; do
  odir="$OUT/shard${s}"
  CUDA_VISIBLE_DEVICES="$s" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
    "${COMMON[@]}" --output-dir "$odir" --num-shards 8 --shard-idx "$s" \
    > "$OUT/shard${s}.log" 2>&1 &
  echo "  [shard${s}] gpu=$s pid=$! -> $odir"
done
echo "launched. tail $OUT/shard*.log"
