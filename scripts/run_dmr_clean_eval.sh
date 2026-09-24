#!/usr/bin/env bash
# DMR clean-protocol 4-arm eval over the full 1173/task with a STOCK reader.
# Arms: pages | pages_memory(naive lexical) | pages_memory(selector) | pages_memory_assoc(selector).
# Retrieval = ColQwen2 ranked top-16 (pretrained). Selector trained on aux pool (disjoint docs).
# Parallelised across 8 GPUs via the harness --num-shards/--shard-idx striding; merge after.
#
# Usage: bash scripts/run_dmr_clean_eval.sh <2b|4b>
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${1:?usage: run_dmr_clean_eval.sh <2b|4b>}"
CFG="configs/dmr_eval_stock_qwen3vl_${SIZE}.yaml"
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
SELDIR="code/runs/20260630/dmr_selector_aux"
OUT="code/runs/20260630/dmr_eval_${SIZE}"
mkdir -p "$OUT"
COMMON=(--eval-config "$CFG" --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved --retrieved-pages-jsonl "$RET" \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 --max-new-tokens 64 \
  --selector-scorer cross_block_attention --selector-hidden-dim 32)

launch() { # gpu logical-name datasets modes extra...
  local gpu="$1" name="$2" ds="$3" modes="$4"; shift 4
  local odir="$OUT/${name}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
    "${COMMON[@]}" --datasets "$ds" --modes "$modes" --output-dir "$odir" "$@" \
    > "$OUT/${name}.log" 2>&1 &
  echo "  [$name] gpu=$gpu pid=$! -> $odir"
}

echo "== DMR clean eval ($SIZE) =="
# Logical run A: pages + naive lexical memory (no selector), both datasets, 4 shards (gpu 0-3)
for s in 0 1 2 3; do
  launch "$s" "runA_pages_naive_shard${s}" "V-MQAR,V-NIAH" "pages,pages_memory" --num-shards 4 --shard-idx "$s"
done
# Logical run B: selector + assoc, V-MQAR, 2 shards (gpu 4-5)
for s in 0 1; do
  launch "$((4+s))" "runB_vmqar_sel_shard${s}" "V-MQAR" "pages_memory,pages_memory_assoc" \
    --selector-ckpt "$SELDIR/vmqar_selector.pt" --num-shards 2 --shard-idx "$s"
done
# Logical run C: selector + assoc, V-NIAH, 2 shards (gpu 6-7)
for s in 0 1; do
  launch "$((6+s))" "runC_vniah_sel_shard${s}" "V-NIAH" "pages_memory,pages_memory_assoc" \
    --selector-ckpt "$SELDIR/vniah_selector.pt" --num-shards 2 --shard-idx "$s"
done
echo "launched. tail $OUT/*.log"
