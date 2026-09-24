#!/usr/bin/env bash
# READER-FAMILY GENERALIZATION GRID for the retrieval-reading gap.
#
# The gap (+0.28 V-MQAR / +0.33 V-NIAH) is currently a Qwen3-VL result. This runs the same two
# arms — retrieved pages as IMAGES vs the same pages as extracted TEXT memory — across four
# stock readers from three families, so the finding stops being a property of one backbone.
#
# Held fixed across families: retrieval (ColQwen2 ranked top-16, where full-recall is 0.998 /
# 1.000 so the gap is pure reading), memory construction, prompts, budget, greedy decoding,
# and the evaluated samples. NOT fixed: per-family image tokenisation, which is structural —
# see configs/dmr_grid_*.yaml. The claim is a WITHIN-model paired gap, so that is sound, but
# absolute scores are not comparable across families and must not be tabled as if they were.
#
# Usage: bash scripts/run_dmr_reader_family_grid.sh [n_samples_per_task]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-400}"   # subsample per task; the gap is ~0.3 so n=400 gives SE ~0.02
# Only the PROCESSOR path was verified offline for the non-Qwen families; the
# AutoModelForImageTextToText load + generate path has never run. Smoke each config at n=4
# first (`bash scripts/run_dmr_reader_family_grid.sh 4`) and only then spend the full pass.
(( N > 8 )) && echo "NOTE: run with N=4 once first — the HF model-load path is unverified for LLaVA-OV/Idefics3"
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
OUT="code/runs/20260810/dmr_reader_grid"
mkdir -p "$OUT"

COMMON=(--benchmark-root data/fovedoc_project \
  --split all --max-samples "$N" --page-source retrieved --retrieved-pages-jsonl "$RET" \
  --retrieved-top-k 16 --memory-topk 16 --max-new-tokens 64 \
  --datasets "V-MQAR,V-NIAH" --modes "pages,pages_memory")

# One model per GPU. Qwen3-VL-2B/4B are the published readers, re-run here at the same
# subsample so the grid has an in-family anchor rather than relying on the 1173-sample table.
i=0
launch() { # config reader-impl name
  local cfg="$1" impl="$2" name="$3"
  CUDA_VISIBLE_DEVICES="$i" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
    --eval-config "$cfg" --reader-impl "$impl" "${COMMON[@]}" \
    --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 &
  echo "  [$name] gpu=$i pid=$! impl=$impl"
  i=$((i+1))
}

echo "== reader-family grid (n=$N per task, top-16 pages) =="
launch configs/dmr_eval_stock_qwen3vl_2b.yaml qwen3vl qwen3vl_2b
launch configs/dmr_eval_stock_qwen3vl_4b.yaml qwen3vl qwen3vl_4b
launch configs/dmr_grid_qwen3vl_8b.yaml       hf      qwen3vl_8b
launch configs/dmr_grid_qwen25vl_7b.yaml      hf      qwen25vl_7b
launch configs/dmr_grid_idefics3_8b.yaml      hf      idefics3_8b
launch configs/dmr_grid_llava_ov_7b.yaml      hf      llava_ov_7b
echo "launched $i runs. tail $OUT/*.log"
echo
echo "next: judge each arm (<=8 concurrent), then pair pages vs pages_memory WITHIN each model:"
echo "  for m in $OUT/*/; do for t in vmqar vniah; do for a in pages pages_memory; do"
echo "    OPENAI_API_KEY=... .venv/bin/python scripts/judge_dmr_derisk_correctness.py \\"
echo "      --predictions \$m/\$t/predictions_\$a.jsonl --output \$m/\$t/judged_\$a.jsonl --concurrency 8"
echo "  done; done; done"
