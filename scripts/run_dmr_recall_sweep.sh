#!/usr/bin/env bash
# CONTROLLED RETRIEVAL-DEGRADATION SWEEP — does the memory advantage depend on retrieval recall?
#
# The external run (MMLongBench-Doc) left this as its most important loose end: on the 132
# questions where top-16 missed some gold evidence, the memory arm was -0.061 BEHIND the image
# arm, against +0.004 overall. If that is real it says the whole retrieval-reading gap is
# conditional on retrieval being solved -- and a deployed system is never in that regime.
#
# FoveDoc cannot answer this observationally: full-recall@16 is 0.998, so there are ~2 rows in
# the partial condition. So manufacture it. scripts/dmr_make_recall_sweep_manifest.py drops j of
# each sample's gold pages from the retrieved set and substitutes pages the retriever ranked
# BELOW top-16, holding the budget at exactly 16 pages. Recall becomes the only thing that moves.
#
# V-MQAR has exactly 2 gold pages for all 1173 samples -> j in {0,1,2} is a 3-point curve
# (recall 1.0 / 0.5 / 0.0) meaning the same thing for every sample. 938 samples are eligible
# (the rest are documents too short to draw substitutes from). V-NIAH has 1 gold page, so it
# gives the two endpoints only, over 1025 samples.
#
# Both arms (`pages`, `pages_memory`) run in ONE pass per level, so they see byte-identical page
# sets and the pairing is exact. Memory is OCR, not the PDF text layer -- this is the deployable
# arm and the one the paper's claim rests on.
#
# V-MQAR runs first: it is the 3-point curve and the paper's figure. V-NIAH is the single-hop
# endpoint check and is expendable if GPU time runs out.
#
# Usage: bash scripts/run_dmr_recall_sweep.sh [vmqar|vniah|all]
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-all}"

CFG="configs/dmr_eval_stock_qwen3vl_2b.yaml"
OCR_DOCS="code/runs/20260810/dmr_ocr_documents"
SWEEP="code/runs/20260902/recall_sweep"
OUT="$SWEEP/eval"
GPUS=(0 1 2 3 4 5 6 7)

[[ -d "$OCR_DOCS" ]] || { echo "missing OCR store: $OCR_DOCS"; exit 1; }
NDOC=$(find "$OCR_DOCS" -name '*.json' | wc -l)
(( NDOC >= 1173 )) || { echo "ABORT: OCR store incomplete ($NDOC/1173)"; exit 1; }

# Refuse to start if another job holds the GPUs -- this run assumes it has all 8, and silently
# sharing them turns a 30-minute sweep into an unbounded one.
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | wc -l)
if (( BUSY > 0 )); then
  echo "WARNING: $BUSY process(es) already on the GPUs. This sweep wants all 8."
  echo "         Ctrl-C now, or set SWEEP_FORCE=1 to share anyway."
  [[ "${SWEEP_FORCE:-0}" == "1" ]] || { sleep 10; echo "aborting."; exit 1; }
fi

COMMON=(--eval-config "$CFG" \
  --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 --max-new-tokens 64 \
  --memory-document-root "$OCR_DOCS" --require-memory-document-root \
  --modes "pages,pages_memory")

run_level() { # task dataset j
  local task="$1" dataset="$2" j="$3"
  local man="$SWEEP/${task}_drop${j}.jsonl"
  local ids="$SWEEP/${task}_sample_ids.txt"
  local dst="$OUT/${task}_drop${j}"
  [[ -s "$man" ]] || { echo "  missing manifest $man — run dmr_make_recall_sweep_manifest.py"; return 1; }
  # Resume on the file the merge actually writes. Checking a name nothing produces would make
  # every interrupted run start over from level 0, which under a deadline is the expensive kind
  # of silent bug.
  if [[ -s "$dst/merged/pages_memory_${task}.jsonl" && -s "$dst/merged/pages_${task}.jsonl" ]]; then
    echo "  skip (already merged): $dst"; return 0
  fi
  echo "== $task drop$j (recall $(awk "BEGIN{printf \"%.2f\", ($4-$j)/$4}" 2>/dev/null || echo '?')) =="
  mkdir -p "$dst"
  local pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONPATH=src nohup .venv/bin/python \
      scripts/run_dmr_oracle_scratchpad_derisk.py "${COMMON[@]}" \
      --datasets "$dataset" --retrieved-pages-jsonl "$man" --only-ids-file "$ids" \
      --num-shards "${#GPUS[@]}" --shard-idx "$i" \
      --output-dir "$dst/shard${i}" > "$dst/shard${i}.log" 2>&1 &
    pids+=($!); echo "  [shard$i] gpu=${GPUS[$i]} pid=$!"
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || { echo "  !! a shard failed — see $dst/shard*.log"; return 1; }
  for arm in pages pages_memory; do
    .venv/bin/python scripts/dmr_merge_arms.py --eval-dir "$dst" --out "$dst/merged" \
      --arm "$arm" --shard-glob 'shard*' --mode-file "predictions_${arm}.jsonl" --tasks "$task" \
      || return 1
  done
  # The whole point of the sweep is that recall is what moved. Assert it actually did, per level,
  # rather than discovering after judging that the manifest was not picked up.
  .venv/bin/python - "$dst/merged/pages_memory_${task}.jsonl" "$j" <<'PY' || return 1
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
j = int(sys.argv[2])
kept = collections.Counter(r["partial_recall_count"] for r in rows)
src = collections.Counter(r["memory_block_source"] for r in rows)
npages = collections.Counter(r["num_selected_pages"] for r in rows)
gold = len(rows[0]["gold_evidence_page_ids"])
print(f"  n={len(rows)} gold_retained={dict(kept)} pages={dict(npages)} memory_src={dict(src)}")
assert set(kept) == {gold - j}, f"recall did not move as intended: {dict(kept)}"
assert set(npages) == {16}, f"page budget drifted: {dict(npages)}"
assert set(src) <= {"ocr"}, f"memory is not 100% OCR: {dict(src)}"
PY
  echo "  ok"
}

case "$STAGE" in
  vmqar|all)
    for j in 0 1 2; do run_level vmqar V-MQAR "$j" 2 || exit 1; done
    ;;&
  vniah|all)
    for j in 0 1; do run_level vniah V-NIAH "$j" 1 || exit 1; done
    ;;
esac

cat <<'NEXT'

done. next:
  1) judge EVERY level and BOTH arms in ONE session (Gate 3 — the levels are compared to each
     other, so a judge change between them would be indistinguishable from a recall effect):
       export OPENAI_API_KEY=... OPENAI_BASE_URL=...
       for d in code/runs/20260902/recall_sweep/eval/*/merged/*.jsonl; do
         out="${d/merged/judged}"; mkdir -p "$(dirname "$out")"
         .venv/bin/python scripts/judge_dmr_derisk_correctness.py \
           --predictions "$d" --output "$out" --concurrency 8
       done
  2) sweep for `verdict: error` rows afterwards (a file can finish WITH errors and then be
     skipped forever as "already judged" — that is how B2 came to report n=981):
       bash scripts/run_dmr_judging.sh repair
  3) analyse: scripts/dmr_recall_sweep_analysis.py
NEXT
