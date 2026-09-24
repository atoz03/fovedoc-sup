#!/usr/bin/env bash
# FOREIGN-MEMORY CONTROL — is the zero-recall residual format, or same-document topical context?
#
# The recall sweep showed the memory advantage survives removing every gold page: +0.030 (V-MQAR)
# and +0.043 (V-NIAH), p < 1e-3. That was read as a FORMAT effect. But at zero recall the memory
# is still text of the same document -- same topic, same entities -- so "related text helps" is an
# equally good explanation, and it implies something much weaker about OCR pipelines.
#
# This run replaces the memory with a DIFFERENT document's text (deranged store: 0 self-maps,
# 0 same-family pairs, so every document's memory comes from another corpus entirely). Images,
# retrieved page ids, packing, budget and sample set are all identical to the sweep, so the only
# variable is whose text it is.
#
#   images only                    -> the sweep's `pages` arm, REUSED verbatim (no re-inference)
#   images + own-document memory   -> the sweep's `pages_memory` arm
#   images + foreign-doc memory    -> THIS RUN
#
# Only `pages_memory` is generated here. The `pages` arm is bit-identical by construction -- same
# manifest, same images, and the harness's `_mode_user_text` returns the base prompt unchanged for
# mode `pages`, so the memory store cannot reach it. Generation on this harness is deterministic
# (verified: the sweep's drop0 pages arm reproduced the parity run byte-for-byte on 1963 ids), so
# re-running it would burn 8 GPUs to reproduce a file we already have.
#
# Levels chosen to bracket the question:
#   drop0  full recall  -- does foreign text HURT when the evidence is present and visible?
#   dropN  zero recall  -- the condition where the +0.030/+0.043 residual was measured
#
# Usage: bash scripts/run_dmr_foreign_memory.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CFG="configs/dmr_eval_stock_qwen3vl_2b.yaml"
FOREIGN="code/runs/20260904/foreign_ocr_documents"
SWEEP="code/runs/20260902/recall_sweep"
OUT="code/runs/20260904/foreign_memory/eval"
GPUS=(0 1 2 3 4 5 6 7)

[[ -d "$FOREIGN" ]] || { echo "missing foreign store: $FOREIGN — run dmr_make_foreign_memory_store.py"; exit 1; }
NDOC=$(find "$FOREIGN" -name '*.json' -not -name '_derangement.json' | wc -l)
(( NDOC >= 1173 )) || { echo "ABORT: foreign store incomplete ($NDOC/1173)"; exit 1; }
# A store that accidentally kept a document's own text is not a control. Re-assert here rather
# than trusting the builder, because this is the one property the whole experiment rests on.
.venv/bin/python - "$FOREIGN" <<'PY' || exit 1
import json, sys
m = json.load(open(f"{sys.argv[1]}/_derangement.json", encoding="utf-8"))["mapping"]
self_maps = sum(1 for k, v in m.items() if k == v)
print(f"  derangement: {len(m)} documents, {self_maps} self-maps")
raise SystemExit(1 if self_maps else 0)
PY

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | wc -l)
if (( BUSY > 0 )); then
  echo "WARNING: $BUSY process(es) already on the GPUs. This run wants all 8."
  echo "         Ctrl-C now, or set FOREIGN_FORCE=1 to share anyway."
  [[ "${FOREIGN_FORCE:-0}" == "1" ]] || { sleep 10; echo "aborting."; exit 1; }
fi

COMMON=(--eval-config "$CFG" \
  --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 --max-new-tokens 64 \
  --memory-document-root "$FOREIGN" --require-memory-document-root \
  --modes "pages_memory")

run_level() { # task dataset j
  local task="$1" dataset="$2" j="$3"
  local man="$SWEEP/${task}_drop${j}.jsonl"
  local ids="$SWEEP/${task}_sample_ids.txt"
  local dst="$OUT/${task}_drop${j}"
  [[ -s "$man" ]] || { echo "  missing manifest $man"; return 1; }
  if [[ -s "$dst/merged/pages_memory_${task}.jsonl" ]]; then
    echo "  skip (already merged): $dst"; return 0
  fi
  echo "== $task drop$j (foreign memory) =="
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
  .venv/bin/python scripts/dmr_merge_arms.py --eval-dir "$dst" --out "$dst/merged" \
    --arm pages_memory --shard-glob 'shard*' --mode-file "predictions_pages_memory.jsonl" \
    --tasks "$task" || return 1

  # Assert the run actually consumed the foreign store and stayed paired with the sweep.
  .venv/bin/python - "$dst/merged/pages_memory_${task}.jsonl" \
                     "$SWEEP/eval/${task}_drop${j}/merged/pages_${task}.jsonl" "$j" <<'PY' || return 1
import json, sys, collections
new = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
base = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8")}
j = int(sys.argv[3])
assert set(new) == set(base), f"id set differs from the sweep baseline: {len(new)} vs {len(base)}"
pages = collections.Counter(r["num_selected_pages"] for r in new.values())
src = collections.Counter(r["memory_block_source"] for r in new.values())
facts = collections.Counter(r["memory_num_facts"] for r in new.values())
kept = collections.Counter(r["partial_recall_count"] for r in new.values())
# Same images as the sweep arm, or the contrast is not paired.
same_pages = sum(1 for i in new if new[i]["selected_page_ids"] == base[i]["selected_page_ids"])
assert same_pages == len(new), f"page sets diverged from the sweep on {len(new)-same_pages} ids"
assert set(pages) == {16}, f"page budget moved: {dict(pages)}"
assert set(src) == {"ocr"}, f"memory not served from the store: {dict(src)}"
print(f"  n={len(new)} pages={dict(pages)} memory_src={dict(src)} "
      f"facts={dict(sorted(facts.items()))} gold_retained={dict(sorted(kept.items()))}")
print("  ok")
PY
}

# V-MQAR: 2 gold pages -> drop0 = full recall, drop2 = zero recall.
run_level vmqar V-MQAR 0 || exit 1
run_level vmqar V-MQAR 2 || exit 1
# V-NIAH: 1 gold page -> drop0 = full recall, drop1 = zero recall.
run_level vniah V-NIAH 0 || exit 1
run_level vniah V-NIAH 1 || exit 1

cat <<'EOF'

done. next:
  1) judge BOTH new files in ONE session, same judge as the sweep (gpt-5.6-luna), because these
     are compared directly against the sweep's arms:
       export OPENAI_API_KEY=... OPENAI_BASE_URL=...
       DMR_JUDGE_MODEL=gpt-5.6-luna bash scripts/run_dmr_judging.sh foreign
  2) bash scripts/run_dmr_judging.sh repair    # until it reports 0
  3) .venv/bin/python scripts/dmr_foreign_memory_analysis.py
EOF
