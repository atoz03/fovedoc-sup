#!/usr/bin/env bash
# OCR-PARITY CONTROL for the retrieval-reading gap.
#
# The headline gap (stock reader on retrieved page IMAGES vs the same pages as extracted TEXT)
# was measured with memory packed from the born-digital PDF text layer. That makes it partly a
# claim about free perfect OCR. This run repacks the SAME memory from RapidOCR over the SAME
# rendered images (--memory-document-root) and re-measures, so the claim becomes one about a
# deployable OCR-then-read pipeline.
#
# Everything except the block store is held fixed: same retrieval (ColQwen2 ranked top-16),
# same stock reader, same budget, same prompts, same 1173 samples -> strictly paired.
# The `pages` (no-memory) arm is unchanged by construction and is reused from the clean run.
#
# Usage: bash scripts/run_dmr_ocr_parity_eval.sh <2b|4b>
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${1:?usage: run_dmr_ocr_parity_eval.sh <2b|4b>}"
CFG="configs/dmr_eval_stock_qwen3vl_${SIZE}.yaml"
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
OCR_DOCS="code/runs/20260810/dmr_ocr_documents"
OUT="code/runs/20260810/dmr_ocr_parity_${SIZE}"

[[ -d "$OCR_DOCS" ]] || { echo "missing OCR store: $OCR_DOCS (run scripts/dmr_build_ocr_document_cache.py)"; exit 1; }
NDOC=$(find "$OCR_DOCS" -name '*.json' | wc -l)
echo "== OCR-parity eval ($SIZE) == OCR store: $NDOC documents"
# Hard gate: a partial store would silently serve PDF-text-layer blocks for the missing
# documents, so the "OCR arm" would not be an OCR arm. --require-memory-document-root below
# also fails per-sample, but failing here costs seconds instead of a wasted GPU pass.
(( NDOC >= 1173 )) || { echo "ABORT: OCR store incomplete ($NDOC/1173). Finish the build first."; exit 1; }

mkdir -p "$OUT"
COMMON=(--eval-config "$CFG" \
  --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved --retrieved-pages-jsonl "$RET" \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 --max-new-tokens 64)

# 8 shards over the two tasks; memory blocks come from the OCR store instead of the PDF layer.
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$s" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
    "${COMMON[@]}" --datasets "V-MQAR,V-NIAH" --modes "pages_memory" \
    --memory-document-root "$OCR_DOCS" --require-memory-document-root \
    --num-shards 8 --shard-idx "$s" \
    --output-dir "$OUT/ocrmem_shard${s}" \
    > "$OUT/ocrmem_shard${s}.log" 2>&1 &
  echo "  [ocrmem shard$s] gpu=$s pid=$!"
done
echo "launched. tail $OUT/*.log"
echo
cat <<NEXT

next:
  1) merge:  .venv/bin/python scripts/dmr_merge_arms.py --eval-dir $OUT --out $OUT/merged \\
               --arm ocrmem --shard-glob 'ocrmem_shard*'

  2) assert the arm really is 100% OCR before judging anything:
       .venv/bin/python -c "import json,collections,glob; \\
         c=collections.Counter(json.loads(l)['memory_block_source'] \\
           for f in glob.glob('$OUT/merged/*.jsonl') for l in open(f)); print(c)"
       # must be {'ocr': N} with no 'pdf_text_layer_fallback'

  3) judge ALL THREE arms in the SAME session (<=8 concurrent; the endpoint 503s above that).
     The claim is "OCR memory still beats reading the pages as images", so the image arm is
     part of the comparison and needs same-session verdicts too. Do NOT pair against the
     2026-06-30 judged files: those were scored six weeks ago on a flaky proxy, and with a
     ~1pt judge-noise floor the OCR-vs-PDF delta is small enough for cross-session drift to
     swamp it.
       for t in vmqar vniah; do
         for arm in ocrmem pdfmem pages; do
           OPENAI_API_KEY=... .venv/bin/python scripts/judge_dmr_derisk_correctness.py \\
             --predictions $OUT/merged/\${arm}_\$t.jsonl \\
             --output $OUT/judged/\${arm}_\$t.jsonl --concurrency 8
         done
       done
     (copy code/runs/20260630/dmr_eval_${SIZE}/merged/{naive,pages}_<task>.jsonl in as
      pdfmem_/pages_<task>.jsonl — same predictions, fresh verdicts)

  4) analyse as a FIDELITY DOSE-RESPONSE, not a scanned-vs-born-digital split. The release
     manifests contain ZERO scanned (docvqa) documents — all 1173 are born-digital, every
     family scores exactly 1.000 PDF best-block recall — so there is no stratum where OCR can
     win and this run measures a pure cost. What it can show instead: OCR best-block recall
     varies 0.55-0.87 by family, so join per-sample fidelity to per-sample correctness and
     test whether the OCR arm fails exactly where OCR lost the evidence.
       .venv/bin/python scripts/dmr_ocr_fidelity_report.py ... \\
         --dump-jsonl code/runs/20260810/dmr_ocr_fidelity_per_evidence.jsonl
     then join on \`sample_id\` (== the \`id\` field in predictions).
NEXT
