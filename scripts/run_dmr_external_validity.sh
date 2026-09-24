#!/usr/bin/env bash
# EXTERNAL VALIDITY: does the retrieval-reading gap exist off our own benchmark?
#
# The gap is currently a FoveDoc result, and FoveDoc is ours — the obvious reviewer question.
# This reruns the identical two arms (retrieved pages as IMAGES vs the same pages as extracted
# TEXT memory) on MMLongBench-Doc: 1091 QA over all 135 documents, median 28 pages (max 468),
# real human questions we did not write. The 1035/127 figure this file used to quote was the
# loader's 120-page cap silently excluding the 8 longest documents — see the prewarm stage.
#
# Note the memory here is OCR-derived for EVERY document (only 25 of 135 ship a PDF-text-layer
# store, and mixing sources across documents would confound the arm). That makes this run the
# external counterpart of the OCR-parity control, not of the text-layer headline.
#
# Stages: 1b CPU (pre-warm image cache) -> 1 CPU (OCR store) -> 2 GPU (retrieval) -> 3 GPU (read).
# Run prewarm BEFORE the OCR store: --require-complete drops documents whose pages are not all
# rendered, and prewarm is what renders them.
#
# Caveat for analysis: MMLongBench documents are long — median 28 pages, and once the loader's
# 120-page cap is lifted they run to 468 — so full-recall@16 will NOT saturate the way it does
# on FoveDoc. The "retrieval is solved, the gap is pure reading" framing does not transfer;
# stratify by full-recall. This is the honest version of the caveat: leaving the cap at 120
# excludes the 8 longest documents, which would understate exactly this problem.
# Usage: bash scripts/run_dmr_external_validity.sh <ocr|prewarm|retrieve|read|all>
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?usage: run_dmr_external_validity.sh <ocr|prewarm|retrieve|read|all>}"
LMU=data/LMUData
OUT=code/runs/20260810/external
MANIFEST="$OUT/mmlongbench_manifest.jsonl"
OCR_DOCS="$OUT/mmlongbench_ocr_documents"
RANKED="$OUT/mmlongbench_colqwen2_ranked.jsonl"
mkdir -p "$OUT"

# Shard index and GPU id are decoupled: the reader-family grid occupies GPUs 0-5 for hours,
# and hard-coding shard s -> GPU s would drop a retrieval job on top of one of them. Override
# with e.g. GPUS="6 7" to run on whatever is actually free; the shard count follows the list.
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NSHARD=${#GPU_LIST[@]}
echo "[gpus] ${GPU_LIST[*]}  (${NSHARD} shards)"

case "$STAGE" in
  ocr|all)
    echo "== stage 1: OCR document store for MMLongBench-Doc (CPU) =="
    # --require-complete: a document missing pages from the image cache would yield a memory
    # built over a partial document, which reads as a memory failure rather than a cache gap.
    .venv/bin/python scripts/dmr_build_ocr_document_cache.py \
      --source tsv --tsv "$LMU/MMLongBench_DOC.tsv" --require-complete \
      --out-dir "$OCR_DOCS" --workers 96
    ;;&
  prewarm|all)
    echo "== stage 1b: pre-warm the page-image cache single-threaded (CPU) =="
    # 56 of 1091 rows have pages missing from the cache. The loader renders those from PDF
    # base64 on demand, so 8 reader shards would race to write the SAME cache paths at
    # startup and can tear each other's files. One serial pass makes the parallel run safe.
    #
    # max_pages=600, not the loader's 120 default: at 120 the 8 documents longer than that
    # render partially, --require-complete drops them, and the external benchmark quietly
    # becomes 127/135 documents — losing exactly the long documents this experiment is about.
    PYTHONPATH=src .venv/bin/python -c "
from eccv26.data.benchmarks import iter_samples, LoadSpec
n = longest = 0
for s in iter_samples(LoadSpec(benchmark_root='$LMU', dataset='MMLongBench_DOC',
                               split='all', max_samples=100000, max_pages=600)):
    n += 1
    longest = max(longest, len(s.images or []))
    if n % 100 == 0:
        print(f'  prewarmed {n}, longest doc {longest} pages', flush=True)
print(f'  prewarm done: {n} samples, longest {longest} pages')
"
    ;;&
  retrieve|all)
    echo "== stage 2: ColQwen2 page retrieval (GPU) =="
    [[ -s "$MANIFEST" ]] || .venv/bin/python scripts/export_external_page_retrieval_manifest.py \
      --tsv "$LMU/MMLongBench_DOC.tsv" --dataset MMLongBench_DOC --out "$MANIFEST"
    for s in $(seq 0 $((NSHARD - 1))); do
      g="${GPU_LIST[$s]}"
      CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=src nohup .venv/bin/python scripts/run_colqwen2_page_retrieval.py \
        --manifest "$MANIFEST" --output-path "$OUT/ranked_shard${s}.jsonl" \
        --device cuda:0 --shard-index "$s" --num-shards "$NSHARD" \
        > "$OUT/retrieve_shard${s}.log" 2>&1 &
      echo "  [retrieve shard$s] gpu=$g pid=$!"
    done
    wait
    cat "$OUT"/ranked_shard*.jsonl > "$RANKED"
    echo "  merged -> $RANKED ($(wc -l < "$RANKED") rows)"
    ;;&
  read|all)
    echo "== stage 3: reader arms, stock Qwen3-VL-2B, top-16 (GPU) =="
    [[ -s "$RANKED" ]] || { echo "missing $RANKED — run the retrieve stage first"; exit 1; }
    for s in $(seq 0 $((NSHARD - 1))); do
      g="${GPU_LIST[$s]}"
      CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
        --eval-config configs/dmr_eval_stock_qwen3vl_2b.yaml \
        --benchmark-root "$LMU" --datasets MMLongBench_DOC --split all --max-samples 100000 \
        --page-source retrieved --retrieved-pages-jsonl "$RANKED" --retrieved-top-k 16 \
        --memory-topk 16 --memory-document-root "$OCR_DOCS" --require-memory-document-root \
        --max-pages 600 \
        --image-max-pixels 262144 --max-new-tokens 64 --modes "pages,pages_memory" \
        --output-dir "$OUT/read_shard${s}" --num-shards "$NSHARD" --shard-idx "$s" \
        > "$OUT/read_shard${s}.log" 2>&1 &
      echo "  [read shard$s] gpu=$g pid=$!"
    done
    ;;
esac
echo "done/launched. artifacts under $OUT"
