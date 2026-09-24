#!/usr/bin/env bash
# SELECTION / LOCALIZATION CONTROLS — is the memory gain the TEXT, or a POINTER to where to look?
#
# The memory arm is not "the same evidence as pixels, but as text". It scores every block of the
# 16 retrieved pages lexically against the question, keeps the top-16, caps each at 300 chars and
# tags each with its page number. So the +13–16 pp OCR gain (dmr_ocr_parity.md) has three
# candidate sources that no existing arm separates:
#   (1) recognition  — the VLM cannot read the characters off the page image;
#   (2) localization — it can read them but cannot find the needle in 16 pages;
#   (3) selection    — query-conditioned block picking is a second-stage retriever that hands the
#                      model the answer region directly.
# The free stratification (paper/scratch, 2026-09-20) already shows the gain is conditional on the
# gold block being INSIDE the packed memory (PDF arm: +0.38/+0.39 where packed, ~0 where not), which
# is compatible with all three. These arms pull them apart. Everything else is held fixed: same
# stock 2B reader, same ColQwen2 top-16 pages, same OCR store, same 1173 samples per task; the
# `pages` and `ocrmem` reference arms are regenerated in the same run so every arm shares the hardware.
#
#   arm      modes            selector      prefix   what it removes / adds vs the paper's OCR arm
#   pages    pages            —             —        the image arm, regenerated on the same hardware as the new arms
#   ocrmem   pages_memory     lexical       keyword  the paper's OCR arm, regenerated likewise (bridge to Table 2)
#   lex16n   pages_memory     lexical       neutral  only the wording ("by keyword / candidate evidence")
#   rand16n  pages_memory     random        neutral  query-conditioning (same 16-block budget)
#   randcn   pages_memory     random_chars  neutral  query-conditioning (same CHARACTER budget)
#   alln     pages_memory     all           neutral  selection and budget: every block, reading order
#   hilite   pages_highlight  lexical       neutral  the text: same 16 blocks, drawn as red boxes on the pages
#   crops    pages_crops      lexical       neutral  the text: same 16 blocks, cropped at the reader's own
#                                                    pixel density and appended after the pages
#   cropsnat pages_crops      lexical       neutral  optional: crops at native render resolution
#                                                    (localization + resolution), ARMS="... cropsnat"
#
# Decision rules (all paired, McNemar, same judge session for every arm incl. pages and ocrmem):
#   lex16n ≈ rand16n/randcn            -> the query-conditioned selection is not the driver
#   lex16n ≫ rand16n and lex16n ≈ alln -> selection is a convenience, not the mechanism: unselected
#                                         full text does the same job
#   hilite/crops recover most of lex16n -> localization; the paper cannot call the whole gap a reading gap
#   hilite/crops ≈ pages, lex16n ≫ both -> the model needs the TEXT, not the pointer: recognition
#
# Usage: bash scripts/run_dmr_selection_controls.sh            # all arms, sequential, one shard per GPU slot
#        ARMS="hilite crops" bash scripts/run_dmr_selection_controls.sh
#        PY=<python> GPUS="0 1 2 3 0 1 2 3" bash scripts/run_dmr_selection_controls.sh   # e.g. 4 GPUs, 2 shards each
set -uo pipefail
cd "$(dirname "$0")/.."

CFG="configs/dmr_eval_stock_qwen3vl_2b.yaml"
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
OCR_DOCS="code/runs/20260810/dmr_ocr_documents"
PARITY="code/runs/20260810/dmr_ocr_parity_2b/merged"
OUT="${OUT:-code/runs/20260920/selection_controls}"
PY="${PY:-.venv/bin/python}"
# One shard per entry; a GPU listed twice runs two shards (fine for a 2B reader on 80 GB cards).
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
ARMS="${ARMS:-pages lex16n hilite crops rand16n randcn alln ocrmem}"

[[ -d "$OCR_DOCS" ]] || { echo "missing OCR store: $OCR_DOCS"; exit 1; }
[[ -s "$PARITY/ocrmem_vmqar.jsonl" && -s "$PARITY/pages_vmqar.jsonl" ]] || { echo "missing parity merged arms under $PARITY"; exit 1; }
NDOC=$(find "$OCR_DOCS" -name '*.json' | wc -l)
(( NDOC >= 1173 )) || { echo "ABORT: OCR store incomplete ($NDOC/1173)"; exit 1; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | wc -l)
if (( BUSY > 0 )); then
  echo "WARNING: $BUSY process(es) already on the GPUs. This run wants GPUs ${GPUS[*]} to itself."
  echo "         Ctrl-C now, or set CONTROLS_FORCE=1 to share anyway."
  [[ "${CONTROLS_FORCE:-0}" == "1" ]] || { sleep 10; echo "aborting."; exit 1; }
fi

COMMON=(--eval-config "$CFG" \
  --benchmark-root data/fovedoc_project \
  --split all --max-samples 100000 --page-source retrieved --retrieved-pages-jsonl "$RET" \
  --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144 --max-new-tokens 64 \
  --memory-document-root "$OCR_DOCS" --require-memory-document-root \
  --memory-prefix-style neutral)

# arm -> (mode, extra flags)
arm_mode()  { case "$1" in pages) echo pages;; ocrmem|lex16n|rand16n|randcn|alln) echo pages_memory;; hilite) echo pages_highlight;; crops|cropsnat) echo pages_crops;; *) return 1;; esac; }
arm_flags() {
  case "$1" in
    pages)    echo "--memory-selector lexical";;
    ocrmem)   echo "--memory-selector lexical --memory-prefix-style keyword";;
    lex16n)   echo "--memory-selector lexical";;
    rand16n)  echo "--memory-selector random";;
    randcn)   echo "--memory-selector random_chars";;
    alln)     echo "--memory-selector all --memory-max-chars-per-block 0";;
    hilite)   echo "--memory-selector lexical";;
    crops)    echo "--memory-selector lexical --crop-scale reader";;
    cropsnat) echo "--memory-selector lexical --crop-scale native";;
    *) return 1;;
  esac
}

run_arm() { # arm
  local arm="$1" mode flags dst
  mode=$(arm_mode "$arm") || { echo "unknown arm: $arm"; return 1; }
  flags=$(arm_flags "$arm")
  dst="$OUT/$arm"
  if [[ -s "$dst/merged/${arm}_vmqar.jsonl" && -s "$dst/merged/${arm}_vniah.jsonl" ]]; then
    echo "  skip (already merged): $dst"; return 0
  fi
  echo "== arm $arm (mode=$mode $flags) =="
  mkdir -p "$dst"
  local pids=()
  for i in "${!GPUS[@]}"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONPATH=src nohup "$PY" \
      scripts/run_dmr_oracle_scratchpad_derisk.py "${COMMON[@]}" $flags \
      --datasets "V-MQAR,V-NIAH" --modes "$mode" \
      --num-shards "${#GPUS[@]}" --shard-idx "$i" \
      --output-dir "$dst/shard${i}" > "$dst/shard${i}.log" 2>&1 &
    pids+=($!); echo "  [shard$i] gpu=${GPUS[$i]} pid=$!"
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || { echo "  !! a shard failed — see $dst/shard*.log"; return 1; }
  "$PY" scripts/dmr_merge_arms.py --eval-dir "$dst" --out "$dst/merged" \
    --arm "$arm" --shard-glob 'shard*' --mode-file "predictions_${mode}.jsonl" --tasks vmqar,vniah || return 1

  # Pairing + construction asserts: the arm must sit on the parity run's exact samples, pages and
  # block store, and must differ from the paper's OCR arm in exactly the one thing it is meant to.
  "$PY" - "$arm" "$dst/merged" "$PARITY" <<'PY' || return 1
import json, sys, collections
arm, mdir, parity = sys.argv[1:4]
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
def keys(r): return [(int(f["page_id"]), str(f["block_id"])) for f in (r.get("memory_facts") or [])]
for task in ("vmqar", "vniah"):
    new = load(f"{mdir}/{arm}_{task}.jsonl"); base = load(f"{parity}/ocrmem_{task}.jsonl"); img = load(f"{parity}/pages_{task}.jsonl")
    assert set(new) == set(base) == set(img), f"{task}: id set differs from parity ({len(new)} vs {len(base)})"
    assert all(new[i]["selected_page_ids"] == img[i]["selected_page_ids"] for i in new), f"{task}: page sets diverged"
    assert collections.Counter(r["num_selected_pages"] for r in new.values()) == {16: len(new)}, f"{task}: page budget moved"
    assert set(r["memory_block_source"] for r in new.values()) == {"ocr"}, f"{task}: memory not served from OCR store"
    if arm != "pages":
        want = "keyword" if arm == "ocrmem" else "neutral"
        assert set(r["memory_prefix_style"] for r in new.values()) == {want}, f"{task}: prefix style != {want}"
    # Cross-hardware drift check for the two regenerated reference arms: identical prediction share vs parity.
    if arm in ("pages", "ocrmem"):
        ref = img if arm == "pages" else base
        ident = sum(1 for i in new if new[i]["pred"] == ref[i]["pred"])
        print(f"  {arm} {task}: predictions identical to the parity run on {ident}/{len(new)} ({ident/len(new):.1%})")
    same = sum(1 for i in new if keys(new[i]) == keys(base[i]))
    nfacts = collections.Counter(r["memory_num_facts"] for r in new.values())
    chars_new = sum(len(f["text"]) for r in new.values() for f in (r.get("memory_facts") or []))
    chars_base = sum(len(f["text"]) for r in base.values() for f in (r.get("memory_facts") or []))
    if arm in ("ocrmem", "lex16n", "hilite", "crops", "cropsnat"):
        assert same == len(new), f"{task}: {arm} must pack exactly the parity arm's blocks ({same}/{len(new)})"
    elif arm == "rand16n":
        assert same < 0.05 * len(new), f"{task}: random arm equals lexical on {same} ids"
        # a handful of documents have no OCR block on their retrieved pages -> 0 facts in EVERY arm
        assert set(nfacts) <= {0, 16}, f"{task}: random arm block count {dict(nfacts)}"
    elif arm == "randcn":
        assert same < 0.05 * len(new), f"{task}: random_chars arm equals lexical on {same} ids"
        assert 1.0 <= chars_new / chars_base <= 1.35, f"{task}: char budget ratio {chars_new/chars_base:.3f}"
    elif arm == "alln":
        assert sum(v for k, v in nfacts.items() if k > 16 or k == 0) >= 0.99 * len(new), f"{task}: all-blocks arm is not all blocks {dict(nfacts)}"
    if arm == "hilite":
        assert set(r["visual_memory_images"] for r in new.values()) == {0}
    if arm in ("crops", "cropsnat"):
        assert all(r["visual_memory_images"] == r["memory_num_facts"] for r in new.values()), f"{task}: crop count != block count"
    print(f"  {arm} {task}: n={len(new)} same_blocks_as_parity={same} facts={sorted(nfacts.items())[:3]}... chars_ratio={chars_new/max(1,chars_base):.3f} ok")
PY
}

for arm in $ARMS; do run_arm "$arm" || exit 1; done

cat <<NEXT

next — judge EVERY arm of the comparison in ONE session (the regenerated pages and ocrmem arms
included; never pair against the 2026-08-24 verdicts, the judge-noise floor is ~1 pt):
  mkdir -p $OUT/judged
  for t in vmqar vniah; do
    for arm in $ARMS; do
      OPENAI_API_KEY=... OPENAI_BASE_URL=... DMR_JUDGE_MODEL=... $PY scripts/judge_dmr_derisk_correctness.py \\
        --predictions $OUT/\$arm/merged/\${arm}_\$t.jsonl --output $OUT/judged/\${arm}_\$t.jsonl --concurrency 8
    done
  done
then:
  $PY scripts/dmr_selection_controls_analysis.py --judged-dir $OUT/judged --out results/tables/dmr_selection_controls.md
NEXT
