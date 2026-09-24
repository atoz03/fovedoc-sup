#!/usr/bin/env bash
# Everything that needs the judge, in one ordered pass. No GPU.
#
# The key comes from the environment and is never written to disk or echoed. Export it for the
# duration of the shell that runs this and nothing else:  export OPENAI_API_KEY=...
#
# JUDGE CALL CONVENTION (standing, set 2026-09-05). For GPT-series graders always use the
# /responses dialect and pass reasoning effort EXPLICITLY:
#     export DMR_JUDGE_API=responses DMR_JUDGE_REASONING_EFFORT=<level>
# Never rely on the provider default -- this campaign has had three graders withdrawn under it,
# and an unpinned default is one more thing that can move between sessions without a trace.
# Levels: none|minimal|low|medium|high|xhigh|max.
#
#   * xhigh is the default choice for NEW work.
#   * medium is REQUIRED for anything pooled with the recall sweep or the foreign-memory control,
#     because those 9728 + 3926 rows were graded at medium. Effort is part of the grader's
#     identity: measured on 48 rows, medium and xhigh agree 0.958 and give identical strict
#     accuracy, but xhigh is 5.4x slower (5 vs 27 rows/min at concurrency 24), so re-grading an
#     existing comparison to reach it buys nothing.
#
# CONC=24 is safe on the 2026-09-05 proxy (24/24 concurrent returned 200). The older proxy 503'd
# above 8; CONC is per-endpoint, not a constant.
#
# Order is by value, so a mid-run outage costs the least important work:
#   canary  5 rows against one file — this endpoint has already produced two distinct failure
#           modes (503 above concurrency 8, and a connection-refused outage that silently wrote
#           192 `verdict: error` rows into B2). Catch a dead endpoint in seconds.
#   parity  the load-bearing control. THREE arms in this one session — ocrmem, pdfmem and the
#           image arm — because the claim is "OCR-then-read beats native visual reading", so
#           the image arm is part of the comparison, not a fixed baseline. Pairing against the
#           2026-06-30 verdicts instead would put a ~1pt/100 judge-noise floor under a delta
#           that may be smaller than that.
#   grid    reader-family generalization, 6 models x 2 tasks x 2 arms.
#   b2      repair the 192 rows lost to the endpoint outage (re-judge, never re-infer).
#   repair  sweep every judged file for `verdict: error` rows and re-judge just those. File-
#           level resumability alone is not enough — a file can finish WITH errors in it and
#           then be skipped forever as "already judged". That is how B2 came to report n=981.
#   levers  re-judge the June lever comparators. Lever effects run as small as +0.026 against a
#           ~1pt/100 judge-noise floor, so pairing a new gpt-5.4 arm against a June gpt-5.5 one
#           can invent or erase the entire result.
#   dose    the fidelity dose-response, which needs parity's verdicts to exist first.
#   sweep   the controlled retrieval-degradation sweep: 5 levels x 2 arms. Every level is
#           compared against every other, so a judge change *between levels* would be
#           indistinguishable from a recall effect — all 10 files must be judged in ONE session.
#           Do not use the ad-hoc for-loop printed by the sweep runner: it does not set
#           DMR_JUDGE_MODEL, so it defaults to gpt-5.5, which hard-503s every row and writes a
#           full file of `verdict: error` that then looks "already judged".
#   foreign the foreign-memory control (same images, another document's text). Compared directly
#           against the sweep's arms, so it must be graded by the SAME judge as the sweep --
#           pass DMR_JUDGE_MODEL=gpt-5.6-luna explicitly rather than relying on the default.
#
# Every stage is resumable: an existing non-empty output is skipped, so re-running after an
# outage picks up where it stopped.
#
#   controls the selection / localization controls (2026-09-20): 8 arms x 2 tasks, all regenerated in one
#           run on the A800 (pages and ocrmem included), so ALL 16 files go through one session. Not part
#           of `all`. Graded with gpt-5.6-luna at medium (gpt-5.4 has been offline since 2026-09-02; on
#           1963 identical predictions luna agrees with it on 0.988 of strict verdicts, delta -0.001, so
#           the bridge arm `ocrmem` still ties to Table 2 within judge noise). Pass DMR_JUDGE_MODEL
#           explicitly. ~18.8k rows, ~12 h at CONC=24 medium; xhigh would take ~2.5 days.
#
# Usage: bash scripts/run_dmr_judging.sh [canary|parity|grid|b2|levers|external|sweep|foreign|repair|dose|all|controls]
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-all}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY in this shell (in-session only; never commit it)}"
: "${OPENAI_BASE_URL:?export OPENAI_BASE_URL (e.g. https://host/v1); /responses is appended}"
# gpt-5.5 -- the model the 2026-06-30 verdicts used -- is listed by the current proxy but
# returns a hard 503 on every path. gpt-5.4 works. That is a JUDGE CHANGE: nothing scored here
# may be pooled with the June numbers, which is already the rule Gate 3 imposes (re-judge every
# arm of a comparison in one session) but now applies to the headline table too.
export DMR_JUDGE_MODEL="${DMR_JUDGE_MODEL:-gpt-5.4}"

PY="${PY:-.venv/bin/python}"   # e.g. PY=python on the A800 box
JUDGE="$PY scripts/judge_dmr_derisk_correctness.py"
CONC="${CONC:-8}"   # the endpoint 503s above this
PARITY=code/runs/20260810/dmr_ocr_parity_2b
GRID=code/runs/20260810/dmr_reader_grid
LEVER=code/runs/20260810/dmr_lever_gaps
SWEEP=code/runs/20260902/recall_sweep/eval
FOREIGN=code/runs/20260904/foreign_memory/eval
JUNE=code/runs/20260630/dmr_eval_2b/merged

judge() { # predictions output
  local pred="$1" out="$2"
  [[ -s "$pred" ]] || { echo "  skip (no predictions): $pred"; return 0; }
  [[ -s "$out"  ]] && { echo "  skip (already judged): $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  echo "  judging $(wc -l < "$pred") rows -> $out"
  $JUDGE --predictions "$pred" --output "$out" --concurrency "$CONC" || {
    echo "  !! judge failed on $pred — fix and re-run this stage (it resumes)"; return 1; }
}

case "$STAGE" in
  canary|all)
    echo "== canary: 5 rows, fail fast if the endpoint is down =="
    tmp=$(mktemp -d); head -5 "$PARITY/merged/ocrmem_vmqar.jsonl" > "$tmp/canary.jsonl"
    $JUDGE --predictions "$tmp/canary.jsonl" --output "$tmp/canary.judged.jsonl" --concurrency 2 || exit 1
    .venv/bin/python - "$tmp/canary.judged.jsonl" <<'PY'
import collections, json, sys
c = collections.Counter(
    (json.loads(l).get("judge") or {}).get("verdict") for l in open(sys.argv[1], encoding="utf-8")
)
print("  canary verdicts:", dict(c))
if c.get("error"):
    print("  !! endpoint is returning errors — stop here, do not spend the sweep")
    raise SystemExit(1)
PY
    rc=$?; rm -rf "$tmp"; [[ $rc -eq 0 ]] || exit 1
    echo "  canary OK"
    ;;&
  parity|all)
    echo "== parity: 3 arms x 2 tasks, same session =="
    # The PDF-memory and image arms reuse June's predictions verbatim; only the verdicts are new.
    mkdir -p "$PARITY/merged"
    for t in vmqar vniah; do
      [[ -s "$PARITY/merged/pdfmem_$t.jsonl" ]] || cp "$JUNE/naive_$t.jsonl" "$PARITY/merged/pdfmem_$t.jsonl"
      [[ -s "$PARITY/merged/pages_$t.jsonl"  ]] || cp "$JUNE/pages_$t.jsonl"  "$PARITY/merged/pages_$t.jsonl"
    done
    for t in vmqar vniah; do for arm in ocrmem pdfmem pages; do
      judge "$PARITY/merged/${arm}_$t.jsonl" "$PARITY/judged/${arm}_$t.jsonl"
    done; done
    ;;&
  grid|all)
    echo "== grid: 6 models x 2 tasks x 2 arms =="
    # Absolute scores are NOT comparable across families (image tokenisation differs by
    # construction); only the within-model pages vs pages_memory pair is.
    for m in "$GRID"/*/; do
      [[ -d "$m" ]] || continue
      for t in vmqar vniah; do for arm in pages pages_memory; do
        judge "$m/$t/predictions_$arm.jsonl" "$m/$t/judged_$arm.jsonl"
      done; done
    done
    ;;&
  b2|all)
    echo "== b2: re-judge all 1173 (not a splice — the June 981 were graded by a different model) =="
    bash scripts/run_dmr_lever_gaps.sh b2
    ;;&
  levers|all)
    echo "== levers: bring the June comparator arms onto the current judge =="
    bash scripts/run_dmr_lever_gaps.sh levers
    ;;&
  external|all)
    echo "== external: MMLongBench-Doc, both arms =="
    # Stratify by full_recall when analysing: external full-recall@16 is 0.844, not FoveDoc's
    # 0.998/1.000, so a pooled delta mixes reading with a retrieval shortfall. And 244 of the
    # 1091 questions are MMLongBench's deliberately unanswerable items, which measure abstention
    # rather than reading and need separate treatment.
    #   Careful: "unanswerable" (244, by answers[0]) and "empty gold evidence set" (246) are
    #   DIFFERENT sets -- 7 unanswerable questions do carry gold pages, and 9 answerable ones
    #   carry none. The clean reading stratum is the 838 that are both. See
    #   scripts/dmr_external_analysis.py, which does all of this and is the only thing that
    #   should be quoted from this run.
    for arm in ext extmem; do
      judge "code/runs/20260810/external/merged/${arm}_mmlongbench_doc.jsonl" \
            "code/runs/20260810/external/judged/${arm}_mmlongbench_doc.jsonl"
    done
    ;;&
  sweep|all)
    echo "== sweep: retrieval-degradation levels, both arms, one session =="
    # V-MQAR has 2 gold pages so it degrades in 3 steps (recall 1.0/0.5/0.0); V-NIAH has 1, so
    # 2 steps (1.0/0.0). Levels are contrasted against each other, hence one session for all.
    for spec in "vmqar 0 1 2" "vniah 0 1"; do
      set -- $spec; task="$1"; shift
      for j in "$@"; do
        for arm in pages pages_memory; do
          judge "$SWEEP/${task}_drop${j}/merged/${arm}_${task}.jsonl" \
                "$SWEEP/${task}_drop${j}/judged/${arm}_${task}.jsonl"
        done
      done
    done
    ;;&
  foreign|all)
    echo "== foreign-memory control: same images, another document's text =="
    # Compared directly against the sweep's arms, so it MUST use the sweep's judge
    # (gpt-5.6-luna). The repair guard below refuses to mix judges, but that only catches a
    # mismatch after the fact -- set DMR_JUDGE_MODEL explicitly when running this stage.
    for spec in "vmqar 0 2" "vniah 0 1"; do
      set -- $spec; task="$1"; shift
      for j in "$@"; do
        judge "$FOREIGN/${task}_drop${j}/merged/pages_memory_${task}.jsonl" \
              "$FOREIGN/${task}_drop${j}/judged/pages_memory_${task}.jsonl"
      done
    done
    ;;&
  repair|all)
    # File-level resumability is not enough: a file can finish WITH error rows in it and then
    # be skipped forever as "already judged". That is precisely how B2 came to report n=981.
    # Sweep every judged file, re-judge only its error rows, splice them back. Two passes,
    # because a transient outage can span one.
    echo "== repair: re-judge error rows in every judged file =="
    for pass in 1 2; do
      total=0
      # -not -name '*.predictions.jsonl': those are extraction scratch files with every verdict
      # deliberately stripped, so the sweep reads them as 100% unjudged and re-judges all of
      # them. That cost ~2346 wasted calls on the first run before it was caught.
      # Scope must list every run dir that holds verdicts. It was `20260810` alone until the
      # 2026-09-02 recall sweep landed under a new date dir and was silently skipped by repair;
      # the 2026-09-04 foreign-memory control landed under another one and needed the same fix.
      # Any new date dir that holds verdicts MUST be added here.
      for jf in $(find code/runs/20260810 code/runs/20260902 code/runs/20260904 code/runs/20260920 -path '*judged*' -name '*.jsonl' \
                    -not -name '*.predictions.jsonl' -not -name '*.summary.json' 2>/dev/null | sort); do
        n=$(.venv/bin/python - "$jf" <<'PY'
import json, sys
print(sum(1 for l in open(sys.argv[1], encoding="utf-8")
          if (json.loads(l).get("judge") or {}).get("verdict") in (None, "error")))
PY
)
        [[ "$n" -gt 0 ]] || continue
        # Repairing across judges is the B2 trap in a new coat: splicing a verdict from the
        # CURRENT model into a file graded by another silently produces a mixed-judge file.
        # Every summary written from 2026-09-03 records judge_model; refuse anything that
        # disagrees, and refuse pre-provenance files too unless explicitly forced.
        # Identity is (model, reasoning effort): the same model at a different effort is a
        # different grader, so both must match before a verdict may be spliced in.
        # `medium` is the provider default, so a file graded before the effort field existed
        # (recorded as absent) is the same grader as one graded at an explicit `medium`.
        # Normalise both sides, or every pre-2026-09-05 file looks like a mismatch and repair
        # silently refuses to touch files it is supposed to fix.
        owner=$(.venv/bin/python - "$jf.summary.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(f"{d.get('judge_model') or ''}@{d.get('judge_reasoning_effort') or 'medium'}")
except Exception: print("@")
PY
)
        mine="$DMR_JUDGE_MODEL@${DMR_JUDGE_REASONING_EFFORT:-medium}"
        if [[ "$owner" != "$mine" && "${DMR_REPAIR_FORCE:-0}" != "1" ]]; then
          echo "  SKIP (judge mismatch: file=${owner:-unknown} current=$mine): $jf"
          # Known edge case: a skipped file also skips the summary refresh below, so a file
          # repaired under one grader and later re-checked under another keeps a stale n_error.
          # Harmless today (skipped files are not modified), but do not read a skipped file's
          # summary as current.
          continue
        fi
        echo "  pass$pass: $n error rows in $jf"
        total=$((total + n))
        tmp=$(mktemp -d)
        .venv/bin/python scripts/dmr_rejudge_errors.py extract --judged "$jf" --out "$tmp/rows.jsonl" >/dev/null
        $JUDGE --predictions "$tmp/rows.jsonl" --output "$tmp/rows.judged.jsonl" --concurrency "$CONC" >/dev/null 2>&1
        .venv/bin/python scripts/dmr_rejudge_errors.py splice --judged "$jf" \
          --patch "$tmp/rows.judged.jsonl" --out "$tmp/fixed.jsonl" --summary "$jf.summary.json"
        mv "$tmp/fixed.jsonl" "$jf"; rm -rf "$tmp"
      done
      echo "  pass$pass repaired $total rows"
      [[ "$total" -eq 0 ]] && break
    done
    ;;&
  dose|all)
    echo "== dose-response: does the OCR arm fail where OCR lost the evidence? =="
    for t in vmqar vniah; do
      [[ -s "$PARITY/judged/ocrmem_$t.jsonl" ]] || { echo "  skip $t (parity not judged)"; continue; }
      .venv/bin/python scripts/dmr_fidelity_dose_response.py \
        --fidelity code/runs/20260810/dmr_ocr_fidelity_per_evidence.jsonl \
        --arm "ocr=$PARITY/judged/ocrmem_$t.jsonl" \
        --arm "pdf=$PARITY/judged/pdfmem_$t.jsonl" \
        --arm "img=$PARITY/judged/pages_$t.jsonl" \
        --metric judge --out "results/tables/dmr_dose_response_$t.md"
    done
    ;;
  controls)
    CTRL="${CTRL:-code/runs/20260920/selection_controls}"
    echo "== controls: selection / localization arms, one session, judge=$DMR_JUDGE_MODEL effort=${DMR_JUDGE_REASONING_EFFORT:-<unset>} =="
    [[ "${DMR_JUDGE_API:-}" == "responses" ]] || { echo "  set DMR_JUDGE_API=responses DMR_JUDGE_REASONING_EFFORT=medium (standing convention)"; exit 1; }
    for arm in pages ocrmem lex16n hilite crops rand16n randcn alln cropsnat; do
      for t in vmqar vniah; do
        pred="$CTRL/$arm/merged/${arm}_$t.jsonl"
        [[ -s "$pred" ]] || continue
        judge "$pred" "$CTRL/judged/${arm}_$t.jsonl" || exit 1
      done
    done
    PYTHONPATH=src $PY scripts/dmr_selection_controls_analysis.py --judged-dir "$CTRL/judged" \
      --out results/tables/dmr_selection_controls.md
    ;;
esac
echo "done."
