#!/usr/bin/env bash
# Close the three gaps left by the lever campaign (see docs/experiments/results/20260630_*).
#
#   niah    B1 (Qwen3-VL-4B-Thinking) and C (LoRA-2B) were only ever run on V-MQAR. Without
#           V-NIAH we cannot say whether the join-closers are join-specific or a general
#           reading gain — which is the whole claim.
#   decode  B1 sampled at t=0.6 while its naive baseline was greedy, so +0.026 confounds
#           reasoning with decoding. This runs stock-4B naive memory at the SAME sampling
#           settings; the residual B1 gain over THIS arm is the reasoning effect.
#   b2      B2-4B reported n=981 because 192 samples were dropped. Those need RE-JUDGING, not
#           re-inference — see the stage below; it uses no GPU.
#   levers  re-judge the JUNE comparator arms under the current judge. Every lever delta pairs a
#           new arm against a June one, and the judge model changed (gpt-5.5 no longer resolves),
#           so without this every lever number is a two-model comparison. No GPU.
#
# Usage: bash scripts/run_dmr_lever_gaps.sh <niah|decode|b2|levers|all>
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:?usage: run_dmr_lever_gaps.sh <niah|decode|b2|levers|all>}"
ROOT=data/fovedoc_project
RET="code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
OUT="code/runs/20260810/dmr_lever_gaps"
mkdir -p "$OUT"

BASE=(--benchmark-root "$ROOT" --split all --max-samples 100000 --page-source retrieved \
  --retrieved-pages-jsonl "$RET" --retrieved-top-k 16 --memory-topk 16 --image-max-pixels 262144)

# Shard index is decoupled from GPU id: other stages of this campaign hold GPUs for hours, so
# hard-coding shard s -> GPU s would stack jobs on a busy device. Override with GPUS="0 1 2 3".
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NSHARD=${#GPU_LIST[@]}

shardN() { # name config datasets modes extra...
  local name="$1" cfg="$2" ds="$3" modes="$4"; shift 4
  mkdir -p "$OUT/$name"
  for s in $(seq 0 $((NSHARD - 1))); do
    local g="${GPU_LIST[$s]}"
    CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=src nohup .venv/bin/python scripts/run_dmr_oracle_scratchpad_derisk.py \
      --eval-config "$cfg" "${BASE[@]}" --datasets "$ds" --modes "$modes" "$@" \
      --output-dir "$OUT/$name/shard${s}" --num-shards "$NSHARD" --shard-idx "$s" \
      > "$OUT/$name/shard${s}.log" 2>&1 &
    echo "  [$name shard$s] gpu=$g pid=$!"
  done
  wait
  echo "  [$name] all shards exited"
}

case "$WHICH" in
  niah|all)
    echo "== V-NIAH coverage for the two join-closers =="
    # B1: reasoning reader. Thinking traces need the long budget + --strip-think.
    shardN B1_thinking_vniah configs/dmr_eval_stock_qwen3vl_4b_thinking.yaml V-NIAH pages_memory \
      --max-new-tokens 1536 --strip-think
    # C: LoRA-2B trained on disjoint aux to compose from memory.
    shardN C_lora2b_vniah configs/dmr_eval_c_2b.yaml V-NIAH pages_memory --max-new-tokens 64
    ;;&
  decode|all)
    echo "== decoding-matched control for B1 (stock 4B, naive memory, sampling not greedy) =="
    # configs/dmr_eval_stock_qwen3vl_4b_sampled.yaml must mirror the 4b config with
    # do_sample: true / temperature: 0.6 / top_p: 0.95 (B1's decoding settings).
    shardN decode_control_4b_sampled configs/dmr_eval_stock_qwen3vl_4b_sampled.yaml V-MQAR pages_memory \
      --max-new-tokens 64
    ;;&
  b2|all)
    echo "== B2 stragglers: the 192 rows lost to endpoint errors (CPU/network only) =="
    # Checked 2026-08-24: all 192 have a non-empty pred AND intermediate_extract, and every
    # judge payload is the identical `URLError: Connection refused`. The reader never failed —
    # the judge endpoint was down. So this is a re-judge, not a re-run. Re-running inference
    # would burn 8 GPUs regenerating text we already have and would silently perturb it (the
    # 2-call arm is not greedy end-to-end), breaking pairing with the other 981.
    #
    # But do NOT splice fresh verdicts into the surviving 981. Those were graded by gpt-5.5,
    # which no longer resolves; anything scored now is gpt-5.4. A spliced file would be graded
    # by two different models — the cross-session pooling Gate 3 forbids, hidden inside one
    # file where it is far harder to spot than across two. Re-judge all 1173 instead: it costs
    # ~8 minutes and the predictions are already on disk.
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment (never write it to disk)}"
    .venv/bin/python scripts/judge_dmr_derisk_correctness.py \
      --predictions code/runs/20260630/dmr_levers/B2_4b/merged/b2_vmqar.jsonl \
      --output "$OUT/judged/b2_vmqar.jsonl" --concurrency "${CONC:-8}"
    echo "  B2 now reports on the full 1173, single-judge, rather than the biased n=981 subset."
    ;;&
  levers|all)
    # Every lever delta pairs a NEW gpt-5.4 arm against a JUNE gpt-5.5 comparator. Same problem
    # as B2, one level up: the judge-noise floor is ~1pt/100 and the lever effects are as small
    # as +0.026, so a model change between arms can invent or erase the entire result. Re-judge
    # the June comparators here, before any lever number is computed.
    echo "== re-judge the June lever comparators under the current judge =="
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment}"
    mkdir -p "$OUT/judged"
    rejudge_fresh() { # name predictions-or-judged
      local name="$1" src="$2"
      [[ -s "$src" ]] || { echo "  skip $name (missing $src)"; return 0; }
      [[ -s "$OUT/judged/$name.jsonl" ]] && { echo "  skip $name (done)"; return 0; }
      local pred="$src"
      if [[ "$src" == *judged* ]]; then   # strip stale verdicts rather than splicing into them
        # Written OUTSIDE judged/ on purpose: a verdict-stripped file sitting in judged/ looks
        # like a fully-unjudged arm to the repair sweep.
        mkdir -p "$OUT/rejudge_src"
        pred="$OUT/rejudge_src/$name.predictions.jsonl"
        .venv/bin/python scripts/dmr_rejudge_errors.py extract --judged "$src" --out "$pred" --all >/dev/null
      fi
      echo "  re-judging $name ($(wc -l < "$pred") rows)"
      .venv/bin/python scripts/judge_dmr_derisk_correctness.py \
        --predictions "$pred" --output "$OUT/judged/$name.jsonl" --concurrency "${CONC:-8}"
    }
    L=code/runs/20260630/dmr_levers
    rejudge_fresh naive4b_vmqar   code/runs/20260630/dmr_eval_4b/merged/naive_vmqar.jsonl
    rejudge_fresh naive4b_vniah   code/runs/20260630/dmr_eval_4b/merged/naive_vniah.jsonl
    rejudge_fresh B1_4b_vmqar     "$L/B1_4b_thinking/merged/b1_vmqar.jsonl"
    rejudge_fresh C_2b_vmqar      "$L/C_2b_eval/merged/c_vmqar.jsonl"
    rejudge_fresh C_4b_vmqar      "$L/C_4b_eval/merged/c_vmqar.jsonl"
    rejudge_fresh A_lex32_vmqar   "$L/A_4b/judged/lex32_vmqar.judged.jsonl"
    rejudge_fresh A_union_vmqar   "$L/A_4b/judged/union_vmqar.judged.jsonl"
    echo "  June comparators now share a judge with the new arms."
    ;;
esac
echo "launched. tail $OUT/*/shard*.log"
