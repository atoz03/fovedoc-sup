#!/usr/bin/env python3
"""Merge sharded DMR clean-eval predictions into one jsonl per (arm, task).

Arms come from three logical runs (see scripts/run_dmr_clean_eval.sh):
  pages    = runA_*/<task>/predictions_pages.jsonl          (no memory baseline)
  naive    = runA_*/<task>/predictions_pages_memory.jsonl   (lexical memory)
  selector = runB_*/vmqar/predictions_pages_memory.jsonl + runC_*/vniah/predictions_pages_memory.jsonl
  assoc    = runB_*/vmqar/predictions_pages_memory_assoc.jsonl + runC_*/vniah/predictions_pages_memory_assoc.jsonl

De-dups by sample id (sharding is disjoint, but be safe). Writes <out>/<arm>_<task>.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _collect(patterns: list[str]) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            for line in open(fp, encoding="utf-8"):
                if not line.strip():
                    continue
                o = json.loads(line)
                if o.get("skip_reason"):
                    continue
                sid = str(o.get("id"))
                if sid in seen:
                    continue
                seen.add(sid)
                rows.append(o)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="e.g. code/runs/20260630/dmr_eval_2b")
    ap.add_argument("--out", required=True, help="output dir for merged per-arm jsonl")
    ap.add_argument("--arm", help="generic mode: name this arm instead of using the clean-eval layout")
    ap.add_argument("--shard-glob", default="*_shard*",
                    help="generic mode: shard-dir glob under --eval-dir")
    ap.add_argument("--mode-file", default="predictions_pages_memory.jsonl",
                    help="generic mode: per-task predictions filename to collect")
    ap.add_argument("--tasks", default="vmqar,vniah",
                    help="generic mode: per-task subdirectory names. The default is the two "
                         "FoveDoc tasks; external benchmarks use their own (e.g. mmlongbench_doc), "
                         "and a mismatch silently merges nothing rather than erroring")
    args = ap.parse_args()
    e = args.eval_dir.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.arm:
        # Generic mode: one arm, one predictions file, sharded the same way for both tasks.
        tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
        spec = {
            (args.arm, task): [f"{e}/{args.shard_glob}/{task}/{args.mode_file}"]
            for task in tasks
        }
        merged_any = False
        for (arm, task), pats in spec.items():
            rows = _collect(pats)
            if not rows:
                continue
            merged_any = True
            fp = out / f"{arm}_{task}.jsonl"
            with open(fp, "w", encoding="utf-8") as w:
                for o in rows:
                    w.write(json.dumps(o, ensure_ascii=False) + "\n")
            print(f"{arm:9s} {task}: {len(rows):5d} -> {fp}")
        if not merged_any:
            # Silence here used to mean "wrong --tasks" and looked identical to success.
            raise SystemExit(
                f"[merge] nothing matched under {e}/{args.shard_glob}/<task>/{args.mode_file} "
                f"for tasks={list(tasks)} — check --tasks against the run's output layout"
            )
        return

    spec = {
        ("pages", "vmqar"): [f"{e}/runA_pages_naive_shard*/vmqar/predictions_pages.jsonl"],
        ("pages", "vniah"): [f"{e}/runA_pages_naive_shard*/vniah/predictions_pages.jsonl"],
        ("naive", "vmqar"): [f"{e}/runA_pages_naive_shard*/vmqar/predictions_pages_memory.jsonl"],
        ("naive", "vniah"): [f"{e}/runA_pages_naive_shard*/vniah/predictions_pages_memory.jsonl"],
        ("selector", "vmqar"): [f"{e}/runB_vmqar_sel_shard*/vmqar/predictions_pages_memory.jsonl"],
        ("selector", "vniah"): [f"{e}/runC_vniah_sel_shard*/vniah/predictions_pages_memory.jsonl"],
        ("assoc", "vmqar"): [f"{e}/runB_vmqar_sel_shard*/vmqar/predictions_pages_memory_assoc.jsonl"],
        ("assoc", "vniah"): [f"{e}/runC_vniah_sel_shard*/vniah/predictions_pages_memory_assoc.jsonl"],
    }
    for (arm, task), pats in spec.items():
        rows = _collect(pats)
        fp = out / f"{arm}_{task}.jsonl"
        with open(fp, "w", encoding="utf-8") as w:
            for o in rows:
                w.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"{arm:9s} {task}: {len(rows):5d} -> {fp}")


if __name__ == "__main__":
    main()
