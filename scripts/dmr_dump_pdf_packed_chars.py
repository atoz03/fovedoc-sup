#!/usr/bin/env python
"""Recover the PDF arm's packing budget without a GPU — the missing half of the parity check.

`memory_packed_chars` was added to the harness on 2026-08-10, so the OCR-parity run records it
but the 2026-06-30 PDF-arm predictions do not. Comparing packed budgets across the two arms
would otherwise need a whole GPU pass re-running a reader whose output we already have.

Memory construction is deterministic and CPU-only: same samples, same retrieval file, same
top-k, same lexical scorer. This replays `_build_memory_scratchpad` for BOTH block stores in a
single pass, so the two budgets are paired per sample by construction rather than merged after
the fact.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/dmr_dump_pdf_packed_chars.py \
      --benchmark-root <builder project> \
      --retrieved-pages-jsonl code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl \
      --ocr-root code/runs/20260810/dmr_ocr_documents \
      --out code/runs/20260810/dmr_packed_chars_paired.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The harness is a script, not a module, and re-implementing its page-subset + memory logic here
# is exactly the kind of drift that makes a control stop controlling anything. Load it directly.
_spec = importlib.util.spec_from_file_location(
    "dmr_derisk", ROOT / "scripts" / "run_dmr_oracle_scratchpad_derisk.py"
)
dmr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dmr)

from eccv26.data.benchmarks import LoadSpec, iter_samples  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--retrieved-pages-jsonl", required=True)
    ap.add_argument("--ocr-root", default=None, help="omit to dump the PDF arm alone")
    ap.add_argument("--datasets", default="V-MQAR,V-NIAH")
    ap.add_argument("--split", default="all")
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--retrieved-top-k", type=int, default=16)
    ap.add_argument("--memory-topk", type=int, default=16)
    ap.add_argument("--memory-max-chars-per-block", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    retrieved_map = dmr._load_retrieved_pages_map(
        [args.retrieved_pages_jsonl], int(args.retrieved_top_k)
    )
    print(f"[packed] retrieval map: {len(retrieved_map)} samples (top_k={args.retrieved_top_k})", flush=True)

    sources: list[tuple[str, str | None]] = [("pdf", None)]
    if args.ocr_root:
        sources.append(("ocr", args.ocr_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, list[int]] = {name: [] for name, _ in sources}
    facts_stats: dict[str, list[int]] = {name: [] for name, _ in sources}
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            spec = LoadSpec(
                benchmark_root=args.benchmark_root,
                dataset=dataset,
                split=str(args.split),
                max_samples=int(args.max_samples),
            )
            for n, sample in enumerate(iter_samples(spec), start=1):
                ranked = retrieved_map.get(str(sample.sample_id), [])
                # _build_page_subset drops ids the sample has no page for, so this matches the
                # reader's memory_pages exactly rather than the raw ranked list.
                _, sel_pids = dmr._build_page_subset(sample, set(ranked))
                if not sel_pids:
                    continue
                row: dict[str, Any] = {
                    "sample_id": str(sample.sample_id),
                    "dataset": dataset,
                    "document_id": str(sample.meta.get("document_id") or ""),
                    "num_pages": len(sel_pids),
                }
                for name, root in sources:
                    scratchpad, facts, block_source = dmr._build_memory_scratchpad(
                        sample=sample,
                        gold_page_ids=sel_pids,
                        benchmark_root=args.benchmark_root,
                        topk=int(args.memory_topk),
                        max_chars_per_block=int(args.memory_max_chars_per_block),
                        document_root=root,
                        strict_document_root=bool(root),
                    )
                    row[f"{name}_packed_chars"] = len(scratchpad)
                    row[f"{name}_num_facts"] = len(facts)
                    row[f"{name}_block_source"] = block_source
                    stats[name].append(len(scratchpad))
                    facts_stats[name].append(len(facts))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                if n % 200 == 0:
                    print(f"  [{dataset}] {n} scanned, {n_written} written", flush=True)

    print(f"\n[written] {out_path} ({n_written} samples)\n")
    print("| arm | packed chars mean | median | p10 | p90 | facts mean |")
    print("|---|---|---|---|---|---|")
    for name, _ in sources:
        v = sorted(stats[name])
        if not v:
            continue
        p = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
        print(
            f"| {name} | {statistics.fmean(v):.0f} | {statistics.median(v):.0f} | "
            f"{p(0.10)} | {p(0.90)} | {statistics.fmean(facts_stats[name]):.1f} |"
        )
    if len(sources) == 2:
        pairs = [
            (a, b) for a, b in zip(stats["pdf"], stats["ocr"]) if a or b
        ]
        wins = sum(1 for a, b in pairs if b > a)
        print(
            f"\nPaired: OCR packs more than PDF on {wins}/{len(pairs)} samples "
            f"({wins / max(1, len(pairs)):.1%}); mean delta "
            f"{statistics.fmean([b - a for a, b in pairs]):+.0f} chars."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
