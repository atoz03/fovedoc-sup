#!/usr/bin/env python
"""Does the OCR memory arm fail exactly where OCR lost the evidence?

This replaces the scanned-vs-born-digital stratification the campaign originally planned, which
turned out to have no scanned stratum to stratify on (the release manifests are 100%
born-digital). What the data does support is a dose axis: OCR best-block recall of the gold
evidence varies from 0.55 to 0.87 across source families, and samples map 1:1 to documents, so
per-sample fidelity joins straight onto per-sample correctness.

If the OCR arm's deficit concentrates in low-fidelity samples, the loss is attributable to the
recogniser and the memory design is unharmed. If it is flat in fidelity, something else is
wrong and the recogniser is not the explanation.

Aggregation across evidence items defaults to `min`, not `mean`: V-MQAR is multi-hop, so a
sample is only answerable if *every* evidence item survived. `mean` would score a sample with
one perfectly-recovered and one destroyed block as comfortably mid-fidelity when it is in fact
unanswerable.

Usage:
  .venv/bin/python scripts/dmr_fidelity_dose_response.py \
      --fidelity code/runs/20260810/dmr_ocr_fidelity_per_evidence.jsonl \
      --arm ocr=code/runs/20260810/dmr_ocr_parity_2b/judged/ocrmem_vmqar.jsonl \
      --arm pdf=code/runs/20260810/dmr_ocr_parity_2b/judged/pdfmem_vmqar.jsonl \
      --arm img=code/runs/20260810/dmr_ocr_parity_2b/judged/pages_vmqar.jsonl \
      --metric judge --out results/tables/dmr_fidelity_dose_response.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]
BIN_LABELS = ["[0, .25)", "[.25, .5)", "[.5, .75)", "[.75, 1)", "1.0"]


def _bin(v: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= v < hi:
            return i
    return len(BINS) - 1


def _score(row: dict[str, Any], metric: str) -> float | None:
    """1.0 / 0.0 correctness, or None when the row carries no usable verdict."""
    if metric == "judge":
        j = row.get("judge")
        if not isinstance(j, dict):
            return None
        verdict = j.get("verdict")
        if verdict in (None, "error"):
            return None
        return 1.0 if verdict == "correct" else 0.0
    s = row.get("scoring")
    if not isinstance(s, dict) or not s.get("scorable"):
        return None
    return float(s.get("score") or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fidelity", required=True, help="dmr_ocr_fidelity_report.py --dump-jsonl output")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; judged (or raw) predictions for one arm")
    ap.add_argument("--metric", choices=["judge", "scoring"], default="judge",
                    help="'scoring' uses the harness's built-in vqa_soft — a proxy, useful "
                         "before the judge has run, not a substitute for it")
    ap.add_argument("--aggregate", choices=["min", "mean"], default="min")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_sample: dict[str, list[float]] = defaultdict(list)
    family: dict[str, str] = {}
    for line in Path(args.fidelity).open("r", encoding="utf-8"):
        r = json.loads(line)
        per_sample[r["sample_id"]].append(float(r["ocr_best_block_recall"]))
        family[r["sample_id"]] = r["family"]
    agg = min if args.aggregate == "min" else statistics.fmean
    fidelity = {k: agg(v) for k, v in per_sample.items()}

    arms: dict[str, dict[str, float]] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            print(f"[error] --arm expects NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        scored: dict[str, float] = {}
        unusable = 0
        for line in Path(path).open("r", encoding="utf-8"):
            row = json.loads(line)
            s = _score(row, args.metric)
            if s is None:
                unusable += 1
                continue
            scored[str(row.get("id"))] = s
        arms[name] = scored
        print(f"[arm] {name:8s} {len(scored)} scored, {unusable} unusable  ({path})")

    # Only samples every arm scored AND fidelity covers — otherwise bin-to-bin deltas compare
    # different sample sets and the dose-response is an artefact of who dropped out where.
    common = set(fidelity)
    for scored in arms.values():
        common &= set(scored)
    print(f"[join] {len(common)} samples common to fidelity and all {len(arms)} arms\n")
    if not common:
        return 1

    names = list(arms)
    lines: list[str] = []
    lines.append("# Fidelity dose-response — does the OCR arm fail where OCR lost the evidence?\n")
    lines.append(
        f"Dose axis = per-sample OCR best-block recall of the gold evidence, aggregated across a\n"
        f"sample's evidence items with `{args.aggregate}` (a multi-hop sample is only answerable if\n"
        f"*every* item survived). Correctness metric = `{args.metric}`. n={len(common)} samples\n"
        f"common to the fidelity dump and all {len(arms)} arms.\n"
    )
    header = "| OCR fidelity | n | " + " | ".join(names) + " |"
    lines.append(header)
    lines.append("|---" * (2 + len(names)) + "|")

    buckets: dict[int, list[str]] = defaultdict(list)
    for sid in common:
        buckets[_bin(fidelity[sid])].append(sid)
    for i, label in enumerate(BIN_LABELS):
        ids = buckets.get(i) or []
        if not ids:
            continue
        cells = [f"{statistics.fmean([arms[n][s] for s in ids]):.3f}" for n in names]
        lines.append(f"| {label} | {len(ids)} | " + " | ".join(cells) + " |")
    cells = [f"{statistics.fmean([arms[n][s] for s in common]):.3f}" for n in names]
    lines.append(f"| **all** | **{len(common)}** | " + " | ".join(f"**{c}**" for c in cells) + " |")

    if len(names) >= 2:
        a, b = names[0], names[1]
        lines.append(f"\n## Paired delta, {a} − {b}, by fidelity bin\n")
        lines.append("| OCR fidelity | n | delta | interpretation |")
        lines.append("|---|---|---|---|")
        for i, label in enumerate(BIN_LABELS):
            ids = buckets.get(i) or []
            if not ids:
                continue
            d = statistics.fmean([arms[a][s] - arms[b][s] for s in ids])
            lines.append(f"| {label} | {len(ids)} | {d:+.3f} | |")
        d_all = statistics.fmean([arms[a][s] - arms[b][s] for s in common])
        lines.append(f"| **all** | **{len(common)}** | **{d_all:+.3f}** | |")
        lines.append(
            f"\nA delta that shrinks toward zero as fidelity rises means the {a} arm's loss is the\n"
            f"recogniser, not the memory design. A flat delta means it is not.\n"
        )

    lines.append("\n## By source family\n")
    lines.append("| family | n | mean OCR fidelity | " + " | ".join(names) + " |")
    lines.append("|---" * (3 + len(names)) + "|")
    by_family: dict[str, list[str]] = defaultdict(list)
    for sid in common:
        by_family[family[sid]].append(sid)
    for fam, ids in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        cells = [f"{statistics.fmean([arms[n][s] for s in ids]):.3f}" for n in names]
        lines.append(
            f"| {fam} | {len(ids)} | {statistics.fmean([fidelity[s] for s in ids]):.3f} | "
            + " | ".join(cells) + " |"
        )

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
        print(f"[written] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
