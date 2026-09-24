#!/usr/bin/env python
"""Re-judge only the rows whose verdict is an endpoint error — no GPU, no re-inference.

A `verdict: "error"` row is a *judging* failure, not a model failure: the prediction is sitting
right there in the file. B2-4B is the worked example — all 192 of its "lost" samples have a
non-empty `pred` and `intermediate_extract`, and every judge payload reads
`URLError: Connection refused`. Re-running the reader for those ids would spend eight GPUs
reproducing text we already have, and would silently change it (the 2-call arm is not greedy
end-to-end), breaking the pairing with the rest of the run.

Dropping them instead is not free either: it is what shrank B2's reported n to 981, and the
error rows are not missing at random (they cluster in whichever wall-clock window the endpoint
was down, which correlates with shard and therefore with sample order).

  extract -> a predictions file holding just the error rows
  judge   -> scripts/judge_dmr_derisk_correctness.py on that file, as usual
  splice  -> fold the fresh verdicts back in, leaving every other row byte-identical

Usage:
  .venv/bin/python scripts/dmr_rejudge_errors.py extract \
      --judged code/runs/20260630/dmr_levers/B2_4b/judged/b2_vmqar.judged.jsonl \
      --out    code/runs/20260810/dmr_lever_gaps/b2_error_rows.jsonl \
      --ids-out code/runs/20260810/dmr_lever_gaps/b2_missing_ids.txt

  OPENAI_API_KEY=... .venv/bin/python scripts/judge_dmr_derisk_correctness.py \
      --predictions code/runs/20260810/dmr_lever_gaps/b2_error_rows.jsonl \
      --output      code/runs/20260810/dmr_lever_gaps/b2_error_rows.judged.jsonl --concurrency 8

  .venv/bin/python scripts/dmr_rejudge_errors.py splice \
      --judged code/runs/20260630/dmr_levers/B2_4b/judged/b2_vmqar.judged.jsonl \
      --patch  code/runs/20260810/dmr_lever_gaps/b2_error_rows.judged.jsonl \
      --out    code/runs/20260810/dmr_lever_gaps/b2_vmqar.judged.repaired.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _verdict(row: dict[str, Any]) -> str | None:
    j = row.get("judge")
    return (j or {}).get("verdict") if isinstance(j, dict) else None


def cmd_extract(args: argparse.Namespace) -> int:
    rows = _rows(Path(args.judged))
    # --all strips every verdict, not just the broken ones. Needed whenever a file has to be
    # re-scored by a DIFFERENT judge: splicing fresh verdicts into stale ones produces a file
    # graded by two models, which is the cross-session pooling the protocol forbids -- only
    # hidden inside a single file where it is much harder to notice.
    bad = rows if args.all else [r for r in rows if _verdict(r) in (None, "error")]
    empty = [r for r in bad if not str(r.get("pred") or "").strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in bad:
            r = {k: v for k, v in r.items() if k != "judge"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[extract] {len(bad)}/{len(rows)} rows need a verdict -> {out}")

    if args.ids_out:
        ids_out = Path(args.ids_out)
        ids_out.parent.mkdir(parents=True, exist_ok=True)
        ids_out.write_text("".join(f"{r.get('id')}\n" for r in bad), encoding="utf-8")
        print(f"[extract] ids -> {ids_out}")

    reasons = collections.Counter(
        str((r.get("judge") or {}).get("reason") or "")[:60] for r in rows if _verdict(r) == "error"
    )
    for reason, n in reasons.most_common(5):
        print(f"           {n:5d}  {reason}")
    if empty:
        # These cannot be repaired by re-judging; they need the reader re-run for those ids.
        print(f"[warn] {len(empty)} of them have an EMPTY pred — re-judging cannot fix those.")
    return 0


def cmd_splice(args: argparse.Namespace) -> int:
    base = _rows(Path(args.judged))
    patch = {str(r.get("id")): r for r in _rows(Path(args.patch))}

    applied = still_bad = 0
    for r in base:
        if _verdict(r) in (None, "error"):
            p = patch.get(str(r.get("id")))
            if p is not None and _verdict(p) not in (None, "error"):
                r["judge"] = p["judge"]
                applied += 1
            else:
                still_bad += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in base:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = collections.Counter(_verdict(r) for r in base)
    print(f"[splice] repaired {applied}, still unjudged {still_bad} -> {out}")
    print(f"[splice] verdicts now: {dict(counts)}")
    scored = [r for r in base if _verdict(r) in ("correct", "partial", "incorrect")]
    if scored:
        acc = sum(1 for r in scored if _verdict(r) == "correct") / len(scored)
        # NOTE: this n counts rows carrying a real verdict, not rows in the file. A file with one
        # stuck row prints n = len-1, which looks like a lost row and is not one.
        print(f"[splice] n={len(scored)}  strict-correct={acc:.4f}")

    # The summary is the only provenance a later reader sees, and until 2026-09-05 splice left it
    # untouched -- so a fully repaired file still advertised the pre-repair `n_error`, i.e. the
    # provenance record lied in the one direction that matters (claiming missing verdicts that are
    # in fact present, or vice versa). Refresh the counts in place, keep the grader identity
    # fields, and record that a repair happened.
    if args.summary:
        sp = Path(args.summary)
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            summary = {}
        summary.update({
            "n": len(base),
            "n_judged": len(scored),
            "n_error": len(base) - len(scored),
            "judge_score_mean": (sum(1.0 if _verdict(r) == "correct"
                                     else 0.5 if _verdict(r) == "partial" else 0.0
                                     for r in scored) / len(scored)) if scored else None,
            "verdict_counts": {k: v for k, v in counts.items() if k},
            "repaired_rows": int(summary.get("repaired_rows", 0)) + applied,
        })
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[splice] refreshed {sp} (n_error now {summary['n_error']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pull error/unjudged rows into a predictions file")
    e.add_argument("--judged", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--ids-out", default=None)
    e.add_argument("--all", action="store_true",
                   help="strip EVERY verdict, not just error rows — use when re-scoring a file "
                        "with a different judge, where splicing would mix two graders in one file")
    e.set_defaults(fn=cmd_extract)

    s = sub.add_parser("splice", help="fold fresh verdicts back into the judged file")
    s.add_argument("--judged", required=True)
    s.add_argument("--patch", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--summary", default=None,
                   help="path to the judged file's .summary.json, refreshed in place so its "
                        "n_error stops contradicting the file it describes")
    s.set_defaults(fn=cmd_splice)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
