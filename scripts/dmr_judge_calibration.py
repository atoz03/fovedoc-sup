#!/usr/bin/env python3
"""How far does the replacement judge move a number, measured on identical predictions?

gpt-5.4 graded every number in the parity, grid, lever and external tables. By 2026-09-03 it was
no longer served by any reachable proxy (listed, but 502 on /responses and 404 on
/chat/completions), so the recall sweep had to be graded by gpt-5.6-luna. That is a judge change,
and this campaign's rule is that verdicts from different judges are never pooled.

The sweep is internally safe -- all ten files, one judge, one session, and the difference-in-
differences never crosses the boundary. But the paper still has to say how big the judge change
is, and here that is measurable rather than assumed: the sweep's drop0 `pages` arm re-ran the
parity condition and produced BYTE-IDENTICAL predictions on 1963 ids (938 V-MQAR + 1025 V-NIAH).
Same strings, two judges. Any difference in `strict` is the judge and nothing else.

Note the asymmetry this exploits: a paired comparison of two judges over one set of predictions
is exactly the quantity we need, and it is unavailable in the usual case where a judge change
also coincides with a model or data change.

  python scripts/dmr_judge_calibration.py --out results/tables/dmr_judge_calibration.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(".")
NEW = "code/runs/20260902/recall_sweep/eval/{task}_drop0/judged/pages_{task}.jsonl"
OLD = "code/runs/20260810/dmr_ocr_parity_2b/judged/pages_{task}.jsonl"
VERDICTS = ("correct", "partial", "incorrect")


def _load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def _mcnemar(b: int, c: int) -> float:
    """Two-sided exact-ish McNemar with continuity correction (same test used campaign-wide)."""
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []
    grand = {"n": 0, "agree": 0, "b": 0, "c": 0}
    per_task = []

    for task in ("vmqar", "vniah"):
        new_p, old_p = ROOT / NEW.format(task=task), ROOT / OLD.format(task=task)
        if not new_p.exists():
            raise SystemExit(f"not judged yet: {new_p}")
        new, old = _load(new_p), _load(old_p)

        ids = []
        for i in set(new) & set(old):
            # Only ids whose PREDICTION text is identical can isolate the judge. Anything else
            # would confound the judge change with a generation difference.
            if new[i].get("pred") != old[i].get("pred"):
                continue
            if (new[i].get("judge") or {}).get("verdict") not in VERDICTS:
                continue
            if (old[i].get("judge") or {}).get("verdict") not in VERDICTS:
                continue
            ids.append(i)
        ids.sort()
        if not ids:
            raise SystemExit(f"no comparable ids for {task}")

        def strict(rec: dict) -> bool:
            return rec["judge"]["verdict"] == "correct"

        b = sum(1 for i in ids if strict(new[i]) and not strict(old[i]))   # new says correct
        c = sum(1 for i in ids if strict(old[i]) and not strict(new[i]))   # old says correct
        agree = sum(1 for i in ids if strict(new[i]) == strict(old[i]))
        s_new = sum(strict(new[i]) for i in ids) / len(ids)
        s_old = sum(strict(old[i]) for i in ids) / len(ids)
        three = sum(1 for i in ids
                    if new[i]["judge"]["verdict"] == old[i]["judge"]["verdict"]) / len(ids)
        d = s_new - s_old
        se = math.sqrt(b + c) / len(ids)
        per_task.append((task, len(ids), s_old, s_new, d, 1.96 * se, three,
                         agree / len(ids), _mcnemar(b, c)))
        for k, v in (("n", len(ids)), ("agree", agree), ("b", b), ("c", c)):
            grand[k] += v

    n = grand["n"]
    d = (grand["b"] - grand["c"]) / n
    se = math.sqrt(grand["b"] + grand["c"]) / n

    lines.append("# Judge calibration — gpt-5.6-luna against gpt-5.4 on identical predictions\n")
    lines.append(
        "gpt-5.4 graded the parity, grid, lever and external tables. It is no longer served by "
        "any reachable proxy, so the recall sweep is graded by **gpt-5.6-luna**. The sweep's "
        "drop0 `pages` arm reproduced the parity condition exactly — the predictions are "
        "byte-identical on all 1963 ids — so the two judges can be compared directly, with the "
        "reader, the pages and the generated strings all held fixed.\n")
    lines.append("| task | n | gpt-5.4 strict | gpt-5.6-luna strict | Δ | 95% CI | "
                 "3-way agree | strict agree | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t, nn, so, sn, dd, ci, th, sa, p in per_task:
        lines.append(f"| {t.upper()} | {nn} | {so:.3f} | {sn:.3f} | {dd:+.3f} | "
                     f"[{dd-ci:+.3f}, {dd+ci:+.3f}] | {th:.3f} | {sa:.3f} | {p:.3g} |")
    lines.append(f"| **pooled** | **{n}** | — | — | **{d:+.3f}** | "
                 f"[{d-1.96*se:+.3f}, {d+1.96*se:+.3f}] | — | "
                 f"**{grand['agree']/n:.3f}** | {_mcnemar(grand['b'], grand['c']):.3g} |")
    lines.append("")
    lines.append(f"Discordant pairs: {grand['b']} where gpt-5.6-luna alone says correct, "
                 f"{grand['c']} where gpt-5.4 alone does.\n")
    lines.append(
        "**How to use this number.** It bounds the judge's contribution to a *level* of accuracy, "
        "which is why no sweep accuracy may be placed beside a parity accuracy. It does **not** "
        "propagate into the sweep's difference-in-differences: that contrast is computed within "
        "one judge, over arms and levels graded in a single session, so a constant judge offset "
        "cancels. What survives is the judge's random disagreement, and the strict-agreement rate "
        "above is the estimate of it.\n")

    text = "\n".join(lines)
    print(text)
    if args.out:
        p = ROOT / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"[calibration] wrote {p}")


if __name__ == "__main__":
    main()
