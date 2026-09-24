#!/usr/bin/env python3
"""Is the zero-recall memory benefit format, or same-document topical context?

Three arms over the SAME samples, images and retrieved page ids, at two recall levels:

  images   the sweep's `pages` arm
  own      the sweep's `pages_memory` arm -- OCR text of the document's own retrieved pages
  foreign  this control      -- OCR text of a DIFFERENT document (deranged store, 0 self-maps,
                                0 same-family pairs)

At full recall `own` carries the answer and `foreign` cannot. At zero recall NEITHER carries the
answer, and that is the row that matters: the sweep measured own − images = +0.030 (V-MQAR) and
+0.043 (V-NIAH) there and attributed it to format. If that is right, `foreign` should reproduce
it, because foreign text is equally text. If instead the benefit came from the document's own
non-gold pages being topically related, `foreign` should fall back toward the image arm.

The decisive number is therefore `foreign − images` at ZERO recall, and its comparison with
`own − images` at the same level, paired within sample.

  python scripts/dmr_foreign_memory_analysis.py --out results/tables/dmr_foreign_memory.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(".")
SWEEP = "code/runs/20260902/recall_sweep/eval/{task}_drop{j}/judged/{arm}_{task}.jsonl"
FOREIGN = "code/runs/20260904/foreign_memory/eval/{task}_drop{j}/judged/pages_memory_{task}.jsonl"
# (task, zero-recall level, n gold pages)
TASKS = [("vmqar", 2, 2), ("vniah", 1, 1)]
VERDICTS = ("correct", "partial", "incorrect")


def _load(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        v = (r.get("judge") or {}).get("verdict")
        if v not in VERDICTS:
            raise SystemExit(f"unjudged/error row in {path} (id={r.get('id')!r}); "
                             f"run: bash scripts/run_dmr_judging.sh repair")
        out[r["id"]] = v == "correct"
    return out


def _mcnemar(a: dict[str, bool], b: dict[str, bool], ids: list[str]) -> tuple[float, float, float]:
    """Paired difference mean(a) - mean(b), its Wald CI half-width, and McNemar p."""
    n = len(ids)
    bb = sum(1 for i in ids if a[i] and not b[i])
    cc = sum(1 for i in ids if b[i] and not a[i])
    d = (bb - cc) / n
    se = math.sqrt(bb + cc) / n
    if bb + cc == 0:
        return d, 0.0, 1.0
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return d, 1.96 * se, math.erfc(math.sqrt(chi / 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = ["# Foreign-memory control — is the zero-recall benefit format or topical context?\n",
             "Same samples, same page images, same retrieved page ids, same lexical packing and "
             "the same 16-block budget at every level. The only variable is **whose text the "
             "memory contains**: the document's own retrieved pages (`own`), or a different "
             "document's (`foreign`, deranged store with 0 self-maps and 0 same-family pairs). "
             "All arms graded by gpt-5.6-luna.\n"]

    for task, zj, ngold in TASKS:
        lines.append(f"\n## {task.upper()}\n")
        lines.append("| recall | arm | strict | Δ vs images | 95% CI | p |")
        lines.append("|---|---|---|---|---|---|")
        store: dict[tuple[int, str], dict[str, bool]] = {}
        for j, label in ((0, "1.00"), (zj, "0.00")):
            img = _load(ROOT / SWEEP.format(task=task, j=j, arm="pages"))
            own = _load(ROOT / SWEEP.format(task=task, j=j, arm="pages_memory"))
            fpath = ROOT / FOREIGN.format(task=task, j=j)
            if not fpath.exists():
                lines.append(f"| {label} | *(foreign not yet run)* | | | | |")
                continue
            frn = _load(fpath)
            ids = sorted(set(img) & set(own) & set(frn))
            if len(ids) != len(img):
                raise SystemExit(f"{task} drop{j}: arms not aligned "
                                 f"({len(ids)} common vs {len(img)} images)")
            store[(j, "img")], store[(j, "own")], store[(j, "frn")] = img, own, frn
            base = sum(img[i] for i in ids) / len(ids)
            lines.append(f"| {label} | images | {base:.3f} | — | | |")
            for key, arm in (("own", own), ("foreign", frn)):
                d, ci, p = _mcnemar(arm, img, ids)
                s = sum(arm[i] for i in ids) / len(ids)
                lines.append(f"| {label} | {key} | {s:.3f} | **{d:+.3f}** | "
                             f"[{d-ci:+.3f}, {d+ci:+.3f}] | {p:.3g} |")

        if (zj, "frn") in store:
            ids = sorted(store[(zj, "img")])
            d_own, ci_o, p_o = _mcnemar(store[(zj, "own")], store[(zj, "img")], ids)
            d_frn, ci_f, p_f = _mcnemar(store[(zj, "frn")], store[(zj, "img")], ids)
            d_of, ci_of, p_of = _mcnemar(store[(zj, "own")], store[(zj, "frn")], ids)
            lines.append(
                f"\n**At zero recall** (neither memory can contain the answer): own is "
                f"{d_own:+.3f} over images, foreign is {d_frn:+.3f}. Own − foreign = "
                f"**{d_of:+.3f}** [{d_of-ci_of:+.3f}, {d_of+ci_of:+.3f}], p = {p_of:.3g}.\n")
            frac = (d_frn / d_own) if d_own else float("nan")
            lines.append(
                f"Foreign text reproduces **{frac:.0%}** of the own-document zero-recall effect. "
                "The closer that is to 100%, the more the residual is pure format — text being "
                "easier for the reader to consume than pixels, regardless of what it says. The "
                "closer to 0%, the more it was the document's own non-gold pages supplying "
                "topical context, and the weaker the format claim becomes.\n")

    text = "\n".join(lines)
    print(text)
    if args.out:
        p = ROOT / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"[foreign] wrote {p}")


if __name__ == "__main__":
    main()
