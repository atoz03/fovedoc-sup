#!/usr/bin/env python3
"""Analyse the controlled retrieval-degradation sweep.

Question: is the memory-over-images advantage a property of reading, or only of the saturated-
retrieval regime? Every recall level contains the SAME samples with the SAME page budget, so
this is a within-sample difference-in-differences and needs no cross-sample assumptions:

    d_i(r) = correct(memory, i, r) - correct(images, i, r)   in {-1, 0, +1}

`d_i(r)` averaged over i is the memory advantage at recall r; the paired contrast
`d_i(r=1.0) - d_i(r=0.0)` is the interaction, i.e. how much of the advantage is bought by having
the evidence present. Both are reported, plus McNemar within each level.

Emits a markdown table and a PDF figure sized for a two-column ICASSP page.

Usage:
  python scripts/dmr_recall_sweep_analysis.py --task vmqar \
      --out results/tables/dmr_recall_sweep.md --fig paper/figures/fig_recall_sweep.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NGOLD = {"vmqar": 2, "vniah": 1}


def _load(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        v = str((r.get("judge") or {}).get("verdict", "")).lower()
        if v in ("", "error"):
            raise SystemExit(
                f"[sweep] {path} still has unjudged/error rows — run "
                f"`bash scripts/run_dmr_judging.sh repair` before analysing"
            )
        out[str(r["id"])] = v == "correct"
    return out


def _chi2_sf_1df(x: float) -> float:
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


def _mcnemar(base: list[bool], arm: list[bool]) -> tuple[float, float, float, float]:
    n = len(base)
    b = sum(1 for x, y in zip(base, arm) if not x and y)
    c = sum(1 for x, y in zip(base, arm) if x and not y)
    d = (b - c) / n
    se = math.sqrt(max(((b + c) - (b - c) ** 2 / n) / (n * n), 0.0))
    p = _chi2_sf_1df((abs(b - c) - 1) ** 2 / (b + c)) if b + c else 1.0
    return d, d - 1.96 * se, d + 1.96 * se, p


def _paired_z(x: list[int], y: list[int]) -> tuple[float, float, float]:
    """Paired contrast of two per-sample effect vectors (same samples, two recall levels)."""
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    m = sum(d) / n
    v = sum((t - m) ** 2 for t in d) / (n - 1)
    se = math.sqrt(v / n)
    z = m / se if se else 0.0
    return m, se, math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vmqar", "vniah"], default="vmqar")
    ap.add_argument("--eval-dir", default="code/runs/20260902/recall_sweep/eval")
    ap.add_argument("--out", default="")
    ap.add_argument("--fig", default="")
    args = ap.parse_args()

    task = args.task
    ngold = NGOLD[task]
    levels: list[tuple[float, int, dict[str, bool], dict[str, bool]]] = []
    for j in range(ngold + 1):
        d = ROOT / args.eval_dir / f"{task}_drop{j}" / "judged"
        img = d / f"pages_{task}.jsonl"
        mem = d / f"pages_memory_{task}.jsonl"
        if not (img.exists() and mem.exists()):
            print(f"[sweep] missing judged files for drop{j} ({d}) — skipping")
            continue
        levels.append(((ngold - j) / ngold, j, _load(img), _load(mem)))
    if not levels:
        raise SystemExit("[sweep] nothing judged yet")

    ids = sorted(set.intersection(*[set(a) & set(b) for _, _, a, b in levels]), key=str)
    print(f"[sweep] {task}: {len(levels)} levels, {len(ids)} samples common to all")

    rows: list[tuple[float, int, float, float, float, float, float, float]] = []
    eff: dict[float, list[int]] = {}
    for recall, j, img, mem in levels:
        bi = [img[i] for i in ids]
        mi = [mem[i] for i in ids]
        d, lo, hi, p = _mcnemar(bi, mi)
        eff[recall] = [int(m) - int(b) for b, m in zip(bi, mi)]
        rows.append((recall, j, sum(bi) / len(bi), sum(mi) / len(mi), d, lo, hi, p))

    L: list[str] = []
    L.append(f"# Controlled retrieval-degradation sweep — {task.upper()}")
    L.append("")
    L.append(
        f"Same {len(ids)} samples at every level, page budget fixed at 16, only the number of "
        "gold pages present varies. Both arms see identical page sets within a level, so the "
        "comparison is paired twice over: across arms and across levels."
    )
    L.append("")
    L.append("| recall | gold pages present | images | +memory | Δ | 95% CI | p |")
    L.append("|---|---|---|---|---|---|---|")
    for recall, j, bi, mi, d, lo, hi, p in rows:
        L.append(f"| {recall:.2f} | {ngold - j}/{ngold} | {bi:.3f} | {mi:.3f} | "
                 f"{d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |")
    L.append("")

    if len(rows) >= 2:
        hi_r, lo_r = max(eff), min(eff)
        m, se, p = _paired_z(eff[hi_r], eff[lo_r])
        L.append(f"**Interaction (recall {hi_r:.2f} vs {lo_r:.2f}), paired within sample:** "
                 f"the memory advantage changes by **{m:+.3f}** "
                 f"[{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}], p = {p:.3g}. "
                 "This is the quantity that says whether the advantage is bought by the evidence "
                 "being present rather than by the format of the prompt.")
        L.append("")
        d_lo = rows[-1][4]
        L.append(
            f"At zero recall the memory arm is {d_lo:+.3f} against the image arm. "
            + ("A negative value here means packed text about the *wrong* pages is actively worse "
               "than pixels the reader can decline to attend to — the failure mode the external "
               "benchmark hinted at." if d_lo < 0 else
               "A non-negative value here means part of the memory benefit is format rather than "
               "evidence delivery, and survives the evidence being removed entirely.")
        )
        L.append("")

    text = "\n".join(L) + "\n"
    if args.out:
        p = ROOT / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"-> {p}")
    print(text)

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        ax.plot(rs, [r[2] for r in rows], "o--", color="0.45", label="images")
        ax.plot(rs, [r[3] for r in rows], "s-", color="C0", label="+ text memory")
        ax.fill_between(rs, [r[2] for r in rows], [r[3] for r in rows], alpha=0.15, color="C0")
        ax.axhline(0, lw=0.5, color="0.8")
        ax.set_xlabel("fraction of gold evidence pages retrieved")
        ax.set_ylabel("strict accuracy")
        ax.set_xticks(rs)
        ax.legend(frameon=False, fontsize=7, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.2)
        fp = ROOT / args.fig
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp)
        print(f"-> {fp}")


if __name__ == "__main__":
    main()
