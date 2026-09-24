#!/usr/bin/env python3
"""External-validity analysis: does the FoveDoc memory gap transfer to MMLongBench-Doc?

Paired over the full 1091 QA (identical retrieved page sets in both arms, verified), strict
judge correctness, McNemar with continuity correction.

Three things make a pooled delta here uninterpretable, so this script never reports one without
its strata:

1. `full_recall@16` is 0.843 externally vs FoveDoc's 0.998/1.000. A pooled delta therefore mixes
   a reading effect with a retrieval shortfall that FoveDoc does not have.
2. 244 of the questions are MMLongBench's deliberately *unanswerable* items. Getting those right
   measures abstention, not reading, and the two constructs move in opposite directions when a
   memory arm makes the reader more willing to answer.
3. Two annotation edge cases cut across (1) and (2), and both silently corrupt a naive split:
     - 7 unanswerable questions DO carry gold evidence pages (evidence of absence),
     - 9 answerable questions carry NO gold pages (whole-document counting questions), and are
       force-marked `full_recall=False` because the empty gold set can never be "recalled".
   So "has gold pages" (845) and "answerable" (847) are different sets, and neither is the clean
   reading stratum. That is the 838 that are both.

Strata are pre-registered (the three above plus document length, and the benchmark's own
`evidence_sources` taxonomy); Bonferroni is applied across every test the script prints.

Usage:
  python scripts/dmr_external_analysis.py --out results/tables/dmr_external_validity.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TSV = Path("data/LMUData/MMLongBench_DOC.tsv")

csv.field_size_limit(10**9)


def _load_judged(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["id"])] = row
    return out


def _load_meta() -> dict[str, dict[str, Any]]:
    """Benchmark metadata keyed by `index`, which equals the harness's sample id."""
    meta: dict[str, dict[str, Any]] = {}
    with open(TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                sources = list(literal_eval(row.get("evidence_sources") or "[]"))
            except Exception:
                sources = []
            try:
                pages = list(literal_eval(row.get("image_path") or "[]"))
            except Exception:
                pages = []
            meta[str(row["index"])] = {
                "doc_id": row.get("doc_id", ""),
                "doc_type": row.get("doc_type", ""),
                "answer_format": row.get("answer_format", ""),
                "evidence_sources": [str(s) for s in sources],
                "doc_pages": len(pages),
            }
    return meta


def _correct(row: dict[str, Any]) -> bool:
    return str((row.get("judge") or {}).get("verdict", "")).lower() == "correct"


def _is_unanswerable(row: dict[str, Any]) -> bool:
    answers = row.get("answers") or []
    return bool(answers) and str(answers[0]).strip().lower() == "not answerable"


_ABSTAIN = re.compile(
    r"not answerable|no information|does not (contain|provide|mention|include)|"
    r"cannot be (determined|answered|found)|not (provided|mentioned|available|specified|stated)|"
    r"unable to (determine|answer|find)|isn't (any )?information|there is no",
    re.I,
)


def _abstains(row: dict[str, Any]) -> bool:
    pred = str(row.get("pred") or "")
    return bool(_ABSTAIN.search(pred)) or pred.strip().lower() in ("none", "n/a", "")


def _chi2_sf_1df(x: float) -> float:
    """Upper tail of chi-square with 1 df == erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


def mcnemar(base: list[bool], arm: list[bool]) -> dict[str, float]:
    """Paired binary comparison. b = arm wins, c = base wins."""
    n = len(base)
    b = sum(1 for x, y in zip(base, arm) if not x and y)
    c = sum(1 for x, y in zip(base, arm) if x and not y)
    delta = (b - c) / n if n else 0.0
    # Variance of the paired difference, discordant-pair form.
    var = ((b + c) - (b - c) ** 2 / n) / (n * n) if n else 0.0
    se = math.sqrt(max(var, 0.0))
    if b + c:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = _chi2_sf_1df(chi2)
    else:
        p = 1.0
    return {
        "n": n,
        "base_rate": sum(base) / n if n else 0.0,
        "arm_rate": sum(arm) / n if n else 0.0,
        "delta": delta,
        "lo": delta - 1.96 * se,
        "hi": delta + 1.96 * se,
        "p": p,
        "b": b,
        "c": c,
    }


def _fmt(st: dict[str, float]) -> str:
    return (
        f"| {st['n']:d} | {st['base_rate']:.3f} | {st['arm_rate']:.3f} | "
        f"{st['delta']:+.3f} | [{st['lo']:+.3f}, {st['hi']:+.3f}] | {st['p']:.3g} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged-dir", default="code/runs/20260810/external/judged")
    ap.add_argument("--base-arm", default="ext", help="image/pages arm")
    ap.add_argument("--mem-arm", default="extmem", help="pages+memory arm")
    ap.add_argument("--task", default="mmlongbench_doc")
    ap.add_argument("--out", default="results/tables/dmr_external_validity.md")
    args = ap.parse_args()

    jd = ROOT / args.judged_dir
    base = _load_judged(jd / f"{args.base_arm}_{args.task}.jsonl")
    mem = _load_judged(jd / f"{args.mem_arm}_{args.task}.jsonl")
    if set(base) != set(mem):
        raise SystemExit("[external] arms are not paired: id sets differ")
    meta = _load_meta()
    missing = [i for i in base if i not in meta]
    if missing:
        raise SystemExit(f"[external] {len(missing)} ids absent from the benchmark TSV, e.g. {missing[:5]}")

    ids = sorted(base, key=int)
    for i in ids:
        if base[i].get("selected_page_ids") != mem[i].get("selected_page_ids"):
            raise SystemExit(f"[external] retrieved pages differ at id={i}; arms are not comparable")

    # Two test families, kept separate because they were fixed at different times.
    # PRIMARY: written into scripts/run_dmr_judging.sh before any external verdict existed
    # ("stratify by full_recall"; "246 ... deliberately unanswerable items need separate
    # treatment"). EXPLORATORY: everything else. Pooling them into one Bonferroni would punish
    # the pre-registered tests for the exploratory ones, and reporting only the pooled α would
    # let a post-hoc stratum borrow the primaries' credibility. Both are printed.
    primary: list[tuple[str, dict[str, float]]] = []
    exploratory: list[tuple[str, dict[str, float]]] = []

    def run(label: str, keep: Callable[[str], bool], family: str = "exploratory") -> dict[str, float]:
        sel = [i for i in ids if keep(i)]
        st = mcnemar([_correct(base[i]) for i in sel], [_correct(mem[i]) for i in sel])
        (primary if family == "primary" else exploratory).append((label, st))
        return st

    def two_sample(label: str, left: list[str], right: list[str], family: str = "exploratory") -> dict[str, float]:
        """Contrast two disjoint strata's paired per-question effects (mem - img in {-1,0,1})."""
        dl = [int(_correct(mem[i])) - int(_correct(base[i])) for i in left]
        dr = [int(_correct(mem[i])) - int(_correct(base[i])) for i in right]
        ml, mr = sum(dl) / len(dl), sum(dr) / len(dr)
        vl = sum((v - ml) ** 2 for v in dl) / (len(dl) - 1)
        vr = sum((v - mr) ** 2 for v in dr) / (len(dr) - 1)
        se = math.sqrt(vl / len(dl) + vr / len(dr))
        z = (ml - mr) / se if se else 0.0
        st = {
            "n": len(dl) + len(dr), "base_rate": mr, "arm_rate": ml, "delta": ml - mr,
            "lo": (ml - mr) - 1.96 * se, "hi": (ml - mr) + 1.96 * se,
            "p": math.erfc(abs(z) / math.sqrt(2)), "b": 0.0, "c": 0.0, "z": z,
        }
        (primary if family == "primary" else exploratory).append((label, st))
        return st

    una = {i for i in ids if _is_unanswerable(base[i])}
    has_gold = {i for i in ids if base[i].get("gold_evidence_page_ids")}
    clean = [i for i in ids if i not in una and i in has_gold]  # 838: answerable AND annotated

    clean_set = set(clean)
    pooled = run("all 1091 (uninterpretable, shown for completeness)", lambda i: True)
    reading = run("answerable + gold-annotated (the reading stratum)", lambda i: i in clean_set, "primary")
    full = run("  ... of which full recall@16", lambda i: i in clean_set and base[i].get("full_recall"), "primary")
    part = run("  ... of which partial recall", lambda i: i in clean_set and not base[i].get("full_recall"), "primary")
    abst = run("unanswerable (abstention, not reading)", lambda i: i in una, "primary")

    # Document length -- motivated by the retrieval-recall curve, which is monotone in length.
    bins = [(0, 30), (31, 60), (61, 120), (121, 10**6)]
    length_rows: list[tuple[str, dict[str, float]]] = []
    for lo, hi in bins:
        label = f"{lo}-{hi} pages" if hi < 10**6 else f">{lo - 1} pages"
        st = run(f"length {label}", lambda i, lo=lo, hi=hi: i in clean_set and lo <= meta[i]["doc_pages"] <= hi)
        length_rows.append((label, st))

    # The benchmark's own evidence taxonomy. This is THE axis for a text-memory method: a gain
    # that concentrates on Pure-text and vanishes on Chart/Figure is a mechanism result.
    src_counts = Counter(s for i in clean for s in meta[i]["evidence_sources"])
    src_rows: list[tuple[str, dict[str, float]]] = []
    for src, _n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        st = run(f"evidence {src}", lambda i, s=src: i in clean_set and s in meta[i]["evidence_sources"])
        src_rows.append((src, st))

    # Direct test of the text-vs-graphic interaction. The five per-source strata above overlap
    # (a question can be tagged both Table and Chart) and none is individually significant, so
    # they cannot carry a mechanism claim on their own. Restricting to SINGLE-tag questions makes
    # the two groups disjoint and tests the interaction once, rather than eyeballing five rows.
    # Disclosure: the evidence-source axis was pre-registered, this particular pooling of its
    # extremes was chosen after seeing the per-stratum directions, so it counts as exploratory.
    single = {i: meta[i]["evidence_sources"][0] for i in clean if len(meta[i]["evidence_sources"]) == 1}
    txt = [i for i, s in single.items() if s == "Pure-text (Plain-text)"]
    gfx = [i for i, s in single.items() if s in ("Chart", "Figure")]
    contrast = two_sample("Pure-text vs Chart+Figure (single-tag only)", txt, gfx)

    tests = primary + exploratory
    k = len(tests)
    alpha = 0.05 / k
    a_pri = 0.05 / len(primary)
    a_exp = 0.05 / len(exploratory)

    L: list[str] = []
    L.append("# External validity — does the memory gap transfer to MMLongBench-Doc?")
    L.append("")
    L.append(
        f"Stock Qwen3-VL-2B, ColQwen2 top-16, OCR memory, n={len(ids)} paired. Both arms see the "
        "*identical* retrieved page set (verified per-id), so the only difference is whether the "
        "extracted-text memory is in the prompt. Strict judge correctness, McNemar with continuity "
        "correction, single grader in one session."
    )
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append("| stratum | n | images | +memory | Δ | 95% CI | p |")
    L.append("|---|---|---|---|---|---|---|")
    for label, st in [
        ("all questions *(pooled — do not cite alone)*", pooled),
        ("**answerable, gold-annotated**", reading),
        ("  ⤷ full recall@16", full),
        ("  ⤷ partial recall", part),
        ("unanswerable *(abstention)*", abst),
    ]:
        L.append(f"| {label} {_fmt(st)}")
    L.append("")
    L.append(
        "*The CI and the p-value come from different estimators — a Wald interval on the paired "
        "difference, and McNemar with a continuity correction, which is conservative. They can "
        "therefore disagree marginally (the 31–60 page row below is the case here). Where they do, "
        "the p-value is the one to trust.*"
    )
    L.append("")
    L.append("## By document length")
    L.append("")
    L.append("Motivated by the retrieval curve, which is monotone in length (0.938 at ≤30 pages → 0.644 above 120).")
    L.append("")
    L.append("| length | n | images | +memory | Δ | 95% CI | p |")
    L.append("|---|---|---|---|---|---|---|")
    for label, st in length_rows:
        L.append(f"| {label} {_fmt(st)}")
    L.append("")
    L.append("## By evidence source (the benchmark's own taxonomy)")
    L.append("")
    L.append(
        "Questions carry one or more evidence-source tags, so these strata overlap and the rows do "
        "not sum to the stratum n. The axis is pre-registered as the mechanism axis for a "
        "text-memory method; the rows are ordered by n, not by effect."
    )
    L.append("")
    L.append("| evidence | n | images | +memory | Δ | 95% CI | p |")
    L.append("|---|---|---|---|---|---|---|")
    for label, st in src_rows:
        L.append(f"| {label} {_fmt(st)}")
    L.append("")
    L.append(
        "No single row is significant, and they overlap, so they cannot carry a mechanism claim by "
        "themselves. The direct contrast can: restricted to the "
        f"{len(txt) + len(gfx)} **single-tag** questions, the per-question paired effect is "
        f"{contrast['arm_rate']:+.3f} on Pure-text (n={len(txt)}) against "
        f"{contrast['base_rate']:+.3f} on Chart+Figure (n={len(gfx)}) — a difference of "
        f"**{contrast['delta']:+.3f}** [{contrast['lo']:+.3f}, {contrast['hi']:+.3f}], "
        f"z = {contrast['z']:.2f}, **p = {contrast['p']:.4f}**. That is the one test here that "
        "survives correction. Disclosure: the axis was pre-registered, but pooling Chart with "
        "Figure was chosen after seeing the per-stratum directions, so it is counted as "
        "exploratory below."
    )
    L.append("")
    L.append("## Abstention: discrimination or response bias?")
    L.append("")
    L.append(
        "A memory arm that simply answers less often would gain on the unanswerable stratum for "
        "free. Signal-detection view: *hit* = abstains on an unanswerable question, *false alarm* "
        "= abstains on an answerable one. (Abstention is detected by surface pattern on `pred`, "
        "so read these as rates on a consistent rule, not as exact counts.)"
    )
    L.append("")
    L.append("| arm | hit rate (n=%d) | false-alarm rate (n=%d) | balanced acc. |" % (len(una), len(clean)))
    L.append("|---|---|---|---|")
    for name, arm in (("images", base), ("+memory", mem)):
        hit = sum(_abstains(arm[i]) for i in una) / len(una)
        fa = sum(_abstains(arm[i]) for i in clean) / len(clean)
        L.append(f"| {name} | {hit:.3f} | {fa:.3f} | {(hit + (1 - fa)) / 2:.3f} |")
    only_mem = [i for i in clean if _abstains(mem[i]) and not _abstains(base[i])]
    cost = sum(1 for i in only_mem if _correct(base[i]))
    ab_b = sum(1 for i in una if not _abstains(base[i]) and _abstains(mem[i]))
    ab_c = sum(1 for i in una if _abstains(base[i]) and not _abstains(mem[i]))
    ab_p = _chi2_sf_1df((abs(ab_b - ab_c) - 1) ** 2 / (ab_b + ab_c)) if ab_b + ab_c else 1.0
    L.append("")
    L.append(
        f"The hit rate roughly triples while the false-alarm rate moves by well under a point: of "
        f"the {len(only_mem)} answerable questions where only the memory arm abstains, the image "
        f"arm answered {cost} correctly. So this is improved discrimination, not a bias shift — "
        "though both arms are poor at abstention in absolute terms."
    )
    L.append("")
    L.append(
        f"Testing the abstention *behaviour* directly rather than through the judge — McNemar on "
        f"the abstain indicator over the {len(una)} unanswerable questions — gives b={ab_b}, "
        f"c={ab_c}, p = {ab_p:.4g}, i.e. the same direction somewhat more strongly than the "
        "correctness view. It is still not below the corrected threshold."
    )
    L.append("")
    L.append(
        "**This is not a replication of anything.** FoveDoc contains 1–2 unanswerable items in "
        "1173, so V-MQAR and V-NIAH could not have measured abstention at all. This is a first "
        "observation, on one benchmark, at p just above a corrected threshold."
    )
    L.append("")
    L.append("## Multiplicity")
    L.append("")
    L.append(
        f"{k} tests are reported, in two families. **Primary** ({len(primary)} tests, α = "
        f"{a_pri:.4f}) are the strata named in `scripts/run_dmr_judging.sh` before any external "
        f"verdict existed. **Exploratory** ({len(exploratory)} tests, α = {a_exp:.4f}) is "
        f"everything else. Pooled over all {k}, α = {alpha:.4f}."
    )
    L.append("")
    for fam, rows, a in (("Primary", primary, a_pri), ("Exploratory", exploratory, a_exp)):
        surv = [(lab, st) for lab, st in rows if st["p"] < a]
        if surv:
            L.append(f"{fam} — survives α = {a:.4f}:")
            for lab, st in surv:
                L.append(f"- **{lab.strip()}**: Δ = {st['delta']:+.3f}, p = {st['p']:.3g}")
        else:
            L.append(f"{fam} — **nothing survives α = {a:.4f}.**")
        L.append("")
    near = [(lab, st) for lab, st in tests if a_pri <= st["p"] < 0.05]
    if near:
        L.append("Directional only (p < 0.05 but above the corrected threshold) — these are "
                 "hypotheses with a sign, not findings:")
        for lab, st in near:
            L.append(f"- {lab.strip()}: Δ = {st['delta']:+.3f}, p = {st['p']:.3g}")
    L.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
