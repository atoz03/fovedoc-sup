#!/usr/bin/env python3
"""Build controlled retrieval-degradation manifests for the recall sweep.

The external run left a directional hypothesis: where top-16 misses some gold evidence, the
memory arm is *worse* than the image arm (-0.061, p=0.099, n=132). FoveDoc cannot test that
observationally -- full-recall@16 is 0.998, so there are ~2 partial-recall rows in 1173. This
script manufactures the condition instead, so the sweep is a controlled manipulation rather than
a subgroup found after the fact.

Design, and why:
  * Hold the page budget at exactly 16 in every level. Varying k instead would change recall AND
    context length together, and the reading gap is known to move with context length, so the
    two effects would be inseparable.
  * Degrade by SUBSTITUTION: drop j of the sample's gold pages from the retrieved set and add j
    pages from the same document that the retriever ranked outside the top-16. Both arms then see
    the identical degraded page set -- one as images, one as images plus the packed text OF THOSE
    SAME PAGES -- so the manipulation is evidence presence, not budget.
  * V-MQAR has exactly 2 gold pages for all 1173 samples, so j in {0,1,2} is a clean 3-point
    recall curve (1.0 / 0.5 / 0.0) that means the same thing for every sample. V-NIAH has 1 gold
    page, so it only has the two endpoints.

Two honest limitations, both consequences of the retrieval manifest being pre-truncated to
top-16 (the full ColQwen2 ranking would need a GPU re-run to recover):
  1. The substituted pages are known to rank below 16, which is the realistic failure mode, but
     they are not rank-ordered among themselves -- this script picks them by ascending page id.
     So "which distractor" is arbitrary within the below-threshold pool, not adversarial.
  2. Samples whose document has fewer than 16 + |gold| pages have nowhere to draw substitutes
     from and are excluded. That is 12.6% of FoveDoc at exactly 16 pages plus part of the
     17-page bucket. The excluded samples are the SHORTEST documents, i.e. the easiest retrieval
     cases, so the sweep runs on a slightly harder-than-average subset. Report the n.

Usage:
  python scripts/dmr_make_recall_sweep_manifest.py --task vmqar --out-dir code/runs/20260902/recall_sweep
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANKED = ROOT / "code/runs/20260630/dmr_retrieval/colqwen2_ranked_all.jsonl"
# Gold page ids are read back from a judged parity file rather than re-derived from the
# benchmark, so the sweep is anchored to exactly the ids the harness itself resolved.
GOLD_SRC = {
    "vmqar": ROOT / "code/runs/20260810/dmr_ocr_parity_2b/judged/ocrmem_vmqar.jsonl",
    "vniah": ROOT / "code/runs/20260810/dmr_ocr_parity_2b/judged/ocrmem_vniah.jsonl",
}
DATASET = {"vmqar": "V-MQAR", "vniah": "V-NIAH"}
_RANGE = re.compile(r":(\d+)-(\d+)$")


def _doc_pages(sample_id: str) -> list[int] | None:
    m = _RANGE.search(sample_id)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return list(range(a, b + 1)) if b >= a else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vmqar", "vniah"], default="vmqar")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--out-dir", default="code/runs/20260902/recall_sweep")
    args = ap.parse_args()

    ds = DATASET[args.task]
    k = int(args.top_k)

    gold: dict[str, list[int]] = {}
    for line in open(GOLD_SRC[args.task], encoding="utf-8"):
        r = json.loads(line)
        gold[str(r["id"])] = [int(p) for p in (r.get("gold_evidence_page_ids") or [])]

    ranked: dict[str, list[int]] = {}
    for line in open(RANKED, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("dataset")) == ds:
            ranked[str(r["sample_id"])] = [int(p) for p in r["ranked_page_ids"]]

    reasons: Counter[str] = Counter()
    eligible: list[tuple[str, list[int], list[int], list[int]]] = []
    for sid, rank in sorted(ranked.items()):
        g = gold.get(sid)
        pages = _doc_pages(sid)
        top = rank[:k]
        if not g:
            reasons["no_gold"] += 1
        elif pages is None:
            reasons["no_page_range"] += 1
        elif not set(g).issubset(top):
            # Degrading a sample whose gold is already missing would confound the level.
            reasons["gold_not_fully_retrieved"] += 1
        elif len(set(pages) - set(top)) < len(g):
            reasons["not_enough_substitutes"] += 1
        else:
            eligible.append((sid, top, g, sorted(set(pages) - set(top))))

    if args.max_samples:
        eligible = eligible[: args.max_samples]

    n_gold = len(eligible[0][2]) if eligible else 0
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {ds}: {len(eligible)} eligible of {len(ranked)}")
    for why, n in reasons.most_common():
        print(f"  excluded {n:5d}  {why}")
    print(f"  gold pages per sample: {n_gold}  ->  levels j = 0..{n_gold}")

    for j in range(n_gold + 1):
        fp = out_dir / f"{args.task}_drop{j}.jsonl"
        with open(fp, "w", encoding="utf-8") as w:
            for sid, top, g, spare in eligible:
                drop = sorted(g)[:j]                     # deterministic: lowest gold ids first
                keep = [p for p in top if p not in drop]
                new = keep + spare[:j]                   # deterministic: lowest spare ids first
                assert len(new) == len(top), (sid, len(new), len(top))
                assert len(set(new)) == len(new), sid
                assert len(set(g) & set(new)) == len(g) - j, sid
                w.write(json.dumps({"sample_id": sid, "dataset": ds,
                                    "ranked_page_ids": new}, ensure_ascii=False) + "\n")
        print(f"  wrote {fp.name:20s} recall = {(n_gold - j) / n_gold:.2f}  ({len(eligible)} rows)")

    ids_fp = out_dir / f"{args.task}_sample_ids.txt"
    ids_fp.write_text("\n".join(sid for sid, _, _, _ in eligible) + "\n", encoding="utf-8")
    print(f"  wrote {ids_fp.name} ({len(eligible)} ids) — pass to --only-ids so every level and "
          f"the reused drop0 predictions cover exactly the same samples")


if __name__ == "__main__":
    main()
