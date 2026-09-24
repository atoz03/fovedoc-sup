#!/usr/bin/env python3
"""DMR Phase-2: train the memory selector (head-only) on cached candidate features.

Reuses the repo's exact-block scorer modules (linear | query_interaction |
cross_block_attention) as the selector, trained with a per-sample listwise margin loss
that pushes gold-evidence blocks above the hardest distractor blocks. Backbone/reader are
NOT involved here (cheap, minutes). Eval = full-recall@budget (V-MQAR needs both gold
blocks in top-B) + AUROC on val, vs the lexical baseline (what naive memory uses).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eccv26.model.dma import (
    DMAExactBlockCrossBlockAttentionScorer,
    DMAExactBlockQueryInteractionScorer,
)
from eccv26.utils.exact_block import EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_TEXT_SKETCH_DIM


def _load(cache_dir: Path, dataset: str, split: str):
    stem = f"{dataset.lower().replace('-', '')}_{split}"
    z = np.load(cache_dir / f"{stem}.npz")
    meta = json.load(open(cache_dir / f"{stem}.meta.json", encoding="utf-8"))
    qsketch = np.asarray(meta["query_sketch"], dtype=np.float32)  # (S,64)
    return {
        "features": z["features"], "block_sketch": z["block_sketch"], "label": z["label"],
        "group": z["group"], "page_id": z["page_id"], "query_sketch_by_sample": qsketch,
        "n_gold": np.asarray(meta["n_gold"]), "n_samples": int(meta["n_samples"]),
    }


def _sample_slices(group: np.ndarray):
    slices = {}
    order = np.argsort(group, kind="stable")
    g_sorted = group[order]
    uniq, starts = np.unique(g_sorted, return_index=True)
    starts = list(starts) + [len(group)]
    for i, gid in enumerate(uniq):
        idx = order[starts[i]:starts[i + 1]]
        slices[int(gid)] = idx
    return slices


def _mann_whitney_auroc(y, s):
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    rank = np.argsort(np.argsort(s)) + 1.0
    return float((rank[y.astype(bool)].sum() - P * (P + 1) / 2) / (P * N))


def _eval(scorer, data, slices, dev, budgets):
    feats = torch.tensor(data["features"], device=dev)
    bsk = torch.tensor(data["block_sketch"], device=dev)
    pid = torch.tensor(data["page_id"], device=dev)
    qsk_s = data["query_sketch_by_sample"]
    all_scores = np.zeros(len(data["label"]), dtype=np.float32)
    rec = {B: {"recall": [], "full": []} for B in budgets}
    scorer.eval()
    with torch.no_grad():
        for gid, idx in slices.items():
            it = torch.tensor(idx, device=dev)
            qs = torch.tensor(qsk_s[gid], device=dev).unsqueeze(0).expand(len(idx), -1)
            logit = scorer(feats[it], page_ids=pid[it], query_sketch=qs, block_sketch=bsk[it]).cpu().numpy()
            all_scores[idx] = logit
            y = data["label"][idx]; tot = int(y.sum())
            if tot == 0:
                continue
            order = np.argsort(-logit)
            ys = y[order]
            for B in budgets:
                hit = int(ys[:B].sum())
                rec[B]["recall"].append(hit / tot)
                rec[B]["full"].append(1.0 if hit >= tot else 0.0)
    auroc = _mann_whitney_auroc(data["label"], all_scores)
    out = {"auroc": auroc}
    for B in budgets:
        out[f"recall@{B}"] = float(np.mean(rec[B]["recall"])) if rec[B]["recall"] else float("nan")
        out[f"full@{B}"] = float(np.mean(rec[B]["full"])) if rec[B]["full"] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", default="V-MQAR")
    ap.add_argument("--scorer", default="cross_block_attention",
                    choices=["cross_block_attention", "query_interaction"])
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--neg-topk", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    budgets = [4, 8, 16]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr = _load(Path(args.cache_dir), args.dataset, "train")
    va = _load(Path(args.cache_dir), args.dataset, "val")
    tr_sl = _sample_slices(tr["group"]); va_sl = _sample_slices(va["group"])
    print(f"[{args.dataset}] train samples={tr['n_samples']} rows={len(tr['label'])} pos={int(tr['label'].sum())}; "
          f"val samples={va['n_samples']} rows={len(va['label'])}", flush=True)

    # lexical baseline (feature[0]=log lexical) on val
    lex = _eval_lexical(va, va_sl, budgets)
    print(f"LEXICAL val: AUROC={lex['auroc']:.3f} full@8={lex['full@8']:.3f} full@16={lex['full@16']:.3f}", flush=True)

    if args.scorer == "cross_block_attention":
        scorer = DMAExactBlockCrossBlockAttentionScorer(EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_TEXT_SKETCH_DIM, args.hidden_dim)
    else:
        scorer = DMAExactBlockQueryInteractionScorer(EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_TEXT_SKETCH_DIM, args.hidden_dim)
    scorer.to(dev)
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr, weight_decay=1e-4)

    feats = torch.tensor(tr["features"], device=dev)
    bsk = torch.tensor(tr["block_sketch"], device=dev)
    pid = torch.tensor(tr["page_id"], device=dev)
    labl = tr["label"]
    qsk_s = tr["query_sketch_by_sample"]
    gids = [g for g in tr_sl if int(labl[tr_sl[g]].sum()) > 0]

    best = {"full@8": -1.0}; best_metrics = None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        scorer.train()
        np.random.shuffle(gids)
        ep_loss = 0.0; nb = 0
        for gid in gids:
            idx = tr_sl[gid]
            it = torch.tensor(idx, device=dev)
            qs = torch.tensor(qsk_s[gid], device=dev).unsqueeze(0).expand(len(idx), -1)
            logit = scorer(feats[it], page_ids=pid[it], query_sketch=qs, block_sketch=bsk[it])
            y = torch.tensor(labl[idx], device=dev)
            pos = logit[y > 0.5]; neg = logit[y < 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            topk_neg = torch.topk(neg, k=min(args.neg_topk, neg.numel())).values
            loss = torch.relu(args.margin - pos[:, None] + topk_neg[None, :]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        if ep % 5 == 0 or ep == args.epochs:
            m = _eval(scorer, va, va_sl, dev, budgets)
            print(f"ep{ep:02d} loss={ep_loss/max(1,nb):.4f} val AUROC={m['auroc']:.3f} "
                  f"recall@8={m['recall@8']:.3f} full@8={m['full@8']:.3f} full@16={m['full@16']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if m["full@8"] > best["full@8"]:
                best = m; best_metrics = m
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                torch.save(scorer.state_dict(), str(Path(args.output).with_suffix(".pt")))

    result = {"dataset": args.dataset, "scorer": args.scorer, "hidden_dim": args.hidden_dim,
              "lexical_baseline": lex, "learned_best": best_metrics}
    json.dump(result, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _eval_lexical(data, slices, budgets):
    s = data["features"][:, 0]
    rec = {B: {"recall": [], "full": []} for B in budgets}
    for gid, idx in slices.items():
        y = data["label"][idx]; tot = int(y.sum())
        if tot == 0:
            continue
        order = np.argsort(-s[idx]); ys = y[order]
        for B in budgets:
            hit = int(ys[:B].sum())
            rec[B]["recall"].append(hit / tot); rec[B]["full"].append(1.0 if hit >= tot else 0.0)
    out = {"auroc": _mann_whitney_auroc(data["label"], s)}
    for B in budgets:
        out[f"recall@{B}"] = float(np.mean(rec[B]["recall"]))
        out[f"full@{B}"] = float(np.mean(rec[B]["full"]))
    return out


if __name__ == "__main__":
    main()
