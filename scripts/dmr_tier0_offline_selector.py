#!/usr/bin/env python3
"""DMR Tier-0：离线判定 selector 可行性（无 reader / 无 judge）。

问题：每个样本 ~400 个候选 block，其中 1-2 个是 gold evidence block。naive memory 用
lexical top-k；我们问——在已有特征（19 结构特征 + query·block sketch）上，一个简单可学习
打分器能不能把 gold block 排得比 distractor 更靠前（AUROC + recall@budget），从而支撑
"distractor-robust memory selector"。GO 线：learned 明显高于 lexical 基线，且 recall@budget
（V-MQAR 要求两 gold block 同时进 top-B）达到可用水平。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eccv26.data.benchmarks import LoadSpec, iter_samples
from eccv26.eval import _resolve_document_json_path
from eccv26.utils.exact_block import build_exact_block_candidate_rows


def _gold_blocks(meta) -> set[tuple[int, str]]:
    ev = meta.get("evidence")
    out: set[tuple[int, str]] = set()
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict) and e.get("block_id") is not None and e.get("page_id") is not None:
                try:
                    out.add((int(e["page_id"]), str(e["block_id"])))
                except (TypeError, ValueError):
                    pass
    return out


def _build_split(benchmark_root: str, dataset: str, split: str, max_samples: int | None):
    """返回 (X[n,F], y[n], group[n], n_gold_per_sample{gid:int})。"""
    spec = LoadSpec(benchmark_root=benchmark_root, dataset=dataset, split=split, max_samples=max_samples)
    feats: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    n_gold: dict[int, int] = {}
    gid = 0
    t0 = time.time()
    for si, sample in enumerate(iter_samples(spec), 1):
        if si % 25 == 0:
            print(f"   [{dataset}/{split}] built {si} samples, rows={len(feats)}, {time.time()-t0:.0f}s", flush=True)
        gold = _gold_blocks(sample.meta)
        if not gold:
            continue
        ipids = [int(x) for x in (sample.meta.get("_input_page_ids") or [])]
        djson = _resolve_document_json_path(sample.meta, benchmark_root)
        rows = build_exact_block_candidate_rows(
            document_json_path=djson, source_page_ids=ipids,
            query_texts=[sample.question, sample.context])
        if not rows:
            continue
        npos = 0
        for r in rows:
            f = list(r.get("features") or [])
            if len(f) != 19:
                continue
            qs = np.asarray(r.get("query_sketch") or [], dtype=np.float32)
            bs = np.asarray(r.get("block_sketch") or [], dtype=np.float32)
            qb = float(np.dot(qs, bs)) if qs.size and bs.size and qs.size == bs.size else 0.0
            feats.append(f + [qb])
            is_pos = 1 if (int(r["page_id"]), str(r["block_id"])) in gold else 0
            labels.append(is_pos)
            groups.append(gid)
            npos += is_pos
        n_gold[gid] = npos
        gid += 1
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    g = np.asarray(groups, dtype=np.int64)
    return X, y, g, n_gold


def _mann_whitney_auroc(y: np.ndarray, s: np.ndarray) -> float:
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    rank = np.argsort(np.argsort(s)) + 1.0
    return float((rank[y.astype(bool)].sum() - P * (P + 1) / 2) / (P * N))


def _recall_at_budget(y, g, s, n_gold, budgets):
    """每个样本按 s 排序，统计 top-B 命中 gold 的 recall 与 full-recall（全部 gold 进 top-B）。"""
    res = {B: {"recall": [], "full": []} for B in budgets}
    for gid in np.unique(g):
        m = g == gid
        ys = y[m]; ss = s[m]
        tot = int(ys.sum())
        if tot == 0:
            continue
        order = np.argsort(-ss)
        ys_sorted = ys[order]
        for B in budgets:
            hit = int(ys_sorted[:B].sum())
            res[B]["recall"].append(hit / tot)
            res[B]["full"].append(1.0 if hit >= tot else 0.0)
    return {B: {"recall": float(np.mean(v["recall"])) if v["recall"] else float("nan"),
                "full": float(np.mean(v["full"])) if v["full"] else float("nan")}
            for B, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/fovedoc_project")
    ap.add_argument("--datasets", default="V-MQAR")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    budgets = [4, 8, 16]
    out = {"datasets": args.datasets, "budgets": budgets, "by_dataset": {}}

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        t0 = time.time()
        Xtr, ytr, gtr, ng_tr = _build_split(args.benchmark_root, dataset, "train", args.max_train)
        Xva, yva, gva, ng_va = _build_split(args.benchmark_root, dataset, "val", args.max_val)
        print(f"[{dataset}] train rows={len(ytr)} pos={int(ytr.sum())} samples={len(np.unique(gtr))}; "
              f"val rows={len(yva)} pos={int(yva.sum())} samples={len(np.unique(gva))}  build={time.time()-t0:.1f}s", flush=True)

        # standardize on train
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd; Xva_n = (Xva - mu) / sd

        # --- lexical baseline: feature[0] = log lexical score ---
        lex_tr = Xtr[:, 0]; lex_va = Xva[:, 0]
        base = {
            "auroc_val": _mann_whitney_auroc(yva, lex_va),
            "recall_val": _recall_at_budget(yva, gva, lex_va, ng_va, budgets),
        }

        # --- learned logistic regression (torch, class-weighted BCE) ---
        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        Xt = torch.tensor(Xtr_n, device=dev); yt = torch.tensor(ytr, device=dev)
        w = torch.zeros(Xt.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(1.0, ytr.sum())], device=dev)
        opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-4)
        lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for ep in range(args.epochs):
            opt.zero_grad()
            logit = Xt @ w + b
            loss = lossfn(logit, yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            s_va = (torch.tensor(Xva_n, device=dev) @ w + b).cpu().numpy()
        learned = {
            "auroc_val": _mann_whitney_auroc(yva, s_va),
            "recall_val": _recall_at_budget(yva, gva, s_va, ng_va, budgets),
            "final_loss": float(loss.item()),
        }
        out["by_dataset"][dataset] = {"lexical_baseline": base, "learned_logreg": learned,
                                      "n_train_samples": int(len(np.unique(gtr))),
                                      "n_val_samples": int(len(np.unique(gva)))}
        print(f"[{dataset}] LEXICAL  AUROC={base['auroc_val']:.3f}  recall@8={base['recall_val'][8]['recall']:.3f} full@8={base['recall_val'][8]['full']:.3f}", flush=True)
        print(f"[{dataset}] LEARNED  AUROC={learned['auroc_val']:.3f}  recall@8={learned['recall_val'][8]['recall']:.3f} full@8={learned['recall_val'][8]['full']:.3f}", flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
