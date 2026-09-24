#!/usr/bin/env python3
"""DMR Phase-2 prep: precompute & cache candidate-block features + gold labels.

The per-block feature build (build_exact_block_candidate_rows + blake2b sketches) is
pure-Python slow (~0.5s/sample). Caching it once lets selector training / sweeps run fast.
Per (dataset, split) we cache, over ALL input pages of each sample:
  features (N,19) + qb (N,) query·block sketch dot + block_sketch (N,64)
  + label (N,) gold-evidence-block or not + group (N,) sample idx + page_id (N,)
  + sidecar json: sample_ids, query_sketch (S,64), n_gold (S,), block_ids (N).
Each split is saved on completion (resumable across splits).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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


import re
from types import SimpleNamespace


def _page_ids_from_images(images) -> list[int]:
    """aux pool has meta._input_page_ids=null; derive page ids from page_NNNN.png names."""
    pids = []
    for p in images or []:
        m = re.search(r"page_(\d+)", str(p))
        if m:
            pids.append(int(m.group(1)))
    return pids


def _iter_aux_samples(benchmark_root: str, dataset: str, split: str, max_samples: int | None):
    """Read paper_protocol/aux_train_qa_pool/<ds>/<split>.jsonl directly (disjoint train pool).
    Yields a sample shim with the same attrs build_cache uses; input page ids derived from images,
    document_id passed into meta so _resolve_document_json_path finds data/documents/<id>.json."""
    fam = {"V-MQAR": "vmqar", "V-NIAH": "vniah"}.get(dataset, dataset.lower().replace("-", ""))
    path = Path(benchmark_root) / "data" / "manifests" / "paper_protocol" / "aux_train_qa_pool" / fam / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"aux manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_samples is not None and i >= max_samples:
                break
            o = json.loads(line)
            meta = dict(o.get("meta") or {})
            meta.setdefault("document_id", o.get("document_id"))
            meta["_input_page_ids"] = _page_ids_from_images(o.get("images"))
            yield SimpleNamespace(sample_id=o.get("id"), question=o.get("question"),
                                  context=o.get("context"), meta=meta)


def build_cache(benchmark_root: str, dataset: str, split: str, out_dir: Path,
                max_samples: int | None, source: str = "benchmark"):
    if source == "aux":
        sample_iter = _iter_aux_samples(benchmark_root, dataset, split, max_samples)
    else:
        spec = LoadSpec(benchmark_root=benchmark_root, dataset=dataset, split=split, max_samples=max_samples)
        sample_iter = iter_samples(spec)
    feats: list[list[float]] = []
    qbs: list[float] = []
    sketches: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    page_ids: list[int] = []
    block_ids: list[str] = []
    sample_ids: list[str] = []
    query_sketches: list[list[float]] = []
    n_gold: list[int] = []
    gid = 0
    t0 = time.time()
    n_seen = 0
    for sample in sample_iter:
        n_seen += 1
        if n_seen % 50 == 0:
            print(f"   [{dataset}/{split}] {n_seen} samples, rows={len(feats)}, {time.time()-t0:.0f}s", flush=True)
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
        qs_sample = None
        npos = 0
        for r in rows:
            f = list(r.get("features") or [])
            if len(f) != 19:
                continue
            qs = np.asarray(r.get("query_sketch") or [], dtype=np.float32)
            bs = np.asarray(r.get("block_sketch") or [], dtype=np.float32)
            if qs_sample is None and qs.size == 64:
                qs_sample = qs
            qb = float(np.dot(qs, bs)) if qs.size and bs.size and qs.size == bs.size else 0.0
            feats.append(f)
            qbs.append(qb)
            sketches.append(bs.tolist() if bs.size == 64 else [0.0] * 64)
            is_pos = 1 if (int(r["page_id"]), str(r["block_id"])) in gold else 0
            labels.append(is_pos)
            groups.append(gid)
            page_ids.append(int(r["page_id"]))
            block_ids.append(str(r["block_id"]))
            npos += is_pos
        sample_ids.append(str(sample.sample_id))
        query_sketches.append((qs_sample.tolist() if qs_sample is not None else [0.0] * 64))
        n_gold.append(npos)
        gid += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.lower().replace('-', '')}_{split}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        features=np.asarray(feats, dtype=np.float32),
        qb=np.asarray(qbs, dtype=np.float32),
        block_sketch=np.asarray(sketches, dtype=np.float32),
        label=np.asarray(labels, dtype=np.float32),
        group=np.asarray(groups, dtype=np.int64),
        page_id=np.asarray(page_ids, dtype=np.int64),
    )
    json.dump(
        {"dataset": dataset, "split": split,
         "sample_ids": sample_ids, "query_sketch": query_sketches, "n_gold": n_gold,
         "block_ids": block_ids,
         "n_rows": len(feats), "n_samples": gid, "n_pos": int(sum(labels))},
        open(out_dir / f"{stem}.meta.json", "w", encoding="utf-8"))
    print(f"[{dataset}/{split}] cached rows={len(feats)} samples={gid} pos={int(sum(labels))} "
          f"-> {out_dir/stem}.npz  ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/fovedoc_project")
    ap.add_argument("--datasets", default="V-MQAR,V-NIAH")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--source", default="benchmark", choices=["benchmark", "aux"],
                    help="benchmark=final_benchmark splits; aux=paper_protocol/aux_train_qa_pool (disjoint train pool)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            stem = f"{dataset.lower().replace('-', '')}_{split}"
            if (out_dir / f"{stem}.npz").exists():
                print(f"[{dataset}/{split}] already cached, skip", flush=True)
                continue
            build_cache(args.benchmark_root, dataset, split, out_dir, args.max_samples, source=args.source)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
