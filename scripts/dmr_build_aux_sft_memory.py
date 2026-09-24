#!/usr/bin/env python3
"""Build memory-augmented SFT data for lever C (train the reader to compose the answer
from page-tagged document memory) on the DISJOINT aux V-MQAR pool.

Each example matches the winning eval arm exactly:
  images       = gold evidence page images
  user_text    = lexical memory scratchpad (top-k blocks, page-tagged) + question
  assistant_text = gold answer

Reuses the harness's `_build_memory_scratchpad` / `_mode_user_text` / `_build_user_text`
so train == eval formatting. Protocol-clean: aux docs are disjoint from the 1173 release set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dmr_oracle_scratchpad_derisk as H  # noqa: E402


def _page_ids_from_images(images):
    out = []
    for p in images or []:
        m = re.search(r"page_(\d+)", str(p))
        if m:
            out.append(int(m.group(1)))
    return out


def _gold_pages(meta):
    ev = meta.get("evidence") or []
    pages = []
    for e in ev:
        if isinstance(e, dict) and e.get("page_id") is not None:
            pages.append(int(e["page_id"]))
    return sorted(set(pages))


def build(split: str, benchmark_root: str, topk: int, out_path: Path, max_samples=None):
    fam_path = (Path(benchmark_root) / "data" / "manifests" / "paper_protocol" /
                "aux_train_qa_pool" / "vmqar" / f"{split}.jsonl")
    n_in = n_out = n_skip = 0
    with out_path.open("w", encoding="utf-8") as w:
        for line in fam_path.open(encoding="utf-8"):
            if max_samples and n_in >= max_samples:
                break
            n_in += 1
            o = json.loads(line)
            ans = [a for a in (o.get("answers") or []) if str(a).strip()]
            if not ans:
                n_skip += 1
                continue
            meta = dict(o.get("meta") or {})
            meta["document_id"] = o.get("document_id")
            imgs = o.get("images") or []
            in_pages = _page_ids_from_images(imgs)
            meta["_input_page_ids"] = in_pages
            gold = _gold_pages(meta)
            if not gold:
                n_skip += 1
                continue
            sample = SimpleNamespace(sample_id=o.get("id"), question=o.get("question"),
                                     context=o.get("context"), meta=meta, dataset="V-MQAR")
            scratchpad, facts = H._build_memory_scratchpad(
                sample=sample, gold_page_ids=gold, benchmark_root=benchmark_root,
                topk=topk, max_chars_per_block=300, selector=None, selector_device=None)
            if not scratchpad.strip():
                n_skip += 1
                continue
            base_user = H._build_user_text(o.get("context"), o.get("question"), None)
            user_text = H._mode_user_text(mode="pages_memory", base_user_text=base_user, scratchpad=scratchpad)
            # gold page images (map page_id -> image path via order)
            pid_to_img = {pid: imgs[i] for i, pid in enumerate(in_pages) if i < len(imgs)}
            gold_imgs = [pid_to_img[p] for p in gold if p in pid_to_img]
            if not gold_imgs:
                n_skip += 1
                continue
            rec = {"id": o.get("id"), "user_text": user_text, "assistant_text": ans[0],
                   "images": gold_imgs,
                   "meta": {"document_id": o.get("document_id"), "gold_pages": gold,
                            "task_type": "vmqar_memory_join"}}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[{split}] in={n_in} out={n_out} skip={n_skip} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/fovedoc_project")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        build(split, args.benchmark_root, args.topk, out / f"{split}.jsonl", args.max_samples)


if __name__ == "__main__":
    main()
