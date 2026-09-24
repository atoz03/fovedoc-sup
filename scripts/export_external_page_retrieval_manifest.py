#!/usr/bin/env python
"""Emit a ColQwen2 page-retrieval manifest for the external long-document benchmarks.

`scripts/run_colqwen2_page_retrieval.py` is manifest-driven and dataset-agnostic: each row
needs `sample_id`, `dataset`, `question` and `pages: [{page_id, image_path}]`. This builds
that from the VLMEvalKit TSVs so MMLongBench-Doc / SlideVQA can go through exactly the same
retrieve-then-read protocol as FoveDoc.

page_id follows the loader's convention — the 1-based POSITION in the row's `image_path`
list (`eccv26.data.benchmarks._iter_mmlongbench_doc_tsv` sets `_input_page_ids` that way).
Verified on MMLongBench-Doc: the `_<n>.jpg` suffix equals the position for all 1091 rows, so
the OCR store keyed by filename number and the reader agree on what "page 7" means.

Samples whose pages are not fully present in the image cache are skipped and counted, rather
than silently retrieved over a partial document.

Usage:
  .venv/bin/python scripts/export_external_page_retrieval_manifest.py \
      --tsv data/LMUData/MMLongBench_DOC.tsv \
      --dataset MMLongBench_DOC --out code/runs/20260810/external/mmlongbench_manifest.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--dataset", required=True, help="dataset name the reader harness will use")
    ap.add_argument("--image-cache-dir", default=None, help="default: <tsv dir>/images/<tsv stem>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pages", type=int, default=0, help="skip documents longer than this (0 = no cap)")
    args = ap.parse_args()

    import pandas as pd

    tsv_path = Path(args.tsv)
    cache_dir = Path(args.image_cache_dir) if args.image_cache_dir else tsv_path.parent / "images" / tsv_path.stem
    df = pd.read_csv(tsv_path, sep="\t", usecols=["index", "question", "doc_id", "image_path"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped_missing = skipped_long = 0
    with out_path.open("w", encoding="utf-8") as w:
        for row in df.itertuples(index=False):
            try:
                names = [Path(str(n)).name for n in ast.literal_eval(row.image_path)]
            except Exception:
                skipped_missing += 1
                continue
            if args.max_pages and len(names) > args.max_pages:
                skipped_long += 1
                continue
            pages = []
            complete = True
            for position, name in enumerate(names, start=1):
                path = cache_dir / name
                if not path.exists():
                    complete = False
                    break
                pages.append({"page_id": position, "image_path": str(path)})
            if not complete or not pages:
                skipped_missing += 1
                continue
            w.write(json.dumps({
                "sample_id": str(row.index),
                "dataset": str(args.dataset),
                "document_id": str(row.doc_id),
                "question": str(row.question),
                "pages": pages,
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[written] {out_path}: kept={kept} skipped_missing_pages={skipped_missing} skipped_too_long={skipped_long}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
