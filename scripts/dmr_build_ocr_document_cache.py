#!/usr/bin/env python
"""OCR-parity control: rebuild the document-block store from RENDERED PAGE IMAGES.

Why: the DMR text memory is currently packed from `data/documents/<doc>.json`, whose blocks
come from the born-digital PDF text layer (PyMuPDF). That makes the headline
retrieval-reading gap partly a statement about *free perfect OCR* rather than about a
deployable pipeline. This script produces a parallel document store with the SAME schema
but blocks recovered by OCR (RapidOCR / PP-OCR ONNX, CPU-only), so the memory arms can be
re-run end-to-end on OCR text and the gap re-measured under realistic extraction noise.

Output schema matches `eccv26.utils.exact_block.load_document_page_blocks` expectations:
    {"document_id", "pages": [{"page_id", "width", "height", "page_text",
                               "blocks": [{"block_id", "bbox", "text", "order_index"}]}]}
Coordinates are in IMAGE PIXELS and width/height are the pixel dims, so bbox-derived
features stay self-consistent (they are normalised by page dims downstream).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/dmr_build_ocr_document_cache.py \
      --benchmark-root <builder project> --out-dir code/runs/20260810/dmr_ocr_documents \
      --workers 48
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Keep every worker single-threaded; parallelism comes from the process pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_PAGE_NUM_RE = re.compile(r"page_(\d+)\.png$", re.IGNORECASE)
_EXTERNAL_PAGE_NUM_RE = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

_OCR = None  # per-process RapidOCR singleton


def _page_id_from_image_path(path: str) -> int | None:
    m = _PAGE_NUM_RE.search(str(path))
    return int(m.group(1)) if m else None


def collect_documents(
    *, benchmark_root: Path, tasks: list[str], splits: list[str]
) -> dict[str, list[tuple[int, str]]]:
    """document_id -> sorted [(page_id, image_path)] over the release manifests."""
    docs: dict[str, dict[int, str]] = {}
    for task in tasks:
        for split in splits:
            manifest = benchmark_root / "data" / "manifests" / "final_benchmark" / task / f"{split}.jsonl"
            if not manifest.exists():
                continue
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    doc_id = str(row.get("document_id") or "").strip()
                    if doc_id == "":
                        continue
                    pages = docs.setdefault(doc_id, {})
                    for image_path in row.get("images") or []:
                        page_id = _page_id_from_image_path(str(image_path))
                        if page_id is not None:
                            pages.setdefault(page_id, str(image_path))
    return {doc_id: sorted(pages.items()) for doc_id, pages in docs.items()}


def collect_documents_from_tsv(
    *, tsv_path: Path, image_cache_dir: Path, doc_column: str = "doc_id"
) -> dict[str, list[tuple[int, str]]]:
    """document_id -> [(page_id, image_path)] for the external long-document benchmarks.

    MMLongBench-Doc / SlideVQA ship page images in a flat cache named `<stem>_<page>.jpg`.
    Only cached pages are enumerated; documents whose pages were never rendered are skipped
    by the caller so a partial store never masquerades as a complete one.
    """
    import ast

    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t", usecols=[doc_column, "image_path"])
    docs: dict[str, dict[int, str]] = {}
    for doc_id, raw_paths in zip(df[doc_column], df["image_path"]):
        try:
            names = ast.literal_eval(raw_paths)
        except Exception:
            continue
        pages = docs.setdefault(str(doc_id), {})
        for name in names:
            base = Path(str(name)).name
            m = _EXTERNAL_PAGE_NUM_RE.search(base)
            if m is None:
                continue
            candidate = image_cache_dir / base
            if candidate.exists():
                pages[int(m.group(1))] = str(candidate)
    return {doc_id: sorted(pages.items()) for doc_id, pages in docs.items() if pages}


def _get_ocr(config_path: str | None):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR  # imported lazily inside the worker

        _OCR = RapidOCR(config_path=config_path)
    return _OCR


def _poly_to_bbox(poly: Any) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


_SPACE_AFTER_PUNCT = re.compile(r'([.,;:!?)\]"”’])(?=[A-Za-z0-9À-ɏ])')
_SPACE_BEFORE_PUNCT = re.compile(r'(?<=[A-Za-z0-9À-ɏ])([(\[“‘])')


def repair_spacing(text: str) -> str:
    """Re-insert spaces the recogniser drops next to punctuation.

    PP-OCR routinely emits `Tempering.Empirically` and `the"strength"of`, which would break
    the downstream word-level lexical scorer for reasons that are an artefact of this
    recogniser rather than of OCR viability. Only punctuation-adjacent boundaries are
    touched (no CamelCase splitting, which would damage real tokens like `arXiv`).
    """
    text = _SPACE_AFTER_PUNCT.sub(r"\1 ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r" \1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _x_overlap_ratio(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter = min(a[2], b[2]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    narrower = min(a[2] - a[0], b[2] - b[0])
    return inter / narrower if narrower > 0 else 0.0


def group_lines_into_blocks(
    lines: list[tuple[tuple[float, float, float, float], str]],
    *,
    gap_factor: float,
    min_x_overlap: float,
) -> list[dict[str, Any]]:
    """Merge OCR lines into paragraph-like blocks.

    A line joins an open block when it sits within `gap_factor` median line-heights below
    it and their x-ranges overlap enough. Multi-column pages fall out for free: a
    right-column line has no x-overlap with the left-column block, so it opens its own.
    """
    if not lines:
        return []
    heights = [b[3] - b[1] for b, _ in lines if b[3] > b[1]]
    med_h = statistics.median(heights) if heights else 1.0
    max_gap = gap_factor * med_h

    ordered = sorted(lines, key=lambda item: (item[0][1], item[0][0]))
    blocks: list[dict[str, Any]] = []
    for bbox, text in ordered:
        best_idx, best_ov = None, 0.0
        for idx, blk in enumerate(blocks):
            gap = bbox[1] - blk["bbox"][3]
            if gap > max_gap or gap < -0.5 * med_h:
                continue
            ov = _x_overlap_ratio(bbox, blk["bbox"])
            if ov >= min_x_overlap and ov > best_ov:
                best_idx, best_ov = idx, ov
        if best_idx is None:
            blocks.append({"bbox": list(bbox), "lines": [text]})
        else:
            blk = blocks[best_idx]
            blk["bbox"] = [
                min(blk["bbox"][0], bbox[0]),
                min(blk["bbox"][1], bbox[1]),
                max(blk["bbox"][2], bbox[2]),
                max(blk["bbox"][3], bbox[3]),
            ]
            blk["lines"].append(text)
    return blocks


def _reading_order_sort(blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    half = page_width / 2.0 if page_width > 0 else 0.0
    # band 0 = left/full-width, band 1 = right column; single-column pages all land in band 0.
    return sorted(blocks, key=lambda b: (1 if (half > 0 and b["bbox"][0] >= half) else 0, b["bbox"][1]))


def ocr_document(args: tuple[str, list[tuple[int, str]], str, dict[str, Any]]) -> dict[str, Any]:
    doc_id, pages, out_dir, opts = args
    out_path = Path(out_dir) / f"{doc_id}.json"
    if out_path.exists() and not opts["overwrite"]:
        return {"document_id": doc_id, "status": "cached", "pages": 0}

    from PIL import Image

    ocr = _get_ocr(opts.get("config_path"))
    t0 = time.time()
    out_pages: list[dict[str, Any]] = []
    n_lines = 0
    for page_id, image_path in pages:
        if not Path(image_path).exists():
            out_pages.append(
                {"page_id": page_id, "width": 0.0, "height": 0.0, "page_text": "",
                 "blocks": [], "_ocr_error": "missing_image"}
            )
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        try:
            result, _elapse = ocr(image_path)
        except Exception as exc:  # a single bad page must not kill the document
            out_pages.append(
                {"page_id": page_id, "width": float(width), "height": float(height),
                 "page_text": "", "blocks": [], "_ocr_error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        lines: list[tuple[tuple[float, float, float, float], str]] = []
        for item in result or []:
            poly, text, score = item[0], str(item[1]), float(item[2])
            if score < opts["min_score"] or text.strip() == "":
                continue
            lines.append((_poly_to_bbox(poly), text.strip()))
        n_lines += len(lines)

        grouped = group_lines_into_blocks(
            lines, gap_factor=opts["gap_factor"], min_x_overlap=opts["min_x_overlap"]
        )
        grouped = _reading_order_sort(grouped, float(width))
        blocks: list[dict[str, Any]] = []
        for order_index, blk in enumerate(grouped, start=1):
            text = " ".join(blk["lines"]).strip()
            if opts["repair_spacing"]:
                text = repair_spacing(text)
            if len(text) < opts["min_block_chars"]:
                continue
            blocks.append(
                {
                    "block_id": f"p{page_id}_b{len(blocks) + 1}",
                    "type": "ocr_text",
                    "bbox": [round(v, 2) for v in blk["bbox"]],
                    "text": text,
                    "order_index": order_index,
                }
            )
        out_pages.append(
            {
                "page_id": page_id,
                "width": float(width),
                "height": float(height),
                "image_path": image_path,
                "page_text": "\n".join(b["text"] for b in blocks),
                "blocks": blocks,
            }
        )

    payload = {
        "schema_version": "1.0",
        "document_id": doc_id,
        "meta": {
            "block_source": "ocr",
            "ocr_engine": "rapidocr_onnxruntime",
            "min_score": opts["min_score"],
            "gap_factor": opts["gap_factor"],
            "min_x_overlap": opts["min_x_overlap"],
            "repair_spacing": opts["repair_spacing"],
        },
        "pages": out_pages,
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    return {
        "document_id": doc_id,
        "status": "ok",
        "pages": len(out_pages),
        "lines": n_lines,
        "blocks": sum(len(p.get("blocks") or []) for p in out_pages),
        "seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("fovedoc", "tsv"), default="fovedoc",
                    help="fovedoc = release manifests; tsv = external long-doc benchmark image cache")
    ap.add_argument("--benchmark-root", help="required for --source fovedoc")
    ap.add_argument("--tsv", help="required for --source tsv, e.g. .../MMLongBench_DOC.tsv")
    ap.add_argument("--image-cache-dir", help="page-image cache for --source tsv "
                                              "(default: <tsv dir>/images/<tsv stem>)")
    ap.add_argument("--doc-column", default="doc_id")
    ap.add_argument("--require-complete", action="store_true",
                    help="--source tsv: skip documents with any page missing from the image cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="vmqar,vniah")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--gap-factor", type=float, default=1.1, help="max inter-line gap in median line heights")
    ap.add_argument("--min-x-overlap", type=float, default=0.5)
    ap.add_argument("--min-block-chars", type=int, default=2)
    ap.add_argument("--no-repair-spacing", action="store_true",
                    help="keep raw recogniser output including dropped spaces around punctuation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "fovedoc":
        if not args.benchmark_root:
            ap.error("--source fovedoc requires --benchmark-root")
        docs = collect_documents(
            benchmark_root=Path(args.benchmark_root),
            tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
            splits=[s.strip() for s in args.splits.split(",") if s.strip()],
        )
    else:
        if not args.tsv:
            ap.error("--source tsv requires --tsv")
        tsv_path = Path(args.tsv)
        cache_dir = (
            Path(args.image_cache_dir)
            if args.image_cache_dir
            else tsv_path.parent / "images" / tsv_path.stem
        )
        docs = collect_documents_from_tsv(
            tsv_path=tsv_path, image_cache_dir=cache_dir, doc_column=str(args.doc_column)
        )
        if args.require_complete:
            import ast

            import pandas as pd

            df = pd.read_csv(tsv_path, sep="\t", usecols=[str(args.doc_column), "image_path"])
            expected: dict[str, set[str]] = {}
            for doc_id, raw in zip(df[str(args.doc_column)], df["image_path"]):
                try:
                    names = ast.literal_eval(raw)
                except Exception:
                    continue
                expected.setdefault(str(doc_id), set()).update(Path(str(n)).name for n in names)
            before = len(docs)
            docs = {
                d: pages
                for d, pages in docs.items()
                if len(pages) >= len(expected.get(d, set()))
            }
            print(f"[filter] --require-complete dropped {before - len(docs)} partially-rendered documents",
                  flush=True)
    doc_items = sorted(docs.items())
    if args.limit > 0:
        doc_items = doc_items[: args.limit]
    total_pages = sum(len(p) for _, p in doc_items)
    print(f"[plan] documents={len(doc_items)} pages={total_pages} workers={args.workers}", flush=True)

    # One shared single-threaded ORT config so N workers do not oversubscribe the box.
    import yaml
    import rapidocr_onnxruntime as rocr

    base_cfg = yaml.safe_load((Path(rocr.__file__).parent / "config.yaml").read_text(encoding="utf-8"))
    for section in ("Global", "Det", "Cls", "Rec"):
        if section in base_cfg:
            base_cfg[section]["intra_op_num_threads"] = 1
            base_cfg[section]["inter_op_num_threads"] = 1
    cfg_path = out_dir / "_rapidocr_1thread.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, allow_unicode=True), encoding="utf-8")

    opts = {
        "overwrite": bool(args.overwrite),
        "min_score": float(args.min_score),
        "gap_factor": float(args.gap_factor),
        "min_x_overlap": float(args.min_x_overlap),
        "min_block_chars": int(args.min_block_chars),
        "repair_spacing": not bool(args.no_repair_spacing),
        "config_path": str(cfg_path),
    }
    payloads = [(doc_id, pages, str(out_dir), opts) for doc_id, pages in doc_items]

    import multiprocessing as mp

    t0 = time.time()
    done = cached = failed = 0
    pages_done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, int(args.workers))) as pool:
        for res in pool.imap_unordered(ocr_document, payloads, chunksize=1):
            if res["status"] == "cached":
                cached += 1
            else:
                done += 1
                pages_done += res.get("pages", 0)
                if res.get("blocks", 0) == 0:
                    failed += 1
            n = done + cached
            if n % 25 == 0 or n == len(payloads):
                rate = pages_done / max(1e-6, time.time() - t0)
                eta = (total_pages - pages_done) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"[{n}/{len(payloads)}] ocr={done} cached={cached} zero_block_docs={failed} "
                    f"{rate:.1f} pages/s eta={eta:.1f} min",
                    flush=True,
                )
    print(f"[done] {done} documents OCR'd, {cached} cached, {failed} produced no blocks, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
