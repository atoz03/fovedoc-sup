#!/usr/bin/env python
"""How much gold evidence survives OCR? — the ceiling of the OCR-parity memory arm.

The DMR memory packs *blocks*. So an OCR store can lose evidence two ways:
  1. the recogniser never produced the words (token loss), and
  2. the words exist but are scattered across blocks, so no single packable block carries
     the evidence (fragmentation).
This report measures both, against the born-digital PDF text layer as the reference, using
the SAME tokenizer the lexical block scorer uses so the numbers speak to retrieval utility
rather than to string prettiness.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/dmr_ocr_fidelity_report.py \
      --benchmark-root <builder project> --ocr-root code/runs/20260810/dmr_ocr_documents \
      --out results/tables/dmr_ocr_fidelity.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eccv26.utils.exact_block import load_document_page_blocks, tokenize_exact_block_text  # noqa: E402


def _tok(text: str | None) -> Counter[str]:
    return Counter(tokenize_exact_block_text(text or ""))


def _recall(need: Counter[str], have: Counter[str]) -> float:
    total = sum(need.values())
    if total == 0:
        return float("nan")
    covered = sum(min(c, have.get(t, 0)) for t, c in need.items())
    return covered / total


def _source_family(document_id: str) -> str:
    for prefix in ("arxiv", "docvqa", "cloudflare", "aws", "wipo", "registry", "gov", "nasa"):
        if document_id.startswith(prefix):
            return prefix
    return document_id.split("_", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--ocr-root", required=True)
    ap.add_argument("--tasks", default="vmqar,vniah")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--recoverable-threshold", type=float, default=0.9)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--dump-jsonl",
        default=None,
        help="Write per-evidence-item rows keyed by sample_id. Samples map 1:1 to documents, so "
        "this joins directly to per-sample judged correctness — the fidelity dose-response "
        "(does the OCR arm fail exactly where OCR lost the evidence?) that replaces the "
        "scanned-vs-born-digital stratification the release set turned out not to support.",
    )
    args = ap.parse_args()

    benchmark_root = Path(args.benchmark_root)
    ocr_root = Path(args.ocr_root)
    thr = float(args.recoverable_threshold)

    doc_cache: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}

    def pages_of(doc_id: str, source: str) -> dict[int, dict[str, Any]]:
        key = (doc_id, source)
        if key not in doc_cache:
            path = (
                ocr_root / f"{doc_id}.json"
                if source == "ocr"
                else benchmark_root / "data" / "documents" / f"{doc_id}.json"
            )
            doc_cache[key] = load_document_page_blocks(str(path)) if path.exists() else {}
        return doc_cache[key]

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_docs: set[str] = set()

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        seen = 0
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            manifest = benchmark_root / "data" / "manifests" / "final_benchmark" / task / f"{split}.jsonl"
            if not manifest.exists():
                continue
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    if args.max_samples and seen >= args.max_samples:
                        break
                    seen += 1
                    r = json.loads(line)
                    doc_id = str(r.get("document_id") or "")
                    if not doc_id:
                        continue
                    sample_id = str(r.get("id") or "")
                    if not (ocr_root / f"{doc_id}.json").exists():
                        missing_docs.add(doc_id)
                        continue
                    ocr_pages = pages_of(doc_id, "ocr")
                    pdf_pages = pages_of(doc_id, "pdf")
                    for ev in r.get("meta", {}).get("evidence") or []:
                        page_id = ev.get("page_id")
                        need = _tok(ev.get("text_excerpt"))
                        if page_id is None or sum(need.values()) == 0:
                            continue
                        entry: dict[str, Any] = {
                            "task": task,
                            "sample_id": sample_id,
                            "document_id": doc_id,
                            "page_id": int(page_id),
                            "family": _source_family(doc_id),
                            "evidence_tokens": sum(need.values()),
                        }
                        for source, pages in (("ocr", ocr_pages), ("pdf", pdf_pages)):
                            page = pages.get(int(page_id)) or {}
                            blocks = page.get("blocks") or []
                            page_tokens: Counter[str] = Counter()
                            best_block = 0.0
                            for b in blocks:
                                bt = _tok(b.get("text"))
                                page_tokens += bt
                                best_block = max(best_block, _recall(need, bt))
                            entry[f"{source}_page_recall"] = _recall(need, page_tokens)
                            entry[f"{source}_best_block_recall"] = best_block
                        rows[task].append(entry)

    def summarize(items: list[dict[str, Any]], key: str) -> tuple[float, float]:
        vals = [x[key] for x in items if x.get(key) == x.get(key)]
        if not vals:
            return float("nan"), float("nan")
        return statistics.fmean(vals), sum(1 for v in vals if v >= thr) / len(vals)

    lines: list[str] = []
    lines.append("# OCR fidelity of gold evidence — ceiling for the OCR-parity memory arm\n")
    lines.append(
        f"Reference = born-digital PDF text layer (`data/documents`), candidate = RapidOCR over the\n"
        f"rendered page images (`{ocr_root}`). Tokenizer is the one the lexical block scorer uses.\n"
        f"*page recall* = fraction of gold-excerpt tokens present anywhere on the page; *best-block\n"
        f"recall* = the most any single packable block covers (what the memory can actually select).\n"
        f"`recoverable` = share of evidence items at or above {thr:.2f} recall.\n"
    )
    lines.append("| task | n | src | page recall | page recoverable | best-block recall | block recoverable |")
    lines.append("|---|---|---|---|---|---|---|")
    for task, items in rows.items():
        for source, label in (("pdf", "PDF text layer"), ("ocr", "OCR")):
            pr, prec = summarize(items, f"{source}_page_recall")
            br, brec = summarize(items, f"{source}_best_block_recall")
            lines.append(
                f"| {task} | {len(items)} | {label} | {pr:.3f} | {prec:.3f} | {br:.3f} | {brec:.3f} |"
            )

    lines.append("\n## By source family (OCR best-block recall)\n")
    lines.append("| family | n | pdf best-block | ocr best-block | delta |")
    lines.append("|---|---|---|---|---|")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for items in rows.values():
        for x in items:
            by_family[x["family"]].append(x)
    for family, items in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        p, _ = summarize(items, "pdf_best_block_recall")
        o, _ = summarize(items, "ocr_best_block_recall")
        lines.append(f"| {family} | {len(items)} | {p:.3f} | {o:.3f} | {o - p:+.3f} |")

    lines.append("\n## Block-budget parity\n")
    lines.append(
        "The two arms pack the same top-k blocks at the same per-block char cap, so if OCR blocks\n"
        "are systematically shorter or more fragmented than PDF paragraphs the OCR arm is simply\n"
        "budget-starved and fidelity is confounded with packing budget. These are the numbers that\n"
        "decide whether a total-chars-matched variant is needed.\n"
    )
    lines.append("| src | blocks/page | chars/block (mean) | chars/block (median) | chars/page |")
    lines.append("|---|---|---|---|---|")
    pack: dict[str, dict[str, list[float]]] = {
        s: {"per_page": [], "chars": [], "page_chars": []} for s in ("pdf", "ocr")
    }
    for (doc_id, source), pages in doc_cache.items():
        for page in pages.values():
            blocks = page.get("blocks") or []
            pack[source]["per_page"].append(len(blocks))
            page_chars = 0
            for b in blocks:
                n = len(str(b.get("text") or ""))
                pack[source]["chars"].append(n)
                page_chars += n
            pack[source]["page_chars"].append(page_chars)
    for source, label in (("pdf", "PDF text layer"), ("ocr", "OCR")):
        d = pack[source]
        if not d["chars"]:
            continue
        lines.append(
            f"| {label} | {statistics.fmean(d['per_page']):.1f} | {statistics.fmean(d['chars']):.0f} | "
            f"{statistics.median(d['chars']):.0f} | {statistics.fmean(d['page_chars']):.0f} |"
        )

    if missing_docs:
        lines.append(f"\n> {len(missing_docs)} documents had no OCR store yet and were skipped.\n")

    if args.dump_jsonl:
        dump_path = Path(args.dump_jsonl)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            for items in rows.values():
                for x in items:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[written] {dump_path} ({sum(len(v) for v in rows.values())} evidence rows)")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[written] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
