from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_PAGE_RE = re.compile(r"page_(\d+)\.[^.]+$")


def _builder_family(dataset: str) -> str:
    aliases = {
        "V-NIAH": "V-NIAH",
        "V-MQAR": "V-MQAR",
    }
    try:
        return aliases[dataset]
    except KeyError as exc:
        raise ValueError(f"当前只支持 V-NIAH / V-MQAR，收到: {dataset}") from exc


def _resolve_manifest_path(benchmark_root: Path, dataset: str, split: str) -> Path:
    family = _builder_family(dataset)
    candidates = [
        benchmark_root / "data" / "benchmarks" / family / f"{split}.jsonl",
        benchmark_root / family.lower() / f"{split}.jsonl",
        benchmark_root / family / f"{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"找不到 FoveDoc manifest: dataset={dataset}, split={split}\n{tried}")


def _page_id_from_path(path: Path, fallback: int) -> int:
    match = _PAGE_RE.search(path.name)
    if match is None:
        return fallback
    return int(match.group(1))


def _resolve_page_paths(benchmark_root: Path, row: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    raw_paths = list(row.get("images") or [])
    document_id = str(row.get("source_document_id") or row.get("document_id") or "").strip()
    context_page_ids = list(row.get("context_page_ids") or [])
    if not raw_paths and document_id and context_page_ids:
        raw_paths = ["" for _ in context_page_ids]
    pages: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_paths, start=1):
        raw_text = str(raw or "").strip()
        path = Path(raw_text)
        if raw_text and not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file() and document_id and idx <= len(context_page_ids):
            try:
                page_num = int(context_page_ids[idx - 1])
            except (TypeError, ValueError):
                page_num = idx
            path = benchmark_root / "data" / "render" / document_id / f"page_{page_num:04d}.png"
        if not path.is_file():
            raise FileNotFoundError(
                f"图片不存在: sample={row.get('task_id') or row.get('id')} idx={idx} raw={raw_text}"
            )
        page_id = _page_id_from_path(path, idx)
        pages.append({"page_id": page_id, "image_path": str(path)})
    return pages


def _gold_page_ids(row: dict[str, Any]) -> list[int]:
    ids: set[int] = set()
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        try:
            page_id = int(item.get("page_id"))
        except (TypeError, ValueError):
            continue
        if page_id > 0:
            ids.add(page_id)
    return sorted(ids)


def export_manifest(
    *,
    benchmark_root: Path,
    datasets: list[str],
    split: str,
    max_samples: int | None,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    dataset_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as out_f:
        for dataset in datasets:
            manifest_path = _resolve_manifest_path(benchmark_root, dataset, split)
            count = 0
            with manifest_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if max_samples is not None and count >= max_samples:
                        break
                    row = json.loads(line)
                    pages = _resolve_page_paths(benchmark_root, row, manifest_path)
                    record = {
                        "dataset": dataset,
                        "sample_id": str(row.get("task_id") or row.get("id") or ""),
                        "document_id": str(row.get("source_document_id") or row.get("document_id") or ""),
                        "question": str(row.get("question") or "").strip(),
                        "answers": [str(row.get("answer"))] if row.get("answer") is not None else [],
                        "gold_page_ids": _gold_page_ids(row),
                        "pages": pages,
                        "num_pages": len(pages),
                        "difficulty": row.get("difficulty") or {},
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    written += 1
            dataset_counts[dataset] = count
    return {
        "output_path": str(output_path),
        "total_rows": written,
        "dataset_counts": dataset_counts,
        "split": split,
        "max_samples_per_dataset": max_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FoveDoc samples into a page-image retrieval manifest for external retrievers."
    )
    parser.add_argument(
        "--benchmark-root",
        default="data/fovedoc_project",
    )
    parser.add_argument("--datasets", nargs="+", default=["V-NIAH", "V-MQAR"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--output-path",
        default="code/runs/20260430/112900_phase_b3_external_retrieval_readiness/fovedoc_page_retrieval_manifest_smoke.jsonl",
    )
    args = parser.parse_args()

    summary = export_manifest(
        benchmark_root=Path(args.benchmark_root),
        datasets=[str(x) for x in args.datasets],
        split=str(args.split),
        max_samples=None if args.max_samples < 0 else int(args.max_samples),
        output_path=Path(args.output_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
