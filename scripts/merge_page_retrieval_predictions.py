from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSONL。") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} 不是 JSON object。")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_id(row: dict[str, Any], *, source: str) -> str:
    value = str(row.get("sample_id") or "").strip()
    if not value:
        raise ValueError(f"{source} 缺少 sample_id。")
    return value


def _manifest_order(manifest_path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(_read_jsonl(manifest_path), start=1):
        sample_id = _sample_id(row, source=f"{manifest_path}:{index}")
        if sample_id in seen:
            raise ValueError(f"{manifest_path} 存在重复 sample_id: {sample_id}")
        seen.add(sample_id)
        order.append(sample_id)
    return order


def merge_predictions(
    *,
    input_paths: list[Path],
    output_path: Path,
    manifest_path: Path | None,
    require_manifest_complete: bool,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("必须至少提供一个 --input。")

    rows_by_id: dict[str, dict[str, Any]] = {}
    sources_by_id: dict[str, str] = {}
    input_counts: dict[str, int] = {}
    for input_path in input_paths:
        rows = _read_jsonl(input_path)
        input_counts[str(input_path)] = len(rows)
        for row_index, row in enumerate(rows, start=1):
            sample_id = _sample_id(row, source=f"{input_path}:{row_index}")
            if sample_id in rows_by_id:
                first_source = sources_by_id[sample_id]
                raise ValueError(
                    f"发现重复 sample_id: {sample_id}，首次来源 {first_source}，重复来源 {input_path}:{row_index}"
                )
            rows_by_id[sample_id] = row
            sources_by_id[sample_id] = f"{input_path}:{row_index}"

    if manifest_path is None:
        merged = [rows_by_id[sample_id] for sample_id in sorted(rows_by_id)]
        missing: list[str] = []
        extra: list[str] = []
        manifest_count = None
    else:
        order = _manifest_order(manifest_path)
        manifest_ids = set(order)
        prediction_ids = set(rows_by_id)
        missing = [sample_id for sample_id in order if sample_id not in prediction_ids]
        extra = sorted(prediction_ids - manifest_ids)
        if require_manifest_complete and missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"合并结果缺少 {len(missing)} 个 manifest 样本，例如: {preview}")
        if extra:
            preview = ", ".join(extra[:5])
            raise ValueError(f"预测文件包含 {len(extra)} 个 manifest 外样本，例如: {preview}")
        # 有 manifest 时按正式样本顺序输出，避免分片顺序影响下游 diff。
        merged = [rows_by_id[sample_id] for sample_id in order if sample_id in rows_by_id]
        manifest_count = len(order)

    _write_jsonl(output_path, merged)
    summary = {
        "output_path": str(output_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_count": manifest_count,
        "num_inputs": len(input_paths),
        "input_counts": input_counts,
        "num_predictions": len(merged),
        "num_missing": len(missing),
        "num_extra": len(extra),
        "missing_preview": missing[:10],
        "extra_preview": extra[:10],
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded page-retrieval prediction JSONL files."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Sharded prediction JSONL paths.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest", default=None, help="Optional manifest JSONL for ordering and checks.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing manifest samples; extra samples still fail.",
    )
    args = parser.parse_args()

    summary = merge_predictions(
        input_paths=[Path(path) for path in args.input],
        output_path=Path(args.output_path),
        manifest_path=Path(args.manifest) if args.manifest else None,
        require_manifest_complete=not bool(args.allow_partial),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
