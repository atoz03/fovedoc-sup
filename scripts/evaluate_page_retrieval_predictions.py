from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _ranked_page_ids_from_prediction(row: dict[str, Any]) -> list[int]:
    for key in ["ranked_page_ids", "selected_page_ids", "page_ids", "top_page_ids"]:
        ids = _as_int_list(row.get(key))
        if ids:
            return ids

    scored = row.get("scores")
    if isinstance(scored, list):
        pairs: list[tuple[int, float]] = []
        for item in scored:
            if not isinstance(item, dict):
                continue
            try:
                page_id = int(item.get("page_id"))
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            pairs.append((page_id, score))
        if pairs:
            return [page_id for page_id, _ in sorted(pairs, key=lambda x: (-x[1], x[0]))]

    return []


def _make_baseline_predictions(
    manifest_rows: list[dict[str, Any]],
    *,
    baseline: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    predictions: list[dict[str, Any]] = []
    for row in manifest_rows:
        page_ids = [int(page["page_id"]) for page in row.get("pages", []) if "page_id" in page]
        gold_ids = _as_int_list(row.get("gold_page_ids"))
        if baseline == "first":
            ranked = page_ids
        elif baseline == "oracle":
            # oracle 只用于验证 evaluator，上游真实 baseline 不能使用 gold。
            gold_first = [page_id for page_id in gold_ids if page_id in page_ids]
            ranked = gold_first + [page_id for page_id in page_ids if page_id not in set(gold_first)]
        elif baseline == "random":
            ranked = list(page_ids)
            rng.shuffle(ranked)
        else:
            raise ValueError(f"未知 baseline: {baseline}")
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row.get("dataset"),
                "ranked_page_ids": ranked,
                "baseline": baseline,
                "latency_s": 0.0,
            }
        )
    return predictions


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "recall_mean": None,
            "full_recall_mean": None,
            "hit_rate": None,
            "latency_s_mean": None,
        }
    return {
        "num_samples": len(records),
        "recall_mean": _safe_mean([float(x["recall"]) for x in records]),
        "full_recall_mean": _safe_mean([float(x["full_recall"]) for x in records]),
        "hit_rate": _safe_mean([float(x["hit"]) for x in records]),
        "latency_s_mean": _safe_mean(
            [float(x["latency_s"]) for x in records if x.get("latency_s") is not None]
        ),
    }


def evaluate(
    *,
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    ks: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_id = {str(row.get("sample_id")): row for row in prediction_rows}
    per_sample: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = str(row.get("sample_id"))
        pred = predictions_by_id.get(sample_id, {})
        ranked = _ranked_page_ids_from_prediction(pred)
        gold_ids = set(_as_int_list(row.get("gold_page_ids")))
        latency = pred.get("latency_s")
        try:
            latency_s = None if latency is None else float(latency)
        except (TypeError, ValueError):
            latency_s = None

        for k in ks:
            selected = ranked[:k]
            selected_set = set(selected)
            if not gold_ids:
                recall = None
                full_recall = None
                hit = None
            else:
                hits = gold_ids & selected_set
                recall = len(hits) / len(gold_ids)
                full_recall = float(gold_ids.issubset(selected_set))
                hit = float(bool(hits))
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "dataset": row.get("dataset"),
                    "k": k,
                    "gold_page_ids": sorted(gold_ids),
                    "selected_page_ids": selected,
                    "recall": recall,
                    "full_recall": full_recall,
                    "hit": hit,
                    "latency_s": latency_s,
                    "num_pages": int(row.get("num_pages", 0)),
                    "num_selected": len(selected),
                }
            )

    summary: dict[str, Any] = {"overall": {}, "by_dataset": {}}
    for k in ks:
        k_records = [x for x in per_sample if int(x["k"]) == k and x["recall"] is not None]
        summary["overall"][f"@{k}"] = _summarize(k_records)
        datasets = sorted({str(x.get("dataset")) for x in k_records})
        summary["by_dataset"][f"@{k}"] = {
            dataset: _summarize([x for x in k_records if str(x.get("dataset")) == dataset])
            for dataset in datasets
        }
    return summary, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate page-level retrieval predictions against a FoveDoc page manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--baseline", choices=["first", "oracle", "random"], default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.predictions is None and args.baseline is None:
        raise ValueError("必须提供 --predictions 或 --baseline。")

    manifest_rows = _read_jsonl(Path(args.manifest))
    if args.predictions is not None:
        prediction_rows = _read_jsonl(Path(args.predictions))
    else:
        prediction_rows = _make_baseline_predictions(
            manifest_rows,
            baseline=str(args.baseline),
            seed=int(args.seed),
        )

    summary, per_sample = evaluate(
        manifest_rows=manifest_rows,
        prediction_rows=prediction_rows,
        ks=[int(k) for k in args.ks],
    )
    summary["manifest"] = str(args.manifest)
    summary["predictions"] = str(args.predictions) if args.predictions else None
    summary["baseline"] = args.baseline
    summary["ks"] = [int(k) for k in args.ks]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "page_retrieval_eval_per_sample.jsonl", per_sample)
    if args.predictions is None:
        _write_jsonl(out_dir / "page_retrieval_predictions.jsonl", prediction_rows)
    (out_dir / "page_retrieval_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
