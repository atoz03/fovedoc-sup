from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
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


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _load_runtime(colpali_package_dir: Path) -> tuple[Any, Any, Any, Any]:
    import sys

    sys.path.insert(0, str(colpali_package_dir))
    import torch
    from peft import PeftModel
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    return torch, PeftModel, Image, (ColQwen2, ColQwen2Processor)


def run_colqwen2(
    *,
    manifest_path: Path,
    output_path: Path,
    colpali_package_dir: Path,
    base_model_path: Path,
    adapter_path: Path,
    device: str,
    image_batch_size: int,
    max_rows: int | None,
    shard_index: int,
    num_shards: int,
    progress_every: int,
) -> dict[str, Any]:
    torch, PeftModel, Image, classes = _load_runtime(colpali_package_dir)
    ColQwen2, ColQwen2Processor = classes

    rows = _read_jsonl(manifest_path)
    if max_rows is not None:
        rows = rows[:max_rows]
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if num_shards > 1:
        rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]

    processor = ColQwen2Processor.from_pretrained(
        str(adapter_path),
        local_files_only=True,
        use_fast=False,
    )
    base = ColQwen2.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        local_files_only=True,
    )
    model.eval()

    predictions: list[dict[str, Any]] = []
    total_pages = 0
    started_all = time.perf_counter()

    with torch.inference_mode():
        for row_index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            pages = row.get("pages") or []
            images = [Image.open(page["image_path"]).convert("RGB") for page in pages]
            page_ids = [int(page["page_id"]) for page in pages]

            query_batch = processor.process_queries([str(row.get("question") or "")]).to(device)
            query_embeddings = model(**query_batch)
            query_list = [query_embeddings[0].detach().cpu()]

            image_embeddings = []
            for image_batch in _batched(images, image_batch_size):
                batch_images = processor.process_images(image_batch).to(device)
                batch_embeddings = model(**batch_images)
                image_embeddings.extend([emb.detach().cpu() for emb in batch_embeddings])

            scores_tensor = processor.score_multi_vector(
                query_list,
                image_embeddings,
                batch_size=max(1, min(64, len(image_embeddings))),
                device=device,
            )[0]
            scored_pages = [
                {"page_id": page_id, "score": float(score)}
                for page_id, score in zip(page_ids, scores_tensor.tolist())
            ]
            scored_pages.sort(key=lambda item: (-float(item["score"]), int(item["page_id"])))
            predictions.append(
                {
                    "sample_id": str(row.get("sample_id")),
                    "dataset": row.get("dataset"),
                    "ranked_page_ids": [int(item["page_id"]) for item in scored_pages],
                    "scores": scored_pages,
                    "latency_s": time.perf_counter() - started,
                    "num_pages": len(pages),
                    "model": "vidore/colqwen2-v1.0",
                }
            )
            total_pages += len(pages)
            if progress_every > 0 and (row_index == 1 or row_index % progress_every == 0 or row_index == len(rows)):
                elapsed = time.perf_counter() - started_all
                print(
                    json.dumps(
                        {
                            "progress": f"{row_index}/{len(rows)}",
                            "sample_id": str(row.get("sample_id")),
                            "latency_s": predictions[-1]["latency_s"],
                            "elapsed_s": elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    _write_jsonl(output_path, predictions)
    summary = {
        "manifest": str(manifest_path),
        "output_path": str(output_path),
        "num_samples": len(predictions),
        "num_pages": total_pages,
        "latency_s_total": time.perf_counter() - started_all,
        "image_batch_size": image_batch_size,
        "device": device,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "colpali_package_dir": str(colpali_package_dir),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ColQwen2 page retrieval on a FoveDoc page-image manifest."
    )
    parser.add_argument(
        "--manifest",
        default="code/runs/20260430/112900_phase_b3_external_retrieval_readiness/fovedoc_page_retrieval_manifest_smoke.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="code/runs/20260430/112900_phase_b3_external_retrieval_readiness/colqwen2_predictions_smoke.jsonl",
    )
    parser.add_argument(
        "--colpali-package-dir",
        default="code/runs/20260430/112900_phase_b3_external_retrieval_readiness/colpali_0310_target",
    )
    parser.add_argument("--base-model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--adapter-path",
        default="code/runs/20260430/112900_phase_b3_external_retrieval_readiness/models/vidore_colqwen2-v1.0",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    summary = run_colqwen2(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output_path),
        colpali_package_dir=Path(args.colpali_package_dir),
        base_model_path=Path(args.base_model_path),
        adapter_path=Path(args.adapter_path),
        device=str(args.device),
        image_batch_size=int(args.image_batch_size),
        max_rows=None if int(args.max_rows) < 0 else int(args.max_rows),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
