from __future__ import annotations

import json
import os
import time
from functools import partial
from functools import lru_cache
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_linear_schedule_with_warmup
from PIL import Image

from eccv26.data.sft_jsonl import load_sft_jsonl
from eccv26.grounding import normalize_grounding_training_phase
from eccv26.model.cross_attn import CrossAttnConfig, apply_cross_attn_adapter_to_qwen3vl_model, cross_modules_to_save
from eccv26.model.dma import (
    DMAConfig,
    _freeze_non_record_retention_readout_modules,
    apply_dma_to_qwen3vl_model,
    clear_dma_query_routing_sketch,
    collect_dma_training_block_logits,
    collect_dma_training_page_logits,
    collect_dma_training_selected_ratios,
    dma_modules_to_save,
    reset_dma_page_stats,
    score_dma_exact_block_features,
    set_dma_query_routing_sketch,
)
from eccv26.model.qwen3vl import (
    _align_peft_modules_to_save_devices,
    _load_local_peft_adapter_config,
    score_qwen3vl_block_relevance,
    _validate_adapter_runtime_modules,
)
from eccv26.utils.exact_block import (
    EXACT_BLOCK_FEATURE_INDEX,
    attach_exact_block_teacher_scores,
    build_exact_block_candidate_rows,
    build_exact_block_page_stats_by_id,
    exact_block_candidate_block_sketches_to_tensor,
    exact_block_candidate_features_to_tensor,
    exact_block_candidate_page_ids_to_tensor,
    exact_block_candidate_query_sketches_to_tensor,
    resolve_exact_block_scorer,
)
from eccv26.utils.exact_block_teacher import (
    VLM_BLOCK_RERANK_SYSTEM_TEXT,
    build_block_rerank_user_text,
    crop_exact_block_from_page_image,
)
from eccv26.utils.image import resize_to_max_pixels
from eccv26.utils.io import append_jsonl, dump_json, ensure_dir


def _safe_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _append_startup_trace(trace_path: Path, stage: str, **fields: Any) -> None:
    """在首条 train_log 之前记录多卡启动阶段，避免 torchrun 卡住时完全无迹可查。"""
    record: dict[str, Any] = {
        "ts": time.time(),
        "stage": stage,
        "pid": os.getpid(),
        "rank": _safe_int_env("RANK"),
        "local_rank": _safe_int_env("LOCAL_RANK"),
        "world_size": _safe_int_env("WORLD_SIZE"),
    }
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    append_jsonl(trace_path, record)
    if record["rank"] in (None, 0):
        extra = ", ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        suffix = "" if extra == "" else f" | {extra}"
        print(f"[train_sft][startup] stage={stage} pid={record['pid']}{suffix}", flush=True)


def _wait_for_file_barrier(
    *,
    barrier_dir: Path,
    stage: str,
    world_size: int,
    rank: int,
    timeout_s: float = 600.0,
    poll_interval_s: float = 0.2,
) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    ready_file = barrier_dir / f"rank_{int(rank):02d}.ready"
    ready_file.write_text(str(time.time()), encoding="utf-8")
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        ready_count = sum(1 for idx in range(int(world_size)) if (barrier_dir / f"rank_{idx:02d}.ready").exists())
        if ready_count >= int(world_size):
            return
        time.sleep(float(poll_interval_s))
    raise TimeoutError(f"文件 barrier 超时: stage={stage}, world_size={world_size}, rank={rank}")


def _build_atomic_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")


def _atomic_dump_json(path: str | Path, obj: Any) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _build_atomic_tmp_path(target_path)
    try:
        dump_json(tmp_path, obj)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_torch_save(obj: Any, path: str | Path) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _build_atomic_tmp_path(target_path)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _read_json_file_with_retry(
    path: str | Path,
    *,
    timeout_s: float = 600.0,
    poll_interval_s: float = 0.05,
) -> dict[str, Any]:
    target_path = Path(path)
    deadline = time.time() + float(timeout_s)
    last_error: Exception | None = None
    while time.time() < deadline:
        if target_path.exists():
            try:
                return json.loads(target_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                last_error = exc
        time.sleep(float(poll_interval_s))
    raise TimeoutError(f"读取 JSON 文件超时: path={target_path}") from last_error


def _load_torch_file_with_retry(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    timeout_s: float = 600.0,
    poll_interval_s: float = 0.05,
):
    target_path = Path(path)
    deadline = time.time() + float(timeout_s)
    last_error: Exception | None = None
    while time.time() < deadline:
        if target_path.exists():
            try:
                return torch.load(target_path, map_location=map_location)
            except (RuntimeError, EOFError, OSError) as exc:
                last_error = exc
        time.sleep(float(poll_interval_s))
    raise TimeoutError(f"读取 Torch 文件超时: path={target_path}") from last_error


def _collect_rank_payloads_via_files(
    *,
    barrier_dir: Path,
    stage: str,
    payload: dict[str, Any],
    world_size: int,
    rank: int,
    timeout_s: float = 600.0,
    poll_interval_s: float = 0.2,
) -> list[dict[str, Any]]:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    payload_path = barrier_dir / f"rank_{int(rank):02d}.json"
    _atomic_dump_json(payload_path, payload)
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        payload_paths = [barrier_dir / f"rank_{idx:02d}.json" for idx in range(int(world_size))]
        if all(path.exists() for path in payload_paths):
            records: list[dict[str, Any]] = []
            for path in payload_paths:
                records.append(
                    _read_json_file_with_retry(
                        path,
                        timeout_s=max(1.0, float(poll_interval_s) * 5.0),
                        poll_interval_s=min(float(poll_interval_s), 0.05),
                    )
                )
            return records
        time.sleep(float(poll_interval_s))
    raise TimeoutError(f"文件聚合超时: stage={stage}, world_size={world_size}, rank={rank}")


class SftDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list, image_max_pixels: int | None) -> None:
        self._examples = examples
        self._image_max_pixels = image_max_pixels

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self._examples[idx]
        images = []
        for im in ex.images:
            if isinstance(im, (str, Path)):
                im = Image.open(im).convert("RGB")
            rim, _ = resize_to_max_pixels(im, self._image_max_pixels)
            images.append(rim)
        meta = dict(ex.meta) if isinstance(ex.meta, dict) else {}
        meta.setdefault("_user_text", ex.user_text)
        return {
            "id": ex.example_id,
            "images": images,
            "user_text": ex.user_text,
            "assistant_text": ex.assistant_text,
            "rejected_assistant_text": getattr(ex, "rejected_assistant_text", None),
            "system_prompt": getattr(ex, "system_prompt", None),
            "meta": meta,
        }


def _build_messages(system_prompt: str, images: list, user_text: str, assistant_text: str | None) -> list[dict[str, Any]]:
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": ([{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]),
        },
    ]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return msgs


def _estimate_prompt_token_lengths(
    processor: Any,
    *,
    full_attention_mask: torch.Tensor,
    assistant_texts: list[str],
) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise ValueError("processor 缺少 tokenizer.encode，无法估计 prompt token 长度。")

    prompt_lens: list[int] = []
    for row_idx, assistant_text in enumerate(assistant_texts):
        full_len = int(full_attention_mask[row_idx].sum().item())
        # Qwen3-VL 的 generation prompt 已经包含 assistant 头部；
        # 因此只需要扣掉 assistant 正文以及结尾的 <|im_end|>\\n，就能得到 prompt 长度。
        completion_suffix = f"{assistant_text}<|im_end|>\n"
        completion_len = len(tokenizer.encode(completion_suffix, add_special_tokens=False))
        if completion_len <= 0 or completion_len > full_len:
            raise ValueError(
                f"assistant completion token 长度异常: row={row_idx}, completion_len={completion_len}, full_len={full_len}"
            )
        prompt_lens.append(full_len - completion_len)
    return prompt_lens


def _build_supervised_multimodal_batch(
    processor: Any,
    system_prompt: str,
    batch: list[dict[str, Any]],
    *,
    assistant_field: str,
) -> dict[str, Any]:
    msgs_full = [
        _build_messages(
            str(b.get("system_prompt") or system_prompt),
            b["images"],
            b["user_text"],
            b[assistant_field],
        )
        for b in batch
    ]
    full = processor.apply_chat_template(
        msgs_full,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    labels = full["input_ids"].clone()
    assistant_texts = [str(b[assistant_field]) for b in batch]
    # 默认只做一次 multimodal template 构造，避免 8 卡下首批 CPU 预处理被重复放大。
    # 如果遇到异常样本，再退回旧的 prompt 模板口径，优先保证标注边界正确。
    try:
        prompt_lens = _estimate_prompt_token_lengths(
            processor,
            full_attention_mask=full["attention_mask"],
            assistant_texts=assistant_texts,
        )
    except Exception:
        msgs_prompt = [
            _build_messages(
                str(b.get("system_prompt") or system_prompt),
                b["images"],
                b["user_text"],
                None,
            )
            for b in batch
        ]
        prompt = processor.apply_chat_template(
            msgs_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()

    query_token_mask = torch.zeros_like(full["attention_mask"])
    for i, pl in enumerate(prompt_lens):
        labels[i, : int(pl)] = -100
        query_token_mask[i, : int(pl)] = 1

    full["labels"] = labels
    full["query_token_mask"] = query_token_mask
    return full


def _collate_fn(processor: Any, system_prompt: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
    # full：包含 assistant 用于 labels
    full = _build_supervised_multimodal_batch(
        processor,
        system_prompt,
        batch,
        assistant_field="assistant_text",
    )
    # 只保留 prompt 侧 token，供 query-conditioned token routing 提取 query sketch；
    # 训练时明确排除 assistant 答案，避免把目标答案泄漏进视觉路由。
    full["batch_meta"] = [b["meta"] for b in batch]
    full["sample_image_counts"] = [len(b["images"]) for b in batch]
    full["batch_images"] = [list(b["images"]) for b in batch]
    has_rejected = any(str(b.get("rejected_assistant_text") or "").strip() != "" for b in batch)
    if has_rejected:
        rejected_batch = []
        for item in batch:
            rejected_item = dict(item)
            rejected_text = str(item.get("rejected_assistant_text") or "").strip()
            rejected_item["rejected_assistant_text"] = rejected_text if rejected_text != "" else str(item["assistant_text"])
            rejected_batch.append(rejected_item)
        rejected_full = _build_supervised_multimodal_batch(
            processor,
            system_prompt,
            rejected_batch,
            assistant_field="rejected_assistant_text",
        )
        for key, value in rejected_full.items():
            full[f"rejected_{key}"] = value
    return full


def _page_id_to_int(value: Any) -> int | None:
    try:
        page_id = int(value)
    except (TypeError, ValueError):
        return None
    return page_id if page_id > 0 else None


def _resolve_sample_input_page_ids(meta: dict[str, Any], *, num_images: int) -> list[int]:
    raw = meta.get("_input_page_ids")
    if isinstance(raw, list):
        page_ids = [_page_id_to_int(value) for value in raw]
        page_ids = [page_id for page_id in page_ids if page_id is not None]
        if len(page_ids) >= num_images:
            return page_ids[:num_images]
    return list(range(1, int(num_images) + 1))


def _extract_target_page_ids(meta: dict[str, Any]) -> set[int]:
    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list):
        return set()
    target_page_ids: set[int] = set()
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        page_id = _page_id_to_int(item.get("page_id"))
        if page_id is not None:
            target_page_ids.add(page_id)
    return target_page_ids


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4096)
def _load_document_page_layouts(document_json_path: str) -> dict[int, dict[str, Any]]:
    path = Path(document_json_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_pages = data.get("pages")
    if not isinstance(raw_pages, list):
        return {}

    page_layouts: dict[int, dict[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        page_id = _page_id_to_int(page.get("page_id"))
        if page_id is None:
            continue
        width = _safe_float(page.get("width"))
        height = _safe_float(page.get("height"))
        raw_blocks = page.get("blocks")
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        block_layouts: dict[str, dict[str, Any]] = {}
        ordered_block_ids: list[str] = []
        for order_index, block in enumerate(raw_blocks):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if block_id == "":
                continue
            bbox = block.get("bbox")
            normalized_bbox: tuple[float, float, float, float] | None = None
            if isinstance(bbox, list) and len(bbox) == 4:
                x0 = _safe_float(bbox[0])
                y0 = _safe_float(bbox[1])
                x1 = _safe_float(bbox[2])
                y1 = _safe_float(bbox[3])
                if None not in {x0, y0, x1, y1} and x1 > x0 and y1 > y0:
                    normalized_bbox = (float(x0), float(y0), float(x1), float(y1))
            ordered_block_ids.append(block_id)
            block_layouts[block_id] = {
                "bbox": normalized_bbox,
                "order_index": order_index,
            }
        page_layouts[page_id] = {
            "width": width,
            "height": height,
            "blocks": block_layouts,
            "ordered_block_ids": ordered_block_ids,
        }
    return page_layouts


def _resolve_document_json_path(meta: dict[str, Any]) -> str | None:
    raw_path = meta.get("document_json")
    if isinstance(raw_path, str) and raw_path.strip() != "":
        return str(Path(raw_path))
    document_id = meta.get("document_id")
    if not isinstance(document_id, str) or document_id.strip() == "":
        return None
    benchmark_root = meta.get("benchmark_root")
    if isinstance(benchmark_root, str) and benchmark_root.strip() != "":
        path = Path(benchmark_root) / "data" / "documents" / f"{document_id}.json"
        return str(path)
    return None


def _normalize_coarse_block_index_from_layout(
    *,
    page_id: int,
    raw_block_id: str,
    page_layouts: dict[int, dict[str, Any]] | None,
    grid_size: int = 3,
) -> int | None:
    if page_layouts is None or grid_size < 1:
        return None
    page_layout = page_layouts.get(page_id)
    if not isinstance(page_layout, dict):
        return None
    block_layouts = page_layout.get("blocks")
    if not isinstance(block_layouts, dict):
        return None
    block_layout = block_layouts.get(raw_block_id)
    if not isinstance(block_layout, dict):
        return None

    bbox = block_layout.get("bbox")
    width = _safe_float(page_layout.get("width"))
    height = _safe_float(page_layout.get("height"))
    if (
        isinstance(bbox, tuple)
        and len(bbox) == 4
        and width is not None
        and height is not None
        and width > 0
        and height > 0
    ):
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        col_bin = min(grid_size - 1, max(0, int((center_x / width) * grid_size)))
        row_bin = min(grid_size - 1, max(0, int((center_y / height) * grid_size)))
        return row_bin * grid_size + col_bin

    ordered_block_ids = page_layout.get("ordered_block_ids")
    if not isinstance(ordered_block_ids, list) or len(ordered_block_ids) == 0:
        return None
    try:
        order_index = ordered_block_ids.index(raw_block_id)
    except ValueError:
        order_index = block_layout.get("order_index")
    if not isinstance(order_index, int) or order_index < 0:
        return None
    coarse_index = min(grid_size * grid_size - 1, int(order_index * (grid_size * grid_size) / len(ordered_block_ids)))
    return coarse_index


def _coarse_block_id_to_index(
    value: Any,
    *,
    page_layouts: dict[int, dict[str, Any]] | None = None,
) -> int | None:
    if not isinstance(value, str):
        return None
    block_id = value.strip()
    if block_id == "":
        return None
    if "_g" in block_id:
        suffix = block_id.rsplit("_g", 1)[-1]
        if len(suffix) == 2 and suffix.isdigit():
            row_bin = int(suffix[0])
            col_bin = int(suffix[1])
            if 0 <= row_bin <= 2 and 0 <= col_bin <= 2:
                return row_bin * 3 + col_bin
    if "_b" in block_id:
        if page_layouts is not None:
            page_part = block_id.split("_", 1)[0]
            page_id = _page_id_to_int(page_part[1:] if page_part.startswith("p") else None)
            if page_id is not None:
                mapped_index = _normalize_coarse_block_index_from_layout(
                    page_id=page_id,
                    raw_block_id=block_id,
                    page_layouts=page_layouts,
                )
                if mapped_index is not None:
                    return mapped_index
        suffix = block_id.rsplit("_b", 1)[-1]
        if suffix.isdigit():
            block_index = int(suffix)
            if 1 <= block_index <= 9:
                return block_index - 1
    return None


def _extract_visible_target_block_indices(meta: dict[str, Any], *, num_images: int) -> tuple[dict[int, set[int]], dict[str, int]]:
    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list):
        return {}, {"num_visible_target_blocks": 0, "num_visible_target_pages_with_blocks": 0}

    input_page_ids = set(_resolve_sample_input_page_ids(meta, num_images=num_images))
    document_json_path = _resolve_document_json_path(meta)
    page_layouts = {} if document_json_path is None else _load_document_page_layouts(document_json_path)
    target_block_indices: dict[int, set[int]] = {}
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        page_id = _page_id_to_int(item.get("page_id"))
        if page_id is None or page_id not in input_page_ids:
            continue
        block_index = _coarse_block_id_to_index(item.get("block_id"), page_layouts=page_layouts)
        if block_index is None:
            continue
        target_block_indices.setdefault(page_id, set()).add(block_index)

    stats = {
        "num_visible_target_blocks": int(sum(len(x) for x in target_block_indices.values())),
        "num_visible_target_pages_with_blocks": int(len(target_block_indices)),
    }
    return target_block_indices, stats


def _extract_exact_target_block_ids(
    meta: dict[str, Any],
    *,
    num_images: int,
    visible_only: bool,
) -> tuple[dict[int, set[str]], dict[str, int]]:
    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list):
        return {}, {
            "num_target_exact_blocks": 0,
            "num_target_exact_pages": 0,
            "num_visible_target_exact_blocks": 0,
            "num_visible_target_exact_pages": 0,
            "has_document_layout": 0,
        }

    input_page_ids = set(_resolve_sample_input_page_ids(meta, num_images=num_images))
    document_json_path = _resolve_document_json_path(meta)
    has_document_layout = int(document_json_path is not None and Path(document_json_path).exists())
    target_block_ids_by_page: dict[int, set[str]] = {}
    visible_target_block_ids_by_page: dict[int, set[str]] = {}
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        page_id = _page_id_to_int(item.get("page_id"))
        block_id = item.get("block_id")
        if page_id is None or block_id is None:
            continue
        normalized_block_id = str(block_id).strip()
        if normalized_block_id == "":
            continue
        target_block_ids_by_page.setdefault(page_id, set()).add(normalized_block_id)
        if page_id in input_page_ids:
            visible_target_block_ids_by_page.setdefault(page_id, set()).add(normalized_block_id)

    selected = visible_target_block_ids_by_page if visible_only else target_block_ids_by_page
    stats = {
        "num_target_exact_blocks": int(sum(len(x) for x in target_block_ids_by_page.values())),
        "num_target_exact_pages": int(len(target_block_ids_by_page)),
        "num_visible_target_exact_blocks": int(sum(len(x) for x in visible_target_block_ids_by_page.values())),
        "num_visible_target_exact_pages": int(len(visible_target_block_ids_by_page)),
        "has_document_layout": has_document_layout,
    }
    return selected, stats


def _summarize_example_exact_target_coverage(examples: list[Any]) -> dict[str, int]:
    summary = {
        "num_examples": 0,
        "num_examples_with_document_layout": 0,
        "num_examples_with_target_pages": 0,
        "num_examples_with_target_exact_blocks": 0,
        "num_examples_with_visible_target_exact_blocks": 0,
        "num_target_pages": 0,
        "num_target_exact_blocks": 0,
        "num_visible_target_exact_blocks": 0,
    }
    for ex in examples:
        meta = ex.meta if hasattr(ex, "meta") and isinstance(ex.meta, dict) else {}
        num_images = len(getattr(ex, "images", []) or [])
        target_page_ids = _extract_target_page_ids(meta)
        _, exact_stats = _extract_exact_target_block_ids(meta, num_images=num_images, visible_only=False)
        _, visible_exact_stats = _extract_exact_target_block_ids(meta, num_images=num_images, visible_only=True)
        summary["num_examples"] += 1
        summary["num_examples_with_document_layout"] += int(exact_stats["has_document_layout"] > 0)
        summary["num_examples_with_target_pages"] += int(len(target_page_ids) > 0)
        summary["num_examples_with_target_exact_blocks"] += int(exact_stats["num_target_exact_blocks"] > 0)
        summary["num_examples_with_visible_target_exact_blocks"] += int(
            visible_exact_stats["num_visible_target_exact_blocks"] > 0
        )
        summary["num_target_pages"] += int(len(target_page_ids))
        summary["num_target_exact_blocks"] += int(exact_stats["num_target_exact_blocks"])
        summary["num_visible_target_exact_blocks"] += int(visible_exact_stats["num_visible_target_exact_blocks"])
    return summary


def _build_dma_page_targets(batch_meta: list[dict[str, Any]], sample_image_counts: list[int]) -> tuple[torch.Tensor, dict[str, int]]:
    labels: list[float] = []
    stats = {
        "num_images": 0,
        "num_visible_positive_pages": 0,
        "num_target_pages": 0,
    }
    for meta, num_images in zip(batch_meta, sample_image_counts):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        stats["num_images"] += len(input_page_ids)
        stats["num_target_pages"] += len(target_page_ids)
        for page_id in input_page_ids:
            is_positive = 1.0 if page_id in target_page_ids else 0.0
            labels.append(is_positive)
            if is_positive > 0.0:
                stats["num_visible_positive_pages"] += 1
    return torch.tensor(labels, dtype=torch.float32), stats


def _split_flat_page_logits(page_logits: torch.Tensor, sample_image_counts: list[int]) -> list[torch.Tensor] | None:
    if page_logits.ndim != 1:
        return None
    expected = sum(int(x) for x in sample_image_counts)
    if int(page_logits.numel()) != int(expected):
        return None
    split_logits: list[torch.Tensor] = []
    start = 0
    for count in sample_image_counts:
        end = start + int(count)
        split_logits.append(page_logits[start:end])
        start = end
    return split_logits


def _infer_sample_retained_page_counts(
    page_selected_ratios: torch.Tensor | None,
    sample_image_counts: list[int],
    budget_ratio_override: float | None = None,
) -> list[int] | None:
    if budget_ratio_override is not None and float(budget_ratio_override) > 0.0:
        retained_counts: list[int] = []
        for count in sample_image_counts:
            num_images = int(count)
            if num_images <= 0:
                retained_counts.append(0)
                continue
            retained_count = max(1, min(num_images, int(round(num_images * float(budget_ratio_override)))))
            retained_counts.append(int(retained_count))
        return retained_counts
    if page_selected_ratios is None:
        return None
    expected = sum(int(x) for x in sample_image_counts)
    if int(page_selected_ratios.numel()) != int(expected):
        return None

    retained_counts: list[int] = []
    start = 0
    for count in sample_image_counts:
        num_images = int(count)
        end = start + num_images
        sample_ratios = page_selected_ratios[start:end]
        start = end
        if num_images <= 0:
            retained_counts.append(0)
            continue
        budget_ratio = float(sample_ratios.float().mean().item()) if int(sample_ratios.numel()) > 0 else 0.0
        retained_count = max(1, min(num_images, int(round(num_images * budget_ratio))))
        retained_counts.append(int(retained_count))
    return retained_counts


def _select_budget_boundary_negative_local_indices(
    *,
    negative_logits: torch.Tensor,
    retained_count: int,
    positive_count: int,
    negative_topk: int,
) -> torch.Tensor | None:
    num_negative = int(negative_logits.numel())
    if num_negative <= 0:
        return None
    positive_slots = min(max(0, int(positive_count)), max(0, int(retained_count)))
    allowed_negative_slots = max(0, int(retained_count) - positive_slots)
    if num_negative <= allowed_negative_slots:
        return None
    sorted_negative_indices = torch.argsort(negative_logits, descending=True)
    boundary_negative_indices = sorted_negative_indices[allowed_negative_slots:]
    if int(boundary_negative_indices.numel()) <= 0:
        return None

    boundary_topk = int(negative_topk)
    if boundary_topk <= 0:
        boundary_topk = max(1, min(int(retained_count), int(boundary_negative_indices.numel())))
    boundary_topk = max(1, min(boundary_topk, int(boundary_negative_indices.numel())))
    return boundary_negative_indices[:boundary_topk]


def _compute_dma_page_ranking_loss(
    *,
    page_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    margin: float,
    ranking_topk: int,
    sample_retained_page_counts: list[int] | None = None,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    stats = {
        "ranking_samples": 0,
        "ranking_pairs": 0,
    }
    if split_logits is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_logits, meta, num_images) in enumerate(zip(split_logits, batch_meta, sample_image_counts)):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        if not positive_indices:
            continue
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not negative_indices:
            continue

        pos_logits = sample_logits[positive_indices]
        neg_logits = sample_logits[negative_indices]
        retained_page_count = None
        if sample_retained_page_counts is not None and sample_idx < len(sample_retained_page_counts):
            retained_page_count = int(sample_retained_page_counts[sample_idx])
        if retained_page_count is not None and retained_page_count > 0:
            allowed_negative_slots = max(0, retained_page_count - len(positive_indices))
            if int(neg_logits.numel()) <= allowed_negative_slots:
                continue
            threshold_rank = min(int(neg_logits.numel()), allowed_negative_slots + 1)
            threshold_neg_logit = torch.topk(neg_logits, k=threshold_rank).values[-1]
            pairwise_margin = torch.relu(float(margin) - pos_logits + threshold_neg_logit)
            sample_losses.append(pairwise_margin.mean())
            stats["ranking_samples"] += 1
            stats["ranking_pairs"] += int(pos_logits.numel())
            continue
        topk = max(1, min(int(ranking_topk), int(neg_logits.numel())))
        topk_neg_logits = torch.topk(neg_logits, k=topk).values
        pairwise_margin = torch.relu(float(margin) - pos_logits[:, None] + topk_neg_logits[None, :])
        sample_losses.append(pairwise_margin.mean())
        stats["ranking_samples"] += 1
        stats["ranking_pairs"] += int(pos_logits.numel() * topk_neg_logits.numel())

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_topk_margin_loss(
    *,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    margin: float,
    topk: int,
) -> tuple[torch.Tensor | None, int]:
    if int(positive_logits.numel()) == 0 or int(negative_logits.numel()) == 0:
        return None, 0
    ranking_topk = max(1, min(int(topk), int(negative_logits.numel())))
    topk_neg_logits = torch.topk(negative_logits, k=ranking_topk).values
    pairwise_margin = torch.relu(float(margin) - positive_logits[:, None] + topk_neg_logits[None, :])
    return pairwise_margin.mean(), int(positive_logits.numel() * topk_neg_logits.numel())


def _compute_budget_margin_loss(
    *,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    margin: float,
    budget_topk: int,
) -> tuple[torch.Tensor | None, int]:
    if int(positive_logits.numel()) == 0 or int(negative_logits.numel()) == 0:
        return None, 0
    allowed_negative_slots = max(0, int(budget_topk) - int(positive_logits.numel()))
    if int(negative_logits.numel()) <= allowed_negative_slots:
        return None, 0
    threshold_rank = min(int(negative_logits.numel()), allowed_negative_slots + 1)
    threshold_neg_logit = torch.topk(negative_logits, k=threshold_rank).values[-1]
    pairwise_margin = torch.relu(float(margin) - positive_logits + threshold_neg_logit)
    return pairwise_margin.mean(), int(positive_logits.numel())


def _compute_soft_topk_membership(
    *,
    logits: torch.Tensor,
    retained_count: int,
    temperature: float,
) -> torch.Tensor | None:
    if int(logits.numel()) <= 0:
        return None
    clipped_retained_count = max(0, min(int(retained_count), int(logits.numel())))
    if clipped_retained_count <= 0:
        return torch.zeros_like(logits)
    if clipped_retained_count >= int(logits.numel()):
        return torch.ones_like(logits)
    boundary_values = torch.topk(logits.detach(), k=clipped_retained_count + 1).values
    threshold = 0.5 * (boundary_values[clipped_retained_count - 1] + boundary_values[clipped_retained_count])
    safe_temperature = max(float(temperature), 1.0e-6)
    return torch.sigmoid((logits - threshold) / safe_temperature)


def _derive_page_focus_scores(block_logits: torch.Tensor | None) -> torch.Tensor | None:
    if block_logits is None or block_logits.ndim != 2 or int(block_logits.shape[0]) == 0:
        return None
    return block_logits.max(dim=1).values


def _compute_dma_page_readout_ranking_loss(
    *,
    page_logits: torch.Tensor,
    page_focus_scores: torch.Tensor | None,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    margin: float,
    ranking_topk: int,
    sample_retained_page_counts: list[int] | None = None,
    focus_loss_weight: float = 0.25,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    split_page_focus = None
    if page_focus_scores is not None:
        split_page_focus = _split_flat_page_logits(page_focus_scores, sample_image_counts)
    stats = {
        "ranking_samples": 0,
        "ranking_pairs": 0,
        "readout_focus_samples": 0,
        "readout_focus_pairs": 0,
    }
    if split_logits is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_logits, meta, num_images) in enumerate(zip(split_logits, batch_meta, sample_image_counts)):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        if not positive_indices:
            continue
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not negative_indices:
            continue

        pos_logits = sample_logits[positive_indices]
        neg_logits = sample_logits[negative_indices]
        retained_page_count = None
        if sample_retained_page_counts is not None and sample_idx < len(sample_retained_page_counts):
            retained_page_count = int(sample_retained_page_counts[sample_idx])

        hard_negative_indices: torch.Tensor
        page_loss: torch.Tensor | None
        ranking_pairs: int
        if retained_page_count is not None and retained_page_count > 0:
            allowed_negative_slots = max(0, retained_page_count - len(positive_indices))
            if int(neg_logits.numel()) <= allowed_negative_slots:
                continue
            threshold_rank = min(int(neg_logits.numel()), allowed_negative_slots + 1)
            topk_result = torch.topk(neg_logits, k=threshold_rank)
            hard_negative_indices = topk_result.indices
            threshold_neg_logit = topk_result.values[-1]
            page_loss = torch.relu(float(margin) - pos_logits + threshold_neg_logit).mean()
            ranking_pairs = int(pos_logits.numel())
        else:
            topk = max(1, min(int(ranking_topk), int(neg_logits.numel())))
            topk_result = torch.topk(neg_logits, k=topk)
            hard_negative_indices = topk_result.indices
            page_loss = torch.relu(float(margin) - pos_logits[:, None] + topk_result.values[None, :]).mean()
            ranking_pairs = int(pos_logits.numel() * topk_result.values.numel())

        sample_loss = page_loss
        stats["ranking_samples"] += 1
        stats["ranking_pairs"] += int(ranking_pairs)

        if (
            split_page_focus is not None
            and float(focus_loss_weight) > 0.0
            and sample_idx < len(split_page_focus)
            and split_page_focus[sample_idx].shape == sample_logits.shape
        ):
            sample_focus = split_page_focus[sample_idx]
            pos_focus = sample_focus[positive_indices]
            neg_focus = sample_focus[negative_indices]
            if int(pos_focus.numel()) > 0 and int(neg_focus.numel()) > 0:
                hard_negative_focus = neg_focus[hard_negative_indices]
                focus_loss = torch.relu(
                    float(margin) - pos_focus[:, None] + hard_negative_focus[None, :]
                ).mean()
                sample_loss = sample_loss + float(focus_loss_weight) * focus_loss
                stats["readout_focus_samples"] += 1
                stats["readout_focus_pairs"] += int(pos_focus.numel() * hard_negative_focus.numel())

        sample_losses.append(sample_loss)

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_dma_single_page_focus_loss(
    *,
    page_focus_scores: torch.Tensor | None,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    margin: float,
    sample_retained_page_counts: list[int] | None = None,
    ranking_topk: int = 1,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    stats = {
        "single_positive_samples": 0,
        "single_positive_pairs": 0,
    }
    split_page_focus = _split_flat_page_logits(page_focus_scores, sample_image_counts)
    if split_page_focus is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_focus, meta, num_images) in enumerate(zip(split_page_focus, batch_meta, sample_image_counts)):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        if len(positive_indices) != 1:
            continue
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not negative_indices:
            continue

        pos_focus = sample_focus[positive_indices]
        neg_focus = sample_focus[negative_indices]
        retained_page_count = None
        if sample_retained_page_counts is not None and sample_idx < len(sample_retained_page_counts):
            retained_page_count = int(sample_retained_page_counts[sample_idx])

        if retained_page_count is not None and retained_page_count > 0:
            allowed_negative_slots = max(0, retained_page_count - len(positive_indices))
            if int(neg_focus.numel()) <= allowed_negative_slots:
                continue
            threshold_rank = min(int(neg_focus.numel()), allowed_negative_slots + 1)
            threshold_neg_focus = torch.topk(neg_focus, k=threshold_rank).values[-1]
            focus_loss = torch.relu(float(margin) - pos_focus + threshold_neg_focus).mean()
            pair_count = int(pos_focus.numel())
        else:
            topk = max(1, min(int(ranking_topk), int(neg_focus.numel())))
            topk_values = torch.topk(neg_focus, k=topk).values
            focus_loss = torch.relu(float(margin) - pos_focus[:, None] + topk_values[None, :]).mean()
            pair_count = int(pos_focus.numel() * topk_values.numel())

        sample_losses.append(focus_loss)
        stats["single_positive_samples"] += 1
        stats["single_positive_pairs"] += int(pair_count)

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_dma_multi_page_coverage_loss(
    *,
    page_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    margin: float,
    sample_retained_page_counts: list[int] | None = None,
    ranking_topk: int = 1,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    stats = {
        "multi_positive_samples": 0,
        "multi_positive_pages": 0,
        "multi_page_pairs": 0,
    }
    if split_logits is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_logits, meta, num_images) in enumerate(zip(split_logits, batch_meta, sample_image_counts)):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        if len(positive_indices) < 2:
            continue
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not negative_indices:
            continue

        pos_logits = sample_logits[positive_indices]
        neg_logits = sample_logits[negative_indices]
        weakest_positive_logit = pos_logits.min()

        retained_page_count = None
        if sample_retained_page_counts is not None and sample_idx < len(sample_retained_page_counts):
            retained_page_count = int(sample_retained_page_counts[sample_idx])
        if retained_page_count is not None and retained_page_count > 0:
            allowed_negative_slots = max(0, retained_page_count - len(positive_indices))
            if int(neg_logits.numel()) <= allowed_negative_slots:
                continue
            threshold_rank = min(int(neg_logits.numel()), allowed_negative_slots + 1)
            threshold_neg_logit = torch.topk(neg_logits, k=threshold_rank).values[-1]
        else:
            topk = max(1, min(int(ranking_topk), int(neg_logits.numel())))
            threshold_neg_logit = torch.topk(neg_logits, k=topk).values[-1]

        coverage_loss = torch.relu(float(margin) - weakest_positive_logit + threshold_neg_logit)
        sample_losses.append(coverage_loss)
        stats["multi_positive_samples"] += 1
        stats["multi_positive_pages"] += int(len(positive_indices))
        stats["multi_page_pairs"] += 1

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_dma_page_retained_set_loss(
    *,
    page_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    sample_retained_page_counts: list[int] | None,
    temperature: float,
    negative_weight: float,
    negative_topk: int,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    stats = {
        "retained_set_samples": 0,
        "retained_set_positive_pages": 0,
    }
    if split_logits is None or sample_retained_page_counts is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_logits, meta, num_images) in enumerate(zip(split_logits, batch_meta, sample_image_counts)):
        if sample_idx >= len(sample_retained_page_counts):
            break
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not positive_indices or not negative_indices:
            continue

        retained_page_count = int(sample_retained_page_counts[sample_idx])
        if retained_page_count <= 0 or retained_page_count >= len(input_page_ids):
            continue
        soft_membership = _compute_soft_topk_membership(
            logits=sample_logits,
            retained_count=retained_page_count,
            temperature=float(temperature),
        )
        if soft_membership is None:
            continue

        positive_target = min(len(positive_indices), retained_page_count) / float(len(positive_indices))
        positive_probs = soft_membership[positive_indices]
        positive_loss = torch.nn.functional.binary_cross_entropy(
            positive_probs,
            torch.full_like(positive_probs, fill_value=float(positive_target)),
        )

        negative_loss = torch.tensor(0.0, device=sample_logits.device, dtype=sample_logits.dtype)
        boundary_negative_local = _select_budget_boundary_negative_local_indices(
            negative_logits=sample_logits[negative_indices],
            retained_count=retained_page_count,
            positive_count=len(positive_indices),
            negative_topk=int(negative_topk),
        )
        if boundary_negative_local is not None and int(boundary_negative_local.numel()) > 0:
            hard_negative_indices = [negative_indices[int(idx)] for idx in boundary_negative_local.tolist()]
            negative_probs = soft_membership[hard_negative_indices]
            negative_loss = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))

        sample_losses.append(positive_loss + float(negative_weight) * negative_loss)
        stats["retained_set_samples"] += 1
        stats["retained_set_positive_pages"] += int(len(positive_indices))

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_dma_single_page_retained_recovery_loss(
    *,
    page_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    sample_retained_page_counts: list[int] | None,
    margin: float,
    negative_topk: int,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    stats = {
        "single_page_retained_samples": 0,
        "single_page_retained_pairs": 0,
        "single_page_positive_in_topk": 0,
    }
    if split_logits is None or sample_retained_page_counts is None:
        return None, stats

    sample_losses: list[torch.Tensor] = []
    for sample_idx, (sample_logits, meta, num_images) in enumerate(zip(split_logits, batch_meta, sample_image_counts)):
        if sample_idx >= len(sample_retained_page_counts):
            break
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        positive_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_page_ids]
        if len(positive_indices) != 1:
            continue
        negative_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]
        if not negative_indices:
            continue

        retained_page_count = int(sample_retained_page_counts[sample_idx])
        if retained_page_count <= 0 or retained_page_count >= len(input_page_ids):
            continue

        topk_indices = torch.topk(sample_logits, k=retained_page_count).indices.tolist()
        stats["single_page_retained_samples"] += 1
        stats["single_page_positive_in_topk"] += int(int(positive_indices[0]) in {int(idx) for idx in topk_indices})

        boundary_negative_local = _select_budget_boundary_negative_local_indices(
            negative_logits=sample_logits[negative_indices],
            retained_count=retained_page_count,
            positive_count=1,
            negative_topk=int(negative_topk),
        )
        if boundary_negative_local is None or int(boundary_negative_local.numel()) <= 0:
            continue
        boundary_negative_indices = [negative_indices[int(idx)] for idx in boundary_negative_local.tolist()]
        pos_logits = sample_logits[positive_indices]
        neg_logits = sample_logits[boundary_negative_indices]
        sample_losses.append(torch.relu(float(margin) - pos_logits[:, None] + neg_logits[None, :]).mean())
        stats["single_page_retained_pairs"] += int(pos_logits.numel() * neg_logits.numel())

    if not sample_losses:
        return None, stats
    return torch.stack(sample_losses).mean(), stats


def _compute_dma_block_readout_ranking_loss(
    *,
    page_logits: torch.Tensor,
    block_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    loss_type: str,
    margin: float,
    topk: int,
    retained_block_topk: int,
    sample_retained_page_counts: list[int] | None = None,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_page_logits = _split_flat_page_logits(page_logits, sample_image_counts)
    split_block_logits = _split_flat_block_logits(block_logits, sample_image_counts)
    stats = {
        "block_supervision_pages": 0,
        "block_ranking_pages": 0,
        "block_ranking_pairs": 0,
        "block_focus_pages": 0,
        "block_readout_pages": 0,
        "block_readout_pairs": 0,
        "visible_positive_blocks": 0,
    }
    if split_page_logits is None or split_block_logits is None:
        return None, stats

    page_losses: list[torch.Tensor] = []
    normalized_loss_type = str(loss_type).strip().lower()
    use_budget_ranking = normalized_loss_type == "readout_budget_ranking"
    for sample_idx, (sample_page_logits, sample_block_logits, meta, num_images) in enumerate(
        zip(split_page_logits, split_block_logits, batch_meta, sample_image_counts)
    ):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        target_block_indices_by_page, block_stats = _extract_visible_target_block_indices(
            sample_meta,
            num_images=int(num_images),
        )
        stats["visible_positive_blocks"] += int(block_stats["num_visible_target_blocks"])

        positive_page_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id in target_block_indices_by_page]
        negative_page_indices = [idx for idx, page_id in enumerate(input_page_ids) if page_id not in target_page_ids]

        hard_negative_page_indices: list[int] = []
        if negative_page_indices:
            negative_page_logits = sample_page_logits[negative_page_indices]
            retained_page_count = None
            if sample_retained_page_counts is not None and sample_idx < len(sample_retained_page_counts):
                retained_page_count = int(sample_retained_page_counts[sample_idx])
            if retained_page_count is not None and retained_page_count > 0:
                allowed_negative_slots = max(0, retained_page_count - len(positive_page_indices))
                if int(negative_page_logits.numel()) > allowed_negative_slots:
                    threshold_rank = min(int(negative_page_logits.numel()), allowed_negative_slots + 1)
                    hard_negative_local = torch.topk(negative_page_logits, k=threshold_rank).indices.tolist()
                    hard_negative_page_indices = [negative_page_indices[int(idx)] for idx in hard_negative_local]
            else:
                negative_topk = max(1, min(int(topk), int(negative_page_logits.numel())))
                hard_negative_local = torch.topk(negative_page_logits, k=negative_topk).indices.tolist()
                hard_negative_page_indices = [negative_page_indices[int(idx)] for idx in hard_negative_local]

        hard_negative_block_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
        if hard_negative_page_indices:
            hard_negative_page_logits = sample_page_logits[hard_negative_page_indices]
            hard_negative_page_weights = torch.softmax(hard_negative_page_logits, dim=0)
            for page_offset, negative_page_idx in enumerate(hard_negative_page_indices):
                negative_block_logits = sample_block_logits[negative_page_idx]
                if int(negative_block_logits.numel()) <= 0:
                    continue
                negative_block_topk = max(1, min(int(retained_block_topk), int(negative_block_logits.numel())))
                top_negative_blocks = torch.topk(negative_block_logits, k=negative_block_topk).values
                page_weight = hard_negative_page_weights[page_offset].to(dtype=top_negative_blocks.dtype)
                hard_negative_block_groups.append((top_negative_blocks, page_weight))

        for page_idx, page_id in enumerate(input_page_ids):
            if page_id not in target_page_ids:
                continue
            page_block_logits = sample_block_logits[page_idx]
            target_block_indices = sorted(target_block_indices_by_page.get(page_id, set()))
            if target_block_indices:
                positive_logits = page_block_logits[target_block_indices]
                negative_mask = torch.ones_like(page_block_logits, dtype=torch.bool)
                negative_mask[target_block_indices] = False
                negative_logits = page_block_logits[negative_mask]

                page_loss_terms: list[torch.Tensor] = []
                if use_budget_ranking:
                    local_loss, local_pairs = _compute_budget_margin_loss(
                        positive_logits=positive_logits,
                        negative_logits=negative_logits,
                        margin=float(margin),
                        budget_topk=int(retained_block_topk),
                    )
                else:
                    local_loss, local_pairs = _compute_topk_margin_loss(
                        positive_logits=positive_logits,
                        negative_logits=negative_logits,
                        margin=float(margin),
                        topk=int(topk),
                    )
                if local_loss is not None:
                    page_loss_terms.append(local_loss)
                    stats["block_ranking_pages"] += 1
                    stats["block_ranking_pairs"] += int(local_pairs)

                if hard_negative_block_groups:
                    readout_losses: list[torch.Tensor] = []
                    readout_pair_count = 0
                    for negative_block_group, page_weight in hard_negative_block_groups:
                        if int(negative_block_group.numel()) <= 0:
                            continue
                        per_group_loss = torch.relu(
                            float(margin) - positive_logits[:, None] + negative_block_group[None, :]
                        ).mean()
                        readout_losses.append(page_weight * per_group_loss)
                        readout_pair_count += int(positive_logits.numel() * negative_block_group.numel())
                    if readout_losses:
                        page_loss_terms.append(torch.stack(readout_losses).sum())
                        stats["block_readout_pages"] += 1
                        stats["block_readout_pairs"] += int(readout_pair_count)

                if page_loss_terms:
                    page_losses.append(torch.stack(page_loss_terms).sum())
                    continue

            focus_topk = max(1, min(int(topk), int(page_block_logits.numel())))
            topk_values, topk_indices = torch.topk(page_block_logits, k=focus_topk)
            focus_mean = topk_values.mean()
            if int(page_block_logits.numel()) > focus_topk:
                background_mask = torch.ones_like(page_block_logits, dtype=torch.bool)
                background_mask[topk_indices] = False
                background_mean = page_block_logits[background_mask].mean()
            else:
                background_mean = page_block_logits.mean()
            page_losses.append(torch.relu(float(margin) - focus_mean + background_mean))
            stats["block_focus_pages"] += 1

    if not page_losses:
        return None, stats
    return torch.stack(page_losses).mean(), stats


def _compute_dma_block_retained_set_loss(
    *,
    block_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    retained_block_topk: int,
    temperature: float,
    negative_weight: float,
    negative_topk: int,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_block_logits = _split_flat_block_logits(block_logits, sample_image_counts)
    stats = {
        "visible_positive_blocks": 0,
        "retained_block_set_pages": 0,
        "retained_block_set_blocks": 0,
    }
    if split_block_logits is None:
        return None, stats

    page_losses: list[torch.Tensor] = []
    for sample_block_logits, meta, num_images in zip(split_block_logits, batch_meta, sample_image_counts):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_block_indices_by_page, block_stats = _extract_visible_target_block_indices(
            sample_meta,
            num_images=int(num_images),
        )
        stats["visible_positive_blocks"] += int(block_stats["num_visible_target_blocks"])

        for page_idx, page_id in enumerate(input_page_ids):
            target_block_indices = sorted(target_block_indices_by_page.get(page_id, set()))
            if not target_block_indices:
                continue
            page_block_logits = sample_block_logits[page_idx]
            if int(page_block_logits.numel()) <= int(retained_block_topk):
                continue
            soft_membership = _compute_soft_topk_membership(
                logits=page_block_logits,
                retained_count=int(retained_block_topk),
                temperature=float(temperature),
            )
            if soft_membership is None:
                continue

            positive_target = min(len(target_block_indices), int(retained_block_topk)) / float(len(target_block_indices))
            positive_probs = soft_membership[target_block_indices]
            positive_loss = torch.nn.functional.binary_cross_entropy(
                positive_probs,
                torch.full_like(positive_probs, fill_value=float(positive_target)),
            )

            negative_mask = torch.ones_like(page_block_logits, dtype=torch.bool)
            negative_mask[target_block_indices] = False
            negative_logits = page_block_logits[negative_mask]
            hard_negative_topk = int(negative_topk)
            if hard_negative_topk <= 0:
                hard_negative_topk = max(1, int(retained_block_topk))
            hard_negative_topk = min(hard_negative_topk, int(negative_logits.numel()))
            negative_probs = torch.empty(0, device=page_block_logits.device, dtype=page_block_logits.dtype)
            if hard_negative_topk > 0:
                hard_negative_local = torch.topk(negative_logits, k=hard_negative_topk).indices
                negative_probs = soft_membership[negative_mask][hard_negative_local]
            negative_loss = torch.tensor(0.0, device=page_block_logits.device, dtype=page_block_logits.dtype)
            if int(negative_probs.numel()) > 0:
                negative_loss = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))

            page_losses.append(positive_loss + float(negative_weight) * negative_loss)
            stats["retained_block_set_pages"] += 1
            stats["retained_block_set_blocks"] += int(len(target_block_indices))

    if not page_losses:
        return None, stats
    return torch.stack(page_losses).mean(), stats


def _split_flat_block_logits(block_logits: torch.Tensor, sample_image_counts: list[int]) -> list[torch.Tensor] | None:
    if block_logits.ndim != 2:
        return None
    expected = sum(int(x) for x in sample_image_counts)
    if int(block_logits.shape[0]) != int(expected):
        return None
    split_logits: list[torch.Tensor] = []
    start = 0
    for count in sample_image_counts:
        end = start + int(count)
        split_logits.append(block_logits[start:end])
        start = end
    return split_logits


def _compute_dma_block_focus_loss(
    *,
    block_logits: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    loss_type: str,
    margin: float,
    topk: int,
    retained_block_topk: int,
) -> tuple[torch.Tensor | None, dict[str, int]]:
    split_logits = _split_flat_block_logits(block_logits, sample_image_counts)
    stats = {
        "block_supervision_pages": 0,
        "block_ranking_pages": 0,
        "block_ranking_pairs": 0,
        "block_focus_pages": 0,
        "visible_positive_blocks": 0,
    }
    if split_logits is None:
        return None, stats

    page_losses: list[torch.Tensor] = []
    for sample_block_logits, meta, num_images in zip(split_logits, batch_meta, sample_image_counts):
        sample_meta = meta if isinstance(meta, dict) else {}
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = _extract_target_page_ids(sample_meta)
        target_block_indices_by_page, block_stats = _extract_visible_target_block_indices(
            sample_meta,
            num_images=int(num_images),
        )
        stats["visible_positive_blocks"] += int(block_stats["num_visible_target_blocks"])
        for page_idx, page_id in enumerate(input_page_ids):
            if page_id not in target_page_ids:
                continue
            page_block_logits = sample_block_logits[page_idx]
            target_block_indices = sorted(target_block_indices_by_page.get(page_id, set()))
            if target_block_indices:
                normalized_loss_type = str(loss_type).strip().lower()
                if normalized_loss_type in {"ranking", "budget_ranking"}:
                    positive_logits = page_block_logits[target_block_indices]
                    negative_mask = torch.ones_like(page_block_logits, dtype=torch.bool)
                    negative_mask[target_block_indices] = False
                    negative_logits = page_block_logits[negative_mask]
                    if normalized_loss_type == "budget_ranking":
                        ranking_loss, ranking_pairs = _compute_budget_margin_loss(
                            positive_logits=positive_logits,
                            negative_logits=negative_logits,
                            margin=float(margin),
                            budget_topk=int(retained_block_topk),
                        )
                    else:
                        ranking_loss, ranking_pairs = _compute_topk_margin_loss(
                            positive_logits=positive_logits,
                            negative_logits=negative_logits,
                            margin=float(margin),
                            topk=int(topk),
                        )
                    if ranking_loss is not None:
                        page_losses.append(ranking_loss)
                        stats["block_ranking_pages"] += 1
                        stats["block_ranking_pairs"] += int(ranking_pairs)
                    continue
                target = torch.zeros_like(page_block_logits)
                target[target_block_indices] = 1.0
                negative_count = max(1, int(target.numel()) - len(target_block_indices))
                pos_weight = torch.tensor(
                    [negative_count / max(1, len(target_block_indices))],
                    device=page_block_logits.device,
                    dtype=page_block_logits.dtype,
                )
                page_losses.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        page_block_logits,
                        target,
                        pos_weight=pos_weight,
                    )
                )
                stats["block_supervision_pages"] += 1
                continue

            focus_topk = max(1, min(int(topk), int(page_block_logits.numel())))
            topk_values, topk_indices = torch.topk(page_block_logits, k=focus_topk)
            focus_mean = topk_values.mean()
            if int(page_block_logits.numel()) > focus_topk:
                background_mask = torch.ones_like(page_block_logits, dtype=torch.bool)
                background_mask[topk_indices] = False
                background_mean = page_block_logits[background_mask].mean()
            else:
                background_mean = page_block_logits.mean()
            page_losses.append(torch.relu(float(margin) - focus_mean + background_mean))
            stats["block_focus_pages"] += 1

    if not page_losses:
        return None, stats
    return torch.stack(page_losses).mean(), stats


def _build_sample_exact_block_page_stats(
    *,
    input_page_ids: list[int],
    sample_page_logits: torch.Tensor | None,
    sample_block_logits: torch.Tensor | None,
    sample_selected_ratios: torch.Tensor | None,
) -> dict[int, dict[str, Any]]:
    page_stats: list[dict[str, Any]] = []
    for page_offset, page_id in enumerate(input_page_ids):
        page_score = 0.0
        if sample_page_logits is not None and 0 <= page_offset < int(sample_page_logits.numel()):
            page_score = float(sample_page_logits[page_offset].detach().item())
        selected_ratio = 0.0
        if sample_selected_ratios is not None and 0 <= page_offset < int(sample_selected_ratios.numel()):
            selected_ratio = float(sample_selected_ratios[page_offset].detach().item())
        coarse_block_scores: list[float] = []
        page_focus = 0.0
        if (
            sample_block_logits is not None
            and int(sample_block_logits.ndim) == 2
            and 0 <= page_offset < int(sample_block_logits.shape[0])
        ):
            coarse_block_scores = [float(x) for x in sample_block_logits[page_offset].detach().cpu().tolist()]
            if coarse_block_scores:
                page_focus = max(coarse_block_scores)
        top_block_ids = [
            f"p{int(page_id)}_g{int(block_index) + 1}"
            for block_index, _ in sorted(
                enumerate(coarse_block_scores),
                key=lambda item: (-float(item[1]), int(item[0])),
            )
        ]
        page_stats.append(
            {
                "page_id": int(page_id),
                "page_score": float(page_score),
                "page_focus": float(page_focus),
                "selected_ratio": float(selected_ratio),
                "top_block_ids": top_block_ids,
                "coarse_block_scores": coarse_block_scores,
            }
        )
    return build_exact_block_page_stats_by_id(page_stats)


def _normalize_exact_block_dataset_tag(sample_meta: dict[str, Any]) -> str:
    benchmark = str(sample_meta.get("benchmark") or sample_meta.get("family") or "").strip().lower()
    if benchmark in {"v-niah", "vniah"}:
        return "vniah"
    if benchmark in {"v-mqar", "vmqar"}:
        return "vmqar"
    return "other"


def _resolve_exact_block_source_page_ids(
    *,
    input_page_ids: list[int],
    target_page_ids: list[int],
    sample_page_logits: torch.Tensor | None,
    sample_retained_page_count: int | None,
) -> list[int]:
    ordered_input_page_ids = [int(page_id) for page_id in input_page_ids]
    if not ordered_input_page_ids:
        return []
    if sample_retained_page_count is None:
        return ordered_input_page_ids
    retained_page_count = int(sample_retained_page_count)
    if retained_page_count <= 0 or retained_page_count >= len(ordered_input_page_ids):
        return ordered_input_page_ids
    if sample_page_logits is None or int(sample_page_logits.numel()) < len(ordered_input_page_ids):
        return ordered_input_page_ids

    top_page_indices = torch.topk(sample_page_logits[: len(ordered_input_page_ids)], k=retained_page_count).indices.tolist()
    retained_page_set = {ordered_input_page_ids[int(page_idx)] for page_idx in top_page_indices}
    target_page_set = {int(page_id) for page_id in target_page_ids}
    candidate_page_set = retained_page_set | target_page_set
    ordered_source_page_ids = [page_id for page_id in ordered_input_page_ids if page_id in candidate_page_set]
    return ordered_source_page_ids or ordered_input_page_ids


def _maybe_dropout_exact_block_lexical_features(
    feature_tensor: torch.Tensor,
    *,
    dropout_prob: float,
) -> torch.Tensor:
    """
    训练时对 exact block 特征中的 lexical 项做 dropout。

    目的：避免线性头被 lexical 特征“锁死”，逼迫它在 bbox/order/page_score 等非纯词面特征上学到可泛化的排序。
    注意：按 sample 级别做一次性 dropout（整页候选一致），避免候选间出现“输入不一致”导致排序不稳定。
    """

    if int(feature_tensor.numel()) == 0:
        return feature_tensor
    prob = float(dropout_prob)
    if prob <= 0.0 or prob >= 1.0:
        return feature_tensor
    # 仅在训练态启用；eval/推理保持确定性。
    if not bool(torch.is_grad_enabled()):
        return feature_tensor
    # sample 级别一次性开关
    if float(torch.rand((), device=feature_tensor.device).item()) >= prob:
        return feature_tensor

    idx_log = int(EXACT_BLOCK_FEATURE_INDEX["lexical_score_log"])
    idx_ratio = int(EXACT_BLOCK_FEATURE_INDEX["lexical_overlap_ratio"])
    dropped = feature_tensor.clone()
    dropped[:, idx_log] = 0.0
    dropped[:, idx_ratio] = 0.0
    return dropped


def _compute_exact_block_infonce_loss(
    *,
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    """
    多正样本 InfoNCE：-log( sum(exp(pos/t)) / sum(exp(all/t)) )
    """

    if int(positive_indices.numel()) == 0:
        return None
    temp = max(1e-6, float(temperature))
    scaled = logits / temp
    if int(negative_indices.numel()) == 0:
        # 全是正样本时没有对比信号，返回 0。
        return scaled.new_zeros(())
    all_indices = torch.cat([positive_indices, negative_indices], dim=0)
    pos_lse = torch.logsumexp(scaled[positive_indices], dim=0)
    all_lse = torch.logsumexp(scaled[all_indices], dim=0)
    return -(pos_lse - all_lse)


def _compute_exact_block_margin_ranking_loss(
    *,
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    margin: float,
    hard_negative_topk: int,
) -> torch.Tensor | None:
    if int(positive_indices.numel()) == 0 or int(negative_indices.numel()) == 0:
        return None
    k = int(hard_negative_topk)
    if k <= 0:
        k = 1
    k = min(k, int(negative_indices.numel()))
    neg_scores = logits[negative_indices]
    hard_negs = torch.topk(neg_scores, k=k).values  # [k]
    pos_scores = logits[positive_indices]  # [p]
    # softplus(m - (pos - neg))：希望 pos 至少比 hard neg 大 margin
    diffs = float(margin) - pos_scores[:, None] + hard_negs[None, :]
    return torch.nn.functional.softplus(diffs).mean()


def _compute_exact_block_distill_loss(
    *,
    logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    if int(teacher_indices.numel()) < 2 or int(teacher_scores.numel()) != int(teacher_indices.numel()):
        return None
    temp = max(1e-6, float(temperature))
    student_logits = logits[teacher_indices] / temp
    teacher_logits = teacher_scores.to(device=logits.device, dtype=logits.dtype) / temp
    teacher_probs = torch.softmax(teacher_logits, dim=0)
    student_log_probs = torch.log_softmax(student_logits, dim=0)
    return torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def _uses_vlm_cross_encoder_exact_block_scorer(scorer_type: str) -> bool:
    normalized = str(scorer_type or "").strip().lower()
    return normalized in {"vlm_cross_encoder", "vlm_rerank"}


def _select_exact_block_vlm_candidates(
    *,
    candidates: list[dict[str, Any]],
    positive_block_ids: set[str],
    lexical_topm: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    topm = max(1, int(lexical_topm))
    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda idx: (
            -float(candidates[idx].get("lexical_score", 0.0)),
            int(candidates[idx].get("page_rank", 0)),
            int(candidates[idx].get("order_index", 0)),
        ),
    )[:topm]
    kept_indices = set(ranked_indices)
    for idx, candidate in enumerate(candidates):
        if str(candidate.get("block_id") or "") in positive_block_ids:
            kept_indices.add(idx)
    ordered_indices = sorted(
        kept_indices,
        key=lambda idx: (
            -float(candidates[idx].get("lexical_score", 0.0)),
            int(candidates[idx].get("page_rank", 0)),
            int(candidates[idx].get("order_index", 0)),
        ),
    )
    return [candidates[idx] for idx in ordered_indices]


def _score_exact_block_candidates_with_vlm_cross_encoder(
    *,
    model,
    processor: Any,
    sample_images: list[Image.Image],
    input_page_ids: list[int],
    candidates: list[dict[str, Any]],
    user_text: str,
    crop_expand_ratio: float,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    if processor is None or not candidates or not sample_images or not input_page_ids:
        return None, []
    image_by_page_id = {
        int(page_id): image.convert("RGB") if isinstance(image, Image.Image) else image
        for page_id, image in zip(input_page_ids, sample_images)
    }
    valid_candidates: list[dict[str, Any]] = []
    candidate_logits: list[torch.Tensor] = []
    for candidate in candidates:
        page_id = _page_id_to_int(candidate.get("page_id"))
        page_image = image_by_page_id.get(int(page_id)) if page_id is not None else None
        if not isinstance(page_image, Image.Image):
            continue
        crop = crop_exact_block_from_page_image(
            page_image=page_image,
            bbox=candidate.get("bbox"),
            page_width=_safe_float(candidate.get("page_width")),
            page_height=_safe_float(candidate.get("page_height")),
            expand_ratio=float(crop_expand_ratio),
        )
        if crop is None:
            continue
        block_user_text = build_block_rerank_user_text(user_text, str(candidate.get("text") or ""))
        candidate_logits.append(
            score_qwen3vl_block_relevance(
                model=model,
                processor=processor,
                images=[crop],
                user_text=block_user_text,
                system_text=VLM_BLOCK_RERANK_SYSTEM_TEXT,
            ).reshape(())
        )
        valid_candidates.append(candidate)
    if not candidate_logits:
        return None, []
    return torch.stack(candidate_logits, dim=0), valid_candidates


def _compute_dma_exact_block_loss(
    *,
    model,
    processor: Any | None,
    batch_meta: list[dict[str, Any]],
    batch_images: list[list[Image.Image]] | None,
    sample_image_counts: list[int],
    page_logits: torch.Tensor | None,
    block_logits: torch.Tensor | None,
    selected_ratios: torch.Tensor | None,
    sample_retained_page_counts: list[int] | None,
    exact_block_topk: int,
    temperature: float,
    negative_weight: float,
    negative_topk: int,
    loss_type: str,
    margin: float,
    lexical_dropout: float,
    distill_weight: float,
    distill_temperature: float,
    scorer_type: str,
    vlm_rerank_topm: int,
    vlm_crop_expand_ratio: float,
) -> tuple[torch.Tensor | None, dict[str, float | int | None]]:
    split_page_logits = _split_flat_page_logits(page_logits, sample_image_counts) if page_logits is not None else None
    split_block_logits = _split_flat_block_logits(block_logits, sample_image_counts) if block_logits is not None else None
    split_selected_ratios = (
        _split_flat_page_logits(selected_ratios, sample_image_counts) if selected_ratios is not None else None
    )
    stats = {
        "exact_block_supervision_samples": 0,
        "exact_block_candidates": 0,
        "exact_block_positive_blocks": 0,
        "exact_block_negative_blocks": 0,
        "exact_block_supervision_samples_vniah": 0,
        "exact_block_supervision_samples_vmqar": 0,
        "exact_block_candidates_vniah": 0,
        "exact_block_candidates_vmqar": 0,
        "exact_block_positive_blocks_vniah": 0,
        "exact_block_positive_blocks_vmqar": 0,
        "exact_block_teacher_samples": 0,
        "exact_block_teacher_blocks": 0,
        "exact_block_distill_loss": None,
    }
    sample_losses: list[torch.Tensor] = []
    sample_distill_losses: list[torch.Tensor] = []
    for sample_idx, (meta, num_images) in enumerate(zip(batch_meta, sample_image_counts)):
        sample_meta = meta if isinstance(meta, dict) else {}
        sample_images = []
        if batch_images is not None and sample_idx < len(batch_images):
            raw_sample_images = batch_images[sample_idx]
            if isinstance(raw_sample_images, list):
                sample_images = raw_sample_images
        document_json_path = _resolve_document_json_path(sample_meta)
        if document_json_path is None or not Path(document_json_path).exists():
            continue
        target_block_ids_by_page, _ = _extract_exact_target_block_ids(
            sample_meta,
            num_images=int(num_images),
            visible_only=True,
        )
        if not target_block_ids_by_page:
            continue
        input_page_ids = _resolve_sample_input_page_ids(sample_meta, num_images=int(num_images))
        target_page_ids = sorted(int(page_id) for page_id in target_block_ids_by_page.keys())
        page_stats_by_id = _build_sample_exact_block_page_stats(
            input_page_ids=input_page_ids,
            sample_page_logits=None if split_page_logits is None else split_page_logits[sample_idx],
            sample_block_logits=None if split_block_logits is None else split_block_logits[sample_idx],
            sample_selected_ratios=None if split_selected_ratios is None else split_selected_ratios[sample_idx],
        )
        candidate_source_page_ids = _resolve_exact_block_source_page_ids(
            input_page_ids=input_page_ids,
            target_page_ids=target_page_ids,
            sample_page_logits=None if split_page_logits is None else split_page_logits[sample_idx],
            sample_retained_page_count=None
            if sample_retained_page_counts is None or sample_idx >= len(sample_retained_page_counts)
            else sample_retained_page_counts[sample_idx],
        )
        if float(distill_weight) > 0.0:
            teacher_payload = sample_meta.get("exact_block_teacher")
            raw_teacher_page_ids = teacher_payload.get("source_page_ids") if isinstance(teacher_payload, dict) else None
            if isinstance(raw_teacher_page_ids, list):
                teacher_page_id_set = {
                    int(page_id)
                    for page_id in (_page_id_to_int(value) for value in raw_teacher_page_ids)
                    if page_id is not None
                }
                if teacher_page_id_set:
                    candidate_source_page_ids = [
                        page_id for page_id in input_page_ids if int(page_id) in teacher_page_id_set or int(page_id) in candidate_source_page_ids
                    ] or candidate_source_page_ids
        candidates = build_exact_block_candidate_rows(
            document_json_path=str(document_json_path),
            source_page_ids=candidate_source_page_ids,
            query_texts=[sample_meta.get("_user_text")],
            page_stats_by_id=page_stats_by_id,
        )
        if not candidates:
            continue
        attach_exact_block_teacher_scores(candidates, sample_meta.get("exact_block_teacher"))
        positive_block_ids = {
            str(block_id)
            for block_ids in target_block_ids_by_page.values()
            for block_id in block_ids
        }
        positive_indices = [
            idx
            for idx, candidate in enumerate(candidates)
            if str(candidate.get("block_id")) in positive_block_ids
        ]
        if not positive_indices:
            continue
        if _uses_vlm_cross_encoder_exact_block_scorer(scorer_type):
            candidates = _select_exact_block_vlm_candidates(
                candidates=candidates,
                positive_block_ids=positive_block_ids,
                lexical_topm=int(vlm_rerank_topm),
            )
            candidate_logits, candidates = _score_exact_block_candidates_with_vlm_cross_encoder(
                model=model,
                processor=processor,
                sample_images=sample_images,
                input_page_ids=input_page_ids,
                candidates=candidates,
                user_text=str(sample_meta.get("_user_text") or ""),
                crop_expand_ratio=float(vlm_crop_expand_ratio),
            )
            positive_indices = [
                idx
                for idx, candidate in enumerate(candidates)
                if str(candidate.get("block_id")) in positive_block_ids
            ]
        else:
            candidate_features = exact_block_candidate_features_to_tensor(candidates)
            candidate_page_ids = exact_block_candidate_page_ids_to_tensor(candidates)
            candidate_query_sketches = exact_block_candidate_query_sketches_to_tensor(candidates)
            candidate_block_sketches = exact_block_candidate_block_sketches_to_tensor(candidates)
            candidate_features = _maybe_dropout_exact_block_lexical_features(
                candidate_features,
                dropout_prob=float(lexical_dropout),
            )
            candidate_logits = score_dma_exact_block_features(
                model,
                candidate_features,
                candidate_page_ids,
                candidate_query_sketches,
                candidate_block_sketches,
            )
        if candidate_logits is None or int(candidate_logits.numel()) != len(candidates):
            continue
        if not positive_indices:
            continue

        normalized_loss_type = str(loss_type).strip().lower()
        positive_idx = torch.tensor(positive_indices, device=candidate_logits.device, dtype=torch.long)
        negative_mask = torch.ones((int(candidate_logits.numel()),), device=candidate_logits.device, dtype=torch.bool)
        negative_mask[positive_idx] = False
        negative_idx = torch.nonzero(negative_mask, as_tuple=False).reshape(-1)
        if int(negative_topk) > 0 and int(negative_idx.numel()) > int(negative_topk):
            # hard negatives：按当前 scorer 打分选 top-k
            negative_scores = candidate_logits[negative_idx]
            keep = torch.topk(negative_scores, k=int(negative_topk)).indices
            negative_idx = negative_idx[keep]

        sample_loss: torch.Tensor | None = None
        if normalized_loss_type == "infonce":
            sample_loss = _compute_exact_block_infonce_loss(
                logits=candidate_logits,
                positive_indices=positive_idx,
                negative_indices=negative_idx,
                temperature=float(temperature),
            )
        elif normalized_loss_type == "margin_ranking":
            sample_loss = _compute_exact_block_margin_ranking_loss(
                logits=candidate_logits,
                positive_indices=positive_idx,
                negative_indices=negative_idx,
                margin=float(margin),
                hard_negative_topk=max(1, int(negative_topk)),
            )
        else:
            retained_count = max(1, min(len(candidates), int(exact_block_topk)))
            soft_membership = _compute_soft_topk_membership(
                logits=candidate_logits,
                retained_count=retained_count,
                temperature=float(temperature),
            )
            if soft_membership is None:
                continue
            positive_target = min(len(positive_indices), retained_count) / float(len(positive_indices))
            positive_probs = soft_membership[positive_idx]
            positive_loss = torch.nn.functional.binary_cross_entropy(
                positive_probs,
                torch.full_like(positive_probs, fill_value=float(positive_target)),
            )
            negative_loss = None
            if int(negative_idx.numel()) > 0:
                negative_loss = torch.nn.functional.binary_cross_entropy(
                    soft_membership[negative_idx],
                    torch.zeros_like(soft_membership[negative_idx]),
                )
            sample_loss = positive_loss if negative_loss is None else positive_loss + float(negative_weight) * negative_loss

        teacher_pairs = [
            (idx, float(candidate.get("teacher_score")))
            for idx, candidate in enumerate(candidates)
            if candidate.get("teacher_score") is not None
        ]
        sample_distill_loss: torch.Tensor | None = None
        if float(distill_weight) > 0.0 and len(teacher_pairs) >= 2:
            teacher_indices = torch.tensor(
                [pair[0] for pair in teacher_pairs],
                device=candidate_logits.device,
                dtype=torch.long,
            )
            teacher_scores = torch.tensor(
                [pair[1] for pair in teacher_pairs],
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
            sample_distill_loss = _compute_exact_block_distill_loss(
                logits=candidate_logits,
                teacher_indices=teacher_indices,
                teacher_scores=teacher_scores,
                temperature=float(distill_temperature),
            )

        sample_terms = [term for term in (sample_loss,) if term is not None]
        if sample_distill_loss is not None:
            sample_terms.append(float(distill_weight) * sample_distill_loss)
            sample_distill_losses.append(sample_distill_loss.detach())
        if not sample_terms:
            continue
        sample_losses.append(torch.stack(sample_terms).sum() if len(sample_terms) > 1 else sample_terms[0])
        dataset_tag = _normalize_exact_block_dataset_tag(sample_meta)
        stats["exact_block_supervision_samples"] += 1
        stats["exact_block_candidates"] += int(len(candidates))
        stats["exact_block_positive_blocks"] += int(len(positive_indices))
        stats["exact_block_negative_blocks"] += int(negative_idx.numel())
        if sample_distill_loss is not None:
            stats["exact_block_teacher_samples"] += 1
            stats["exact_block_teacher_blocks"] += int(len(teacher_pairs))
        if dataset_tag in {"vniah", "vmqar"}:
            stats[f"exact_block_supervision_samples_{dataset_tag}"] += 1
            stats[f"exact_block_candidates_{dataset_tag}"] += int(len(candidates))
            stats[f"exact_block_positive_blocks_{dataset_tag}"] += int(len(positive_indices))
    if not sample_losses:
        return None, stats
    if sample_distill_losses:
        stats["exact_block_distill_loss"] = float(torch.stack(sample_distill_losses).mean().item())
    return torch.stack(sample_losses).mean(), stats


def _split_model_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    model_batch = dict(batch)
    batch_meta = list(model_batch.pop("batch_meta", []))
    sample_image_counts = [int(x) for x in model_batch.pop("sample_image_counts", [])]
    return model_batch, batch_meta, sample_image_counts


def _resolve_model_execution_device(model) -> torch.device:
    base_model = getattr(model, "module", model)
    parameter = next(base_model.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


def _move_model_batch_tensors_to_device(model_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved_batch: dict[str, Any] = {}
    for key, value in model_batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device=device)
        else:
            moved_batch[key] = value
    return moved_batch


def _compute_supervised_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("sequence log prob 计算要求 logits=[B,T,V], labels=[B,T]")
    if int(logits.shape[0]) != int(labels.shape[0]) or int(logits.shape[1]) != int(labels.shape[1]):
        raise ValueError("logits/labels 维度不匹配，无法计算 sequence log prob")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(device=logits.device)
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask.to(device=token_log_probs.device, dtype=token_log_probs.dtype)
    token_counts = valid_mask.sum(dim=1).clamp_min(1)
    return token_log_probs.sum(dim=1) / token_counts.to(device=token_log_probs.device, dtype=token_log_probs.dtype)


def _forward_language_model_batch(
    *,
    model,
    model_batch: dict[str, Any],
    query_token_mask: torch.Tensor | None,
    query_routing_enabled: bool,
    image_token_id: int | None,
):
    if query_routing_enabled:
        set_dma_query_routing_sketch(
            model,
            input_ids=model_batch["input_ids"],
            attention_mask=model_batch.get("attention_mask"),
            text_mask=query_token_mask,
            image_token_id=image_token_id,
        )
    try:
        return model(**model_batch)
    finally:
        if query_routing_enabled:
            clear_dma_query_routing_sketch(model)


def _configure_training_memory_optimizations(model, cfg: DictConfig, dma_cfg: DMAConfig | None = None) -> None:
    if isinstance(cfg, dict):
        train_cfg = cfg.get("train", {})
    else:
        train_cfg = cfg.train if "train" in cfg else {}
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()
        gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
        if callable(gradient_checkpointing_enable):
            use_reentrant = bool(train_cfg.get("gradient_checkpointing_use_reentrant", False))
            gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )

        # Vision-DMA 的 page/block 训练统计依赖视觉块前向中的 side-effect 张量。
        # reentrant checkpoint 的第一遍前向处于 no_grad，会让这些统计张量失去梯度。
        # 这里仅关闭视觉编码器自身的 checkpoint，保留语言侧 checkpoint，避免整模显存回退过大。
        vision_dma_train_stats_enabled = bool(
            dma_cfg is not None
            and dma_cfg.enable
            and dma_cfg.apply_to_vision
            and (
                bool(getattr(dma_cfg, "aux_page_supervision_enable", False))
                or bool(getattr(dma_cfg, "aux_block_supervision_enable", False))
                or bool(getattr(dma_cfg, "retention_readout_enable", False))
                or bool(getattr(dma_cfg, "exact_block_readout_enable", False))
            )
        )
        if vision_dma_train_stats_enabled:
            record_layer_idx: int | None = None
            for module in model.modules():
                if module.__class__.__name__ != "Qwen3VLVisionBlock":
                    continue
                layer_idx = getattr(getattr(module, "attn", None), "_dma_layer_idx", None)
                if isinstance(layer_idx, int):
                    record_layer_idx = layer_idx if record_layer_idx is None else max(record_layer_idx, layer_idx)
            for module in model.modules():
                if module.__class__.__name__ != "Qwen3VLVisionBlock" or not hasattr(module, "gradient_checkpointing"):
                    continue
                layer_idx = getattr(getattr(module, "attn", None), "_dma_layer_idx", None)
                if record_layer_idx is None or layer_idx == record_layer_idx:
                    module.gradient_checkpointing = False

    # 训练态不需要 KV cache；长视觉序列下保留它只会白白占显存。
    for maybe_cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
    ):
        if maybe_cfg is not None and hasattr(maybe_cfg, "use_cache"):
            maybe_cfg.use_cache = False


def _build_zero_grad_anchor_from_module(module: torch.nn.Module | None) -> torch.Tensor | None:
    """给模块补一个零权重项，避免当前 batch 没命中监督样本时该模块完全不参与 autograd。"""
    if module is None:
        return None

    anchor_terms: list[torch.Tensor] = []
    for parameter in module.parameters():
        if parameter.requires_grad:
            anchor_terms.append(parameter.reshape(-1).sum() * 0.0)
    if not anchor_terms:
        return None
    return torch.stack(anchor_terms).sum()


def _resolve_ddp_graph_mode(
    train_cfg,
    dma_cfg: DMAConfig,
    *,
    exact_block_manual_sync_enabled: bool = False,
    modules_to_save_manual_sync_enabled: bool = False,
    gradient_checkpointing_use_reentrant: bool = False,
) -> dict[str, bool | str | None]:
    """
    统一解析 DDP 图模式。

    exact-block scorer 当前是在主 `model.forward(...)` 之后单独调用的。
    这类“参数在 forward 之外参与梯度”的场景，PyTorch 官方要求走 static-graph 路径；
    否则容易在多卡下出现 `unused parameter`、`unfinished reduction` 或 `ready twice`。
    """

    user_static_graph = bool(getattr(train_cfg, "ddp_static_graph", False))
    user_find_unused_override = getattr(train_cfg, "ddp_find_unused_parameters", None)
    exact_block_scorer_outside_forward = (
        bool(getattr(dma_cfg, "enable", False))
        and bool(getattr(dma_cfg, "apply_to_vision", False))
        and bool(getattr(dma_cfg, "exact_block_readout_enable", False))
        and float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) > 0.0
    )
    reentrant_checkpoint_manual_sync = (
        bool(modules_to_save_manual_sync_enabled) and bool(gradient_checkpointing_use_reentrant)
    )
    ddp_static_graph = user_static_graph or (
        exact_block_scorer_outside_forward and not exact_block_manual_sync_enabled
    )
    if reentrant_checkpoint_manual_sync:
        ddp_static_graph = True
    reason: str | None = None
    if exact_block_scorer_outside_forward and not exact_block_manual_sync_enabled and not user_static_graph:
        reason = "exact_block_scorer_outside_forward"
    elif reentrant_checkpoint_manual_sync and not user_static_graph:
        reason = "reentrant_checkpoint_manual_grad_sync"

    if user_find_unused_override is None:
        ddp_find_unused_parameters = False if (
            exact_block_manual_sync_enabled or modules_to_save_manual_sync_enabled
        ) else (not ddp_static_graph)
    else:
        ddp_find_unused_parameters = bool(user_find_unused_override)
        if exact_block_scorer_outside_forward and not exact_block_manual_sync_enabled and ddp_find_unused_parameters:
            ddp_find_unused_parameters = False
            if reason is None:
                reason = "exact_block_scorer_outside_forward"
        if reentrant_checkpoint_manual_sync and ddp_find_unused_parameters:
            ddp_find_unused_parameters = False
            if reason is None:
                reason = "reentrant_checkpoint_manual_grad_sync"
    if exact_block_manual_sync_enabled and reason is None:
        reason = "exact_block_manual_grad_sync"

    return {
        "ddp_static_graph": ddp_static_graph,
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
        "ddp_graph_mode_reason": reason,
    }


def _collect_exact_block_trainable_param_names(model, dma_cfg: DMAConfig) -> list[str]:
    """
    收集 exact-block scorer 在当前模型里的真实可训练参数名。

    这些参数会在主 `model.forward(...)` 外单独参与 exact-block loss。
    多卡训练下把它们交给手动梯度同步，比要求 DDP reducer 追踪这段额外图更稳定。
    """

    exact_block_training_enabled = (
        bool(getattr(dma_cfg, "enable", False))
        and bool(getattr(dma_cfg, "apply_to_vision", False))
        and bool(getattr(dma_cfg, "exact_block_readout_enable", False))
        and float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) > 0.0
    )
    if not exact_block_training_enabled:
        return []
    scorer = resolve_exact_block_scorer(model)
    if scorer is None:
        return []
    scorer_param_ids = {id(param) for param in scorer.parameters() if param.requires_grad}
    if not scorer_param_ids:
        return []
    param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) in scorer_param_ids
    ]
    return sorted(set(param_names))


def _collect_modules_to_save_trainable_param_names(model) -> list[str]:
    """
    收集当前模型里所有活动 `modules_to_save` 可训练参数名。

    这类参数通常是 DMA/Cross-Attn 额外小头，在 PEFT 下表现为
    `*.modules_to_save.default.*`。它们在多卡 + reentrant checkpoint 组合下
    更容易触发 DDP reducer 的 `ready twice`，因此统一走手动梯度同步更稳。
    """

    param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and ".modules_to_save." in name and ".original_module." not in name
    ]
    return sorted(set(param_names))


def _merge_manual_sync_param_names(*param_name_groups: list[str]) -> list[str]:
    """
    合并多组手动同步参数名，并保持稳定顺序去重。

    exact-block scorer 与 `modules_to_save` 小头可能部分重叠；
    这里统一合并成单一口径，避免 trace、DDP ignore 与真实同步集合不一致。
    """

    merged_names: list[str] = []
    seen_names: set[str] = set()
    for group in param_name_groups:
        for name in group:
            if name in seen_names:
                continue
            seen_names.add(name)
            merged_names.append(name)
    return merged_names


def _collect_all_trainable_param_names(model) -> list[str]:
    """收集当前模型里所有可训练参数名。"""

    return [name for name, param in model.named_parameters() if param.requires_grad]


def _resolve_parallel_training_mode(
    train_cfg,
    dma_cfg: DMAConfig,
    *,
    world_size: int,
    manual_sync_param_names: list[str],
    all_trainable_param_names: list[str],
    all_trainable_param_numel: int,
) -> dict[str, bool | str | int | None]:
    """
    在小规模 adapter 训练下，优先使用“纯手动梯度同步”替代 DDP reducer。

    当前 2B + LoRA + DMA 口径里，全部可训练参数仅约十几 MB。
    如果还让 DDP 管剩余 LoRA 参数，首轮 backward 会先走一次 local-used-map all-reduce；
    一旦动态分支和 static-graph 组合不稳定，就可能直接卡在 reducer，而不是卡在真实梯度同步。
    这里对“小训练参数集”自动切到“每步扁平化 all-reduce 全部可训练梯度”的口径，
    保持梯度平均语义不变，同时绕开 DDP 的参数参与图同步。
    """

    if int(world_size) <= 1:
        return {
            "manual_data_parallel_all_trainable": False,
            "parallel_mode_reason": None,
            "all_trainable_param_numel": int(all_trainable_param_numel),
        }

    user_override = getattr(train_cfg, "manual_data_parallel_all_trainable", None)
    eligible = (
        bool(getattr(dma_cfg, "enable", False))
        and bool(getattr(dma_cfg, "apply_to_vision", False))
        and bool(all_trainable_param_names)
    )
    small_trainable_set = int(all_trainable_param_numel) > 0 and int(all_trainable_param_numel) <= 16_777_216
    if user_override is None:
        manual_data_parallel_all_trainable = bool(eligible and small_trainable_set)
        reason = "small_trainable_set_manual_all_reduce" if manual_data_parallel_all_trainable else None
    else:
        manual_data_parallel_all_trainable = bool(user_override) and bool(eligible)
        reason = "manual_all_trainable_user_override" if manual_data_parallel_all_trainable else None

    return {
        "manual_data_parallel_all_trainable": manual_data_parallel_all_trainable,
        "parallel_mode_reason": reason,
        "all_trainable_param_numel": int(all_trainable_param_numel),
    }


def _resolve_adamw_runtime_kwargs(train_cfg, dma_cfg: DMAConfig) -> dict[str, bool | str | None | dict[str, bool]]:
    """
    统一解析 AdamW 的实现细节。

    exact-block scorer 是小头、参数量不大，但会在多卡下走“forward 外梯度 + 手动同步”路径。
    这里优先选择更保守的单 tensor AdamW，避免 foreach/fused 多张量 kernel 在首步状态初始化时卡住。
    """

    user_foreach = getattr(train_cfg, "adamw_foreach", None)
    user_fused = getattr(train_cfg, "adamw_fused", None)
    exact_block_training_enabled = (
        bool(getattr(dma_cfg, "enable", False))
        and bool(getattr(dma_cfg, "apply_to_vision", False))
        and bool(getattr(dma_cfg, "exact_block_readout_enable", False))
        and float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) > 0.0
    )

    foreach = None if user_foreach is None else bool(user_foreach)
    fused = None if user_fused is None else bool(user_fused)
    reason: str | None = None
    if exact_block_training_enabled:
        foreach = False
        fused = False
        reason = "exact_block_optimizer_stability"

    optimizer_kwargs: dict[str, bool] = {}
    if foreach is not None:
        optimizer_kwargs["foreach"] = foreach
    if fused is not None:
        optimizer_kwargs["fused"] = fused
    return {
        "optimizer_kwargs": optimizer_kwargs,
        "adamw_foreach": foreach,
        "adamw_fused": fused,
        "adamw_runtime_reason": reason,
    }


def _mark_ddp_ignored_param_names(model, param_names: list[str]) -> None:
    if not param_names:
        return
    existing_names = getattr(model, "_ddp_params_and_buffers_to_ignore", None)
    merged_names = set(existing_names or [])
    merged_names.update(str(name) for name in param_names)
    model._ddp_params_and_buffers_to_ignore = merged_names


def _resolve_named_parameters_for_manual_sync(
    model,
    param_names: list[str],
) -> list[tuple[str, torch.nn.Parameter]]:
    base_model = getattr(model, "module", model)
    named_parameters = dict(base_model.named_parameters())
    sync_parameters: list[tuple[str, torch.nn.Parameter]] = []
    for name in param_names:
        parameter = named_parameters.get(name)
        if parameter is None or (not parameter.requires_grad):
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        sync_parameters.append((name, parameter))
    return sync_parameters


def _sync_named_parameter_gradients(model, param_names: list[str]) -> int:
    """
    对一组已从 DDP 自动归约中排除的参数做显式梯度同步。
    """

    if (not param_names) or (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        return 0

    world_size = int(torch.distributed.get_world_size())
    sync_parameters = _resolve_named_parameters_for_manual_sync(model, param_names)
    if not sync_parameters:
        return 0

    # 逐参数 all_reduce 在 8 卡下会形成大量极小通信，既慢也更容易放大 rank 顺序差异。
    # 这里按稳定名字顺序把梯度打平后只做一次归约，再拷回各自参数。
    flat_grad = torch.cat([parameter.grad.reshape(-1) for _, parameter in sync_parameters], dim=0)
    torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM)
    flat_grad.div_(float(world_size))

    offset = 0
    for _, parameter in sync_parameters:
        numel = parameter.grad.numel()
        parameter.grad.copy_(flat_grad[offset : offset + numel].view_as(parameter.grad))
        offset += numel
    return len(sync_parameters)


def _sync_named_parameter_gradients_via_files(
    model,
    param_names: list[str],
    *,
    barrier_dir: Path,
    stage: str,
    world_size: int,
    rank: int,
) -> int:
    """
    用共享文件系统同步小头梯度，规避 `reentrant checkpoint + NCCL` 下的额外 collective 挂死。
    """

    if (not param_names) or int(world_size) <= 1:
        return 0

    sync_parameters = _resolve_named_parameters_for_manual_sync(model, param_names)
    if not sync_parameters:
        return 0

    stage_dir = barrier_dir / stage
    tensor_dir = stage_dir / "tensors"
    meta_dir = stage_dir / "meta"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    local_manifest = [
        {
            "name": name,
            "numel": int(parameter.grad.numel()),
            "shape": list(parameter.grad.shape),
            "dtype": str(parameter.grad.dtype),
        }
        for name, parameter in sync_parameters
    ]
    local_flat_grad = torch.cat(
        [parameter.grad.detach().reshape(-1).to(dtype=torch.float32, device="cpu") for _, parameter in sync_parameters],
        dim=0,
    )
    tensor_path = tensor_dir / f"rank_{int(rank):02d}.pt"
    _atomic_torch_save(local_flat_grad, tensor_path)
    records = _collect_rank_payloads_via_files(
        barrier_dir=meta_dir,
        stage=stage,
        payload={
            "rank": int(rank),
            "tensor_path": str(tensor_path),
            "manifest": local_manifest,
            "total_numel": int(local_flat_grad.numel()),
        },
        world_size=int(world_size),
        rank=int(rank),
    )

    reference_manifest = records[0]["manifest"]
    reference_numel = int(records[0]["total_numel"])
    for record in records[1:]:
        if record["manifest"] != reference_manifest or int(record["total_numel"]) != reference_numel:
            raise RuntimeError(
                f"manual grad sync 参数布局不一致: stage={stage}, rank={rank}, "
                f"reference_total_numel={reference_numel}, current_total_numel={record['total_numel']}"
            )

    avg_path = stage_dir / "avg.pt"
    if int(rank) == 0:
        flat_sum = torch.zeros(reference_numel, dtype=torch.float32)
        for record in records:
            flat_sum.add_(_load_torch_file_with_retry(record["tensor_path"], map_location="cpu"))
        flat_sum.div_(float(world_size))
        _atomic_torch_save(flat_sum, avg_path)

    averaged_flat_grad = _load_torch_file_with_retry(
        avg_path,
        map_location="cpu",
        timeout_s=600.0,
        poll_interval_s=0.05,
    )

    offset = 0
    for _, parameter in sync_parameters:
        numel = parameter.grad.numel()
        parameter.grad.copy_(
            averaged_flat_grad[offset : offset + numel]
            .to(device=parameter.grad.device, dtype=parameter.grad.dtype)
            .view_as(parameter.grad)
        )
        offset += numel
    return len(sync_parameters)


def _build_dma_runtime_zero_anchors(model, dma_cfg: DMAConfig) -> list[torch.Tensor]:
    """
    给 DMA 的条件分支统一补零锚点。

    这些分支会随 batch 内容动态跳过，例如：
    - query routing 依赖 query sketch 是否成功构造
    - block feedback 依赖上一层是否产生 block cache
    - retention readout 只在记录层参与真实监督

    DDP 下如果某个 rank 本 step 没命中分支，就会把对应参数看成 unused。
    这里统一挂零权重项，保证参数集合在各 rank 上稳定一致。
    """

    runtime_attr_names = ["dma_dt_proj", "dma_gate"]
    if bool(getattr(dma_cfg, "vision_page_bias_enable", False)):
        runtime_attr_names.extend(["dma_page_token_proj", "dma_page_context_proj", "dma_page_gate"])
    if bool(getattr(dma_cfg, "vision_layout_bias_enable", False)):
        runtime_attr_names.extend(["dma_layout_proj", "dma_layout_gate"])
    if bool(getattr(dma_cfg, "vision_query_routing_enable", False)):
        runtime_attr_names.extend(["dma_query_token_proj", "dma_query_context_proj", "dma_query_gate"])
    if bool(getattr(dma_cfg, "vision_block_feedback_enable", False)):
        runtime_attr_names.extend(["dma_block_token_proj", "dma_block_context_proj", "dma_block_gate"])
    if bool(getattr(dma_cfg, "retention_readout_enable", False)):
        runtime_attr_names.extend(["dma_retention_page_proj", "dma_retention_block_proj"])

    anchors: list[torch.Tensor] = []
    seen_module_ids: set[int] = set()
    for module in model.modules():
        for attr_name in runtime_attr_names:
            runtime_module = getattr(module, attr_name, None)
            if runtime_module is None:
                continue
            module_id = id(runtime_module)
            if module_id in seen_module_ids:
                continue
            seen_module_ids.add(module_id)
            anchor = _build_zero_grad_anchor_from_module(runtime_module)
            if anchor is not None:
                anchors.append(anchor)
    return anchors


def _exact_block_block_only_enabled(dma_cfg: DMAConfig) -> bool:
    return (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "exact_block_block_only_enable", False))
    )


def _dma_mask_only_enabled(dma_cfg: DMAConfig) -> bool:
    return (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "mask_modules_only_enable", False))
    )


def _dma_runtime_only_enabled(dma_cfg: DMAConfig) -> bool:
    return (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "runtime_modules_only_enable", False))
    )


def _configure_dma_mask_only_training(model, dma_cfg: DMAConfig) -> list[str]:
    """
    DMA mask-only 模式下，仅保留 `dma_dt_proj` / `dma_gate` 可训练。

    目标是验证：
    - 不训练 query/page/layout/block routing 小头
    - 不训练 retention / exact-block readout
    - 只让主任务 CE 通过 attention logits 路径更新 sparse mask 参数
    """

    if not _dma_mask_only_enabled(dma_cfg):
        return []

    trainable_param_names: list[str] = []
    for name, param in model.named_parameters():
        keep_trainable = (
            (("dma_dt_proj" in name) or ("dma_gate" in name))
            and ".original_module." not in name
        )
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable_param_names.append(name)
    if not trainable_param_names:
        raise ValueError("DMA mask-only 模式下未找到可训练参数：dma_dt_proj / dma_gate")
    return trainable_param_names


def _configure_dma_runtime_only_training(model, dma_cfg: DMAConfig) -> list[str]:
    """
    DMA runtime-only 模式下，仅保留 `dma_*` 小头可训练。

    目标是把主任务 CE 的学习信号尽量约束在 DMA 分支里，
    方便验证 query-conditioned soft gating 是否本身就能起作用。
    """

    if not _dma_runtime_only_enabled(dma_cfg):
        return []

    trainable_param_names: list[str] = []
    for name, param in model.named_parameters():
        keep_trainable = ("dma_" in name) and (".original_module." not in name)
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable_param_names.append(name)
    if not trainable_param_names:
        raise ValueError("DMA runtime-only 模式下未找到可训练参数：dma_*")
    return trainable_param_names


def _configure_block_only_reranker_training(model, dma_cfg: DMAConfig) -> list[str]:
    """
    block-only reranker 模式下，仅保留 exact-block scorer 可训练。
    """

    if not _exact_block_block_only_enabled(dma_cfg):
        return []
    if not bool(getattr(dma_cfg, "exact_block_readout_enable", False)):
        raise ValueError("启用 exact_block_block_only_enable 时，必须同时开启 dma.exact_block_readout_enable=true")
    if float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) <= 0.0:
        raise ValueError("启用 exact_block_block_only_enable 时，必须设置 aux_exact_block_loss_weight > 0")
    if _uses_vlm_cross_encoder_exact_block_scorer(getattr(dma_cfg, "exact_block_scorer_type", "")):
        raise ValueError("vlm_cross_encoder scorer 暂不支持 exact_block_block_only_enable；请先关闭 block-only 模式")

    trainable_param_names: list[str] = []
    for name, param in model.named_parameters():
        keep_trainable = "dma_exact_block_proj" in name and ".original_module." not in name
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable_param_names.append(name)
    if not trainable_param_names:
        raise ValueError("block-only reranker 模式下未找到可训练参数：dma_exact_block_proj")
    return trainable_param_names


def _apply_grounded_freeze_policy(
    model,
    *,
    structured_output_mode: str | None,
    training_phase: str | None,
    freeze_retrieval: bool,
) -> list[str]:
    if str(structured_output_mode or "").strip().lower() != "grounded_json":
        return []
    normalize_grounding_training_phase(training_phase)
    if not freeze_retrieval:
        return []

    frozen_param_names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        should_freeze = (
            "dma_" in lower_name
            or "cross_attn" in lower_name
            or "dmaexactblock" in lower_name
            or "dma_exact_block" in lower_name
        )
        if not should_freeze:
            continue
        param.requires_grad_(False)
        frozen_param_names.append(name)
    return frozen_param_names


def _load_model_and_processor_for_training(
    *,
    model_path: str,
    model_dtype: str,
    attn_implementation: str | None,
    peft_enable: bool,
    peft_cfg: DictConfig,
    dma_cfg: DMAConfig,
    cross_cfg: CrossAttnConfig,
) -> tuple[Any, torch.nn.Module]:
    adapter_cfg = _load_local_peft_adapter_config(model_path)
    base_model_path = model_path
    if adapter_cfg is not None:
        _validate_adapter_runtime_modules(
            model_path=model_path,
            adapter_cfg=adapter_cfg,
            dma=dma_cfg,
            cross_attn=cross_cfg,
        )
        base_model_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
        if not base_model_path:
            raise ValueError(f"{model_path} 的 adapter_config.json 缺少 base_model_name_or_path，无法加载基座模型。")

    processor = AutoProcessor.from_pretrained(model_path)
    dtype = torch.bfloat16 if str(model_dtype) in {"bf16", "bfloat16"} else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    # 与评测保持同口径：先补 DMA / Cross-Attn 结构，再决定是否叠旧 adapter 或创建新 adapter。
    apply_dma_to_qwen3vl_model(model, dma_cfg)
    apply_cross_attn_adapter_to_qwen3vl_model(model, cross_cfg)

    if peft_enable:
        modules_to_save: list[str] = []
        if dma_cfg.enable:
            modules_to_save.extend(dma_modules_to_save(dma_cfg))
        if cross_cfg.enable:
            modules_to_save.extend(cross_modules_to_save())
        modules_to_save = modules_to_save or None
        lora_cfg = LoraConfig(
            r=int(peft_cfg.lora_r),
            lora_alpha=int(peft_cfg.lora_alpha),
            lora_dropout=float(peft_cfg.lora_dropout),
            target_modules=list(peft_cfg.target_modules),
            modules_to_save=modules_to_save,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if adapter_cfg is not None:
            current_adapter_state = get_peft_model_state_dict(model, adapter_name="default")
            adapter_state = load_peft_weights(model_path, device="cpu")
            current_adapter_state.update(adapter_state)
            set_peft_model_state_dict(model, current_adapter_state, adapter_name="default")
            _align_peft_modules_to_save_devices(model)
        _freeze_non_record_retention_readout_modules(model)
    elif adapter_cfg is not None:
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        _align_peft_modules_to_save_devices(model)
        _freeze_non_record_retention_readout_modules(model)

    return processor, model


def _compute_total_loss(
    *,
    model,
    processor: Any | None = None,
    model_batch: dict[str, Any],
    batch_meta: list[dict[str, Any]],
    sample_image_counts: list[int],
    dma_cfg: DMAConfig,
    grounding_cfg: Any | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    model_device = _resolve_model_execution_device(model)
    runtime_model_batch = _move_model_batch_tensors_to_device(dict(model_batch), model_device)
    raw_batch_images = runtime_model_batch.pop("batch_images", None)
    batch_images = raw_batch_images if isinstance(raw_batch_images, list) else None
    raw_query_token_mask = runtime_model_batch.pop("query_token_mask", None)
    query_token_mask = raw_query_token_mask if isinstance(raw_query_token_mask, torch.Tensor) else None
    rejected_model_batch: dict[str, Any] = {}
    for key in list(runtime_model_batch.keys()):
        if not key.startswith("rejected_"):
            continue
        rejected_model_batch[key.removeprefix("rejected_")] = runtime_model_batch.pop(key)
    raw_rejected_query_token_mask = rejected_model_batch.pop("query_token_mask", None)
    rejected_query_token_mask = (
        raw_rejected_query_token_mask if isinstance(raw_rejected_query_token_mask, torch.Tensor) else None
    )
    labels = runtime_model_batch.get("labels") if isinstance(runtime_model_batch.get("labels"), torch.Tensor) else None
    rejected_labels = (
        rejected_model_batch.get("labels") if isinstance(rejected_model_batch.get("labels"), torch.Tensor) else None
    )
    grounded_structured_output = (
        str(getattr(grounding_cfg, "structured_output_mode", "plain")).strip().lower() == "grounded_json"
    )
    grounding_phase = normalize_grounding_training_phase(getattr(grounding_cfg, "training_phase", "final"))
    grounding_freeze_retrieval = bool(getattr(grounding_cfg, "freeze_retrieval", False))
    pref_mode = grounded_structured_output and grounding_phase == "pref" and rejected_labels is not None
    pref_beta = float(getattr(grounding_cfg, "preference_beta", 0.1))
    preference_loss_type = str(getattr(grounding_cfg, "preference_loss_type", "dpo")).strip().lower()
    preference_gamma = float(getattr(grounding_cfg, "preference_gamma", 0.0))
    generator_only_preference = bool(getattr(grounding_cfg, "generator_only_preference", True))
    sparse_lm_target_positions: torch.Tensor | None = None
    sparse_lm_target_labels: torch.Tensor | None = None
    if not pref_mode and labels is not None and labels.ndim == 2 and int(labels.shape[0]) == 1:
        supervised_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
        if int(supervised_positions.numel()) > 0:
            shifted_positions = supervised_positions - 1
            valid_mask = shifted_positions >= 0
            if bool(valid_mask.any()):
                sparse_lm_target_positions = shifted_positions[valid_mask].to(device=labels.device, dtype=torch.long)
                sparse_lm_target_labels = labels[:, supervised_positions[valid_mask]].contiguous()
    if dma_cfg.enable and dma_cfg.apply_to_vision:
        reset_dma_page_stats(model)
    query_routing_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "vision_query_routing_enable", False))
    )
    image_token_id = None
    if processor is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            except Exception:
                image_token_id = None
    sparse_lm_enabled = False
    if pref_mode:
        chosen_model_batch = dict(runtime_model_batch)
        rejected_forward_batch = dict(rejected_model_batch)
        chosen_out = _forward_language_model_batch(
            model=model,
            model_batch=chosen_model_batch,
            query_token_mask=query_token_mask,
            query_routing_enabled=query_routing_enabled,
            image_token_id=image_token_id,
        )
        rejected_out = _forward_language_model_batch(
            model=model,
            model_batch=rejected_forward_batch,
            query_token_mask=rejected_query_token_mask if rejected_query_token_mask is not None else query_token_mask,
            query_routing_enabled=query_routing_enabled,
            image_token_id=image_token_id,
        )
        chosen_sequence_logp = _compute_supervised_sequence_log_probs(chosen_out.logits, labels)
        rejected_sequence_logp = _compute_supervised_sequence_log_probs(rejected_out.logits, rejected_labels)
        preference_margin = chosen_sequence_logp - rejected_sequence_logp
        if preference_loss_type in {"dpo", "pairwise"}:
            # 当前仓库的 DPO 口径是无 reference 的 pairwise loss；sequence_logp 已按 token 数归一。
            lm_loss = -torch.nn.functional.logsigmoid(float(pref_beta) * preference_margin).mean()
        elif preference_loss_type == "simpo":
            lm_loss = -torch.nn.functional.logsigmoid(
                float(pref_beta) * (preference_margin - float(preference_gamma))
            ).mean()
        elif preference_loss_type == "kto":
            if not batch_meta:
                raise ValueError("KTO 需要 batch_meta 中的 ref_chosen_logp/ref_rejected_logp")
            try:
                ref_chosen = torch.tensor(
                    [float(meta["ref_chosen_logp"]) for meta in batch_meta],
                    device=chosen_sequence_logp.device,
                    dtype=chosen_sequence_logp.dtype,
                )
                ref_rejected = torch.tensor(
                    [float(meta["ref_rejected_logp"]) for meta in batch_meta],
                    device=rejected_sequence_logp.device,
                    dtype=rejected_sequence_logp.dtype,
                )
            except KeyError as exc:
                raise ValueError("KTO 需要样本 meta 预先写入 ref_chosen_logp/ref_rejected_logp") from exc
            chosen_reward = chosen_sequence_logp - ref_chosen
            rejected_reward = rejected_sequence_logp - ref_rejected
            chosen_loss = -torch.nn.functional.logsigmoid(
                float(pref_beta) * (chosen_reward - float(preference_gamma))
            )
            rejected_loss = -torch.nn.functional.logsigmoid(
                float(pref_beta) * (-rejected_reward - float(preference_gamma))
            )
            lm_loss = 0.5 * (chosen_loss + rejected_loss).mean()
        else:
            raise ValueError(f"不支持的 preference_loss_type: {preference_loss_type}")
    else:
        if query_routing_enabled:
            set_dma_query_routing_sketch(
                model,
                input_ids=runtime_model_batch["input_ids"],
                attention_mask=runtime_model_batch.get("attention_mask"),
                text_mask=query_token_mask,
                image_token_id=image_token_id,
            )
        try:
            if sparse_lm_target_positions is not None and sparse_lm_target_labels is not None:
                sparse_model_batch = dict(runtime_model_batch)
                sparse_model_batch.pop("labels", None)
                out = model(**sparse_model_batch, logits_to_keep=sparse_lm_target_positions)
                vocab_size = int(out.logits.shape[-1])
                lm_loss = torch.nn.functional.cross_entropy(
                    out.logits.reshape(-1, vocab_size).float(),
                    sparse_lm_target_labels.reshape(-1).to(out.logits.device),
                )
                sparse_lm_enabled = True
            else:
                out = model(**runtime_model_batch)
                lm_loss = out.loss
        finally:
            if query_routing_enabled:
                clear_dma_query_routing_sketch(model)
    aux_stats: dict[str, float | int | None] = {
        "lm_loss": float(lm_loss.detach().item()),
        "lm_logits_kept": int(sparse_lm_target_positions.numel()) if sparse_lm_enabled and sparse_lm_target_positions is not None else None,
        "grounding_phase": grounding_phase,
        "pref_loss": None,
        "pref_beta": float(pref_beta) if pref_mode else None,
        "preference_loss_type": preference_loss_type if pref_mode else None,
        "preference_gamma": float(preference_gamma) if pref_mode else None,
        "chosen_logp": None,
        "rejected_logp": None,
        "preference_margin": None,
        "aux_loss": None,
        "aux_loss_type": None,
        "aux_page_loss": None,
        "aux_block_loss": None,
        "aux_exact_block_loss": None,
        "aux_exact_block_distill_loss": None,
        "aux_single_page_focus_loss": None,
        "aux_single_page_retained_loss": None,
        "aux_multi_page_loss": None,
        "aux_retained_set_loss": None,
        "aux_retained_block_set_loss": None,
        "visible_positive_pages": 0,
        "target_pages": 0,
        "target_exact_blocks": 0,
        "target_exact_pages": 0,
        "visible_target_exact_blocks": 0,
        "visible_target_exact_pages": 0,
        "document_layout_samples": 0,
        "ranking_samples": 0,
        "ranking_pairs": 0,
        "readout_focus_samples": 0,
        "readout_focus_pairs": 0,
        "single_positive_samples": 0,
        "single_positive_pairs": 0,
        "single_page_retained_samples": 0,
        "single_page_retained_pairs": 0,
        "single_page_positive_in_topk": 0,
        "multi_positive_samples": 0,
        "multi_positive_pages": 0,
        "multi_page_pairs": 0,
        "retained_set_samples": 0,
        "retained_set_positive_pages": 0,
        "visible_positive_blocks": 0,
        "block_supervision_pages": 0,
        "block_ranking_pages": 0,
        "block_ranking_pairs": 0,
        "block_focus_pages": 0,
        "block_readout_pages": 0,
        "block_readout_pairs": 0,
        "retained_block_set_pages": 0,
        "retained_block_set_blocks": 0,
        "exact_block_supervision_samples": 0,
        "exact_block_candidates": 0,
        "exact_block_positive_blocks": 0,
        "exact_block_negative_blocks": 0,
        "exact_block_supervision_samples_vniah": 0,
        "exact_block_supervision_samples_vmqar": 0,
        "exact_block_candidates_vniah": 0,
        "exact_block_candidates_vmqar": 0,
        "exact_block_positive_blocks_vniah": 0,
        "exact_block_positive_blocks_vmqar": 0,
        "exact_block_teacher_samples": 0,
        "exact_block_teacher_blocks": 0,
        "exact_block_proj_weight_l2": None,
        "exact_block_proj_delta_l2": None,
        "exact_block_zero_anchor": 0,
    }
    if pref_mode:
        aux_stats["pref_loss"] = float(lm_loss.detach().item())
        aux_stats["chosen_logp"] = float(chosen_sequence_logp.detach().mean().item())
        aux_stats["rejected_logp"] = float(rejected_sequence_logp.detach().mean().item())
        aux_stats["preference_margin"] = float((chosen_sequence_logp - rejected_sequence_logp).detach().mean().item())

    weighted_aux_terms: list[torch.Tensor] = []
    disable_retrieval_aux_losses = grounded_structured_output and grounding_freeze_retrieval
    exact_block_block_only_enabled = _exact_block_block_only_enabled(dma_cfg)
    page_supervision_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and dma_cfg.aux_page_supervision_enable
        and float(dma_cfg.aux_page_loss_weight) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    block_supervision_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "aux_block_supervision_enable", False))
        and float(getattr(dma_cfg, "aux_block_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    multi_page_coverage_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and float(getattr(dma_cfg, "aux_multi_page_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    single_page_focus_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and float(getattr(dma_cfg, "aux_single_page_focus_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    single_page_retained_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and float(getattr(dma_cfg, "aux_single_page_retained_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    retained_set_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and float(getattr(dma_cfg, "aux_retained_set_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    retained_block_set_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and float(getattr(dma_cfg, "aux_retained_block_set_loss_weight", 0.0)) > 0.0
        and not exact_block_block_only_enabled
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    exact_block_supervision_enabled = (
        dma_cfg.enable
        and dma_cfg.apply_to_vision
        and bool(getattr(dma_cfg, "exact_block_readout_enable", False))
        and float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) > 0.0
        and not disable_retrieval_aux_losses
        and not (pref_mode and generator_only_preference)
    )
    if (
        (
            page_supervision_enabled
            or block_supervision_enabled
            or multi_page_coverage_enabled
            or single_page_focus_enabled
            or single_page_retained_enabled
            or retained_set_enabled
            or retained_block_set_enabled
            or exact_block_supervision_enabled
        )
        and sample_image_counts
        and batch_meta
    ):
        page_loss_type = str(getattr(dma_cfg, "aux_page_loss_type", "bce")).strip().lower()
        block_loss_type = str(getattr(dma_cfg, "aux_block_loss_type", "bce")).strip().lower()
        need_block_logits_for_page_loss = page_loss_type in {"readout_ranking", "readout_budget_ranking"}
        need_page_logits_for_block_loss = block_loss_type in {"readout_ranking", "readout_budget_ranking"}
        need_page_logits_for_exact_block_loss = exact_block_supervision_enabled
        need_block_logits_for_exact_block_loss = exact_block_supervision_enabled
        need_retained_page_counts_for_block_loss = block_loss_type == "readout_budget_ranking"
        need_retained_page_counts_for_page_loss = page_loss_type in {"budget_ranking", "readout_budget_ranking"}
        need_retained_page_counts = (
            need_retained_page_counts_for_page_loss
            or need_retained_page_counts_for_block_loss
            or multi_page_coverage_enabled
            or single_page_focus_enabled
            or single_page_retained_enabled
            or retained_set_enabled
        )

        page_targets, target_stats = _build_dma_page_targets(batch_meta, sample_image_counts)
        aux_stats["visible_positive_pages"] = int(target_stats["num_visible_positive_pages"])
        aux_stats["target_pages"] = int(target_stats["num_target_pages"])
        for meta, num_images in zip(batch_meta, sample_image_counts):
            sample_meta = meta if isinstance(meta, dict) else {}
            _, exact_stats = _extract_exact_target_block_ids(
                sample_meta,
                num_images=int(num_images),
                visible_only=False,
            )
            _, visible_exact_stats = _extract_exact_target_block_ids(
                sample_meta,
                num_images=int(num_images),
                visible_only=True,
            )
            aux_stats["target_exact_blocks"] += int(exact_stats["num_target_exact_blocks"])
            aux_stats["target_exact_pages"] += int(exact_stats["num_target_exact_pages"])
            aux_stats["visible_target_exact_blocks"] += int(visible_exact_stats["num_visible_target_exact_blocks"])
            aux_stats["visible_target_exact_pages"] += int(visible_exact_stats["num_visible_target_exact_pages"])
            aux_stats["document_layout_samples"] += int(exact_stats["has_document_layout"])
        block_logits = None
        if (
            need_block_logits_for_page_loss
            or block_supervision_enabled
            or single_page_focus_enabled
            or retained_block_set_enabled
            or need_block_logits_for_exact_block_loss
        ):
            block_logits = collect_dma_training_block_logits(model, num_images=sum(sample_image_counts))
        page_logits = None
        if (
            page_supervision_enabled
            or need_page_logits_for_block_loss
            or multi_page_coverage_enabled
            or single_page_retained_enabled
            or retained_set_enabled
            or need_page_logits_for_exact_block_loss
        ):
            page_logits = collect_dma_training_page_logits(model, num_images=sum(sample_image_counts))
        selected_ratios = None
        if need_retained_page_counts or exact_block_supervision_enabled:
            selected_ratios = collect_dma_training_selected_ratios(model, num_images=sum(sample_image_counts))
        sample_retained_page_counts = None
        if need_retained_page_counts:
            budget_ratio_override = None
            if float(getattr(dma_cfg, "aux_page_budget_ratio", 0.0)) > 0.0:
                budget_ratio_override = float(getattr(dma_cfg, "aux_page_budget_ratio", 0.0))
            elif float(getattr(dma_cfg, "retained_page_budget_ratio", 0.0)) > 0.0:
                budget_ratio_override = float(getattr(dma_cfg, "retained_page_budget_ratio", 0.0))
            sample_retained_page_counts = _infer_sample_retained_page_counts(
                selected_ratios,
                sample_image_counts,
                budget_ratio_override=budget_ratio_override,
            )
        page_focus_scores = _derive_page_focus_scores(block_logits)

        # 先用零权重锚点把当前 step 里开启的读出头挂进图，避免 rank 间因为“本 batch 没命中证据”而出现不同的参数参与集合。
        weighted_aux_terms.extend(_build_dma_runtime_zero_anchors(model, dma_cfg))
        if page_logits is not None:
            weighted_aux_terms.append(page_logits.sum() * 0.0)
        if block_logits is not None:
            weighted_aux_terms.append(block_logits.sum() * 0.0)
        if exact_block_supervision_enabled:
            exact_block_anchor = _build_zero_grad_anchor_from_module(resolve_exact_block_scorer(model))
            if exact_block_anchor is not None:
                weighted_aux_terms.append(exact_block_anchor)
                aux_stats["exact_block_zero_anchor"] = 1

        if int(target_stats["num_visible_positive_pages"]) > 0:
            if page_supervision_enabled:
                loss_type = page_loss_type
                aux_stats["aux_loss_type"] = loss_type
                if (
                    page_logits is not None
                    and int(page_targets.numel()) == int(page_logits.numel())
                    and loss_type in {"ranking", "budget_ranking", "readout_ranking", "readout_budget_ranking"}
                ):
                    if loss_type in {"readout_ranking", "readout_budget_ranking"}:
                        ranking_loss, ranking_stats = _compute_dma_page_readout_ranking_loss(
                            page_logits=page_logits,
                            page_focus_scores=page_focus_scores,
                            batch_meta=batch_meta,
                            sample_image_counts=sample_image_counts,
                            margin=float(getattr(dma_cfg, "aux_page_margin", 0.5)),
                            ranking_topk=int(getattr(dma_cfg, "aux_page_ranking_topk", 2)),
                            sample_retained_page_counts=sample_retained_page_counts,
                            focus_loss_weight=float(getattr(dma_cfg, "aux_page_focus_loss_weight", 0.25)),
                        )
                    else:
                        ranking_loss, ranking_stats = _compute_dma_page_ranking_loss(
                            page_logits=page_logits,
                            batch_meta=batch_meta,
                            sample_image_counts=sample_image_counts,
                            margin=float(getattr(dma_cfg, "aux_page_margin", 0.5)),
                            ranking_topk=int(getattr(dma_cfg, "aux_page_ranking_topk", 2)),
                            sample_retained_page_counts=sample_retained_page_counts,
                        )
                    aux_stats["ranking_samples"] = int(ranking_stats["ranking_samples"])
                    aux_stats["ranking_pairs"] = int(ranking_stats["ranking_pairs"])
                    aux_stats["readout_focus_samples"] = int(ranking_stats.get("readout_focus_samples", 0))
                    aux_stats["readout_focus_pairs"] = int(ranking_stats.get("readout_focus_pairs", 0))
                    if ranking_loss is not None:
                        aux_stats["aux_page_loss"] = float(ranking_loss.detach().item())
                        weighted_aux_terms.append(float(dma_cfg.aux_page_loss_weight) * ranking_loss)
                elif page_logits is not None and int(page_targets.numel()) == int(page_logits.numel()):
                    page_targets = page_targets.to(device=page_logits.device, dtype=page_logits.dtype)
                    negative_count = max(1, int(page_targets.numel()) - int(target_stats["num_visible_positive_pages"]))
                    pos_weight = torch.tensor(
                        [negative_count / max(1, int(target_stats["num_visible_positive_pages"]))],
                        device=page_logits.device,
                        dtype=page_logits.dtype,
                    )
                    page_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        page_logits,
                        page_targets,
                        pos_weight=pos_weight,
                    )
                    aux_stats["aux_page_loss"] = float(page_bce_loss.detach().item())
                    weighted_aux_terms.append(float(dma_cfg.aux_page_loss_weight) * page_bce_loss)

            if single_page_focus_enabled:
                single_page_focus_loss, single_page_stats = _compute_dma_single_page_focus_loss(
                    page_focus_scores=page_focus_scores,
                    batch_meta=batch_meta,
                    sample_image_counts=sample_image_counts,
                    margin=float(getattr(dma_cfg, "aux_single_page_margin", 0.5)),
                    sample_retained_page_counts=sample_retained_page_counts,
                    ranking_topk=int(getattr(dma_cfg, "aux_page_ranking_topk", 1)),
                )
                aux_stats["single_positive_samples"] = int(single_page_stats["single_positive_samples"])
                aux_stats["single_positive_pairs"] = int(single_page_stats["single_positive_pairs"])
                if single_page_focus_loss is not None:
                    aux_stats["aux_single_page_focus_loss"] = float(single_page_focus_loss.detach().item())
                    weighted_aux_terms.append(
                        float(getattr(dma_cfg, "aux_single_page_focus_loss_weight", 0.0)) * single_page_focus_loss
                    )

            if page_logits is not None and single_page_retained_enabled:
                single_page_retained_loss, single_page_retained_stats = _compute_dma_single_page_retained_recovery_loss(
                    page_logits=page_logits,
                    batch_meta=batch_meta,
                    sample_image_counts=sample_image_counts,
                    sample_retained_page_counts=sample_retained_page_counts,
                    margin=float(getattr(dma_cfg, "aux_single_page_retained_margin", 0.5)),
                    negative_topk=int(getattr(dma_cfg, "aux_retained_set_negative_topk", 0)),
                )
                aux_stats["single_page_retained_samples"] = int(single_page_retained_stats["single_page_retained_samples"])
                aux_stats["single_page_retained_pairs"] = int(single_page_retained_stats["single_page_retained_pairs"])
                aux_stats["single_page_positive_in_topk"] = int(single_page_retained_stats["single_page_positive_in_topk"])
                if single_page_retained_loss is not None:
                    aux_stats["aux_single_page_retained_loss"] = float(single_page_retained_loss.detach().item())
                    weighted_aux_terms.append(
                        float(getattr(dma_cfg, "aux_single_page_retained_loss_weight", 0.0)) * single_page_retained_loss
                    )

            if page_logits is not None and multi_page_coverage_enabled:
                multi_page_loss, multi_page_stats = _compute_dma_multi_page_coverage_loss(
                    page_logits=page_logits,
                    batch_meta=batch_meta,
                    sample_image_counts=sample_image_counts,
                    margin=float(getattr(dma_cfg, "aux_multi_page_margin", 0.5)),
                    sample_retained_page_counts=sample_retained_page_counts,
                    ranking_topk=int(getattr(dma_cfg, "aux_page_ranking_topk", 1)),
                )
                aux_stats["multi_positive_samples"] = int(multi_page_stats["multi_positive_samples"])
                aux_stats["multi_positive_pages"] = int(multi_page_stats["multi_positive_pages"])
                aux_stats["multi_page_pairs"] = int(multi_page_stats["multi_page_pairs"])
                if multi_page_loss is not None:
                    aux_stats["aux_multi_page_loss"] = float(multi_page_loss.detach().item())
                    weighted_aux_terms.append(float(getattr(dma_cfg, "aux_multi_page_loss_weight", 0.0)) * multi_page_loss)

            if page_logits is not None and retained_set_enabled:
                retained_set_loss, retained_set_stats = _compute_dma_page_retained_set_loss(
                    page_logits=page_logits,
                    batch_meta=batch_meta,
                    sample_image_counts=sample_image_counts,
                    sample_retained_page_counts=sample_retained_page_counts,
                    temperature=float(getattr(dma_cfg, "aux_retained_set_temperature", 0.25)),
                    negative_weight=float(getattr(dma_cfg, "aux_retained_set_negative_weight", 0.25)),
                    negative_topk=int(getattr(dma_cfg, "aux_retained_set_negative_topk", 0)),
                )
                aux_stats["retained_set_samples"] = int(retained_set_stats["retained_set_samples"])
                aux_stats["retained_set_positive_pages"] = int(retained_set_stats["retained_set_positive_pages"])
                if retained_set_loss is not None:
                    aux_stats["aux_retained_set_loss"] = float(retained_set_loss.detach().item())
                    weighted_aux_terms.append(
                        float(getattr(dma_cfg, "aux_retained_set_loss_weight", 0.0)) * retained_set_loss
                    )

            if block_supervision_enabled:
                if block_logits is not None:
                    if block_loss_type in {"readout_ranking", "readout_budget_ranking"} and page_logits is not None:
                        block_loss, block_stats = _compute_dma_block_readout_ranking_loss(
                            page_logits=page_logits,
                            block_logits=block_logits,
                            batch_meta=batch_meta,
                            sample_image_counts=sample_image_counts,
                            loss_type=block_loss_type,
                            margin=float(getattr(dma_cfg, "aux_block_margin", 0.2)),
                            topk=int(getattr(dma_cfg, "aux_block_topk", 2)),
                            retained_block_topk=int(getattr(dma_cfg, "retained_block_topk", 2)),
                            sample_retained_page_counts=sample_retained_page_counts,
                        )
                    else:
                        block_loss, block_stats = _compute_dma_block_focus_loss(
                            block_logits=block_logits,
                            batch_meta=batch_meta,
                            sample_image_counts=sample_image_counts,
                            loss_type=block_loss_type,
                            margin=float(getattr(dma_cfg, "aux_block_margin", 0.2)),
                            topk=int(getattr(dma_cfg, "aux_block_topk", 2)),
                            retained_block_topk=int(getattr(dma_cfg, "retained_block_topk", 2)),
                        )
                    aux_stats["visible_positive_blocks"] = int(block_stats["visible_positive_blocks"])
                    aux_stats["block_supervision_pages"] = int(block_stats["block_supervision_pages"])
                    aux_stats["block_ranking_pages"] = int(block_stats["block_ranking_pages"])
                    aux_stats["block_ranking_pairs"] = int(block_stats["block_ranking_pairs"])
                    aux_stats["block_focus_pages"] = int(block_stats["block_focus_pages"])
                    aux_stats["block_readout_pages"] = int(block_stats.get("block_readout_pages", 0))
                    aux_stats["block_readout_pairs"] = int(block_stats.get("block_readout_pairs", 0))
                    if block_loss is not None:
                        aux_stats["aux_block_loss"] = float(block_loss.detach().item())
                        weighted_aux_terms.append(float(getattr(dma_cfg, "aux_block_loss_weight", 0.0)) * block_loss)

            if block_logits is not None and retained_block_set_enabled:
                retained_block_set_loss, retained_block_set_stats = _compute_dma_block_retained_set_loss(
                    block_logits=block_logits,
                    batch_meta=batch_meta,
                    sample_image_counts=sample_image_counts,
                    retained_block_topk=int(getattr(dma_cfg, "retained_block_topk", 2)),
                    temperature=float(getattr(dma_cfg, "aux_retained_block_set_temperature", 0.25)),
                    negative_weight=float(getattr(dma_cfg, "aux_retained_block_set_negative_weight", 0.25)),
                    negative_topk=int(getattr(dma_cfg, "aux_retained_block_set_negative_topk", 0)),
                )
                aux_stats["visible_positive_blocks"] = max(
                    int(aux_stats["visible_positive_blocks"] or 0),
                    int(retained_block_set_stats["visible_positive_blocks"]),
                )
                aux_stats["retained_block_set_pages"] = int(retained_block_set_stats["retained_block_set_pages"])
                aux_stats["retained_block_set_blocks"] = int(retained_block_set_stats["retained_block_set_blocks"])
                if retained_block_set_loss is not None:
                    aux_stats["aux_retained_block_set_loss"] = float(retained_block_set_loss.detach().item())
                    weighted_aux_terms.append(
                        float(getattr(dma_cfg, "aux_retained_block_set_loss_weight", 0.0)) * retained_block_set_loss
                    )

            if exact_block_supervision_enabled:
                exact_block_loss, exact_block_stats = _compute_dma_exact_block_loss(
                    model=model,
                    processor=processor,
                    batch_meta=batch_meta,
                    batch_images=batch_images,
                    sample_image_counts=sample_image_counts,
                    page_logits=page_logits,
                    block_logits=block_logits,
                    selected_ratios=selected_ratios,
                    sample_retained_page_counts=sample_retained_page_counts,
                    exact_block_topk=int(getattr(dma_cfg, "exact_block_topk", 4)),
                    temperature=float(getattr(dma_cfg, "exact_block_temperature", 0.25)),
                    negative_weight=float(getattr(dma_cfg, "exact_block_negative_weight", 0.25)),
                    negative_topk=int(getattr(dma_cfg, "exact_block_negative_topk", 0)),
                    loss_type=str(getattr(dma_cfg, "exact_block_loss_type", "soft_topk_bce")),
                    margin=float(getattr(dma_cfg, "exact_block_margin", 0.2)),
                    lexical_dropout=float(getattr(dma_cfg, "exact_block_lexical_dropout", 0.0)),
                    distill_weight=float(getattr(dma_cfg, "exact_block_distill_loss_weight", 0.0)),
                    distill_temperature=float(getattr(dma_cfg, "exact_block_distill_temperature", 1.0)),
                    scorer_type=str(getattr(dma_cfg, "exact_block_scorer_type", "page_interaction")),
                    vlm_rerank_topm=int(getattr(dma_cfg, "exact_block_vlm_rerank_topm", 8)),
                    vlm_crop_expand_ratio=float(getattr(dma_cfg, "exact_block_vlm_crop_expand_ratio", 0.05)),
                )
                aux_stats["exact_block_supervision_samples"] = int(exact_block_stats["exact_block_supervision_samples"])
                aux_stats["exact_block_candidates"] = int(exact_block_stats["exact_block_candidates"])
                aux_stats["exact_block_positive_blocks"] = int(exact_block_stats["exact_block_positive_blocks"])
                aux_stats["exact_block_negative_blocks"] = int(exact_block_stats["exact_block_negative_blocks"])
                aux_stats["exact_block_supervision_samples_vniah"] = int(
                    exact_block_stats["exact_block_supervision_samples_vniah"]
                )
                aux_stats["exact_block_supervision_samples_vmqar"] = int(
                    exact_block_stats["exact_block_supervision_samples_vmqar"]
                )
                aux_stats["exact_block_candidates_vniah"] = int(exact_block_stats["exact_block_candidates_vniah"])
                aux_stats["exact_block_candidates_vmqar"] = int(exact_block_stats["exact_block_candidates_vmqar"])
                aux_stats["exact_block_positive_blocks_vniah"] = int(
                    exact_block_stats["exact_block_positive_blocks_vniah"]
                )
                aux_stats["exact_block_positive_blocks_vmqar"] = int(
                    exact_block_stats["exact_block_positive_blocks_vmqar"]
                )
                aux_stats["exact_block_teacher_samples"] = int(exact_block_stats.get("exact_block_teacher_samples", 0) or 0)
                aux_stats["exact_block_teacher_blocks"] = int(exact_block_stats.get("exact_block_teacher_blocks", 0) or 0)
                aux_stats["aux_exact_block_distill_loss"] = exact_block_stats.get("exact_block_distill_loss")
                if exact_block_loss is not None:
                    aux_stats["aux_exact_block_loss"] = float(exact_block_loss.detach().item())
                    weighted_aux_terms.append(float(getattr(dma_cfg, "aux_exact_block_loss_weight", 0.0)) * exact_block_loss)

    aux_loss = None if not weighted_aux_terms else torch.stack(weighted_aux_terms).sum()
    if aux_loss is not None:
        aux_stats["aux_loss"] = float(aux_loss.detach().item())
    total_loss = lm_loss if aux_loss is None else (lm_loss + aux_loss)
    aux_stats["total_loss"] = float(total_loss.detach().item())

    # 训练诊断：观测 exact block scorer 参数是否真的在动（避免“看起来跑了，但头没学到东西”）。
    scorer = resolve_exact_block_scorer(model)
    if scorer is not None:
        try:
            param_chunks = [param.detach().float().reshape(-1) for param in scorer.parameters()]
            if param_chunks:
                current_param_vector = torch.cat(param_chunks, dim=0)
                aux_stats["exact_block_proj_weight_l2"] = float(current_param_vector.norm().item())
                init_param_vector = getattr(scorer, "_dma_init_param_vector", None)
                if not isinstance(init_param_vector, torch.Tensor):
                    for submodule in scorer.modules():
                        candidate_init_vector = getattr(submodule, "_dma_init_param_vector", None)
                        if isinstance(candidate_init_vector, torch.Tensor):
                            init_param_vector = candidate_init_vector
                            break
                if (
                    isinstance(init_param_vector, torch.Tensor)
                    and tuple(init_param_vector.shape) == tuple(current_param_vector.shape)
                ):
                    aux_stats["exact_block_proj_delta_l2"] = float(
                        (current_param_vector - init_param_vector.detach().float()).norm().item()
                    )
        except Exception:
            # 诊断项不应影响训练主流程
            pass
    return total_loss, aux_stats


def _build_dataloader(
    *,
    examples: list,
    image_max_pixels: int | None,
    processor: Any,
    system_prompt: str,
    batch_size: int,
    shuffle: bool,
    world_size: int,
    rank: int,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = SftDataset(examples, image_max_pixels)
    collator = partial(_collate_fn, processor, system_prompt)
    sampler: DistributedSampler | None = None
    if int(world_size) > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=False,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        # NCCL 进程组建立后再 fork dataloader worker 容易把首轮 barrier/collective 弄僵。
        # 这里先收成主线程加载，配合“首步预取 + barrier”统一首轮节奏，优先保住 8 卡稳定性。
        num_workers=0,
        collate_fn=collator,
    )
    return dataloader, sampler


@torch.inference_mode()
def _evaluate_loss(
    *,
    model,
    processor,
    dataloader,
    accelerator: Accelerator,
    max_batches: int | None,
    dma_cfg: DMAConfig,
    grounding_cfg: Any | None = None,
    sync_dir: Path | None = None,
    sync_prefix: str = "val",
) -> float | None:
    model_was_training = model.training
    model.eval()
    loss_sum = 0.0
    loss_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        model_batch, batch_meta, sample_image_counts = _split_model_batch(batch)
        total_loss, _ = _compute_total_loss(
            model=model,
            processor=processor,
            model_batch=model_batch,
            batch_meta=batch_meta,
            sample_image_counts=sample_image_counts,
            dma_cfg=dma_cfg,
            grounding_cfg=grounding_cfg,
        )
        loss_sum += float(total_loss.detach().float().item())
        loss_count += 1
    if model_was_training:
        model.train()
    world_size = int(getattr(accelerator, "num_processes", 1))
    rank = int(getattr(accelerator, "process_index", 0))
    if world_size > 1:
        if sync_dir is None:
            raise ValueError("多卡验证需要提供 sync_dir。")
        records = _collect_rank_payloads_via_files(
            barrier_dir=sync_dir / sync_prefix,
            stage=sync_prefix,
            payload={"loss_sum": float(loss_sum), "loss_count": int(loss_count)},
            world_size=world_size,
            rank=rank,
        )
        loss_sum = float(sum(float(record.get("loss_sum", 0.0)) for record in records))
        loss_count = int(sum(int(record.get("loss_count", 0)) for record in records))
    if loss_count == 0:
        return None
    return loss_sum / loss_count


def _save_adapter_snapshot(
    *,
    model,
    processor,
    accelerator: Accelerator,
    save_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not accelerator.is_main_process:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    if metadata:
        dump_json(save_dir / "metadata.json", metadata)


def _maybe_run_validation(
    *,
    model,
    processor,
    val_dataloader,
    accelerator: Accelerator,
    step: int,
    elapsed_s: float,
    log_path: Path,
    eval_every: int,
    max_eval_batches: int | None,
    dma_cfg: DMAConfig,
    grounding_cfg: Any | None,
    best_val_loss: float | None,
    best_val_dir: Path | None,
    save_best_on_val: bool,
    sync_dir: Path | None = None,
) -> float | None:
    if val_dataloader is None or eval_every <= 0 or step % eval_every != 0:
        return best_val_loss
    val_loss = _evaluate_loss(
        model=model,
        processor=processor,
        dataloader=val_dataloader,
        accelerator=accelerator,
        max_batches=max_eval_batches,
        dma_cfg=dma_cfg,
        grounding_cfg=grounding_cfg,
        sync_dir=sync_dir,
        sync_prefix=f"val_step_{int(step):06d}",
    )
    if accelerator.is_main_process and val_loss is not None:
        append_jsonl(
            log_path,
            {
                "step": step,
                "split": "val",
                "loss": float(val_loss),
                "elapsed_s": elapsed_s,
            },
        )
        if save_best_on_val and best_val_dir is not None and (best_val_loss is None or float(val_loss) < float(best_val_loss)):
            _save_adapter_snapshot(
                model=model,
                processor=processor,
                accelerator=accelerator,
                save_dir=best_val_dir,
                metadata={
                    "step": int(step),
                    "selected_by": "val_loss",
                    "val_loss": float(val_loss),
                    "elapsed_s": float(elapsed_s),
                },
            )
            return float(val_loss)
    return best_val_loss


def _evaluate_final_val_loss(
    *,
    model,
    processor,
    val_dataloader,
    accelerator: Accelerator,
    max_eval_batches: int | None,
    dma_cfg: DMAConfig,
    grounding_cfg: Any | None = None,
    sync_dir: Path | None = None,
) -> float | None:
    if val_dataloader is None:
        return None
    return _evaluate_loss(
        model=model,
        processor=processor,
        dataloader=val_dataloader,
        accelerator=accelerator,
        max_batches=max_eval_batches,
        dma_cfg=dma_cfg,
        grounding_cfg=grounding_cfg,
        sync_dir=sync_dir,
        sync_prefix="val_final",
    )


def _wait_for_rank_sync(
    *,
    accelerator: Accelerator,
    barrier_root: Path | None,
    stage: str,
) -> None:
    world_size = int(getattr(accelerator, "num_processes", 1))
    if world_size <= 1:
        return
    if barrier_root is None:
        accelerator.wait_for_everyone()
        return
    _wait_for_file_barrier(
        barrier_dir=barrier_root / stage,
        stage=stage,
        world_size=world_size,
        rank=int(getattr(accelerator, "process_index", 0)),
    )


def _shutdown_distributed_if_needed() -> None:
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_path="../../configs", config_name="train_sft")
def main(cfg: DictConfig) -> None:
    out_dir = ensure_dir(cfg.output.dir)
    out_dir_path = Path(out_dir)
    OmegaConf.save(cfg, out_dir_path / "config.yaml")
    startup_trace_path = out_dir_path / "startup_trace.jsonl"
    _append_startup_trace(startup_trace_path, "config_saved", output_dir=str(out_dir_path))

    accelerator = Accelerator(gradient_accumulation_steps=int(cfg.train.grad_accum_steps))
    _append_startup_trace(startup_trace_path, "accelerator_ready", grad_accum_steps=int(cfg.train.grad_accum_steps))
    torch.manual_seed(int(cfg.train.seed))
    _append_startup_trace(startup_trace_path, "seed_ready", seed=int(cfg.train.seed))

    dma_cfg = DMAConfig()
    if "dma" in cfg:
        dma_cfg = DMAConfig(**OmegaConf.to_container(cfg.dma, resolve=True))
    _append_startup_trace(
        startup_trace_path,
        "dma_cfg_ready",
        apply_to_vision=bool(dma_cfg.apply_to_vision),
        query_routing=bool(dma_cfg.vision_query_routing_enable),
        block_feedback=bool(dma_cfg.vision_block_feedback_enable),
        exact_block=bool(dma_cfg.exact_block_readout_enable),
    )

    cross_cfg = CrossAttnConfig()
    if "cross_attn" in cfg:
        cross_cfg = CrossAttnConfig(**OmegaConf.to_container(cfg.cross_attn, resolve=True))
    _append_startup_trace(startup_trace_path, "cross_cfg_ready", cross_attn=bool(cross_cfg.enable))

    _append_startup_trace(startup_trace_path, "before_model_load", model_path=str(cfg.model.path))
    processor, model = _load_model_and_processor_for_training(
        model_path=str(cfg.model.path),
        model_dtype=str(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
        peft_enable=bool(cfg.peft.enable),
        peft_cfg=cfg.peft,
        dma_cfg=dma_cfg,
        cross_cfg=cross_cfg,
    )
    _append_startup_trace(startup_trace_path, "after_model_load")

    _configure_training_memory_optimizations(model, cfg, dma_cfg)
    _append_startup_trace(
        startup_trace_path,
        "after_memory_optimizations",
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        gradient_checkpointing_use_reentrant=bool(cfg.train.gradient_checkpointing_use_reentrant),
    )
    dma_mask_only_param_names = _configure_dma_mask_only_training(model, dma_cfg)
    dma_runtime_only_param_names = [] if dma_mask_only_param_names else _configure_dma_runtime_only_training(model, dma_cfg)
    if dma_mask_only_param_names and _dma_runtime_only_enabled(dma_cfg):
        raise ValueError("不能同时开启 dma.mask_modules_only_enable 与 dma.runtime_modules_only_enable")
    if (dma_mask_only_param_names or dma_runtime_only_param_names) and _exact_block_block_only_enabled(dma_cfg):
        raise ValueError("不能同时开启 dma.mask_modules_only_enable 与 dma.exact_block_block_only_enable")
    block_only_trainable_param_names = (
        []
        if (dma_mask_only_param_names or dma_runtime_only_param_names)
        else _configure_block_only_reranker_training(model, dma_cfg)
    )
    grounded_phase = normalize_grounding_training_phase(getattr(cfg.grounding, "training_phase", "final"))
    frozen_grounded_param_names = _apply_grounded_freeze_policy(
        model,
        structured_output_mode=str(getattr(cfg.grounding, "structured_output_mode", "plain")),
        training_phase=grounded_phase,
        freeze_retrieval=bool(getattr(cfg.grounding, "freeze_retrieval", False)),
    )
    _append_startup_trace(
        startup_trace_path,
        "after_trainable_param_config",
        trainable_tensor_count=len([param for param in model.parameters() if param.requires_grad]),
        block_only_reranker=bool(block_only_trainable_param_names),
        dma_mask_only=bool(dma_mask_only_param_names),
        dma_runtime_only=bool(dma_runtime_only_param_names),
        grounding_training_phase=grounded_phase,
        grounding_freeze_retrieval=bool(getattr(cfg.grounding, "freeze_retrieval", False)),
        grounded_frozen_param_count=len(frozen_grounded_param_names),
    )
    if accelerator.device.type == "cuda":
        if int(getattr(accelerator, "num_processes", 1)) > 1:
            torch.cuda.set_device(accelerator.device)
        first_param = next(model.parameters(), None)
        before_device = str(first_param.device) if first_param is not None else "unknown"
        # 单进程训练也必须显式迁移到 Accelerator 设备；否则 device_map=null 加载会留在 CPU。
        model = model.to(accelerator.device)
        first_param = next(model.parameters(), None)
        after_device = str(first_param.device) if first_param is not None else "unknown"
        _append_startup_trace(
            startup_trace_path,
            "after_model_to_device",
            device=str(accelerator.device),
            before_device=before_device,
            after_device=after_device,
        )

    _append_startup_trace(startup_trace_path, "before_train_jsonl_load", train_jsonl=str(cfg.data.train_jsonl))
    examples = load_sft_jsonl(
        jsonl_path=str(cfg.data.train_jsonl),
        image_root=str(cfg.data.image_root),
        max_images=int(cfg.data.max_images),
        system_prompt=str(cfg.text.system_prompt),
        structured_output_mode=str(getattr(cfg.get("grounding"), "structured_output_mode", "plain")),
        training_phase=str(getattr(cfg.get("grounding"), "training_phase", "final")),
        page_selection_manifest_jsonl=(
            None
            if str(cfg.data.get("train_page_selection_manifest_jsonl") or "").strip() == ""
            else str(cfg.data.get("train_page_selection_manifest_jsonl"))
        ),
    )
    _append_startup_trace(startup_trace_path, "after_train_jsonl_load", num_train_examples=len(examples))
    if len(examples) == 0:
        raise ValueError(f"训练集为空: {cfg.data.train_jsonl}")
    world_size = int(getattr(accelerator, "num_processes", 1))
    process_rank = int(getattr(accelerator, "process_index", 0))
    train_dl, train_sampler = _build_dataloader(
        examples=examples,
        image_max_pixels=cfg.data.image_max_pixels,
        processor=processor,
        system_prompt=str(cfg.text.system_prompt),
        batch_size=int(cfg.train.per_device_batch_size),
        shuffle=True,
        world_size=world_size,
        rank=process_rank,
        seed=int(cfg.train.seed),
    )
    _append_startup_trace(
        startup_trace_path,
        "after_train_dataloader_build",
        manual_distributed_sampler=bool(train_sampler is not None),
    )
    val_examples = None
    val_dl = None
    val_sampler = None
    if cfg.data.get("val_jsonl") is not None and str(cfg.data.val_jsonl).strip() != "":
        _append_startup_trace(startup_trace_path, "before_val_jsonl_load", val_jsonl=str(cfg.data.val_jsonl))
        val_examples = load_sft_jsonl(
            jsonl_path=str(cfg.data.val_jsonl),
            image_root=str(cfg.data.image_root),
            max_images=int(cfg.data.max_images),
            system_prompt=str(cfg.text.system_prompt),
            structured_output_mode=str(getattr(cfg.get("grounding"), "structured_output_mode", "plain")),
            training_phase=str(getattr(cfg.get("grounding"), "training_phase", "final")),
            page_selection_manifest_jsonl=(
                None
                if str(cfg.data.get("val_page_selection_manifest_jsonl") or "").strip() == ""
                else str(cfg.data.get("val_page_selection_manifest_jsonl"))
            ),
        )
        _append_startup_trace(startup_trace_path, "after_val_jsonl_load", num_val_examples=len(val_examples))
        if len(val_examples) == 0:
            raise ValueError(f"验证集为空: {cfg.data.val_jsonl}")
        val_dl, val_sampler = _build_dataloader(
            examples=val_examples,
            image_max_pixels=cfg.data.image_max_pixels,
            processor=processor,
            system_prompt=str(cfg.text.system_prompt),
            batch_size=int(cfg.train.per_device_batch_size),
            shuffle=False,
            world_size=world_size,
            rank=process_rank,
            seed=int(cfg.train.seed),
        )
        _append_startup_trace(
            startup_trace_path,
            "after_val_dataloader_build",
            manual_distributed_sampler=bool(val_sampler is not None),
        )

    optimizer_params = [param for param in model.parameters() if param.requires_grad]
    if not optimizer_params:
        raise ValueError("当前配置下没有可训练参数，无法创建 optimizer")
    adamw_runtime = _resolve_adamw_runtime_kwargs(cfg.train, dma_cfg)
    optimizer = AdamW(
        optimizer_params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        **adamw_runtime["optimizer_kwargs"],
    )
    total_steps = int(cfg.train.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.train.warmup_steps), total_steps)
    _append_startup_trace(
        startup_trace_path,
        "after_optimizer_scheduler",
        optimizer_param_count=len(optimizer_params),
        total_steps=total_steps,
        adamw_foreach=adamw_runtime["adamw_foreach"],
        adamw_fused=adamw_runtime["adamw_fused"],
        adamw_runtime_reason=adamw_runtime["adamw_runtime_reason"],
    )

    has_val = val_dl is not None
    _append_startup_trace(startup_trace_path, "before_prepare_model", has_val=has_val)
    if accelerator.device.type == "cuda" and int(getattr(accelerator, "num_processes", 1)) > 1:
        exact_block_manual_sync_param_names = _collect_exact_block_trainable_param_names(model, dma_cfg)
        modules_to_save_manual_sync_param_names = _collect_modules_to_save_trainable_param_names(model)
        all_trainable_param_names = _collect_all_trainable_param_names(model)
        all_trainable_param_numel = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
        manual_sync_param_names = _merge_manual_sync_param_names(
            modules_to_save_manual_sync_param_names,
            exact_block_manual_sync_param_names,
        )
        parallel_training_mode = _resolve_parallel_training_mode(
            cfg.train,
            dma_cfg,
            world_size=int(getattr(accelerator, "num_processes", 1)),
            manual_sync_param_names=manual_sync_param_names,
            all_trainable_param_names=all_trainable_param_names,
            all_trainable_param_numel=all_trainable_param_numel,
        )
        manual_data_parallel_all_trainable = bool(parallel_training_mode["manual_data_parallel_all_trainable"])
        if manual_data_parallel_all_trainable:
            manual_sync_param_names = list(all_trainable_param_names)
            # 这条路径已经完全绕开 DDP reducer；继续用 NCCL 做 flat all-reduce 仍可能把未完成的
            # collective 留到 optimizer 前的 CUDA synchronize 才暴露出来。小参数集下直接走文件同步更稳。
            use_file_manual_grad_sync = True
            _append_startup_trace(
                startup_trace_path,
                "after_prepare_model",
                manual_ddp=False,
                manual_data_parallel_all_trainable=True,
                parallel_mode_reason=parallel_training_mode["parallel_mode_reason"],
                manual_sync_param_count=len(manual_sync_param_names),
                manual_sync_param_numel=all_trainable_param_numel,
                manual_sync_param_numel_mb=round(float(all_trainable_param_numel) * 2.0 / 1024.0 / 1024.0, 4),
                manual_modules_to_save_params=len(modules_to_save_manual_sync_param_names),
                manual_exact_block_params=len(exact_block_manual_sync_param_names),
                file_manual_grad_sync=use_file_manual_grad_sync,
            )
        else:
            use_file_manual_grad_sync = bool(modules_to_save_manual_sync_param_names) and bool(
                cfg.train.gradient_checkpointing_use_reentrant
            )
            ddp_graph_mode = _resolve_ddp_graph_mode(
                cfg.train,
                dma_cfg,
                exact_block_manual_sync_enabled=bool(exact_block_manual_sync_param_names),
                modules_to_save_manual_sync_enabled=bool(modules_to_save_manual_sync_param_names),
                gradient_checkpointing_use_reentrant=bool(cfg.train.gradient_checkpointing_use_reentrant),
            )
            ddp_static_graph = bool(ddp_graph_mode["ddp_static_graph"])
            ddp_find_unused_parameters = bool(ddp_graph_mode["ddp_find_unused_parameters"])
            _mark_ddp_ignored_param_names(model, manual_sync_param_names)
            ddp_init_sync = bool(getattr(cfg.train, "ddp_init_sync", False))
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[accelerator.local_process_index],
                output_device=accelerator.local_process_index,
                broadcast_buffers=False,
                # block/page/exact-block 辅助监督会按样本命中情况动态缺失。
                # 若显式启用 static graph，则交给 DDP 的 static-graph 路径处理；否则保持 unused 参数探测。
                find_unused_parameters=ddp_find_unused_parameters,
                static_graph=ddp_static_graph,
                init_sync=ddp_init_sync,
            )
            _append_startup_trace(
                startup_trace_path,
                "after_prepare_model",
                manual_ddp=True,
                ddp_init_sync=ddp_init_sync,
                ddp_find_unused_parameters=ddp_find_unused_parameters,
                ddp_static_graph=ddp_static_graph,
                ddp_graph_mode_reason=ddp_graph_mode["ddp_graph_mode_reason"],
                ddp_ignored_manual_sync_params=len(manual_sync_param_names),
                ddp_ignored_modules_to_save_params=len(modules_to_save_manual_sync_param_names),
                ddp_ignored_exact_block_params=len(exact_block_manual_sync_param_names),
                file_manual_grad_sync=use_file_manual_grad_sync,
            )
    else:
        exact_block_manual_sync_param_names = []
        modules_to_save_manual_sync_param_names = []
        manual_sync_param_names = []
        manual_data_parallel_all_trainable = False
        use_file_manual_grad_sync = False
        model = accelerator.prepare_model(model, device_placement=False)
        _append_startup_trace(startup_trace_path, "after_prepare_model", manual_ddp=False)
    optimizer = accelerator.prepare_optimizer(optimizer)
    _append_startup_trace(startup_trace_path, "after_prepare_optimizer")
    _append_startup_trace(startup_trace_path, "after_prepare_train_dataloader", manual_dataloader=True)
    if has_val:
        _append_startup_trace(startup_trace_path, "after_prepare_val_dataloader", manual_dataloader=True)
    scheduler = accelerator.prepare_scheduler(scheduler)
    _append_startup_trace(startup_trace_path, "after_prepare_scheduler")

    log_path = out_dir_path / "train_log.jsonl"
    val_metrics_path = out_dir_path / "val_metrics.json"
    _append_startup_trace(startup_trace_path, "before_train_log_setup")
    if accelerator.is_main_process:
        if log_path.exists():
            log_path.unlink()
        if val_metrics_path.exists():
            val_metrics_path.unlink()
        append_jsonl(
            log_path,
            {
                "step": -1,
                "split": "train_setup",
                "block_only_reranker": bool(block_only_trainable_param_names),
                "trainable_param_names": block_only_trainable_param_names,
                "grounded_frozen_param_names": frozen_grounded_param_names,
                "num_trainable_tensors": len(optimizer_params),
            },
        )
    if world_size > 1:
        _append_startup_trace(startup_trace_path, "before_train_log_barrier", barrier_type="file")
        _wait_for_file_barrier(
            barrier_dir=out_dir_path / "barriers" / "train_log_ready",
            stage="train_log_ready",
            world_size=world_size,
            rank=process_rank,
        )
        _append_startup_trace(startup_trace_path, "after_train_log_barrier", barrier_type="file")

    eval_every = int(cfg.train.get("eval_every", 0) or 0)
    max_eval_batches = None if cfg.train.get("max_eval_batches") is None else int(cfg.train.max_eval_batches)
    save_every = int(cfg.output.get("save_every", 0) or 0)
    save_best_on_val = bool(cfg.output.get("save_best_on_val", True))
    checkpoints_dir = out_dir_path / "checkpoints"
    best_val_dir = out_dir_path / "best_val"
    best_val_loss: float | None = None

    step = 0
    train_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iterator = iter(train_dl)
    prefetched_batch = None
    if world_size > 1:
        _append_startup_trace(startup_trace_path, "before_step0_prefetch")
        prefetched_batch = next(train_iterator)
        _append_startup_trace(startup_trace_path, "after_step0_prefetch")
        _append_startup_trace(startup_trace_path, "before_step0_prefetch_barrier", barrier_type="file")
        _wait_for_file_barrier(
            barrier_dir=out_dir_path / "barriers" / "step0_prefetch",
            stage="step0_prefetch",
            world_size=world_size,
            rank=process_rank,
        )
        _append_startup_trace(startup_trace_path, "after_step0_prefetch_barrier", barrier_type="file")
    model.train()
    t0 = time.time()
    debug_cuda_sync = str(os.environ.get("ECCV26_DEBUG_CUDA_SYNC", "0")).strip().lower() in {"1", "true", "yes", "on"}
    _append_startup_trace(startup_trace_path, "before_train_loop")
    while step < total_steps:
        if prefetched_batch is not None:
            batch = prefetched_batch
            prefetched_batch = None
        else:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(train_epoch)
                train_iterator = iter(train_dl)
                continue

        if step == 0:
            batch_size = None
            batch_meta = batch.get("batch_meta") if isinstance(batch, dict) else None
            if isinstance(batch_meta, list):
                batch_size = len(batch_meta)
            _append_startup_trace(startup_trace_path, "step0_batch_ready", batch_size=batch_size)
        with accelerator.accumulate(model):
            model_batch, batch_meta, sample_image_counts = _split_model_batch(batch)
            if step == 0:
                _append_startup_trace(
                    startup_trace_path,
                    "step0_after_split_model_batch",
                    sample_image_count=sum(int(x) for x in sample_image_counts),
                )
            loss, aux_stats = _compute_total_loss(
                model=model,
                processor=processor,
                model_batch=model_batch,
                batch_meta=batch_meta,
                sample_image_counts=sample_image_counts,
                dma_cfg=dma_cfg,
                grounding_cfg=cfg.grounding,
            )
            if step == 0:
                _append_startup_trace(
                    startup_trace_path,
                    "step0_after_forward_loss",
                    loss=float(loss.detach().float().item()),
                )
            if step == 0 and world_size > 1:
                _append_startup_trace(startup_trace_path, "before_step0_backward_barrier", barrier_type="file")
                _wait_for_file_barrier(
                    barrier_dir=out_dir_path / "barriers" / "step0_backward_ready",
                    stage="step0_backward_ready",
                    world_size=world_size,
                    rank=process_rank,
                )
                _append_startup_trace(startup_trace_path, "after_step0_backward_barrier", barrier_type="file")
            accelerator.backward(loss)
            if step == 0:
                _append_startup_trace(startup_trace_path, "step0_after_backward")
            if manual_sync_param_names:
                if accelerator.device.type == "cuda":
                    if step == 0:
                        _append_startup_trace(startup_trace_path, "step0_before_manual_grad_cuda_sync")
                    # manual sync 直接读取 parameter.grad；这里先等 CUDA 把 backward 尾部真正做完，
                    # 避免 rank 间有人还在写梯度、有人已经开始同步，导致后续 NCCL/文件路径都卡住。
                    torch.cuda.synchronize(device=accelerator.device)
                    if step == 0:
                        _append_startup_trace(startup_trace_path, "step0_after_manual_grad_cuda_sync")
                if step == 0:
                    _append_startup_trace(
                        startup_trace_path,
                        "step0_before_manual_grad_sync",
                        manual_sync_param_count=len(manual_sync_param_names),
                        manual_sync_backend="file" if use_file_manual_grad_sync else "dist",
                    )
                if use_file_manual_grad_sync:
                    synced_grad_tensors = _sync_named_parameter_gradients_via_files(
                        model,
                        manual_sync_param_names,
                        barrier_dir=out_dir_path / "barriers",
                        stage=f"manual_grad_sync_step_{int(step):06d}",
                        world_size=world_size,
                        rank=process_rank,
                    )
                else:
                    synced_grad_tensors = _sync_named_parameter_gradients(model, manual_sync_param_names)
                if step == 0:
                    _append_startup_trace(
                        startup_trace_path,
                        "step0_after_manual_grad_sync",
                        synced_grad_tensors=synced_grad_tensors,
                    )
                _append_startup_trace(startup_trace_path, "step0_before_optimizer_step")
            if debug_cuda_sync and accelerator.device.type == "cuda":
                torch.cuda.synchronize(device=accelerator.device)
                if step == 0:
                    _append_startup_trace(startup_trace_path, "step0_after_pre_optimizer_cuda_sync")
            optimizer.step()
            if step == 0:
                _append_startup_trace(startup_trace_path, "step0_after_optimizer_step")
            if debug_cuda_sync and accelerator.device.type == "cuda":
                torch.cuda.synchronize(device=accelerator.device)
                if step == 0:
                    _append_startup_trace(startup_trace_path, "step0_after_post_optimizer_cuda_sync")
            scheduler.step()
            optimizer.zero_grad()

            elapsed = time.time() - t0
            if step % int(cfg.train.log_every) == 0:
                logged_loss = float(loss.detach().float().item())
                if world_size > 1:
                    records = _collect_rank_payloads_via_files(
                        barrier_dir=out_dir_path / "barriers" / "train_loss" / f"step_{int(step):06d}",
                        stage=f"train_loss_step_{int(step):06d}",
                        payload={"loss": float(logged_loss)},
                        world_size=world_size,
                        rank=process_rank,
                    )
                    logged_loss = float(sum(float(record.get("loss", 0.0)) for record in records) / len(records))
                if accelerator.is_main_process:
                    rec = {
                        "step": step,
                        "split": "train",
                        "loss": float(logged_loss),
                        "elapsed_s": elapsed,
                    }
                    for key in (
                        "lm_loss",
                        "grounding_phase",
                        "pref_loss",
                        "pref_beta",
                        "chosen_logp",
                        "rejected_logp",
                        "preference_margin",
                        "aux_loss",
                        "aux_page_loss",
                        "aux_block_loss",
                        "aux_exact_block_loss",
                        "aux_exact_block_distill_loss",
                        "aux_single_page_focus_loss",
                        "aux_single_page_retained_loss",
                        "aux_multi_page_loss",
                        "aux_retained_set_loss",
                        "aux_retained_block_set_loss",
                        "total_loss",
                        "visible_positive_pages",
                        "target_pages",
                        "target_exact_blocks",
                        "target_exact_pages",
                        "visible_target_exact_blocks",
                        "visible_target_exact_pages",
                        "document_layout_samples",
                        "single_positive_samples",
                        "single_positive_pairs",
                        "single_page_retained_samples",
                        "single_page_retained_pairs",
                        "single_page_positive_in_topk",
                        "multi_positive_samples",
                        "multi_positive_pages",
                        "multi_page_pairs",
                        "retained_set_samples",
                        "retained_set_positive_pages",
                        "visible_positive_blocks",
                        "block_supervision_pages",
                        "block_ranking_pages",
                        "block_ranking_pairs",
                        "block_focus_pages",
                        "block_readout_pages",
                        "block_readout_pairs",
                        "retained_block_set_pages",
                        "retained_block_set_blocks",
                        "exact_block_supervision_samples",
                        "exact_block_candidates",
                        "exact_block_positive_blocks",
                        "exact_block_negative_blocks",
                        "exact_block_supervision_samples_vniah",
                        "exact_block_supervision_samples_vmqar",
                        "exact_block_candidates_vniah",
                        "exact_block_candidates_vmqar",
                        "exact_block_positive_blocks_vniah",
                        "exact_block_positive_blocks_vmqar",
                        "exact_block_teacher_samples",
                        "exact_block_teacher_blocks",
                        "exact_block_zero_anchor",
                        "exact_block_proj_weight_l2",
                        "exact_block_proj_delta_l2",
                        "ranking_samples",
                        "ranking_pairs",
                        "readout_focus_samples",
                        "readout_focus_pairs",
                    ):
                        if key in aux_stats and aux_stats[key] is not None:
                            rec[key] = aux_stats[key]
                    append_jsonl(log_path, rec)
            if save_every > 0 and (step + 1) % save_every == 0:
                save_step = int(step + 1)
                _save_adapter_snapshot(
                    model=model,
                    processor=processor,
                    accelerator=accelerator,
                    save_dir=checkpoints_dir / f"step_{save_step:06d}",
                    metadata={
                        "step": save_step,
                        "selected_by": "save_every",
                        "elapsed_s": float(elapsed),
                    },
                )
            best_val_loss = _maybe_run_validation(
                model=model,
                processor=processor,
                val_dataloader=val_dl,
                accelerator=accelerator,
                step=step,
                elapsed_s=elapsed,
                log_path=log_path,
                eval_every=eval_every,
                max_eval_batches=max_eval_batches,
                dma_cfg=dma_cfg,
                grounding_cfg=cfg.get("grounding"),
                best_val_loss=best_val_loss,
                best_val_dir=best_val_dir,
                save_best_on_val=save_best_on_val,
                sync_dir=out_dir_path / "barriers",
            )

            step += 1
            if step >= total_steps:
                break

    final_val_loss = _evaluate_final_val_loss(
        model=model,
        processor=processor,
        val_dataloader=val_dl,
        accelerator=accelerator,
        max_eval_batches=max_eval_batches,
        dma_cfg=dma_cfg,
        grounding_cfg=cfg.get("grounding"),
        sync_dir=out_dir_path / "barriers",
    )

    if world_size > 1:
        _append_startup_trace(startup_trace_path, "before_final_val_barrier", barrier_type="file")
    _wait_for_rank_sync(
        accelerator=accelerator,
        barrier_root=out_dir_path / "barriers",
        stage="final_val_ready",
    )
    if world_size > 1:
        _append_startup_trace(startup_trace_path, "after_final_val_barrier", barrier_type="file")
    if accelerator.is_main_process:
        if final_val_loss is not None:
            dump_json(
                val_metrics_path,
                {
                    "num_train_examples": len(examples),
                    "num_val_examples": len(val_examples) if val_dl is not None else 0,
                    "train_exact_target_summary": _summarize_example_exact_target_coverage(examples),
                    "val_exact_target_summary": (
                        {} if val_examples is None else _summarize_example_exact_target_coverage(val_examples)
                    ),
                    "max_eval_batches": max_eval_batches,
                    "val_loss": float(final_val_loss),
                    "best_val_loss": None if best_val_loss is None else float(best_val_loss),
                },
            )
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "split": "val_final",
                    "loss": float(final_val_loss),
                    "elapsed_s": time.time() - t0,
                },
            )
        _save_adapter_snapshot(
            model=model,
            processor=processor,
            accelerator=accelerator,
            save_dir=out_dir_path / "final",
            metadata={
                "step": int(step),
                "selected_by": "final",
                "val_loss": None if final_val_loss is None else float(final_val_loss),
                "best_val_loss": None if best_val_loss is None else float(best_val_loss),
                "elapsed_s": float(time.time() - t0),
            },
        )
    if world_size > 1:
        _append_startup_trace(startup_trace_path, "before_final_save_barrier", barrier_type="file")
    _wait_for_rank_sync(
        accelerator=accelerator,
        barrier_root=out_dir_path / "barriers",
        stage="final_save_ready",
    )
    if world_size > 1:
        _append_startup_trace(startup_trace_path, "after_final_save_barrier", barrier_type="file")
    _shutdown_distributed_if_needed()


if __name__ == "__main__":
    main()
