from __future__ import annotations

import json
import hashlib
import math
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from eccv26.data.benchmarks import LoadSpec, iter_samples
from eccv26.grounding import (
    apply_grounded_output_format,
    build_grounded_output_prefill,
    compute_grounding_hard_eval,
    finalize_grounding_aggregate as _finalize_grounding_aggregate_metrics,
    init_grounding_aggregate as _init_grounding_aggregate_metrics,
    normalize_structured_prediction_text,
    replace_structured_prediction_result,
    should_use_grounded_output,
    update_grounding_aggregate as _update_grounding_aggregate_metrics,
)
from eccv26.metrics.vqa import anls_score, relaxed_numeric_match, vqa_soft_accuracy
from eccv26.model.cross_attn import CrossAttnConfig
from eccv26.model.dma import DMAConfig, score_dma_exact_block_features
from eccv26.model.qwen3vl import GenerateConfig, Qwen3VL, VisualTokenPruningConfig
from eccv26.utils.exact_block import (
    build_exact_block_candidate_rows,
    build_exact_block_page_stats_by_id,
    exact_block_candidate_block_sketches_to_tensor,
    exact_block_candidate_features_to_tensor,
    exact_block_candidate_page_ids_to_tensor,
    exact_block_candidate_query_sketches_to_tensor,
)
from eccv26.utils.exact_block_teacher import (
    VLM_BLOCK_RERANK_SYSTEM_TEXT,
    build_block_rerank_user_text as _shared_build_block_rerank_user_text,
    crop_exact_block_from_page_image as _shared_crop_exact_block_from_page_image,
)
from eccv26.utils.image import resize_to_max_pixels
from eccv26.utils.io import append_jsonl, dump_json, ensure_dir
from eccv26.utils.text import normalize_answer


def _build_user_text(
    context: str | None,
    question: str | None,
    source_kind: str | None,
    dataset: str | None = None,
) -> str:
    if source_kind == "throughput":
        return "请阅读这些页面。只输出 OK。"
    q = (question or "").strip()
    if q == "":
        return "请根据图像内容回答。只输出最终答案。"
    if dataset in {"MMLongBench_DOC", "SLIDEVQA_MINI", "SLIDEVQA"}:
        # 外部长文 VQA 使用 ANLS，解释句会被强惩罚；这里显式约束为短答案抽取。
        if context is None or context.strip() == "":
            return (
                f"Question: {q}\n"
                "Answer with the shortest exact answer span only. "
                "Do not explain, do not repeat the question.\n"
                "Answer:"
            )
        return (
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer with the shortest exact answer span only. "
            "Do not explain, do not repeat the question.\n"
            "Answer:"
        )
    if context is None or context.strip() == "":
        return f"{q}\nAnswer:"
    return f"Context:\n{context}\n\nQuestion:\n{q}\nAnswer:"


def _build_grounded_result_refine_prompt(
    *,
    question: str | None,
    verified_evidence: list[dict[str, Any]],
) -> tuple[str, str] | None:
    snippets: list[str] = []
    for item in verified_evidence:
        if not isinstance(item, dict):
            continue
        snippet = str(
            item.get("matched_snippet_raw")
            or item.get("matched_snippet")
            or item.get("quote")
            or ""
        ).strip()
        if snippet == "":
            continue
        snippets.append(snippet)
    if not snippets:
        return None
    system_text = (
        "你是一个严格的基于证据的答案抽取器。"
        " 只能依据给定证据回答，不要补充常识，不要解释。"
        " 如果问题询问编号、数值、日期、标题、专有名词、缩写、项目号、ISSN、百分比、面积或短语，"
        " 必须尽量从证据中原样复制最短答案，保留大小写、连字符、空格和标点。"
        " 如果证据不足，就输出最短可证实短语。"
        " 只输出最终答案，不要输出 JSON，不要输出引号。"
    )
    evidence_lines = [f"[证据{i}] {text}" for i, text in enumerate(snippets[:3], start=1)]
    user_text = (
        f"问题：{(question or '').strip()}\n\n"
        "证据：\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "请只输出最终答案。"
    )
    return system_text, user_text


def _maybe_refine_grounded_result(
    *,
    model: Qwen3VL,
    question: str | None,
    grounding_hard_eval: dict[str, Any] | None,
    mode: str,
    max_new_tokens: int,
) -> str | None:
    if mode != "quote_extract" or not isinstance(grounding_hard_eval, dict):
        return None
    citation_score = grounding_hard_eval.get("citation_score")
    if citation_score is None or float(citation_score) <= 0.0:
        return None
    prompt = _build_grounded_result_refine_prompt(
        question=question,
        verified_evidence=list(grounding_hard_eval.get("verified_evidence") or []),
    )
    if prompt is None:
        return None
    system_text, user_text = prompt
    refine_out = model.generate_one(
        images=[],
        input_page_ids=None,
        user_text=user_text,
        system_text=system_text,
        gen=GenerateConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            assistant_prefill_text=None,
        ),
    )
    refined = str(refine_out.get("pred") or "").strip()
    return refined or None


def _system_prompt_for_dataset(dataset: str, source_kind: str | None) -> str:
    if source_kind == "throughput":
        return "你是一个严谨的长文档视觉模型。请仅输出 OK，不要输出解释。"
    if dataset == "RealWorldQA":
        return "你是一个严谨的视觉问答模型。请只输出最终选项字母（A/B/C/D/E），不要输出解释。"
    if dataset in {"MMLongBench_DOC", "SLIDEVQA_MINI", "SLIDEVQA"}:
        return (
            "You are a strict document VQA model. "
            "Return only the shortest final answer span. "
            "Do not explain, do not add a full sentence."
        )
    return "你是一个严谨的视觉问答模型。请只输出最终答案，不要输出解释。"


_PAGE_SELECTOR_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PAGE_SELECTOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}


def _resolve_effective_image_max_pixels(cfg: DictConfig) -> int | None:
    """
    统一解析评测时真正生效的图像像素预算。

    dense / Vision-DMA 直接使用 data.image_max_pixels；
    weak sparse baseline 通过 model.sparse_keep_ratio 进一步收紧预算，
    复用同一条评测链路而不引入额外推理分支。
    """
    base_max_pixels = cfg.data.image_max_pixels
    if base_max_pixels is None:
        return None

    keep_ratio = 1.0
    if "model" in cfg and getattr(cfg.model, "sparse_keep_ratio", None) is not None:
        keep_ratio = float(cfg.model.sparse_keep_ratio)
    if keep_ratio <= 0.0:
        raise ValueError(f"model.sparse_keep_ratio 必须 > 0，当前={keep_ratio}")
    if keep_ratio >= 0.999999:
        return int(base_max_pixels)
    return max(1, int(int(base_max_pixels) * keep_ratio))


def _tokenize_for_page_selector(text: str | None) -> list[str]:
    if text is None:
        return []
    tokens: list[str] = []
    for raw in _PAGE_SELECTOR_TOKEN_RE.findall(str(text).lower()):
        token = raw.strip()
        if len(token) <= 1:
            continue
        if token in _PAGE_SELECTOR_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _score_one(dataset: str, pred: str, answers: list[str] | None) -> dict[str, Any]:
    if answers is None or len(answers) == 0:
        return {"scorable": False, "score": None, "metric": None}

    if dataset == "RealWorldQA":
        gt = normalize_answer(answers[0]).upper()
        pd = normalize_answer(pred).upper()
        ok = 1.0 if gt in pd.split() or pd == gt else 0.0
        return {"scorable": True, "score": ok, "metric": "choice_em"}

    if dataset == "OCRBench-v2":
        pred_n = normalize_answer(pred)
        pred_tokens = set(pred_n.split())
        ok = 0.0
        for a in answers:
            a_n = normalize_answer(a)
            if a_n == "":
                continue
            if " " in a_n:
                if a_n in pred_n:
                    ok = 1.0
                    break
            else:
                if a_n in pred_tokens:
                    ok = 1.0
                    break
        return {"scorable": True, "score": ok, "metric": "ocr_contains_em"}

    if dataset == "ChartQA":
        s = relaxed_numeric_match(pred, answers)
        return {"scorable": True, "score": s, "metric": "relaxed_numeric"}

    if dataset in {"SLIDEVQA_MINI", "SLIDEVQA", "MMLongBench_DOC"}:
        s = anls_score(pred, answers)
        return {"scorable": True, "score": s, "metric": "anls"}

    s = vqa_soft_accuracy(pred, answers)
    return {"scorable": True, "score": s, "metric": "vqa_soft"}


def _page_id_to_int(value: Any) -> int | None:
    try:
        page_id = int(value)
    except (TypeError, ValueError):
        return None
    return page_id if page_id > 0 else None


def _normalize_page_id_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    page_ids: list[int] = []
    for value in values:
        page_id = _page_id_to_int(value)
        if page_id is not None:
            page_ids.append(page_id)
    return sorted(set(page_ids))


def _normalize_block_id_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    block_ids: list[str] = []
    for value in values:
        if value is None:
            continue
        block_id = str(value).strip()
        if block_id != "":
            block_ids.append(block_id)
    return sorted(set(block_ids))


def _parse_page_and_block_part(value: Any) -> tuple[int, str] | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    if not raw.startswith("p") or "_" not in raw:
        return None
    page_part, block_part = raw.split("_", 1)
    page_id = _page_id_to_int(page_part[1:])
    if page_id is None:
        return None
    return page_id, block_part


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_random_block_score(
    *,
    document_json_path: str,
    question: str | None,
    context: str | None,
    block: dict[str, Any],
) -> float:
    """为 random-block 消融生成可复现的伪随机分数。"""

    seed_parts = [
        str(document_json_path),
        str(question or ""),
        str(context or ""),
        str(block.get("page_id", "")),
        str(block.get("block_id", "")),
        str(block.get("order_index", "")),
    ]
    digest = hashlib.sha256("\x1f".join(seed_parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


@lru_cache(maxsize=4096)
def _load_document_page_layouts(document_json_path: str) -> dict[int, dict[str, Any]]:
    path = Path(document_json_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    pages = doc.get("pages")
    if not isinstance(pages, list):
        return {}

    page_layouts: dict[int, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_id = _page_id_to_int(page.get("page_id"))
        if page_id is None:
            continue
        width = _safe_float(page.get("width"))
        height = _safe_float(page.get("height"))
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            blocks = []
        block_layouts: dict[str, dict[str, Any]] = {}
        ordered_block_ids: list[str] = []
        for order_index, block in enumerate(blocks):
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


@lru_cache(maxsize=4096)
def _load_document_page_texts(document_json_path: str) -> dict[int, str]:
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

    page_texts: dict[int, str] = {}
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        page_id = _page_id_to_int(page.get("page_id"))
        if page_id is None:
            continue
        page_text = page.get("page_text")
        if page_text is None:
            page_text = ""
        page_texts[page_id] = str(page_text)
    return page_texts


@lru_cache(maxsize=4096)
def _load_document_page_blocks(document_json_path: str) -> dict[int, dict[str, Any]]:
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

    page_blocks: dict[int, dict[str, Any]] = {}
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
        blocks: list[dict[str, Any]] = []
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
            blocks.append(
                {
                    "block_id": block_id,
                    "type": str(block.get("type") or "").strip(),
                    "bbox": normalized_bbox,
                    "text": str(block.get("text") or ""),
                    "order_index": order_index,
                }
            )
        page_blocks[page_id] = {
            "width": width,
            "height": height,
            "blocks": blocks,
        }
    return page_blocks


def _resolve_document_json_path(meta: dict[str, Any], benchmark_root: str | None) -> str | None:
    raw_path = meta.get("document_json")
    if isinstance(raw_path, str) and raw_path.strip() != "":
        path = Path(raw_path)
        if not path.is_absolute() and benchmark_root is not None:
            path = Path(str(benchmark_root)) / str(path)
        return str(path)
    document_id = meta.get("document_id")
    if benchmark_root is None or not isinstance(document_id, str) or document_id.strip() == "":
        return None
    path = Path(str(benchmark_root)) / "data" / "documents" / f"{document_id}.json"
    return str(path)


def _serialize_coarse_block_id(page_id: int, row_bin: int, col_bin: int) -> str:
    return f"p{page_id}_g{row_bin}{col_bin}"


def _coarse_block_id_from_page_layout(
    page_id: int,
    raw_block_id: str,
    *,
    page_layouts: dict[int, dict[str, Any]] | None,
    grid_size: int = 3,
) -> str | None:
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
        return _serialize_coarse_block_id(page_id, row_bin, col_bin)

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
    row_bin = coarse_index // grid_size
    col_bin = coarse_index % grid_size
    return _serialize_coarse_block_id(page_id, row_bin, col_bin)


def _normalize_coarse_block_id(value: Any, *, page_layouts: dict[int, dict[str, Any]] | None = None) -> str | None:
    parsed = _parse_page_and_block_part(value)
    if parsed is None:
        return None
    page_id, block_part = parsed
    if block_part.startswith("g") and len(block_part) == 3:
        row_char = block_part[1]
        col_char = block_part[2]
        if row_char in "012" and col_char in "012":
            return f"p{page_id}_g{row_char}{col_char}"
        return None
    if not block_part.startswith("b"):
        return None
    coarse_block_id = _coarse_block_id_from_page_layout(
        page_id,
        f"p{page_id}_{block_part}",
        page_layouts=page_layouts,
    )
    if coarse_block_id is not None:
        return coarse_block_id
    try:
        block_number = int(block_part[1:])
    except ValueError:
        return None
    if not 1 <= block_number <= 9:
        return None
    block_index = block_number - 1
    row_bin = block_index // 3
    col_bin = block_index % 3
    return _serialize_coarse_block_id(page_id, row_bin, col_bin)


def _normalize_coarse_block_id_list(
    values: Any,
    *,
    page_layouts: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    if not isinstance(values, list):
        return []
    coarse_block_ids: list[str] = []
    for value in values:
        coarse_block_id = _normalize_coarse_block_id(value, page_layouts=page_layouts)
        if coarse_block_id is not None:
            coarse_block_ids.append(coarse_block_id)
    return sorted(set(coarse_block_ids))


def _resolve_input_page_ids(meta: dict[str, Any], *, num_input_pages: int) -> list[int]:
    input_page_ids = _normalize_page_id_list(meta.get("_input_page_ids"))
    if len(input_page_ids) >= num_input_pages:
        return input_page_ids[:num_input_pages]
    return list(range(1, int(num_input_pages) + 1))


def _ranked_page_ids_from_external_prediction(row: dict[str, Any]) -> list[int]:
    ranked = []
    seen: set[int] = set()
    raw_ranked = row.get("ranked_page_ids")
    if isinstance(raw_ranked, list):
        for value in raw_ranked:
            try:
                page_id = int(value)
            except (TypeError, ValueError):
                continue
            if page_id in seen:
                continue
            ranked.append(page_id)
            seen.add(page_id)
    if ranked:
        return ranked
    scores = row.get("scores")
    if not isinstance(scores, list):
        return []
    scored_pages: list[tuple[float, int]] = []
    for item in scores:
        if not isinstance(item, dict):
            continue
        try:
            page_id = int(item.get("page_id"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        scored_pages.append((score, page_id))
    scored_pages.sort(key=lambda item: (-item[0], item[1]))
    return [page_id for _, page_id in scored_pages]


def _load_external_page_predictions(path: str | None) -> dict[str, list[int]]:
    if path is None or str(path).strip() == "":
        return {}
    pred_path = Path(str(path))
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到外部页检索结果: {pred_path}")

    rows: list[dict[str, Any]] = []
    if pred_path.suffix.lower() == ".jsonl":
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        payload = json.loads(pred_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            rows = [row for row in payload["predictions"] if isinstance(row, dict)]
        else:
            raise ValueError(f"不支持的外部页检索结果格式: {pred_path}")

    predictions: dict[str, list[int]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        if sample_id == "":
            continue
        ranked = _ranked_page_ids_from_external_prediction(row)
        if ranked:
            predictions[sample_id] = ranked
    return predictions


def _select_input_pages_with_external_selector(
    *,
    images: list[Any],
    meta: dict[str, Any],
    sample_id: str,
    predictions: dict[str, list[int]],
    top_k: int,
    fallback_max_images: int,
) -> tuple[list[Any], list[int], dict[str, Any]]:
    capped_top_k = max(1, int(top_k))
    fallback_k = max(1, int(fallback_max_images))
    candidate_page_ids = _resolve_input_page_ids(meta, num_input_pages=len(images))
    image_by_page_id = {
        page_id: image
        for page_id, image in zip(candidate_page_ids[: len(images)], images, strict=False)
    }
    ranked = list(predictions.get(str(sample_id)) or [])
    selected_page_ids: list[int] = []
    missing_page_ids: list[int] = []
    for page_id in ranked:
        if page_id in selected_page_ids:
            continue
        if page_id not in image_by_page_id:
            missing_page_ids.append(page_id)
            continue
        selected_page_ids.append(page_id)
        if len(selected_page_ids) >= capped_top_k:
            break

    used = bool(selected_page_ids)
    reason = None
    if not selected_page_ids:
        reason = "missing_external_prediction" if str(sample_id) not in predictions else "no_valid_external_page"
        selected_page_ids = candidate_page_ids[: min(fallback_k, len(candidate_page_ids), len(images))]

    selected_images = [image_by_page_id[page_id] for page_id in selected_page_ids if page_id in image_by_page_id]
    return selected_images, selected_page_ids, {
        "type": "external_manifest",
        "used": used,
        "reason": reason,
        "top_k": capped_top_k,
        "order": "retrieval_rank",
        "num_ranked_pages": len(ranked),
        "selected_page_ids": selected_page_ids,
        "missing_page_ids_preview": missing_page_ids[:10],
    }


def _select_input_pages_with_lexical_selector(
    *,
    images: list[Any],
    meta: dict[str, Any],
    question: str | None,
    context: str | None,
    max_images: int,
    benchmark_root: str | None,
) -> tuple[list[Any], list[int], dict[str, Any]]:
    """
    用简单的 query-page lexical overlap 做非训练式 top-k 页选择。

    这条线的目的不是追求最强，而是提供一个真正会“选页”的弱基线，
    以区别于只压像素预算、但不改变输入页集合的弱稀疏基线。
    """
    capped_max_images = max(1, int(max_images))
    candidate_page_ids = _resolve_input_page_ids(meta, num_input_pages=len(images))
    if len(images) <= capped_max_images:
        return list(images), candidate_page_ids[: len(images)], {
            "type": "lexical",
            "used": False,
            "reason": "num_images_leq_budget",
        }

    document_json_path = _resolve_document_json_path(meta, benchmark_root)
    if document_json_path is None:
        return list(images[:capped_max_images]), candidate_page_ids[:capped_max_images], {
            "type": "lexical",
            "used": False,
            "reason": "missing_document_json",
        }

    page_texts = _load_document_page_texts(document_json_path)
    if not page_texts:
        return list(images[:capped_max_images]), candidate_page_ids[:capped_max_images], {
            "type": "lexical",
            "used": False,
            "reason": "missing_page_text",
        }

    query_tokens = _tokenize_for_page_selector(question) + _tokenize_for_page_selector(context)
    if not query_tokens:
        return list(images[:capped_max_images]), candidate_page_ids[:capped_max_images], {
            "type": "lexical",
            "used": False,
            "reason": "empty_query_tokens",
        }

    available_page_ids = candidate_page_ids[: len(images)]
    page_token_counts: dict[int, Counter[str]] = {}
    doc_freq: Counter[str] = Counter()
    for page_id in available_page_ids:
        tokens = _tokenize_for_page_selector(page_texts.get(page_id))
        counts = Counter(tokens)
        page_token_counts[page_id] = counts
        for token in counts.keys():
            doc_freq[token] += 1

    num_pages = max(1, len(available_page_ids))
    query_counts = Counter(query_tokens)
    ranked: list[tuple[float, int, int]] = []
    for order_idx, page_id in enumerate(available_page_ids):
        token_counts = page_token_counts.get(page_id, Counter())
        score = 0.0
        for token, q_tf in query_counts.items():
            p_tf = token_counts.get(token, 0)
            if p_tf <= 0:
                continue
            idf = math.log((1.0 + num_pages) / (1.0 + float(doc_freq.get(token, 0)))) + 1.0
            score += float(q_tf) * min(3.0, float(p_tf)) * idf
        ranked.append((score, page_id, order_idx))

    top_ranked = sorted(ranked, key=lambda item: (-item[0], item[2]))[:capped_max_images]
    if len(top_ranked) == 0 or top_ranked[0][0] <= 0.0:
        return list(images[:capped_max_images]), candidate_page_ids[:capped_max_images], {
            "type": "lexical",
            "used": False,
            "reason": "all_zero_scores",
        }

    selected_page_ids = {page_id for _, page_id, _ in top_ranked}
    selected_indices = [idx for idx, page_id in enumerate(available_page_ids) if page_id in selected_page_ids]
    selected_images = [images[idx] for idx in selected_indices]
    ordered_selected_page_ids = [available_page_ids[idx] for idx in selected_indices]
    top_scores = [
        {"page_id": page_id, "score": round(score, 4)}
        for score, page_id, _ in sorted(top_ranked, key=lambda item: (-item[0], item[2]))
    ]
    return selected_images, ordered_selected_page_ids, {
        "type": "lexical",
        "used": True,
        "reason": "ok",
        "selected_page_ids": ordered_selected_page_ids,
        "top_scores": top_scores[: min(8, len(top_scores))],
    }


def _resolve_source_page_ids_for_block_selector(
    *,
    input_page_ids: list[int],
    model_out: dict[str, Any] | None,
    source_page_mode: str,
) -> list[int]:
    normalized_mode = str(source_page_mode or "retained").strip().lower()
    input_page_ids = list(input_page_ids)
    if normalized_mode == "input":
        return input_page_ids
    if model_out is None:
        return input_page_ids
    if normalized_mode in {"retained_plus_input", "retained_input", "hybrid"}:
        retained_page_ids = _normalize_page_id_list(model_out.get("retained_page_ids"))
        if retained_page_ids:
            # A5 hybrid：block 阶段从 PBC-DMA retained pages 与输入页的并集选块。
            # 按 input_page_ids 排序，避免集合去重导致页序不稳定。
            hybrid_page_set = set(retained_page_ids) | set(input_page_ids)
            ordered = [page_id for page_id in input_page_ids if page_id in hybrid_page_set]
            if ordered:
                return ordered
        return input_page_ids
    if normalized_mode == "selector_input":
        selector_page_ids = _normalize_page_id_list(model_out.get("selector_page_ids"))
        if selector_page_ids:
            selector_page_set = set(selector_page_ids)
            ordered = [page_id for page_id in input_page_ids if page_id in selector_page_set]
            if ordered:
                return ordered
        return input_page_ids

    retained_page_ids = _normalize_page_id_list(model_out.get("retained_page_ids"))
    if retained_page_ids:
        retained_page_set = set(retained_page_ids)
        ordered = [page_id for page_id in input_page_ids if page_id in retained_page_set]
        if ordered:
            return ordered
    return input_page_ids


def _crop_exact_block_from_page_image(
    *,
    page_image: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    page_width: float | None,
    page_height: float | None,
    expand_ratio: float,
) -> Image.Image | None:
    return _shared_crop_exact_block_from_page_image(
        page_image=page_image,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        expand_ratio=expand_ratio,
    )


def _pack_block_crops_into_canvases(
    block_crops: list[Image.Image],
    *,
    max_blocks_per_canvas: int,
    padding: int = 16,
    max_block_edge: int = 768,
) -> list[Image.Image]:
    if not block_crops:
        return []
    canvas_crops: list[Image.Image] = []
    capped_max_blocks = max(1, int(max_blocks_per_canvas))
    for start_idx in range(0, len(block_crops), capped_max_blocks):
        chunk = block_crops[start_idx : start_idx + capped_max_blocks]
        prepared: list[Image.Image] = []
        max_width = 1
        total_height = padding
        for crop in chunk:
            current = crop.convert("RGB")
            w, h = current.size
            longest_edge = max(w, h)
            if longest_edge > max_block_edge:
                scale = float(max_block_edge) / float(longest_edge)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                current = current.resize((new_w, new_h), resample=Image.BICUBIC)
                w, h = current.size
            prepared.append(current)
            max_width = max(max_width, w)
            total_height += h + padding
        canvas = Image.new("RGB", (max_width + 2 * padding, total_height), color=(255, 255, 255))
        cursor_y = padding
        for crop in prepared:
            offset_x = padding + max(0, (max_width - crop.size[0]) // 2)
            canvas.paste(crop, (offset_x, cursor_y))
            cursor_y += crop.size[1] + padding
        canvas_crops.append(canvas)
    return canvas_crops


def _build_block_pack_user_text(
    base_user_text: str,
    *,
    block_texts: list[str] | None = None,
    text_max_chars_per_block: int = 800,
) -> str:
    prefix = "下面给出的图像是从候选页面中裁剪出的高相关证据块。请优先依据这些证据块回答问题。"
    evidence_texts: list[str] = []
    max_chars = max(0, int(text_max_chars_per_block))
    for idx, text in enumerate(block_texts or [], start=1):
        normalized = str(text or "").strip()
        if normalized == "":
            continue
        if max_chars > 0 and len(normalized) > max_chars:
            normalized = normalized[:max_chars].rstrip() + "..."
        evidence_texts.append(f"[候选证据块{idx}] {normalized}")
    if evidence_texts:
        prefix = (
            f"{prefix}\n"
            "下面同时附上这些候选证据块的解析文本；这些文本来自当前 selector 选中的 block，"
            "不代表标注答案。回答时必须只抽取问题所需的最短答案。"
            f"\n{chr(10).join(evidence_texts)}"
        )
    return f"{prefix}\n\n{base_user_text}"


def _build_block_rerank_user_text(base_user_text: str, block_text: str | None) -> str:
    return _shared_build_block_rerank_user_text(base_user_text, block_text)


def _build_exact_block_stage_inputs(
    *,
    model,
    vlm_reranker=None,
    images: list[Any],
    input_page_ids: list[int],
    meta: dict[str, Any],
    question: str | None,
    context: str | None,
    benchmark_root: str | None,
    model_out: dict[str, Any] | None,
    selector_type: str,
    source_page_mode: str,
    block_topk: int,
    crop_expand_ratio: float,
    pack_max_blocks_per_canvas: int,
    pack_enable: bool,
    vlm_rerank_topm: int = 16,
    include_block_text: bool = False,
    include_block_ids_in_text: bool = False,
    block_text_max_chars_per_block: int = 800,
) -> tuple[list[Image.Image], dict[str, Any]]:
    normalized_selector_type = str(selector_type or "none").strip().lower()
    if normalized_selector_type == "none":
        return [], {"type": "none", "used": False, "reason": "disabled"}
    if normalized_selector_type not in {"lexical", "model", "vlm_rerank", "random"}:
        return [], {"type": normalized_selector_type, "used": False, "reason": "unsupported_selector"}

    document_json_path = _resolve_document_json_path(meta, benchmark_root)
    if document_json_path is None:
        return [], {"type": normalized_selector_type, "used": False, "reason": "missing_document_json"}

    image_by_page_id = {
        page_id: image.convert("RGB") if isinstance(image, Image.Image) else image
        for page_id, image in zip(input_page_ids, images)
    }
    source_page_ids = _resolve_source_page_ids_for_block_selector(
        input_page_ids=input_page_ids,
        model_out=model_out,
        source_page_mode=source_page_mode,
    )
    page_stats_by_id = build_exact_block_page_stats_by_id(
        None if model_out is None else model_out.get("retained_page_stats")
    )
    candidate_blocks = build_exact_block_candidate_rows(
        document_json_path=document_json_path,
        source_page_ids=source_page_ids,
        query_texts=[question, context],
        page_stats_by_id=page_stats_by_id,
    )
    if not candidate_blocks:
        return [], {"type": normalized_selector_type, "used": False, "reason": "no_candidate_blocks"}

    ranked_blocks = list(candidate_blocks)
    if normalized_selector_type == "model":
        candidate_features = exact_block_candidate_features_to_tensor(candidate_blocks)
        candidate_page_ids = exact_block_candidate_page_ids_to_tensor(candidate_blocks)
        candidate_query_sketches = exact_block_candidate_query_sketches_to_tensor(candidate_blocks)
        candidate_block_sketches = exact_block_candidate_block_sketches_to_tensor(candidate_blocks)
        candidate_logits = score_dma_exact_block_features(
            model,
            candidate_features,
            candidate_page_ids,
            candidate_query_sketches,
            candidate_block_sketches,
        )
        if candidate_logits is None or int(candidate_logits.numel()) != len(candidate_blocks):
            return [], {
                "type": "model",
                "used": False,
                "reason": "missing_model_exact_block_readout",
                "source_page_ids": source_page_ids,
            }
        for block, score in zip(ranked_blocks, candidate_logits.detach().cpu().tolist()):
            block["score"] = float(score)
    elif normalized_selector_type == "vlm_rerank":
        if vlm_reranker is None or not hasattr(vlm_reranker, "score_block_relevance_one"):
            return [], {
                "type": "vlm_rerank",
                "used": False,
                "reason": "missing_vlm_reranker",
                "source_page_ids": source_page_ids,
            }
        lexical_prefilter = sorted(
            candidate_blocks,
            key=lambda item: (-float(item.get("lexical_score", 0.0)), int(item["page_rank"]), int(item["order_index"])),
        )[: max(1, int(vlm_rerank_topm))]
        ranked_blocks = []
        base_user_text = _build_user_text(context, question, None)
        for block in lexical_prefilter:
            page_image = image_by_page_id.get(int(block["page_id"]))
            if not isinstance(page_image, Image.Image):
                continue
            crop = _crop_exact_block_from_page_image(
                page_image=page_image,
                bbox=block.get("bbox"),
                page_width=_safe_float(block.get("page_width")),
                page_height=_safe_float(block.get("page_height")),
                expand_ratio=crop_expand_ratio,
            )
            if crop is None:
                continue
            score = vlm_reranker.score_block_relevance_one(
                images=[crop],
                user_text=_build_block_rerank_user_text(base_user_text, str(block.get("text") or "")),
                system_text=VLM_BLOCK_RERANK_SYSTEM_TEXT,
            )
            reranked_block = dict(block)
            reranked_block["score"] = float(score)
            ranked_blocks.append(reranked_block)
        if not ranked_blocks:
            return [], {
                "type": "vlm_rerank",
                "used": False,
                "reason": "no_valid_vlm_rerank_candidates",
                "source_page_ids": source_page_ids,
            }
    elif normalized_selector_type == "random":
        # 这里的 random 只用于消融边界：同一文档/问题/块得到稳定分数，避免复跑抖动。
        for block in ranked_blocks:
            block["score"] = _stable_random_block_score(
                document_json_path=str(document_json_path),
                question=question,
                context=context,
                block=block,
            )
    else:
        for block in ranked_blocks:
            block["score"] = float(block.get("lexical_score", 0.0))
    ranked_blocks.sort(key=lambda item: (-float(item["score"]), int(item["page_rank"]), int(item["order_index"])))
    selected_blocks = ranked_blocks[: max(1, int(block_topk))]
    selected_crops: list[Image.Image] = []
    selected_infos: list[dict[str, Any]] = []
    selected_texts: list[str] = []
    for block in selected_blocks:
        page_image = image_by_page_id.get(int(block["page_id"]))
        if not isinstance(page_image, Image.Image):
            continue
        crop = _crop_exact_block_from_page_image(
            page_image=page_image,
            bbox=block.get("bbox"),
            page_width=_safe_float(block.get("page_width")),
            page_height=_safe_float(block.get("page_height")),
            expand_ratio=crop_expand_ratio,
        )
        if crop is None:
            continue
        selected_crops.append(crop)
        if include_block_text:
            block_text = str(block.get("text") or "").strip()
            max_text_chars = max(0, int(block_text_max_chars_per_block))
            if max_text_chars > 0 and len(block_text) > max_text_chars:
                block_text = block_text[:max_text_chars].rstrip() + "..."
            if include_block_ids_in_text:
                block_text = f"page_id={int(block['page_id'])}, block_id={str(block['block_id'])}: {block_text}"
            selected_texts.append(block_text)
        selected_infos.append(
            {
                "page_id": int(block["page_id"]),
                "block_id": str(block["block_id"]),
                "score": round(float(block["score"]), 4),
                "order_index": int(block["order_index"]),
                "text_chars": len(str(block.get("text") or "")),
            }
        )
    if not selected_crops:
        return [], {
            "type": normalized_selector_type,
            "used": False,
            "reason": "selected_blocks_have_no_valid_crop",
            "source_page_ids": source_page_ids,
        }

    packed_images = (
        _pack_block_crops_into_canvases(
            selected_crops,
            max_blocks_per_canvas=pack_max_blocks_per_canvas,
        )
        if pack_enable
        else selected_crops
    )
    debug = {
        "type": normalized_selector_type,
        "used": True,
        "reason": "ok",
        "source_page_mode": str(source_page_mode),
        "source_page_ids": list(source_page_ids),
        "selected_blocks": selected_infos,
        "exact_retained_block_ids": [item["block_id"] for item in selected_infos],
        "pack_enable": bool(pack_enable),
        "num_block_images": len(packed_images),
        "vlm_rerank_topm": max(1, int(vlm_rerank_topm)) if normalized_selector_type == "vlm_rerank" else None,
    }
    if include_block_text:
        debug["selected_block_texts"] = selected_texts
        debug["block_prompt_text_max_chars_per_block"] = max(0, int(block_text_max_chars_per_block))
    return packed_images, debug


def _compute_evidence_eval(
    meta: dict[str, Any],
    *,
    num_input_pages: int,
    visual_tokens: int,
    model_out: dict[str, Any] | None = None,
    benchmark_root: str | None = None,
) -> dict[str, Any]:
    input_page_ids = _resolve_input_page_ids(meta, num_input_pages=num_input_pages)
    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list) or len(raw_evidence) == 0:
        return {
            "available": False,
            "mode": "unavailable",
            "num_input_pages": len(input_page_ids),
            "input_page_ids": input_page_ids,
            "target_page_ids": [],
            "target_block_ids": [],
            "target_coarse_block_ids": [],
            "visible_target_page_ids": [],
            "visible_target_block_ids": [],
            "visible_target_coarse_block_ids": [],
            "retained_page_ids": [],
            "retained_block_ids": [],
            "exact_retained_block_ids": [],
            "retained_coarse_block_ids": [],
            "evidence_page_ids": [],
            "evidence_block_ids": [],
            "exact_blocks_covered": [],
            "exact_blocks_missed": [],
            "visible_exact_blocks_covered": [],
            "visible_exact_blocks_missed": [],
            "visible_pages_covered": [],
            "visible_pages_missed": [],
            "num_target_pages": 0,
            "num_target_blocks": 0,
            "num_visible_target_pages": 0,
            "num_visible_target_blocks": 0,
            "input_page_recall": None,
            "page_recall": None,
            "page_full_recall": None,
            "visible_page_recall": None,
            "visible_page_full_recall": None,
            "exact_block_recall": None,
            "exact_block_full_recall": None,
            "visible_exact_block_recall": None,
            "visible_exact_block_full_recall": None,
            "block_hit": None,
            "visual_tokens": int(visual_tokens),
        }

    target_page_ids: list[int] = []
    target_block_ids: list[str] = []
    visible_target_block_ids: list[str] = []
    input_page_set = set(input_page_ids)
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        page_id = _page_id_to_int(item.get("page_id"))
        if page_id is not None:
            target_page_ids.append(page_id)
        block_id = item.get("block_id")
        if block_id is not None and str(block_id).strip() != "":
            target_block_ids.append(str(block_id))
            if page_id is not None and page_id in input_page_set:
                visible_target_block_ids.append(str(block_id))

    target_page_ids = sorted(set(target_page_ids))
    target_block_ids = sorted(set(target_block_ids))
    document_json_path = _resolve_document_json_path(meta, benchmark_root)
    page_layouts = {} if document_json_path is None else _load_document_page_layouts(document_json_path)
    target_coarse_block_ids = _normalize_coarse_block_id_list(target_block_ids, page_layouts=page_layouts)
    visible_target_page_ids = sorted(page_id for page_id in target_page_ids if page_id in input_page_set)
    visible_target_block_ids = sorted(set(visible_target_block_ids))
    visible_target_coarse_block_ids = _normalize_coarse_block_id_list(
        visible_target_block_ids,
        page_layouts=page_layouts,
    )

    retained_page_ids = _normalize_page_id_list(None if model_out is None else model_out.get("retained_page_ids"))
    selector_page_ids = _normalize_page_id_list(None if model_out is None else model_out.get("selector_page_ids"))
    retained_block_ids = _normalize_block_id_list(None if model_out is None else model_out.get("retained_block_ids"))
    exact_retained_block_ids = _normalize_block_id_list(
        None if model_out is None else model_out.get("exact_retained_block_ids")
    )
    retained_coarse_block_ids = _normalize_coarse_block_id_list(
        retained_block_ids + exact_retained_block_ids,
        page_layouts=page_layouts,
    )
    mode = "model_retained_pages" if retained_page_ids else "input_pages_fallback"
    if not retained_page_ids and selector_page_ids:
        retained_page_ids = list(selector_page_ids)
        mode = "selector_input_pages"
    if not retained_page_ids:
        retained_page_ids = list(input_page_ids)

    retained_page_set = set(retained_page_ids)
    covered_pages = sorted(page_id for page_id in target_page_ids if page_id in retained_page_set)
    visible_covered_pages = sorted(page_id for page_id in visible_target_page_ids if page_id in retained_page_set)

    input_page_recall: float | None = None
    page_recall: float | None = None
    page_full_recall: float | None = None
    visible_page_recall: float | None = None
    visible_page_full_recall: float | None = None
    if target_page_ids:
        input_page_recall = len(visible_target_page_ids) / len(target_page_ids)
        page_recall = len(covered_pages) / len(target_page_ids)
        page_full_recall = 1.0 if len(covered_pages) == len(target_page_ids) else 0.0
    if visible_target_page_ids:
        visible_page_recall = len(visible_covered_pages) / len(visible_target_page_ids)
        visible_page_full_recall = 1.0 if len(visible_covered_pages) == len(visible_target_page_ids) else 0.0

    exact_retained_block_set = set(exact_retained_block_ids)
    exact_blocks_covered = sorted(block_id for block_id in target_block_ids if block_id in exact_retained_block_set)
    visible_exact_blocks_covered = sorted(
        block_id for block_id in visible_target_block_ids if block_id in exact_retained_block_set
    )
    exact_block_recall: float | None = None
    exact_block_full_recall: float | None = None
    visible_exact_block_recall: float | None = None
    visible_exact_block_full_recall: float | None = None
    if target_block_ids and exact_retained_block_ids:
        exact_block_recall = len(exact_blocks_covered) / len(target_block_ids)
        exact_block_full_recall = 1.0 if len(exact_blocks_covered) == len(target_block_ids) else 0.0
    if visible_target_block_ids and exact_retained_block_ids:
        visible_exact_block_recall = len(visible_exact_blocks_covered) / len(visible_target_block_ids)
        visible_exact_block_full_recall = (
            1.0 if len(visible_exact_blocks_covered) == len(visible_target_block_ids) else 0.0
        )

    block_hit: float | None = None
    if target_coarse_block_ids and retained_coarse_block_ids:
        block_hit = 1.0 if set(target_coarse_block_ids) & set(retained_coarse_block_ids) else 0.0

    return {
        "available": True,
        "mode": mode,
        "num_input_pages": len(input_page_ids),
        "input_page_ids": input_page_ids,
        "target_page_ids": target_page_ids,
        "target_block_ids": target_block_ids,
        "target_coarse_block_ids": target_coarse_block_ids,
        "visible_target_page_ids": visible_target_page_ids,
        "visible_target_block_ids": visible_target_block_ids,
        "visible_target_coarse_block_ids": visible_target_coarse_block_ids,
        "retained_page_ids": retained_page_ids,
        "retained_block_ids": retained_block_ids,
        "exact_retained_block_ids": exact_retained_block_ids,
        "retained_coarse_block_ids": retained_coarse_block_ids,
        "evidence_page_ids": target_page_ids,
        "evidence_block_ids": target_block_ids,
        "exact_blocks_covered": exact_blocks_covered,
        "exact_blocks_missed": [block_id for block_id in target_block_ids if block_id not in exact_retained_block_set],
        "visible_exact_blocks_covered": visible_exact_blocks_covered,
        "visible_exact_blocks_missed": [
            block_id for block_id in visible_target_block_ids if block_id not in exact_retained_block_set
        ],
        "evidence_pages_covered": covered_pages,
        "evidence_pages_missed": [page_id for page_id in target_page_ids if page_id not in retained_page_set],
        "visible_pages_covered": visible_covered_pages,
        "visible_pages_missed": [page_id for page_id in visible_target_page_ids if page_id not in retained_page_set],
        "num_target_pages": len(target_page_ids),
        "num_target_blocks": len(target_block_ids),
        "num_visible_target_pages": len(visible_target_page_ids),
        "num_visible_target_blocks": len(visible_target_block_ids),
        "input_page_recall": input_page_recall,
        "page_recall": page_recall,
        "page_full_recall": page_full_recall,
        "visible_page_recall": visible_page_recall,
        "visible_page_full_recall": visible_page_full_recall,
        "exact_block_recall": exact_block_recall,
        "exact_block_full_recall": exact_block_full_recall,
        "visible_exact_block_recall": visible_exact_block_recall,
        "visible_exact_block_full_recall": visible_exact_block_full_recall,
        "block_hit": block_hit,
        "visual_tokens": int(visual_tokens),
    }


def _init_evidence_aggregate() -> dict[str, float | int]:
    return {
        "num_samples": 0,
        "input_page_recall_sum": 0.0,
        "page_recall_sum": 0.0,
        "page_full_recall_sum": 0.0,
        "exact_block_recall_sum": 0.0,
        "exact_block_full_recall_sum": 0.0,
        "exact_block_count": 0,
        "visible_num_samples": 0,
        "visible_page_recall_sum": 0.0,
        "visible_page_full_recall_sum": 0.0,
        "visible_exact_block_recall_sum": 0.0,
        "visible_exact_block_full_recall_sum": 0.0,
        "visible_exact_block_count": 0,
        "block_hit_sum": 0.0,
        "block_hit_count": 0,
    }


def _update_evidence_aggregate(aggregate: dict[str, float | int], evidence_eval: dict[str, Any]) -> None:
    if not evidence_eval.get("available"):
        return
    input_page_recall = evidence_eval.get("input_page_recall")
    page_recall = evidence_eval.get("page_recall")
    page_full_recall = evidence_eval.get("page_full_recall")
    if input_page_recall is None or page_recall is None or page_full_recall is None:
        return
    aggregate["num_samples"] += 1
    aggregate["input_page_recall_sum"] += float(input_page_recall)
    aggregate["page_recall_sum"] += float(page_recall)
    aggregate["page_full_recall_sum"] += float(page_full_recall)
    visible_page_recall = evidence_eval.get("visible_page_recall")
    visible_page_full_recall = evidence_eval.get("visible_page_full_recall")
    if visible_page_recall is not None and visible_page_full_recall is not None:
        aggregate["visible_num_samples"] += 1
        aggregate["visible_page_recall_sum"] += float(visible_page_recall)
        aggregate["visible_page_full_recall_sum"] += float(visible_page_full_recall)
    exact_block_recall = evidence_eval.get("exact_block_recall")
    exact_block_full_recall = evidence_eval.get("exact_block_full_recall")
    if exact_block_recall is not None and exact_block_full_recall is not None:
        aggregate["exact_block_recall_sum"] += float(exact_block_recall)
        aggregate["exact_block_full_recall_sum"] += float(exact_block_full_recall)
        aggregate["exact_block_count"] += 1
    visible_exact_block_recall = evidence_eval.get("visible_exact_block_recall")
    visible_exact_block_full_recall = evidence_eval.get("visible_exact_block_full_recall")
    if visible_exact_block_recall is not None and visible_exact_block_full_recall is not None:
        aggregate["visible_exact_block_recall_sum"] += float(visible_exact_block_recall)
        aggregate["visible_exact_block_full_recall_sum"] += float(visible_exact_block_full_recall)
        aggregate["visible_exact_block_count"] += 1
    block_hit = evidence_eval.get("block_hit")
    if block_hit is not None:
        aggregate["block_hit_sum"] += float(block_hit)
        aggregate["block_hit_count"] += 1


def _finalize_evidence_aggregate(
    aggregate: dict[str, float | int], *, visual_tokens_mean: float | None
) -> dict[str, Any]:
    evidence_num_samples = int(aggregate["num_samples"])
    evidence_input_page_recall_mean: float | None = None
    evidence_page_recall_mean: float | None = None
    evidence_page_full_recall_mean: float | None = None
    evidence_exact_block_recall_mean: float | None = None
    evidence_exact_block_full_recall_mean: float | None = None
    evidence_visible_num_samples = int(aggregate["visible_num_samples"])
    evidence_visible_page_recall_mean: float | None = None
    evidence_visible_page_full_recall_mean: float | None = None
    evidence_visible_exact_block_recall_mean: float | None = None
    evidence_visible_exact_block_full_recall_mean: float | None = None
    evidence_block_hit_rate: float | None = None
    evidence_efficiency: float | None = None
    evidence_visible_efficiency: float | None = None

    if evidence_num_samples > 0:
        evidence_input_page_recall_mean = float(aggregate["input_page_recall_sum"]) / evidence_num_samples
        evidence_page_recall_mean = float(aggregate["page_recall_sum"]) / evidence_num_samples
        evidence_page_full_recall_mean = float(aggregate["page_full_recall_sum"]) / evidence_num_samples
    if int(aggregate["exact_block_count"]) > 0:
        evidence_exact_block_recall_mean = (
            float(aggregate["exact_block_recall_sum"]) / int(aggregate["exact_block_count"])
        )
        evidence_exact_block_full_recall_mean = (
            float(aggregate["exact_block_full_recall_sum"]) / int(aggregate["exact_block_count"])
        )
    if evidence_visible_num_samples > 0:
        evidence_visible_page_recall_mean = float(aggregate["visible_page_recall_sum"]) / evidence_visible_num_samples
        evidence_visible_page_full_recall_mean = (
            float(aggregate["visible_page_full_recall_sum"]) / evidence_visible_num_samples
        )
    if int(aggregate["visible_exact_block_count"]) > 0:
        evidence_visible_exact_block_recall_mean = (
            float(aggregate["visible_exact_block_recall_sum"]) / int(aggregate["visible_exact_block_count"])
        )
        evidence_visible_exact_block_full_recall_mean = (
            float(aggregate["visible_exact_block_full_recall_sum"]) / int(aggregate["visible_exact_block_count"])
        )
    if int(aggregate["block_hit_count"]) > 0:
        evidence_block_hit_rate = float(aggregate["block_hit_sum"]) / int(aggregate["block_hit_count"])
    if evidence_page_recall_mean is not None and visual_tokens_mean is not None and visual_tokens_mean > 0:
        evidence_efficiency = evidence_page_recall_mean / visual_tokens_mean
    if (
        evidence_visible_page_recall_mean is not None
        and visual_tokens_mean is not None
        and visual_tokens_mean > 0
    ):
        evidence_visible_efficiency = evidence_visible_page_recall_mean / visual_tokens_mean

    return {
        "evidence_num_samples": evidence_num_samples,
        "evidence_input_page_recall_mean": evidence_input_page_recall_mean,
        "evidence_page_recall_mean": evidence_page_recall_mean,
        "evidence_page_full_recall_mean": evidence_page_full_recall_mean,
        "evidence_exact_block_recall_mean": evidence_exact_block_recall_mean,
        "evidence_exact_block_full_recall_mean": evidence_exact_block_full_recall_mean,
        "evidence_visible_num_samples": evidence_visible_num_samples,
        "evidence_visible_page_recall_mean": evidence_visible_page_recall_mean,
        "evidence_visible_page_full_recall_mean": evidence_visible_page_full_recall_mean,
        "evidence_visible_exact_block_recall_mean": evidence_visible_exact_block_recall_mean,
        "evidence_visible_exact_block_full_recall_mean": evidence_visible_exact_block_full_recall_mean,
        "evidence_block_hit_rate": evidence_block_hit_rate,
        "evidence_efficiency": evidence_efficiency,
        "evidence_visible_efficiency": evidence_visible_efficiency,
    }


def _sample_belongs_to_shard(sample_idx: int, *, shard_index: int, num_shards: int) -> bool:
    if num_shards < 1:
        raise ValueError(f"num_shards 必须 >= 1，当前={num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index 必须满足 0 <= shard_index < num_shards，当前={shard_index}, num_shards={num_shards}")
    return (sample_idx % num_shards) == shard_index


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    out_dir = ensure_dir(cfg.output.dir)
    OmegaConf.save(cfg, Path(out_dir) / cfg.output.config_snapshot)

    torch.manual_seed(int(cfg.runtime.seed))

    dma_cfg = DMAConfig()
    if "dma" in cfg:
        dma_cfg = DMAConfig(**OmegaConf.to_container(cfg.dma, resolve=True))

    cross_cfg = CrossAttnConfig()
    if "cross_attn" in cfg:
        cross_cfg = CrossAttnConfig(**OmegaConf.to_container(cfg.cross_attn, resolve=True))

    pruning_cfg = VisualTokenPruningConfig()
    if "visual_token_pruning" in cfg:
        pruning_cfg = VisualTokenPruningConfig(**OmegaConf.to_container(cfg.visual_token_pruning, resolve=True))

    model = Qwen3VL(
        cfg.model.path,
        dtype=str(cfg.model.dtype),
        device_map=cfg.model.device_map,
        attn_implementation=cfg.model.attn_implementation,
        dma=dma_cfg,
        cross_attn=cross_cfg,
        visual_token_pruning=pruning_cfg,
    )

    base_gen_cfg = GenerateConfig(
        max_new_tokens=int(cfg.model.max_new_tokens),
        do_sample=bool(cfg.model.do_sample),
        temperature=float(cfg.model.temperature),
        top_p=float(cfg.model.top_p),
        repetition_penalty=float(getattr(cfg.model, "repetition_penalty", 1.05)),
        assistant_prefill_text=None,
    )
    effective_image_max_pixels = _resolve_effective_image_max_pixels(cfg)
    page_selector_type = str(getattr(cfg.model, "page_selector_type", "none") or "none").strip().lower()
    external_page_predictions: dict[str, list[int]] = {}
    external_page_top_k = max(1, int(getattr(cfg.model, "external_page_top_k", cfg.data.max_images)))
    if page_selector_type in {"external", "external_manifest", "external_page"}:
        external_page_predictions_path = str(
            getattr(cfg.model, "external_page_predictions_path", "") or ""
        ).strip()
        if external_page_predictions_path == "":
            raise ValueError("model.page_selector_type=external_manifest 时必须设置 model.external_page_predictions_path")
        external_page_predictions = _load_external_page_predictions(external_page_predictions_path)
        if not external_page_predictions:
            raise ValueError(f"外部页检索结果为空: {external_page_predictions_path}")
    block_selector_type = str(getattr(cfg.model, "block_selector_type", "none") or "none").strip().lower()
    block_source_page_mode = str(getattr(cfg.model, "block_source_page_mode", "retained") or "retained").strip().lower()
    block_topk = max(1, int(getattr(cfg.model, "block_topk", 4)))
    block_vlm_rerank_topm = max(1, int(getattr(cfg.model, "block_vlm_rerank_topm", 16)))
    block_pack_enable = bool(getattr(cfg.model, "block_pack_enable", False))
    # 默认保持历史行为：只要开启 block selector，就允许二阶段重生成。
    # 若该开关为 true，则仅在 grounded_json 模式下执行 block-stage 重生成；
    # retrieval/block 指标所需的 block 选择仍会执行，不受该开关影响。
    block_stage_generate_for_grounded_only = bool(
        getattr(cfg.model, "block_stage_generate_for_grounded_only", False)
    )
    block_prompt_include_text = bool(getattr(cfg.model, "block_prompt_include_text", False))
    block_prompt_include_ids = bool(getattr(cfg.model, "block_prompt_include_ids", False))
    block_prompt_text_max_chars_per_block = max(
        0,
        int(getattr(cfg.model, "block_prompt_text_max_chars_per_block", 800)),
    )
    block_stage_text_only = bool(getattr(cfg.model, "block_stage_text_only", False))
    block_stage_prefill_page_only = bool(getattr(cfg.model, "block_stage_prefill_page_only", False))
    block_crop_expand_ratio = max(0.0, float(getattr(cfg.model, "block_crop_expand_ratio", 0.06)))
    block_pack_max_blocks_per_canvas = max(1, int(getattr(cfg.model, "block_pack_max_blocks_per_canvas", 6)))

    pred_path = Path(out_dir) / cfg.output.predictions_file
    if pred_path.exists():
        pred_path.unlink()

    summary: dict[str, Any] = {
        "model": str(cfg.model.path),
        "datasets": list(cfg.data.datasets),
        "split": str(cfg.data.split),
        "num_shards": int(cfg.runtime.num_shards),
        "shard_index": int(cfg.runtime.shard_index),
        "results": {},
    }

    for dataset_name in cfg.data.datasets:
        num_shards = int(cfg.runtime.num_shards)
        shard_index = int(cfg.runtime.shard_index)
        max_samples = None if cfg.runtime.max_samples is None else int(cfg.runtime.max_samples)
        spec = LoadSpec(
            benchmark_root=str(cfg.data.benchmark_root),
            dataset=str(dataset_name),
            split=str(cfg.data.split),
            max_samples=None if num_shards > 1 else max_samples,
        )

        n_total = 0
        n_kept = 0
        n_scored = 0
        score_sum = 0.0
        visual_tokens_sum = 0
        t_sum = 0.0
        visual_tokens_before_values: list[int] = []
        visual_tokens_after_values: list[int] = []
        input_len_before_values: list[int] = []
        input_len_after_values: list[int] = []
        resolved_split: str | None = None
        metric_name: str | None = None
        sample_dataset_counts: dict[str, int] = {}
        source_kind_counts: dict[str, int] = {}
        evidence_aggregate = _init_evidence_aggregate()
        grounding_aggregate = _init_grounding_aggregate_metrics()
        structured_output_mode = str(
            OmegaConf.select(cfg, "grounding.structured_output_mode", default="plain")
        ).strip().lower()
        structured_constraint_mode = str(
            OmegaConf.select(cfg, "grounding.constraint_mode", default="repair")
        ).strip().lower()
        use_grounding_prefill = structured_constraint_mode in {"prefill", "prefill_repair"}
        use_grounding_repair = structured_constraint_mode in {"repair", "prefill_repair"}
        result_refinement_mode = str(
            OmegaConf.select(cfg, "grounding.result_refinement_mode", default="none")
        ).strip().lower()
        result_refinement_max_new_tokens = int(
            OmegaConf.select(cfg, "grounding.result_refinement_max_new_tokens", default=48)
        )

        for raw_idx, sample in enumerate(iter_samples(spec)):
            if not _sample_belongs_to_shard(raw_idx, shard_index=shard_index, num_shards=num_shards):
                continue
            if max_samples is not None and n_kept >= max_samples:
                break
            n_kept += 1
            n_total += 1
            if resolved_split is None:
                resolved_split = str(sample.meta.get("_resolved_split", cfg.data.split))
            sample_dataset = str(sample.dataset)
            sample_source_kind = (
                None if sample.meta.get("_source_kind") is None else str(sample.meta.get("_source_kind"))
            )
            sample_dataset_counts[sample_dataset] = sample_dataset_counts.get(sample_dataset, 0) + 1
            source_kind_key = sample_source_kind or "unknown"
            source_kind_counts[source_kind_key] = source_kind_counts.get(source_kind_key, 0) + 1

            selector_debug: dict[str, Any] | None = None
            if page_selector_type == "lexical":
                images, input_page_ids, selector_debug = _select_input_pages_with_lexical_selector(
                    images=sample.images,
                    meta=sample.meta,
                    question=sample.question,
                    context=sample.context,
                    max_images=int(cfg.data.max_images),
                    benchmark_root=str(cfg.data.benchmark_root),
                )
            elif page_selector_type in {"external", "external_manifest", "external_page"}:
                images, input_page_ids, selector_debug = _select_input_pages_with_external_selector(
                    images=sample.images,
                    meta=sample.meta,
                    sample_id=sample.sample_id,
                    predictions=external_page_predictions,
                    top_k=external_page_top_k,
                    fallback_max_images=int(cfg.data.max_images),
                )
            else:
                images = sample.images[: int(cfg.data.max_images)]
                input_page_ids = _resolve_input_page_ids(sample.meta, num_input_pages=len(images))
            user_text = _build_user_text(
                sample.context,
                sample.question,
                sample_source_kind,
                sample_dataset,
            )
            sys_prompt = _system_prompt_for_dataset(sample_dataset, sample_source_kind)
            record_dma_stats = not str(sample_dataset).startswith("throughput_")
            grounded_output_enabled = should_use_grounded_output(
                source_kind=sample_source_kind,
                structured_output_mode=structured_output_mode,
            )
            if grounded_output_enabled:
                sys_prompt, user_text = apply_grounded_output_format(
                    system_text=sys_prompt,
                    user_text=user_text,
                )
                sample_gen_cfg = GenerateConfig(
                    max_new_tokens=base_gen_cfg.max_new_tokens,
                    do_sample=base_gen_cfg.do_sample,
                    temperature=base_gen_cfg.temperature,
                    top_p=base_gen_cfg.top_p,
                    repetition_penalty=base_gen_cfg.repetition_penalty,
                    assistant_prefill_text=(
                        build_grounded_output_prefill(training_phase="final")
                        if use_grounding_prefill
                        else None
                    ),
                )
            else:
                sample_gen_cfg = GenerateConfig(
                    max_new_tokens=base_gen_cfg.max_new_tokens,
                    do_sample=base_gen_cfg.do_sample,
                    temperature=base_gen_cfg.temperature,
                    top_p=base_gen_cfg.top_p,
                    repetition_penalty=base_gen_cfg.repetition_penalty,
                    assistant_prefill_text=None,
                )

            t0 = time.time()
            resized_images = []
            resize_metas = []
            for im in images:
                rim, meta = resize_to_max_pixels(im, effective_image_max_pixels)
                resized_images.append(rim)
                resize_metas.append(meta)

            # throughput_* 只测固定上下文下的模型吞吐，没有 question/evidence；
            # block 选择在这里没有语义，继续构建候选块只会把无关 I/O 计入速度。
            block_selector_enabled_for_sample = block_selector_type != "none" and not str(sample_dataset).startswith(
                "throughput_"
            )
            block_stage_generate_allowed = (
                block_selector_enabled_for_sample
                and (grounded_output_enabled or not block_stage_generate_for_grounded_only)
            )
            page_stage_prefill_only = bool(block_stage_prefill_page_only and block_stage_generate_allowed)
            if page_stage_prefill_only:
                page_stage_out = model.prefill_one(
                    images=resized_images,
                    input_page_ids=input_page_ids,
                    user_text=user_text,
                    system_text=sys_prompt,
                )
            else:
                page_stage_out = model.generate_one(
                    images=resized_images,
                    input_page_ids=input_page_ids,
                    user_text=user_text,
                    system_text=sys_prompt,
                    gen=sample_gen_cfg,
                    record_dma_stats=record_dma_stats,
                )
            out = dict(page_stage_out)
            block_selector_debug: dict[str, Any] | None = None
            block_stage_resize_metas: list[dict[str, Any]] = []
            block_stage_generated = False
            if block_selector_enabled_for_sample:
                block_stage_images, block_selector_debug = _build_exact_block_stage_inputs(
                    model=model.model,
                    vlm_reranker=model,
                    images=images,
                    input_page_ids=input_page_ids,
                    meta=sample.meta,
                    question=sample.question,
                    context=sample.context,
                    benchmark_root=str(cfg.data.benchmark_root),
                    model_out=page_stage_out,
                    selector_type=block_selector_type,
                    source_page_mode=block_source_page_mode,
                    block_topk=block_topk,
                    vlm_rerank_topm=block_vlm_rerank_topm,
                    crop_expand_ratio=block_crop_expand_ratio,
                    pack_max_blocks_per_canvas=block_pack_max_blocks_per_canvas,
                    pack_enable=block_pack_enable,
                    include_block_text=block_prompt_include_text,
                    include_block_ids_in_text=block_prompt_include_ids,
                    block_text_max_chars_per_block=block_prompt_text_max_chars_per_block,
                )
                exact_retained_block_ids = _normalize_block_id_list(
                    None if block_selector_debug is None else block_selector_debug.get("exact_retained_block_ids")
                )
                if exact_retained_block_ids:
                    out["exact_retained_block_ids"] = exact_retained_block_ids
                    out["retained_block_ids"] = exact_retained_block_ids
                if block_selector_debug is not None:
                    block_selector_debug["stage_generate_allowed"] = bool(block_stage_generate_allowed)
                    block_selector_debug["stage_generate_mode"] = (
                        "grounded_only" if block_stage_generate_for_grounded_only else "always"
                    )
                    block_selector_debug["prompt_include_text"] = bool(block_prompt_include_text)
                    block_selector_debug["prompt_include_ids"] = bool(block_prompt_include_ids)
                    block_selector_debug["stage_text_only"] = bool(block_stage_text_only)
                    block_selector_debug["stage_prefill_page_only"] = bool(page_stage_prefill_only)
                if block_stage_images and block_stage_generate_allowed:
                    resized_block_stage_images = []
                    if not block_stage_text_only:
                        for im in block_stage_images:
                            rim, meta = resize_to_max_pixels(im, effective_image_max_pixels)
                            resized_block_stage_images.append(rim)
                            block_stage_resize_metas.append(meta)
                    block_prompt_texts = (
                        list(block_selector_debug.get("selected_block_texts") or [])
                        if block_prompt_include_text and isinstance(block_selector_debug, dict)
                        else None
                    )
                    block_stage_out = model.generate_one(
                        images=resized_block_stage_images,
                        input_page_ids=None,
                        user_text=_build_block_pack_user_text(
                            user_text,
                            block_texts=block_prompt_texts,
                            text_max_chars_per_block=block_prompt_text_max_chars_per_block,
                        ),
                        system_text=sys_prompt,
                        gen=sample_gen_cfg,
                        record_dma_stats=record_dma_stats,
                    )
                    block_stage_generated = True
                    out.update(block_stage_out)
                    out["pred"] = block_stage_out["pred"]
                    out["visual_tokens"] = int(page_stage_out["visual_tokens"]) + int(block_stage_out["visual_tokens"])
                    out["input_len"] = int(page_stage_out["input_len"]) + int(block_stage_out["input_len"])
                    out["gen_len"] = int(block_stage_out["gen_len"])
                    out["retained_page_ids"] = page_stage_out.get("retained_page_ids", [])
                    out["exact_retained_block_ids"] = exact_retained_block_ids
                    out["retained_block_ids"] = exact_retained_block_ids
                    out["page_stage"] = {
                        "visual_tokens": int(page_stage_out["visual_tokens"]),
                        "input_len": int(page_stage_out["input_len"]),
                        "gen_len": int(page_stage_out["gen_len"]),
                        "retained_page_ids": _normalize_page_id_list(page_stage_out.get("retained_page_ids")),
                    }
                    out["block_stage"] = {
                        "visual_tokens": int(block_stage_out["visual_tokens"]),
                        "input_len": int(block_stage_out["input_len"]),
                        "gen_len": int(block_stage_out["gen_len"]),
                        "num_block_images": len(resized_block_stage_images),
                        "text_only": bool(block_stage_text_only),
                        "prompt_include_text": bool(block_prompt_include_text),
                    }
                elif block_stage_images and block_selector_debug is not None:
                    block_selector_debug["stage_generate_skipped_reason"] = "grounded_only_gate"
            if page_stage_prefill_only and not block_stage_generated:
                page_stage_fallback_out = model.generate_one(
                    images=resized_images,
                    input_page_ids=input_page_ids,
                    user_text=user_text,
                    system_text=sys_prompt,
                    gen=sample_gen_cfg,
                    record_dma_stats=record_dma_stats,
                )
                out = dict(page_stage_fallback_out)
                if block_selector_debug is not None:
                    block_selector_debug["stage_prefill_fallback_to_page_generate"] = True
            dt = time.time() - t0

            pred = out["pred"]
            if grounded_output_enabled and use_grounding_repair:
                pred = normalize_structured_prediction_text(pred)
                out["pred"] = pred
            if selector_debug is not None:
                out.setdefault("selector_page_ids", list(input_page_ids))
            scoring_pred = pred
            grounding_hard_eval: dict[str, Any] | None = None
            if grounded_output_enabled:
                grounding_hard_eval = compute_grounding_hard_eval(
                    meta=sample.meta,
                    answers=sample.answers,
                    pred_text=pred,
                    benchmark_root=str(cfg.data.benchmark_root),
                )
                refined_result = _maybe_refine_grounded_result(
                    model=model,
                    question=sample.question,
                    grounding_hard_eval=grounding_hard_eval,
                    mode=result_refinement_mode,
                    max_new_tokens=result_refinement_max_new_tokens,
                )
                if refined_result is not None:
                    pred = replace_structured_prediction_result(pred, new_result=refined_result)
                    out["pred"] = pred
                    grounding_hard_eval = compute_grounding_hard_eval(
                        meta=sample.meta,
                        answers=sample.answers,
                        pred_text=pred,
                        benchmark_root=str(cfg.data.benchmark_root),
                    )
                scoring_pred = str(grounding_hard_eval.get("result_text") or pred)
            if sample_dataset == "RealWorldQA":
                scoring_pred = model.postprocess_multiple_choice(scoring_pred)
                if not grounded_output_enabled:
                    pred = scoring_pred

            scoring = _score_one(sample_dataset, scoring_pred, sample.answers)
            evidence_eval = _compute_evidence_eval(
                sample.meta,
                num_input_pages=len(images),
                visual_tokens=int(out["visual_tokens"]),
                model_out=out,
                benchmark_root=str(cfg.data.benchmark_root),
            )
            if metric_name is None and scoring["metric"] is not None:
                metric_name = str(scoring["metric"])

            rec = {
                "dataset": sample.dataset,
                "id": sample.sample_id,
                "question": sample.question,
                "context": sample.context,
                "answers": sample.answers,
                "pred": pred,
                "scoring_pred": scoring_pred,
                "scoring": scoring,
                "evidence_eval": evidence_eval,
                "grounding_hard_eval": grounding_hard_eval,
                "meta": sample.meta,
                "resize": resize_metas,
                "page_selector": selector_debug,
                "block_selector": block_selector_debug,
                "block_resize": block_stage_resize_metas,
                "stats": {
                    "visual_tokens": out["visual_tokens"],
                    "visual_tokens_before": out.get("visual_tokens_before"),
                    "visual_tokens_after": out.get("visual_tokens_after"),
                    "input_len": out["input_len"],
                    "input_len_before": out.get("input_len_before"),
                    "input_len_after": out.get("input_len_after"),
                    "gen_len": out["gen_len"],
                    "page_keep_counts": out.get("page_keep_counts"),
                    "latency_s": dt,
                },
            }
            if out.get("pruning_score_debug") is not None:
                rec["stats"]["pruning_score_debug"] = out.get("pruning_score_debug")
            if bool(cfg.output.save_predictions):
                append_jsonl(pred_path, rec)

            visual_tokens_sum += int(out["visual_tokens"])
            t_sum += dt
            visual_tokens_before_values.append(int(out.get("visual_tokens_before", out["visual_tokens"])))
            visual_tokens_after_values.append(int(out.get("visual_tokens_after", out["visual_tokens"])))
            input_len_before_values.append(int(out.get("input_len_before", out["input_len"])))
            input_len_after_values.append(int(out.get("input_len_after", out["input_len"])))

            if scoring["scorable"]:
                n_scored += 1
                score_sum += float(scoring["score"])
            _update_evidence_aggregate(evidence_aggregate, evidence_eval)
            _update_grounding_aggregate_metrics(grounding_aggregate, grounding_hard_eval)

        visual_tokens_mean = None if n_total == 0 else (visual_tokens_sum / n_total)
        ds_result = {
            "num_samples": n_total,
            "num_scored": n_scored,
            "resolved_split": resolved_split,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "metric": metric_name,
            "score_mean": None if n_scored == 0 else (score_sum / n_scored),
            "visual_tokens_mean": visual_tokens_mean,
            "visual_tokens_before_mean": (
                None if not visual_tokens_before_values else (sum(visual_tokens_before_values) / len(visual_tokens_before_values))
            ),
            "visual_tokens_after_mean": (
                None if not visual_tokens_after_values else (sum(visual_tokens_after_values) / len(visual_tokens_after_values))
            ),
            "input_len_before_mean": (
                None if not input_len_before_values else (sum(input_len_before_values) / len(input_len_before_values))
            ),
            "input_len_after_mean": (
                None if not input_len_after_values else (sum(input_len_after_values) / len(input_len_after_values))
            ),
            "latency_s_mean": None if n_total == 0 else (t_sum / n_total),
            "sample_dataset_counts": sample_dataset_counts,
            "source_kind_counts": source_kind_counts,
        }
        ds_result.update(
            _finalize_evidence_aggregate(
                evidence_aggregate,
                visual_tokens_mean=visual_tokens_mean,
            )
        )
        ds_result.update(_finalize_grounding_aggregate_metrics(grounding_aggregate))
        summary["results"][str(dataset_name)] = ds_result

    dump_json(Path(out_dir) / cfg.output.metrics_file, summary)


if __name__ == "__main__":
    main()
