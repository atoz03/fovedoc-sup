from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "who",
    "why",
    "with",
}

EXACT_BLOCK_FEATURE_NAMES = (
    "lexical_score_log",
    "lexical_overlap_ratio",
    "page_score",
    "page_focus",
    "selected_ratio",
    "coarse_block_score",
    "coarse_rank_score",
    "coarse_focus_match",
    "coarse_focus_dx",
    "coarse_focus_dy",
    "coarse_focus_l1",
    "page_rank_score",
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "order_fraction",
    "block_text_len_log",
)
EXACT_BLOCK_FEATURE_INDEX = {name: idx for idx, name in enumerate(EXACT_BLOCK_FEATURE_NAMES)}
EXACT_BLOCK_FEATURE_DIM = len(EXACT_BLOCK_FEATURE_NAMES)
EXACT_BLOCK_TEXT_SKETCH_DIM = 64


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _page_id_to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def tokenize_exact_block_text(text: str | None) -> list[str]:
    if text is None:
        return []
    return [
        token
        for token in _TOKEN_RE.findall(str(text).lower())
        if token and token not in _STOPWORDS
    ]


def _stable_token_bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % max(1, int(dim))
    sign = 1.0 if int(digest[4]) % 2 == 0 else -1.0
    return bucket, sign


def _counter_to_text_sketch(counter: Counter[str], dim: int = EXACT_BLOCK_TEXT_SKETCH_DIM) -> list[float]:
    dense_dim = max(1, int(dim))
    sketch = [0.0] * dense_dim
    if not counter:
        return sketch
    for token, count in counter.items():
        if not token:
            continue
        bucket, sign = _stable_token_bucket(token, dense_dim)
        sketch[bucket] += sign * math.log1p(max(0.0, float(count)))
    norm = math.sqrt(sum(value * value for value in sketch))
    if norm > 0.0:
        sketch = [value / norm for value in sketch]
    return sketch


@lru_cache(maxsize=4096)
def load_document_page_blocks(document_json_path: str) -> dict[int, dict[str, Any]]:
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
                if None not in {x0, y0, x1, y1} and float(x1) > float(x0) and float(y1) > float(y0):
                    normalized_bbox = (float(x0), float(y0), float(x1), float(y1))
            blocks.append(
                {
                    "block_id": block_id,
                    "bbox": normalized_bbox,
                    "text": str(block.get("text") or ""),
                    "order_index": int(block.get("order_index", order_index)),
                }
            )
        page_blocks[page_id] = {
            "width": width,
            "height": height,
            "page_text": str(page.get("page_text") or ""),
            "blocks": blocks,
        }
    return page_blocks


def build_exact_block_page_stats_by_id(page_stats: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(page_stats, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for page_rank, item in enumerate(page_stats):
        if not isinstance(item, dict):
            continue
        page_id = _page_id_to_int(item.get("page_id"))
        if page_id is None:
            continue
        normalized = dict(item)
        normalized["page_rank"] = int(page_rank)
        out[page_id] = normalized
    return out


def exact_block_bbox_to_coarse_index(
    bbox: tuple[float, float, float, float] | None,
    *,
    page_width: float | None,
    page_height: float | None,
) -> int | None:
    if bbox is None:
        return None
    if page_width is None or page_height is None or page_width <= 0.0 or page_height <= 0.0:
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    cx = max(0.0, min(page_width - 1e-6, (x0 + x1) * 0.5))
    cy = max(0.0, min(page_height - 1e-6, (y0 + y1) * 0.5))
    col = min(2, max(0, int((cx / page_width) * 3.0)))
    row = min(2, max(0, int((cy / page_height) * 3.0)))
    return int(row * 3 + col)


def _resolve_coarse_features(
    page_stat: dict[str, Any] | None,
    *,
    coarse_index: int | None,
) -> tuple[float, float]:
    if coarse_index is None or not isinstance(page_stat, dict):
        return 0.0, 0.0
    coarse_scores = page_stat.get("coarse_block_scores")
    if isinstance(coarse_scores, list) and 0 <= coarse_index < len(coarse_scores):
        try:
            score = float(coarse_scores[coarse_index])
        except (TypeError, ValueError):
            score = 0.0
        ranked_indices = sorted(
            range(len(coarse_scores)),
            key=lambda idx: (-float(coarse_scores[idx]), int(idx)),
        )
        try:
            rank = ranked_indices.index(int(coarse_index))
            rank_score = 1.0 / float(rank + 1)
        except ValueError:
            rank_score = 0.0
        return score, rank_score

    top_block_ids = page_stat.get("top_block_ids")
    page_id = _page_id_to_int(page_stat.get("page_id"))
    coarse_block_id = None if page_id is None else f"p{page_id}_g{int(coarse_index) + 1}"
    if isinstance(top_block_ids, list) and coarse_block_id is not None:
        for rank, block_id in enumerate(top_block_ids):
            if str(block_id) == coarse_block_id:
                return 1.0 / float(rank + 1), 1.0 / float(rank + 1)
    return 0.0, 0.0


def _coarse_index_to_normalized_center(coarse_index: int | None) -> tuple[float, float] | None:
    if coarse_index is None or not (0 <= int(coarse_index) < 9):
        return None
    row = int(coarse_index) // 3
    col = int(coarse_index) % 3
    return ((float(col) + 0.5) / 3.0, (float(row) + 0.5) / 3.0)


def _resolve_coarse_focus_features(
    *,
    page_stat: dict[str, Any] | None,
    coarse_index: int | None,
    bbox_cx: float,
    bbox_cy: float,
) -> tuple[float, float, float, float]:
    if not isinstance(page_stat, dict):
        return 0.0, 0.0, 0.0, 0.0
    coarse_scores = page_stat.get("coarse_block_scores")
    if not isinstance(coarse_scores, list) or not coarse_scores:
        return 0.0, 0.0, 0.0, 0.0
    try:
        focus_index = max(
            range(len(coarse_scores)),
            key=lambda idx: (float(coarse_scores[idx]), -int(idx)),
        )
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0
    focus_center = _coarse_index_to_normalized_center(int(focus_index))
    if focus_center is None:
        return 0.0, 0.0, 0.0, 0.0
    focus_cx, focus_cy = focus_center
    dx = float(bbox_cx) - float(focus_cx)
    dy = float(bbox_cy) - float(focus_cy)
    l1 = abs(dx) + abs(dy)
    match = 1.0 if coarse_index is not None and int(coarse_index) == int(focus_index) else 0.0
    return match, dx, dy, l1


def build_exact_block_candidate_rows(
    *,
    document_json_path: str,
    source_page_ids: list[int],
    query_texts: list[str | None],
    page_stats_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    page_blocks = load_document_page_blocks(document_json_path)
    if not page_blocks:
        return []

    query_tokens = []
    for text in query_texts:
        query_tokens.extend(tokenize_exact_block_text(text))
    if not query_tokens:
        return []
    query_counts = Counter(query_tokens)
    unique_query_count = max(1, len(query_counts))
    query_sketch = _counter_to_text_sketch(query_counts)

    page_stats_by_id = page_stats_by_id or {}
    candidates: list[dict[str, Any]] = []
    token_counters: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()
    for page_rank, page_id in enumerate(source_page_ids):
        page_info = page_blocks.get(int(page_id))
        if not isinstance(page_info, dict):
            continue
        width = _safe_float(page_info.get("width"))
        height = _safe_float(page_info.get("height"))
        raw_blocks = page_info.get("blocks")
        if not isinstance(raw_blocks, list):
            continue
        num_blocks_in_page = max(1, len(raw_blocks))
        for fallback_order, block in enumerate(raw_blocks):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if block_id == "":
                continue
            order_index = int(block.get("order_index", fallback_order))
            block_text = str(block.get("text") or "")
            token_counter = Counter(tokenize_exact_block_text(block_text))
            token_counters.append(token_counter)
            for token in token_counter.keys():
                doc_freq[token] += 1
            candidates.append(
                {
                    "page_id": int(page_id),
                    "page_rank": int(page_rank),
                    "page_width": width,
                    "page_height": height,
                    "block_id": block_id,
                    "bbox": block.get("bbox"),
                    "text": block_text,
                    "order_index": order_index,
                    "num_blocks_in_page": num_blocks_in_page,
                    "page_stat": page_stats_by_id.get(int(page_id), {"page_rank": int(page_rank)}),
                }
            )

    if not candidates:
        return []

    num_candidates = max(1, len(candidates))
    rows: list[dict[str, Any]] = []
    for candidate, token_counter in zip(candidates, token_counters):
        matched_query_terms = 0
        lexical_score = 0.0
        for token, q_tf in query_counts.items():
            b_tf = token_counter.get(token, 0)
            if b_tf <= 0:
                continue
            matched_query_terms += 1
            idf = math.log((1.0 + num_candidates) / (1.0 + float(doc_freq.get(token, 0)))) + 1.0
            lexical_score += float(q_tf) * min(3.0, float(b_tf)) * idf

        bbox = candidate.get("bbox")
        page_width = _safe_float(candidate.get("page_width"))
        page_height = _safe_float(candidate.get("page_height"))
        coarse_index = exact_block_bbox_to_coarse_index(
            bbox if isinstance(bbox, tuple) else bbox,
            page_width=page_width,
            page_height=page_height,
        )
        page_stat = candidate.get("page_stat")
        coarse_block_score, coarse_rank_score = _resolve_coarse_features(
            page_stat if isinstance(page_stat, dict) else None,
            coarse_index=coarse_index,
        )

        bbox_cx = 0.5
        bbox_cy = 0.5
        bbox_w = 1.0
        bbox_h = 1.0
        bbox_area = 1.0
        if (
            isinstance(bbox, tuple)
            and page_width is not None
            and page_height is not None
            and page_width > 0.0
            and page_height > 0.0
        ):
            x0, y0, x1, y1 = bbox
            bbox_cx = max(0.0, min(1.0, ((x0 + x1) * 0.5) / page_width))
            bbox_cy = max(0.0, min(1.0, ((y0 + y1) * 0.5) / page_height))
            bbox_w = max(0.0, min(1.0, (x1 - x0) / page_width))
            bbox_h = max(0.0, min(1.0, (y1 - y0) / page_height))
            bbox_area = max(0.0, min(1.0, bbox_w * bbox_h))

        coarse_focus_match, coarse_focus_dx, coarse_focus_dy, coarse_focus_l1 = _resolve_coarse_focus_features(
            page_stat=page_stat if isinstance(page_stat, dict) else None,
            coarse_index=coarse_index,
            bbox_cx=float(bbox_cx),
            bbox_cy=float(bbox_cy),
        )

        page_rank = int(candidate.get("page_rank", 0))
        order_index = int(candidate.get("order_index", 0))
        num_blocks_in_page = max(1, int(candidate.get("num_blocks_in_page", 1)))
        block_sketch = _counter_to_text_sketch(token_counter)
        features = [
            math.log1p(max(0.0, lexical_score)),
            float(matched_query_terms) / float(unique_query_count),
            float(page_stat.get("page_score", 0.0)) if isinstance(page_stat, dict) else 0.0,
            float(page_stat.get("page_focus", 0.0)) if isinstance(page_stat, dict) else 0.0,
            float(page_stat.get("selected_ratio", 0.0)) if isinstance(page_stat, dict) else 0.0,
            float(coarse_block_score),
            float(coarse_rank_score),
            float(coarse_focus_match),
            float(coarse_focus_dx),
            float(coarse_focus_dy),
            float(coarse_focus_l1),
            1.0 / float(page_rank + 1),
            float(bbox_cx),
            float(bbox_cy),
            float(bbox_w),
            float(bbox_h),
            float(bbox_area),
            min(1.0, float(order_index + 1) / float(num_blocks_in_page)),
            math.log1p(float(sum(token_counter.values()))),
        ]
        rows.append(
            {
                **candidate,
                "coarse_index": coarse_index,
                "lexical_score": float(lexical_score),
                "matched_query_terms": int(matched_query_terms),
                "query_sketch": list(query_sketch),
                "block_sketch": list(block_sketch),
                "features": features,
            }
        )
    return rows


def exact_block_candidate_features_to_tensor(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not candidates:
        return torch.empty((0, EXACT_BLOCK_FEATURE_DIM), device=device, dtype=dtype)
    rows = [list(item.get("features") or []) for item in candidates]
    return torch.tensor(rows, device=device, dtype=dtype)


def exact_block_candidate_page_ids_to_tensor(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not candidates:
        return torch.empty((0,), device=device, dtype=torch.long)
    page_ids = [int(item.get("page_id", 0)) for item in candidates]
    return torch.tensor(page_ids, device=device, dtype=torch.long)


def exact_block_candidate_query_sketches_to_tensor(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not candidates:
        return torch.empty((0, EXACT_BLOCK_TEXT_SKETCH_DIM), device=device, dtype=dtype)
    rows = [list(item.get("query_sketch") or [0.0] * EXACT_BLOCK_TEXT_SKETCH_DIM) for item in candidates]
    return torch.tensor(rows, device=device, dtype=dtype)


def exact_block_candidate_block_sketches_to_tensor(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not candidates:
        return torch.empty((0, EXACT_BLOCK_TEXT_SKETCH_DIM), device=device, dtype=dtype)
    rows = [list(item.get("block_sketch") or [0.0] * EXACT_BLOCK_TEXT_SKETCH_DIM) for item in candidates]
    return torch.tensor(rows, device=device, dtype=dtype)


def exact_block_candidate_key(*, page_id: Any, block_id: Any) -> str | None:
    normalized_page_id = _page_id_to_int(page_id)
    normalized_block_id = str(block_id or "").strip()
    if normalized_page_id is None or normalized_block_id == "":
        return None
    return f"{int(normalized_page_id)}::{normalized_block_id}"


def build_exact_block_teacher_score_map(teacher_payload: Any) -> dict[str, float]:
    if teacher_payload is None:
        return {}

    score_map: dict[str, float] = {}
    if isinstance(teacher_payload, dict):
        raw_score_map = teacher_payload.get("score_by_block_key")
        if isinstance(raw_score_map, dict):
            for raw_key, raw_score in raw_score_map.items():
                key = str(raw_key or "").strip()
                score = _safe_float(raw_score)
                if key != "" and score is not None:
                    score_map[key] = float(score)
        raw_items = teacher_payload.get("scores")
    elif isinstance(teacher_payload, list):
        raw_items = teacher_payload
    else:
        return score_map

    if not isinstance(raw_items, list):
        return score_map

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = exact_block_candidate_key(page_id=item.get("page_id"), block_id=item.get("block_id"))
        if key is None:
            raw_key = str(item.get("key") or "").strip()
            key = raw_key or None
        score = _safe_float(item.get("score"))
        if key is not None and score is not None:
            score_map[key] = float(score)
    return score_map


def attach_exact_block_teacher_scores(candidates: list[dict[str, Any]], teacher_payload: Any) -> int:
    if not candidates:
        return 0
    score_map = build_exact_block_teacher_score_map(teacher_payload)
    if not score_map:
        return 0
    matched = 0
    for candidate in candidates:
        key = exact_block_candidate_key(page_id=candidate.get("page_id"), block_id=candidate.get("block_id"))
        if key is None:
            continue
        score = score_map.get(key)
        if score is None:
            continue
        candidate["teacher_score"] = float(score)
        matched += 1
    return matched


def resolve_exact_block_scorer(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "dma_exact_block_proj"):
        scorer = getattr(model, "dma_exact_block_proj")
        if isinstance(scorer, nn.Module):
            return scorer
    wrapped_model = getattr(model, "module", None)
    if isinstance(wrapped_model, nn.Module) and wrapped_model is not model:
        scorer = resolve_exact_block_scorer(wrapped_model)
        if scorer is not None:
            return scorer
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not model:
            scorer = resolve_exact_block_scorer(base_model)
            if scorer is not None:
                return scorer
    return None


def score_exact_block_candidates(model: nn.Module, candidate_features: torch.Tensor) -> torch.Tensor | None:
    scorer = resolve_exact_block_scorer(model)
    if scorer is None:
        return None
    if int(candidate_features.shape[0]) == 0:
        return candidate_features.new_empty((0,))
    ref_param = next(scorer.parameters(), None)
    if ref_param is None:
        return None
    return scorer(candidate_features.to(device=ref_param.device, dtype=ref_param.dtype)).reshape(-1)
