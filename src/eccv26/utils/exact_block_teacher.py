from __future__ import annotations

import math
from typing import Any

from PIL import Image

from eccv26.utils.exact_block import (
    build_exact_block_candidate_rows,
    exact_block_candidate_key,
)


VLM_BLOCK_RERANK_SYSTEM_TEXT = "你是一个严格的证据相关性判别器。只能回答“是”或“否”。"


def crop_exact_block_from_page_image(
    *,
    page_image: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    page_width: float | None,
    page_height: float | None,
    expand_ratio: float,
) -> Image.Image | None:
    if bbox is None or page_width is None or page_height is None or page_width <= 0 or page_height <= 0:
        return None
    img_w, img_h = page_image.size
    if img_w <= 0 or img_h <= 0:
        return None
    x0, y0, x1, y1 = bbox
    scale_x = float(img_w) / float(page_width)
    scale_y = float(img_h) / float(page_height)
    left = x0 * scale_x
    top = y0 * scale_y
    right = x1 * scale_x
    bottom = y1 * scale_y
    expand_x = max(2.0, (right - left) * max(0.0, float(expand_ratio)))
    expand_y = max(2.0, (bottom - top) * max(0.0, float(expand_ratio)))
    crop_left = max(0, int(math.floor(left - expand_x)))
    crop_top = max(0, int(math.floor(top - expand_y)))
    crop_right = min(img_w, int(math.ceil(right + expand_x)))
    crop_bottom = min(img_h, int(math.ceil(bottom + expand_y)))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None
    return page_image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")


def build_block_rerank_user_text(base_user_text: str, block_text: str | None) -> str:
    clipped_block_text = str(block_text or "").strip()
    if len(clipped_block_text) > 256:
        clipped_block_text = clipped_block_text[:256]
    return (
        "请判断这张证据块图像及其对应文本，是否与问题直接相关。"
        "如果它足以作为回答问题的关键证据，回答“是”；否则回答“否”。\n\n"
        f"问题：\n{base_user_text}\n\n"
        f"候选证据块文本：\n{clipped_block_text or '（无文本）'}"
    )


def build_exact_block_vlm_teacher_payload(
    *,
    vlm_reranker: Any,
    document_json_path: str,
    source_page_ids: list[int],
    user_text: str,
    image_by_page_id: dict[int, Image.Image],
    lexical_topm: int,
    crop_expand_ratio: float,
    page_stats_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = build_exact_block_candidate_rows(
        document_json_path=document_json_path,
        source_page_ids=[int(page_id) for page_id in source_page_ids],
        query_texts=[user_text],
        page_stats_by_id=page_stats_by_id,
    )
    if not candidates:
        return {
            "type": "vlm_rerank",
            "version": 1,
            "topm": max(1, int(lexical_topm)),
            "source_page_ids": [int(page_id) for page_id in source_page_ids],
            "num_candidates": 0,
            "num_scored": 0,
            "scores": [],
        }

    lexical_prefilter = sorted(
        candidates,
        key=lambda item: (-float(item.get("lexical_score", 0.0)), int(item["page_rank"]), int(item["order_index"])),
    )[: max(1, int(lexical_topm))]
    scored_rows: list[dict[str, Any]] = []
    num_failed = 0
    for block in lexical_prefilter:
        page_id = int(block.get("page_id", 0))
        page_image = image_by_page_id.get(page_id)
        if not isinstance(page_image, Image.Image):
            continue
        crop = crop_exact_block_from_page_image(
            page_image=page_image,
            bbox=block.get("bbox"),
            page_width=block.get("page_width"),
            page_height=block.get("page_height"),
            expand_ratio=float(crop_expand_ratio),
        )
        if crop is None:
            continue
        try:
            score = float(
                vlm_reranker.score_block_relevance_one(
                    images=[crop],
                    user_text=build_block_rerank_user_text(user_text, str(block.get("text") or "")),
                    system_text=VLM_BLOCK_RERANK_SYSTEM_TEXT,
                )
            )
        except Exception:
            num_failed += 1
            continue
        scored_rows.append(
            {
                "key": exact_block_candidate_key(page_id=page_id, block_id=block.get("block_id")),
                "page_id": page_id,
                "block_id": str(block.get("block_id") or ""),
                "score": score,
                "lexical_score": float(block.get("lexical_score", 0.0)),
                "page_rank": int(block.get("page_rank", 0)),
                "order_index": int(block.get("order_index", 0)),
            }
        )
    scored_rows.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("lexical_score", 0.0)),
            int(item.get("page_rank", 0)),
            int(item.get("order_index", 0)),
        )
    )
    for rank, item in enumerate(scored_rows):
        item["teacher_rank"] = int(rank)
    return {
        "type": "vlm_rerank",
        "version": 1,
        "topm": max(1, int(lexical_topm)),
        "source_page_ids": [int(page_id) for page_id in source_page_ids],
        "num_candidates": int(len(candidates)),
        "num_scored": int(len(scored_rows)),
        "num_failed": int(num_failed),
        "scores": scored_rows,
    }
