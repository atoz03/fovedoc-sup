from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class VisualTokenPruningResult:
    pruned_image_embeds: Tensor
    pruned_grid_thw: Tensor
    page_keep_indices: Tensor
    page_keep_counts: Tensor
    split_sizes: Tensor
    score_debug: dict[str, Any] | None = None
    position_strategy: str = "select_existing"


def _compute_split_sizes(image_grid_thw: Tensor, spatial_merge_size: int) -> Tensor:
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError("image_grid_thw 必须是 [num_images, 3] 张量。")
    split_sizes = image_grid_thw.prod(dim=-1) // int(spatial_merge_size) ** 2
    if not bool((split_sizes > 0).all()):
        raise ValueError("image_grid_thw 解析出的每页 visual token 数必须大于 0。")
    return split_sizes.to(dtype=torch.long)


def _pack_token_count_as_pseudo_grid(token_count: int, spatial_merge_size: int, *, device: torch.device) -> Tensor:
    if token_count <= 0:
        raise ValueError("token_count 必须大于 0。")

    rows = int(math.isqrt(token_count))
    while rows > 1 and token_count % rows != 0:
        rows -= 1
    cols = token_count // rows

    return torch.tensor(
        [1, rows * int(spatial_merge_size), cols * int(spatial_merge_size)],
        dtype=torch.long,
        device=device,
    )


def _pad_keep_indices(page_keep_indices: list[Tensor], *, device: torch.device) -> Tensor:
    max_kept = max((int(item.numel()) for item in page_keep_indices), default=0)
    if max_kept == 0:
        return torch.empty((len(page_keep_indices), 0), dtype=torch.long, device=device)

    padded = torch.full((len(page_keep_indices), max_kept), -1, dtype=torch.long, device=device)
    for page_idx, keep_idx in enumerate(page_keep_indices):
        if int(keep_idx.numel()) == 0:
            continue
        padded[page_idx, : int(keep_idx.numel())] = keep_idx.to(device=device, dtype=torch.long)
    return padded


def _build_page_score_stats(
    *,
    page_index: int,
    scores: Tensor,
    token_count: int,
    keep_count: int,
) -> dict[str, Any]:
    scores_cpu = scores.detach().to(dtype=torch.float32, device="cpu")
    return {
        "page_index": int(page_index),
        "token_count": int(token_count),
        "keep_count": int(keep_count),
        "mean": float(scores_cpu.mean().item()),
        "std": float(scores_cpu.std(unbiased=False).item()),
        "min": float(scores_cpu.min().item()),
        "max": float(scores_cpu.max().item()),
    }


def _build_uniform_axis_indices(size: int, *, stride: int | None = None, target_count: int | None = None) -> Tensor:
    if size <= 0:
        raise ValueError("axis size 必须大于 0。")
    if stride is not None and stride <= 0:
        raise ValueError("stride 必须大于 0。")
    if target_count is not None and target_count <= 0:
        raise ValueError("target_count 必须大于 0。")

    if stride is not None:
        indices = torch.arange(0, size, stride, dtype=torch.long)
        if int(indices[-1].item()) != size - 1:
            indices = torch.cat([indices, torch.tensor([size - 1], dtype=torch.long)], dim=0)
        return torch.unique_consecutive(indices)

    if target_count is None or target_count >= size:
        return torch.arange(size, dtype=torch.long)

    indices_f = torch.linspace(0, size - 1, steps=target_count, dtype=torch.float32)
    indices = indices_f.round().to(dtype=torch.long)
    indices = torch.unique_consecutive(indices)
    if int(indices[0].item()) != 0:
        indices = torch.cat([torch.tensor([0], dtype=torch.long), indices], dim=0)
    if int(indices[-1].item()) != size - 1:
        indices = torch.cat([indices, torch.tensor([size - 1], dtype=torch.long)], dim=0)
    return torch.unique_consecutive(indices)


def _build_spatial_keep_indices(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    stride: int | None = None,
    keep_ratio: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_t/grid_h/grid_w 必须都大于 0。")

    if stride is None:
        if keep_ratio is None:
            raise ValueError("spatial downsampling 需要 stride 或 keep_ratio 二选一。")
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("spatial downsampling 的 keep_ratio 必须在 (0, 1] 内。")
        target_h = min(grid_h, max(1, int(math.ceil(grid_h * math.sqrt(float(keep_ratio))))))
        target_w = min(grid_w, max(1, int(math.ceil(grid_w * math.sqrt(float(keep_ratio))))))
        keep_rows = _build_uniform_axis_indices(grid_h, target_count=target_h)
        keep_cols = _build_uniform_axis_indices(grid_w, target_count=target_w)
    else:
        keep_rows = _build_uniform_axis_indices(grid_h, stride=int(stride))
        keep_cols = _build_uniform_axis_indices(grid_w, stride=int(stride))

    frame_offsets = torch.arange(grid_t, dtype=torch.long) * int(grid_h * grid_w)
    keep_indices: list[Tensor] = []
    for frame_offset in frame_offsets.tolist():
        base = int(frame_offset)
        row_offsets = keep_rows * int(grid_w)
        frame_grid = row_offsets[:, None] + keep_cols[None, :]
        keep_indices.append(frame_grid.reshape(-1) + base)

    return torch.cat(keep_indices, dim=0), keep_rows, keep_cols


def prune_visual_tokens_by_scores_with_metadata(
    image_embeds: Tensor,
    image_grid_thw: Tensor,
    token_scores: Tensor | None,
    keep_ratio: float = 0.5,
    min_tokens_per_page: int = 16,
    *,
    spatial_merge_size: int = 2,
    record_score_stats: bool = False,
    score_source: str = "external",
) -> VisualTokenPruningResult:
    """
    对 visual encoder 输出做按页 top-k pruning，分数由调用方外部提供。

    说明：
    - `token_scores` 必须与 `image_embeds` 一一对齐，长度等于总 visual token 数。
    - 若 `keep_ratio == 1.0` 且 `record_score_stats == False`，允许 `token_scores=None`，
      此时直接保留全部 token，不额外做打分计算。
    """

    if image_embeds.ndim != 2:
        raise ValueError("image_embeds 必须是 [total_visual_tokens, hidden_dim] 张量。")
    if keep_ratio <= 0.0:
        raise ValueError("keep_ratio 必须大于 0。")
    if min_tokens_per_page <= 0:
        raise ValueError("min_tokens_per_page 必须大于 0。")

    split_sizes = _compute_split_sizes(image_grid_thw, spatial_merge_size)
    total_visual_tokens = int(image_embeds.shape[0])
    if int(split_sizes.sum().item()) != total_visual_tokens:
        raise ValueError(
            "image_embeds 总 token 数与 image_grid_thw 解析出的 split_sizes 不一致："
            f"{total_visual_tokens} vs {int(split_sizes.sum().item())}。"
        )

    scores_flat: Tensor | None = None
    if token_scores is not None:
        if token_scores.ndim == 2:
            if token_scores.shape[0] != 1:
                raise ValueError("token_scores 若为二维张量，batch 维必须等于 1。")
            token_scores = token_scores.squeeze(0)
        if token_scores.ndim != 1:
            raise ValueError("token_scores 必须是一维张量。")
        if int(token_scores.shape[0]) != total_visual_tokens:
            raise ValueError(
                "token_scores 长度必须与 image_embeds 的总 visual token 数一致："
                f"{int(token_scores.shape[0])} vs {total_visual_tokens}。"
            )
        scores_flat = token_scores.to(device=image_embeds.device, dtype=torch.float32)

    page_keep_indices: list[Tensor] = []
    page_keep_counts: list[int] = []
    pruned_pages: list[Tensor] = []
    pruned_grid_rows: list[Tensor] = []
    page_score_stats: list[dict[str, Any]] = []
    all_page_scores: list[Tensor] = []

    offset = 0
    for page_idx, page_tokens in enumerate(split_sizes.tolist()):
        page_size = int(page_tokens)
        page_embeds = image_embeds[offset : offset + page_size]
        page_scores = None if scores_flat is None else scores_flat[offset : offset + page_size]
        offset += page_size

        keep_count = min(
            page_size,
            max(int(min_tokens_per_page), int(page_size * float(keep_ratio))),
        )
        if keep_count == page_size:
            keep_idx = torch.arange(page_size, device=page_embeds.device, dtype=torch.long)
        else:
            if page_scores is None:
                raise RuntimeError("visual token pruning 缺少可用于 top-k 的 token_scores。")
            keep_idx = torch.topk(page_scores, k=keep_count, dim=0, largest=True, sorted=False).indices
            keep_idx = keep_idx.sort().values

        if record_score_stats:
            if page_scores is None:
                raise RuntimeError("visual token pruning 缺少可用于诊断的 token_scores。")
            page_score_stats.append(
                _build_page_score_stats(
                    page_index=page_idx,
                    scores=page_scores,
                    token_count=page_size,
                    keep_count=keep_count,
                )
            )
            all_page_scores.append(page_scores.detach().to(dtype=torch.float32, device="cpu"))

        page_keep_indices.append(keep_idx)
        page_keep_counts.append(int(keep_count))
        pruned_pages.append(page_embeds.index_select(0, keep_idx))
        pruned_grid_rows.append(
            _pack_token_count_as_pseudo_grid(
                token_count=int(keep_count),
                spatial_merge_size=int(spatial_merge_size),
                device=image_embeds.device,
            )
        )

    pruned_image_embeds = torch.cat(pruned_pages, dim=0) if pruned_pages else image_embeds.new_empty((0, image_embeds.shape[-1]))
    pruned_grid_thw = torch.stack(pruned_grid_rows, dim=0) if pruned_grid_rows else image_grid_thw.new_empty((0, 3))
    keep_indices_padded = _pad_keep_indices(page_keep_indices, device=image_embeds.device)
    keep_counts_tensor = torch.tensor(page_keep_counts, dtype=torch.long, device=image_embeds.device)
    score_debug: dict[str, Any] | None = None
    if record_score_stats and all_page_scores:
        all_scores = torch.cat(all_page_scores, dim=0)
        page_mean_values = [float(item["mean"]) for item in page_score_stats]
        page_std_values = [float(item["std"]) for item in page_score_stats]
        global_min = float(all_scores.min().item())
        global_max = float(all_scores.max().item())
        score_debug = {
            "score_source": str(score_source),
            "num_pages": len(page_score_stats),
            "global_mean": float(all_scores.mean().item()),
            "global_std": float(all_scores.std(unbiased=False).item()),
            "global_min": global_min,
            "global_max": global_max,
            "global_range": float(global_max - global_min),
            "page_mean_min": (min(page_mean_values) if page_mean_values else None),
            "page_mean_max": (max(page_mean_values) if page_mean_values else None),
            "page_mean_spread": (
                float(max(page_mean_values) - min(page_mean_values)) if page_mean_values else 0.0
            ),
            "page_std_mean": (
                float(sum(page_std_values) / len(page_std_values)) if page_std_values else 0.0
            ),
            "page_stats": page_score_stats,
        }

    return VisualTokenPruningResult(
        pruned_image_embeds=pruned_image_embeds,
        pruned_grid_thw=pruned_grid_thw,
        page_keep_indices=keep_indices_padded,
        page_keep_counts=keep_counts_tensor,
        split_sizes=split_sizes,
        score_debug=score_debug,
        position_strategy="select_existing",
    )


def prune_visual_tokens_random_with_metadata(
    image_embeds: Tensor,
    image_grid_thw: Tensor,
    keep_ratio: float = 0.5,
    min_tokens_per_page: int = 16,
    *,
    spatial_merge_size: int = 2,
    record_score_stats: bool = False,
    random_seed: int = 42,
) -> VisualTokenPruningResult:
    token_scores: Tensor | None = None
    if keep_ratio < 0.999999 or record_score_stats:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))
        token_scores = torch.rand(
            (int(image_embeds.shape[0]),),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=image_embeds.device)

    return prune_visual_tokens_by_scores_with_metadata(
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        token_scores=token_scores,
        keep_ratio=keep_ratio,
        min_tokens_per_page=min_tokens_per_page,
        spatial_merge_size=spatial_merge_size,
        record_score_stats=record_score_stats,
        score_source="random",
    )


def prune_visual_tokens_spatial_downsample_with_metadata(
    image_embeds: Tensor,
    image_grid_thw: Tensor,
    keep_ratio: float = 0.25,
    min_tokens_per_page: int = 16,
    *,
    spatial_merge_size: int = 2,
    stride: int | None = None,
) -> VisualTokenPruningResult:
    if image_embeds.ndim != 2:
        raise ValueError("image_embeds 必须是 [total_visual_tokens, hidden_dim] 张量。")
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("spatial downsampling 的 keep_ratio 必须在 (0, 1] 内。")
    if min_tokens_per_page <= 0:
        raise ValueError("min_tokens_per_page 必须大于 0。")

    split_sizes = _compute_split_sizes(image_grid_thw, spatial_merge_size)
    total_visual_tokens = int(image_embeds.shape[0])
    if int(split_sizes.sum().item()) != total_visual_tokens:
        raise ValueError(
            "image_embeds 总 token 数与 image_grid_thw 解析出的 split_sizes 不一致："
            f"{total_visual_tokens} vs {int(split_sizes.sum().item())}。"
        )

    page_keep_indices: list[Tensor] = []
    page_keep_counts: list[int] = []
    pruned_pages: list[Tensor] = []
    pruned_grid_rows: list[Tensor] = []
    page_debug_rows: list[dict[str, Any]] = []

    offset = 0
    for page_idx, page_tokens in enumerate(split_sizes.tolist()):
        page_size = int(page_tokens)
        page_embeds = image_embeds[offset : offset + page_size]
        offset += page_size

        grid_t = int(image_grid_thw[page_idx, 0].item())
        grid_h = int(image_grid_thw[page_idx, 1].item()) // int(spatial_merge_size)
        grid_w = int(image_grid_thw[page_idx, 2].item()) // int(spatial_merge_size)
        if int(grid_t * grid_h * grid_w) != page_size:
            raise ValueError(
                "spatial downsampling 页内 grid 与 token 数不一致："
                f"page_idx={page_idx}, grid={grid_t}x{grid_h}x{grid_w}, tokens={page_size}。"
            )

        keep_idx, keep_rows, keep_cols = _build_spatial_keep_indices(
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            stride=stride,
            keep_ratio=(None if stride is not None else float(keep_ratio)),
        )
        if int(keep_idx.numel()) < int(min_tokens_per_page):
            keep_idx = torch.arange(page_size, dtype=torch.long)
            keep_rows = torch.arange(grid_h, dtype=torch.long)
            keep_cols = torch.arange(grid_w, dtype=torch.long)

        keep_idx = keep_idx.to(device=page_embeds.device, dtype=torch.long)
        kept_h = int(keep_rows.numel())
        kept_w = int(keep_cols.numel())
        keep_count = int(keep_idx.numel())

        page_keep_indices.append(keep_idx)
        page_keep_counts.append(keep_count)
        pruned_pages.append(page_embeds.index_select(0, keep_idx))
        pruned_grid_rows.append(
            torch.tensor(
                [grid_t, kept_h * int(spatial_merge_size), kept_w * int(spatial_merge_size)],
                dtype=torch.long,
                device=image_embeds.device,
            )
        )
        page_debug_rows.append(
            {
                "page_index": int(page_idx),
                "token_count": int(page_size),
                "keep_count": keep_count,
                "grid_t": grid_t,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "kept_grid_h": kept_h,
                "kept_grid_w": kept_w,
                "effective_keep_ratio": float(keep_count / max(1, page_size)),
                "stride": (None if stride is None else int(stride)),
            }
        )

    pruned_image_embeds = torch.cat(pruned_pages, dim=0) if pruned_pages else image_embeds.new_empty((0, image_embeds.shape[-1]))
    pruned_grid_thw = torch.stack(pruned_grid_rows, dim=0) if pruned_grid_rows else image_grid_thw.new_empty((0, 3))
    keep_indices_padded = _pad_keep_indices(page_keep_indices, device=image_embeds.device)
    keep_counts_tensor = torch.tensor(page_keep_counts, dtype=torch.long, device=image_embeds.device)

    return VisualTokenPruningResult(
        pruned_image_embeds=pruned_image_embeds,
        pruned_grid_thw=pruned_grid_thw,
        page_keep_indices=keep_indices_padded,
        page_keep_counts=keep_counts_tensor,
        split_sizes=split_sizes,
        score_debug={
            "score_source": "spatial_downsample",
            "spatial_stride": (None if stride is None else int(stride)),
            "page_stats": page_debug_rows,
        },
        position_strategy="recompute_from_pruned_grid",
    )


def prune_visual_tokens_with_metadata(
    image_embeds: Tensor,
    image_grid_thw: Tensor,
    query_embeds: Tensor,
    keep_ratio: float = 0.5,
    min_tokens_per_page: int = 16,
    *,
    spatial_merge_size: int = 2,
    record_score_stats: bool = False,
) -> VisualTokenPruningResult:
    """
    对 visual encoder 输出做按页的 query-aware top-k pruning。

    说明：
    - `pruned_grid_thw` 只保证“每页保留 token 数可被精确表达成一个伪网格”，
      便于统计/调试；任意 top-k 子集并不再对应原始规则 2D 网格。
    - 真正需要完全保真的位置对齐时，应直接保留被选 token 的原始 position ids。
    """

    if image_embeds.ndim != 2:
        raise ValueError("image_embeds 必须是 [total_visual_tokens, hidden_dim] 张量。")
    if query_embeds.ndim == 2:
        if query_embeds.shape[0] != 1:
            raise ValueError("query_embeds 若为二维张量，batch 维必须等于 1。")
        query_embeds = query_embeds.squeeze(0)
    if query_embeds.ndim != 1:
        raise ValueError("query_embeds 必须是一维 pooled embedding。")
    if image_embeds.shape[-1] != query_embeds.shape[-1]:
        raise ValueError("image_embeds 与 query_embeds 的 hidden dim 必须一致。")
    if keep_ratio <= 0.0:
        raise ValueError("keep_ratio 必须大于 0。")
    if min_tokens_per_page <= 0:
        raise ValueError("min_tokens_per_page 必须大于 0。")

    token_scores: Tensor | None = None
    if keep_ratio < 0.999999 or record_score_stats:
        query = query_embeds.to(device=image_embeds.device, dtype=image_embeds.dtype)
        token_scores = F.cosine_similarity(
            image_embeds.to(dtype=torch.float32),
            query.to(dtype=torch.float32).unsqueeze(0),
            dim=-1,
            eps=1e-6,
        )

    return prune_visual_tokens_by_scores_with_metadata(
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        token_scores=token_scores,
        keep_ratio=keep_ratio,
        min_tokens_per_page=min_tokens_per_page,
        spatial_merge_size=spatial_merge_size,
        record_score_stats=record_score_stats,
        score_source="cosine",
    )


def prune_visual_tokens(
    image_embeds: Tensor,
    image_grid_thw: Tensor,
    query_embeds: Tensor,
    keep_ratio: float = 0.5,
    min_tokens_per_page: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    result = prune_visual_tokens_with_metadata(
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        query_embeds=query_embeds,
        keep_ratio=keep_ratio,
        min_tokens_per_page=min_tokens_per_page,
    )
    return result.pruned_image_embeds, result.pruned_grid_thw, result.page_keep_indices
