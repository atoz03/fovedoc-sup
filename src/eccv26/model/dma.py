from __future__ import annotations

"""
DMA（Dynamic Mask Attention）在本项目中的最小可用实现与接入工具。

实现依据：docs/local/dynamic_mask_attention.md 中的公式 (3)(4) 与 Listing 1（PyTorch 参考实现）。

注意：
- 该实现优先保证“可训练/可复现/与 Transformers(Qwen3-VL) 兼容”，未做 CUDA kernel 级别加速。
- DMA 的 top-k 选择是离散操作（论文中也说明只用于 forward 的稀疏选择），梯度通过被选中的路径传播。
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Callable, Optional
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from eccv26.utils.exact_block import (
    EXACT_BLOCK_FEATURE_DIM,
    EXACT_BLOCK_FEATURE_INDEX,
    EXACT_BLOCK_TEXT_SKETCH_DIM,
    resolve_exact_block_scorer,
)


@dataclass(frozen=True)
class DMAConfig:
    enable: bool = False
    # 论文中用 w 表示每个 head 保留的 top-w key；这里直接用 keep_window_size 表达。
    keep_window_size: int = 2048
    # 为避免一次性构造 (B, H, Q, K) 的大张量导致显存/内存峰值过高，按 query 分块处理。
    query_block_size: int = 128
    # 推理优化：DMA bias 只参与 top-k 选择，选中后 attention logits 不再重复加入该 bias。
    # 默认关闭，避免改变既有训练口径；只在 canary 验证通过后用于加速评测。
    selection_bias_only: bool = False
    # gating 参数 A 的初始化（每个 head 一个标量）。
    a_init: float = 1.0
    # 作用范围：论文主线可以聚焦 Vision-DMA 或外接 cross-attn，因此默认不自动开启任何侧。
    apply_to_text: bool = False
    apply_to_vision: bool = False
    # 训练侧最小验证：只保留 `dma_dt_proj` / `dma_gate` 可训练，
    # 其余 LoRA、routing bias、小读出头全部冻结，只让主任务 CE 去更新 sparse mask 参数。
    mask_modules_only_enable: bool = False
    # 若开启，则训练时只更新 DMA 运行时小头（`dma_*`），冻结 LoRA 与其它参数。
    # 用于验证“只靠主任务 CE 训练所有 DMA 参数”的 query-conditioned 版本。
    runtime_modules_only_enable: bool = False
    # 视觉侧可选：加入“页内全局摘要条件化”的 trainable bias。
    # 它仍然只改 DMA 的打分，不引入 dense attention。
    vision_page_bias_enable: bool = False
    vision_page_bias_init: float = 0.0
    # 视觉侧可选：显式接入页内 2D 位置特征，做更强的 layout-aware 稀疏打分。
    vision_layout_bias_enable: bool = False
    vision_layout_bias_init: float = 0.0
    # 视觉侧可选：把文本 query 草图接入视觉 token 的 DMA 稀疏打分。
    # 这是“query-conditioned token routing”的最小落点：
    # 不改 top-k 稀疏范式，只在 top-k 前给每个视觉 token 一个 query-conditioned bias。
    vision_query_routing_enable: bool = False
    vision_query_routing_init: float = 0.0
    # 推理/评测侧消融：保留 query-routing adapter 模块加载，但不向视觉 DMA 注入 query sketch。
    # 这只用于估计已训练 adapter 对 query-conditioned routing 的运行时依赖，不等同于 paired retrain。
    vision_query_routing_runtime_ablate: bool = False
    # 视觉侧可选：把前层聚合出的 coarse block prior 回灌给后层 token routing。
    # 这是把“同权重 block prior + token routing + block bottleneck”压进同一次前向的最小混合版。
    vision_block_feedback_enable: bool = False
    vision_block_feedback_init: float = 0.0
    vision_block_feedback_start_layer: int = 1
    # 若开启，则不再做离散 hard top-k，而是改成 soft sigmoid gate，
    # 并把 `log(sigmoid(score))` 作为 attention log-bias 加回 logits。
    soft_gate_enable: bool = False
    # soft gate 版本下，给 `dma_dt_proj.bias` 的初始化偏置；+2.0 对应 sigmoid≈0.88，接近 dense。
    soft_gate_bias_init: float = 2.0
    # 训练侧可选：对“证据页是否被保住”加入辅助监督。
    aux_page_supervision_enable: bool = False
    aux_page_loss_weight: float = 0.0
    aux_page_loss_type: str = "bce"
    aux_page_margin: float = 0.5
    aux_page_ranking_topk: int = 2
    aux_page_budget_ratio: float = 0.0
    # readout-aligned page ranking 时，对 page_focus（二级排序项）的监督权重。
    aux_page_focus_loss_weight: float = 0.25
    # 单证据页 anti-collapse 监督：只在单正页样本上，要求 page_focus 不要被静态锚点页压过去。
    aux_single_page_focus_loss_weight: float = 0.0
    aux_single_page_margin: float = 0.5
    # 多证据页 retained-set 覆盖监督：专门盯住“最弱正页”不要被 budget 边界吞掉。
    aux_multi_page_loss_weight: float = 0.0
    aux_multi_page_margin: float = 0.5
    # 直接对齐最终 retained top-page 集合：不再只做 pairwise margin，而是监督“谁应该进 top-k retained set”。
    aux_retained_set_loss_weight: float = 0.0
    aux_retained_set_temperature: float = 0.25
    aux_retained_set_negative_weight: float = 0.25
    aux_retained_set_negative_topk: int = 0
    # 单证据页 retained recovery：直接用 page logit 盯住 budget 边界，避免证据页掉出 retained top-k。
    aux_single_page_retained_loss_weight: float = 0.0
    aux_single_page_retained_margin: float = 0.5
    # 是否启用“summary readout 对齐”的页/块级打分头；训练与评测共用同一读出。
    retention_readout_enable: bool = False
    # 评测/推理时 retained page budget 的显式覆盖值；若 >0，则与训练的 budget 口径保持一致。
    retained_page_budget_ratio: float = 0.0
    # 训练侧可选：对证据页内的热点 block 再加一层弱监督。
    # 支持：
    # - bce / ranking / budget_ranking：页内 block 监督
    # - readout_ranking / readout_budget_ranking：对齐最终 retained page/block 读出
    # 现阶段数据里的 block_id 与 DMA coarse block 空间并不完全同构，
    # 因此这里仍保留“无法精确监督时退化为 focus margin”的兜底路径。
    aux_block_supervision_enable: bool = False
    aux_block_loss_weight: float = 0.0
    aux_block_loss_type: str = "bce"
    aux_block_margin: float = 0.2
    aux_block_topk: int = 2
    # 直接对齐最终 retained top-block 集合：对正页内哪些 coarse block 应该留在 retained set 做显式监督。
    aux_retained_block_set_loss_weight: float = 0.0
    aux_retained_block_set_temperature: float = 0.25
    aux_retained_block_set_negative_weight: float = 0.25
    aux_retained_block_set_negative_topk: int = 0
    # retained block summary 的 top-k；训练和评测都应共享这一定义。
    retained_block_topk: int = 2
    # block retained readout 的特征口径：
    # - legacy: 旧版 coarse 稳定特征 [mean, mass, token_ratio]
    # - sharp: 尖特征 [max, topk_mean, token_ratio]
    # - mixed: 混合特征 [mean, mass, token_ratio, max, topk_mean]
    retention_block_feature_mode: str = "legacy"
    # 是否启用真实 exact block 的可训练读出头；训练与 eval 的 model selector 共用。
    exact_block_readout_enable: bool = False
    aux_exact_block_loss_weight: float = 0.0
    # exact block loss 的形式：
    # - soft_topk_bce：可微 top-k membership + BCE（默认，较“稳”但信号偏弱）
    # - infonce：多正样本 InfoNCE（更强的排序信号，优先用于提 block 命中）
    # - margin_ranking：页内 pos vs hard-neg 的 margin ranking（对“边界样本”更直接）
    exact_block_loss_type: str = "soft_topk_bce"
    exact_block_topk: int = 4
    exact_block_temperature: float = 0.25
    exact_block_negative_weight: float = 0.25
    exact_block_negative_topk: int = 0
    # 仅用于 margin_ranking
    exact_block_margin: float = 0.2
    # 训练时对 lexical 特征做 dropout，强制 exact block scorer 学 layout/page 等非纯词面特征。
    # 取值范围 [0,1]；0 表示关闭。
    exact_block_lexical_dropout: float = 0.0
    # 若离线标了 VLM rerank teacher，则可在 exact-block 正负监督之外再加一项 listwise distillation。
    exact_block_distill_loss_weight: float = 0.0
    exact_block_distill_temperature: float = 1.0
    # 若为 true，则把 exact-block 训练切成 block-only reranker：
    # 只更新 `dma_exact_block_proj`，不再对 page/block stage 施加辅助监督。
    exact_block_block_only_enable: bool = False
    # exact-block scorer 结构：
    # - linear：旧版逐 block 独立线性头
    # - page_interaction：按 page_id 做页内上下文汇聚，再用 residual MLP 打分
    # - query_interaction：引入 query/block text sketch，并做 query-conditioned 页内交互
    # - cross_block_attention：在 query 条件下做页内 cross-attention + self-attention rerank
    # - vlm_cross_encoder：直接用 Qwen3VL 对 block crop 做“是/否”teacher-forcing 打分
    exact_block_scorer_type: str = "page_interaction"
    exact_block_scorer_hidden_dim: int = 32
    # `vlm_cross_encoder` 仅对 lexical top-m 候选（并强制保留正样本）做显式 rerank。
    exact_block_vlm_rerank_topm: int = 8
    exact_block_vlm_crop_expand_ratio: float = 0.05
    # exact-block 线性头的初始化权重。默认弱化 lexical，强化 coarse/page 几何对齐。
    exact_block_init_lexical_weight: float = 0.35
    exact_block_init_overlap_weight: float = 0.15
    exact_block_init_coarse_weight: float = 0.75
    exact_block_init_page_weight: float = 0.15
    exact_block_init_focus_match_weight: float = 0.75
    exact_block_init_focus_l1_weight: float = -0.5


class DMAGate(nn.Module):
    """论文中的 gating 系数 A（每个 head 一个标量）。"""

    def __init__(self, num_heads: int, init: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.full((num_heads,), float(init)))

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.a


def _align_module_like(module: nn.Module | None, ref_tensor: torch.Tensor | None) -> None:
    """
    让动态注入的新模块显式跟随参考参数的 device/dtype。

    这对 `device_map=auto` 尤其重要；否则新增模块可能留在 CPU，
    与 attention 原参数落在不同设备，导致 matmul 时报错。
    """

    if module is None or ref_tensor is None:
        return

    target_kwargs: dict[str, Any] = {"device": ref_tensor.device}
    if ref_tensor.is_floating_point():
        target_kwargs["dtype"] = ref_tensor.dtype
    module.to(**target_kwargs)


def _resolve_runtime_module_param(module: nn.Module | None) -> torch.Tensor | None:
    """
    优先返回当前真正参与前向/训练的参数。

    在 PEFT `modules_to_save` 场景下，模块会被 `ModulesToSaveWrapper` 包裹；
    这时应优先读取活动 adapter 副本的参数，而不是被冻结的 original_module。
    """

    if module is None:
        return None

    modules_to_save = getattr(module, "modules_to_save", None)
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(modules_to_save, nn.ModuleDict) and active_adapters:
        adapter_name = active_adapters[0]
        if adapter_name in modules_to_save:
            active_module = modules_to_save[adapter_name]
            try:
                return next(active_module.parameters())
            except StopIteration:
                pass

    try:
        return next(module.parameters())
    except StopIteration:
        return None


def dma_modules_to_save(cfg: DMAConfig | None = None) -> list[str]:
    """
    用于 PEFT（LoRA）场景：把 DMA 的新增模块随 adapter 一起保存/加载。
    """

    modules = ["dma_dt_proj", "dma_gate"]
    if cfg is not None and bool(cfg.apply_to_vision) and bool(cfg.vision_page_bias_enable):
        modules.extend(["dma_page_token_proj", "dma_page_context_proj", "dma_page_gate"])
    if cfg is not None and bool(cfg.apply_to_vision) and bool(cfg.vision_layout_bias_enable):
        modules.extend(["dma_layout_proj", "dma_layout_gate"])
    if cfg is not None and bool(cfg.apply_to_vision) and bool(cfg.vision_query_routing_enable):
        modules.extend(["dma_query_token_proj", "dma_query_context_proj", "dma_query_gate"])
    if cfg is not None and bool(cfg.apply_to_vision) and bool(cfg.vision_block_feedback_enable):
        modules.extend(["dma_block_token_proj", "dma_block_context_proj", "dma_block_gate"])
    if cfg is not None and bool(cfg.apply_to_vision) and bool(cfg.retention_readout_enable):
        modules.extend(["dma_retention_page_proj", "dma_retention_block_proj"])
    scorer_type = ""
    if cfg is not None:
        scorer_type = str(getattr(cfg, "exact_block_scorer_type", "") or "").strip().lower()
    if (
        cfg is not None
        and bool(cfg.apply_to_vision)
        and bool(cfg.exact_block_readout_enable)
        and scorer_type not in {"vlm_cross_encoder", "vlm_rerank"}
    ):
        modules.append("dma_exact_block_proj")
    return modules


def _find_dma_stats_owner(model: nn.Module) -> nn.Module | None:
    if bool(model.__dict__.get("_dma_stats_owner_marker", False)):
        return model
    if "_dma_page_stats" in model.__dict__ or "_dma_last_page_summary" in model.__dict__:
        return model

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        owner = _find_dma_stats_owner(base_model)
        if owner is not None:
            return owner

    for module in model.modules():
        if bool(module.__dict__.get("_dma_stats_owner_marker", False)):
            return module
        owner_ref = getattr(module, "_dma_owner_ref", None)
        owner = owner_ref() if callable(owner_ref) else None
        if isinstance(owner, nn.Module) and bool(owner.__dict__.get("_dma_stats_owner_marker", False)):
            return owner
    return None


def _resolve_dma_query_routing_owner(model: nn.Module) -> nn.Module:
    owner = _find_dma_stats_owner(model)
    return owner if owner is not None else model


def _resolve_dma_input_embedding_module(model: nn.Module) -> nn.Module | None:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if embeddings is not None:
            return embeddings

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not model:
            embeddings = _resolve_dma_input_embedding_module(base_model)
            if embeddings is not None:
                return embeddings
    return None


def clear_dma_query_routing_sketch(model: nn.Module) -> None:
    owner = _resolve_dma_query_routing_owner(model)
    owner._dma_query_routing_sketch = None


def clear_dma_block_feedback_cache(model: nn.Module) -> None:
    owner = _resolve_dma_query_routing_owner(model)
    owner._dma_block_feedback_cache = {}


def _update_dma_block_feedback_cache(
    owner: nn.Module,
    *,
    image_index: int,
    layer_idx: int,
    block_logits: torch.Tensor,
) -> None:
    if not bool(getattr(owner, "_dma_block_feedback_enable", False)):
        return
    if not isinstance(block_logits, torch.Tensor) or int(block_logits.ndim) != 1 or int(block_logits.numel()) <= 0:
        return

    raw_cache = getattr(owner, "_dma_block_feedback_cache", None)
    if not isinstance(raw_cache, dict):
        raw_cache = {}
        owner._dma_block_feedback_cache = raw_cache

    runtime_logits = block_logits.detach().to(dtype=torch.float32)
    count = 1
    previous = raw_cache.get(int(image_index))
    if isinstance(previous, dict):
        previous_logits = previous.get("block_logits")
        previous_count = int(previous.get("count", 1) or 1)
        if isinstance(previous_logits, torch.Tensor) and tuple(previous_logits.shape) == tuple(runtime_logits.shape):
            runtime_logits = (
                previous_logits.to(device=runtime_logits.device, dtype=runtime_logits.dtype) * float(previous_count)
                + runtime_logits
            ) / float(previous_count + 1)
            count = previous_count + 1

    raw_cache[int(image_index)] = {
        "block_logits": runtime_logits,
        "count": int(count),
        "layer_idx": int(layer_idx),
    }


def set_dma_query_routing_sketch(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    image_token_id: int | None = None,
) -> bool:
    """
    从当前 batch 的文本 token 里提取一个 query sketch，供视觉 DMA 路由使用。

    训练时优先传入 prompt-only `text_mask`，避免把答案 token 泄漏进 routing；
    推理时没有答案，因此直接用 prompt 的 `attention_mask` 即可。
    """

    owner = _resolve_dma_query_routing_owner(model)
    if not bool(getattr(owner, "_dma_query_routing_enable", False)) or bool(
        getattr(owner, "_dma_query_routing_runtime_ablate", False)
    ):
        owner._dma_query_routing_sketch = None
        return False

    if not isinstance(input_ids, torch.Tensor) or int(input_ids.ndim) != 2:
        owner._dma_query_routing_sketch = None
        return False

    embedding_module = _resolve_dma_input_embedding_module(model)
    if embedding_module is None:
        owner._dma_query_routing_sketch = None
        return False
    try:
        embedding_param = next(embedding_module.parameters())
    except StopIteration:
        embedding_param = None
    embedding_device = input_ids.device if embedding_param is None else embedding_param.device

    if text_mask is not None and (
        not isinstance(text_mask, torch.Tensor) or tuple(text_mask.shape) != tuple(input_ids.shape)
    ):
        raise ValueError("text_mask 的形状必须与 input_ids 一致")
    if attention_mask is not None and (
        not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != tuple(input_ids.shape)
    ):
        raise ValueError("attention_mask 的形状必须与 input_ids 一致")

    routing_input_ids = input_ids.to(device=embedding_device, dtype=torch.long)
    valid_mask = torch.ones_like(routing_input_ids, dtype=torch.bool)
    if attention_mask is not None:
        valid_mask = valid_mask & attention_mask.to(device=routing_input_ids.device).bool()
    if text_mask is not None:
        valid_mask = valid_mask & text_mask.to(device=routing_input_ids.device).bool()
    if image_token_id is not None:
        valid_mask = valid_mask & routing_input_ids.ne(int(image_token_id))

    if not bool(valid_mask.any()):
        owner._dma_query_routing_sketch = None
        return False

    token_embeddings = embedding_module(routing_input_ids)
    mask_f = valid_mask.unsqueeze(-1).to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    pooled = (token_embeddings * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    owner._dma_query_routing_sketch = pooled / denom
    return True


def _should_record_dma_retention_stats(module: nn.Module, owner: nn.Module) -> bool:
    if bool(getattr(owner, "_dma_retention_stats_disabled", False)):
        return False
    target_layer_idx = getattr(owner, "_dma_retention_record_layer_idx", None)
    layer_idx = getattr(module, "_dma_layer_idx", None)
    if isinstance(target_layer_idx, int) and isinstance(layer_idx, int):
        return int(layer_idx) == int(target_layer_idx)
    return True


def reset_dma_page_stats(model: nn.Module) -> None:
    owner = _find_dma_stats_owner(model)
    if owner is None:
        return
    owner._dma_page_stats = []
    owner._dma_last_page_summary = None
    owner._dma_train_image_logits = []
    owner._dma_train_block_logits = []
    owner._dma_page_readout_logits = []
    owner._dma_block_feedback_cache = {}


def _freeze_non_record_retention_readout_modules(model: nn.Module) -> None:
    """
    retention 读出当前只在记录层参与监督。

    其余视觉层若继续保持可训练，会在 DDP 下稳定表现为 unused parameters。
    这里按记录层索引把非目标层读出头冻结，只保留真正参与 loss 的那一层。

    在 PEFT `modules_to_save` 包装下，活动 adapter 的副本模块才是真正参与前向的分支；
    `original_module` 必须始终保持冻结，否则 DDP 会把它也当成可训练参数追踪，
    在 checkpoint/reentrant backward 下容易再次命中 `ready twice`。
    """

    record_layer_idx = getattr(model, "_dma_retention_record_layer_idx", None)
    if not isinstance(record_layer_idx, int):
        return

    for module in model.modules():
        layer_idx = getattr(module, "_dma_layer_idx", None)
        if not isinstance(layer_idx, int):
            continue
        keep_trainable = int(layer_idx) == int(record_layer_idx)
        for attr_name in ("dma_retention_page_proj", "dma_retention_block_proj"):
            submodule = getattr(module, attr_name, None)
            if not isinstance(submodule, nn.Module):
                continue
            modules_to_save = getattr(submodule, "modules_to_save", None)
            original_module = getattr(submodule, "original_module", None)
            active_adapters = getattr(submodule, "active_adapters", None)
            if isinstance(modules_to_save, nn.ModuleDict) and original_module is not None:
                original_module.requires_grad_(False)
                if isinstance(active_adapters, str):
                    active_adapter_names = {active_adapters}
                else:
                    active_adapter_names = {str(name) for name in (active_adapters or [])}
                for adapter_name, adapter_module in modules_to_save.items():
                    adapter_module.requires_grad_(keep_trainable and adapter_name in active_adapter_names)
                continue
            for parameter in submodule.parameters():
                parameter.requires_grad_(keep_trainable)


class DMAExactBlockPageInteractionScorer(nn.Module):
    """
    页内交互式 exact-block scorer。

    设计目标：
    - 保持训练/评测接口仍然是 “candidate feature tensor -> score”
    - 但让每个 block 的得分显式依赖同页其他 block 的上下文，而不是逐 block 独立线性打分
    """

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.anchor_proj = nn.Linear(feature_dim, 1, bias=True)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feature_tensor: torch.Tensor,
        *,
        page_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(feature_tensor.shape[0]) == 0:
            return feature_tensor.new_empty((0,))
        x = feature_tensor
        if x.dtype != torch.float32:
            x = x.float()
        anchor = self.anchor_proj(x).reshape(-1)
        hidden = self.feature_mlp(self.feature_norm(x))
        if page_ids is None or int(page_ids.numel()) != int(hidden.shape[0]):
            page_ids = torch.zeros((int(hidden.shape[0]),), device=hidden.device, dtype=torch.long)
        else:
            page_ids = page_ids.to(device=hidden.device, dtype=torch.long).reshape(-1)

        page_mean = torch.zeros_like(hidden)
        page_max = torch.zeros_like(hidden)
        for page_id in torch.unique(page_ids, sorted=False):
            mask = page_ids == page_id
            if not bool(mask.any()):
                continue
            group_hidden = hidden[mask]
            group_mean = group_hidden.mean(dim=0, keepdim=True)
            group_max = group_hidden.max(dim=0, keepdim=True).values
            page_mean[mask] = group_mean.expand(int(mask.sum().item()), -1)
            page_max[mask] = group_max.expand(int(mask.sum().item()), -1)

        residual_input = torch.cat(
            [
                x,
                hidden,
                page_mean,
                page_max,
                hidden - page_mean,
                hidden - page_max,
            ],
            dim=-1,
        )
        residual = self.residual_proj(residual_input).reshape(-1)
        return anchor + residual


class DMAExactBlockQueryInteractionScorer(nn.Module):
    """
    带 query 条件化的 exact-block scorer。

    设计目标：
    - 不改现有 exact-block 评测协议，只扩 scorer 输入
    - 让模型看到 query/block 的轻量文本 sketch，而不是只看两个 lexical 标量
    - 仍保留页内 group interaction，避免退回逐 block 独立打分
    """

    def __init__(self, feature_dim: int, text_sketch_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.anchor_proj = nn.Linear(feature_dim, 1, bias=True)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.query_proj = nn.Linear(text_sketch_dim, hidden_dim, bias=False)
        self.block_proj = nn.Linear(text_sketch_dim, hidden_dim, bias=False)
        self.interaction_proj = nn.Linear(text_sketch_dim, hidden_dim, bias=False)
        self.residual_proj = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feature_tensor: torch.Tensor,
        *,
        page_ids: torch.Tensor | None = None,
        query_sketch: torch.Tensor | None = None,
        block_sketch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(feature_tensor.shape[0]) == 0:
            return feature_tensor.new_empty((0,))
        x = feature_tensor.float() if feature_tensor.dtype != torch.float32 else feature_tensor
        anchor = self.anchor_proj(x).reshape(-1)

        num_candidates = int(x.shape[0])
        hidden_dim = int(self.feature_proj.out_features)
        if page_ids is None or int(page_ids.numel()) != num_candidates:
            page_ids = torch.zeros((num_candidates,), device=x.device, dtype=torch.long)
        else:
            page_ids = page_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        if query_sketch is None or int(query_sketch.shape[0]) != num_candidates:
            query_sketch = torch.zeros((num_candidates, self.query_proj.in_features), device=x.device, dtype=x.dtype)
        else:
            query_sketch = query_sketch.to(device=x.device, dtype=x.dtype)
        if block_sketch is None or int(block_sketch.shape[0]) != num_candidates:
            block_sketch = torch.zeros((num_candidates, self.block_proj.in_features), device=x.device, dtype=x.dtype)
        else:
            block_sketch = block_sketch.to(device=x.device, dtype=x.dtype)

        query_hidden = self.query_proj(query_sketch)
        block_hidden = self.block_proj(block_sketch)
        interaction_hidden = self.interaction_proj(query_sketch * block_sketch)
        hidden = torch.nn.functional.silu(self.feature_proj(self.feature_norm(x)) + query_hidden + block_hidden + interaction_hidden)

        page_context = torch.zeros((num_candidates, hidden_dim), device=hidden.device, dtype=hidden.dtype)
        for page_id in torch.unique(page_ids, sorted=False):
            mask = page_ids == page_id
            if not bool(mask.any()):
                continue
            group_hidden = hidden[mask]
            group_query = query_hidden[mask]
            attn_logits = (group_hidden * group_query).sum(dim=-1) / math.sqrt(float(hidden_dim))
            attn = torch.softmax(attn_logits, dim=0).unsqueeze(-1)
            group_context = (attn * group_hidden).sum(dim=0, keepdim=True)
            page_context[mask] = group_context.expand(int(mask.sum().item()), -1)

        residual_input = torch.cat(
            [
                x,
                hidden,
                query_hidden,
                interaction_hidden,
                page_context,
                hidden - page_context,
            ],
            dim=-1,
        )
        residual = self.residual_proj(residual_input).reshape(-1)
        return anchor + residual


class DMAExactBlockCrossBlockAttentionScorer(nn.Module):
    """
    更强的页内 cross-block attention reranker。

    设计目标：
    - 保持 exact-block 的输入接口不变
    - 先用 query sketch 在页内做一次 cross-attention summary
    - 再让同页 block 之间经过 self-attention 做真正的相对排序交互
    """

    def __init__(self, feature_dim: int, text_sketch_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(16, int(hidden_dim))
        num_heads = 4 if hidden_dim % 4 == 0 else (2 if hidden_dim % 2 == 0 else 1)
        self.anchor_proj = nn.Linear(feature_dim, 1, bias=True)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.query_proj = nn.Linear(text_sketch_dim, hidden_dim, bias=False)
        self.block_proj = nn.Linear(text_sketch_dim, hidden_dim, bias=False)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feature_tensor: torch.Tensor,
        *,
        page_ids: torch.Tensor | None = None,
        query_sketch: torch.Tensor | None = None,
        block_sketch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(feature_tensor.shape[0]) == 0:
            return feature_tensor.new_empty((0,))
        x = feature_tensor.float() if feature_tensor.dtype != torch.float32 else feature_tensor
        anchor = self.anchor_proj(x).reshape(-1)

        num_candidates = int(x.shape[0])
        if page_ids is None or int(page_ids.numel()) != num_candidates:
            page_ids = torch.zeros((num_candidates,), device=x.device, dtype=torch.long)
        else:
            page_ids = page_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        if query_sketch is None or int(query_sketch.shape[0]) != num_candidates:
            query_sketch = torch.zeros((num_candidates, self.query_proj.in_features), device=x.device, dtype=x.dtype)
        else:
            query_sketch = query_sketch.to(device=x.device, dtype=x.dtype)
        if block_sketch is None or int(block_sketch.shape[0]) != num_candidates:
            block_sketch = torch.zeros((num_candidates, self.block_proj.in_features), device=x.device, dtype=x.dtype)
        else:
            block_sketch = block_sketch.to(device=x.device, dtype=x.dtype)

        query_hidden = self.query_proj(query_sketch)
        block_hidden = self.block_proj(block_sketch)
        hidden = torch.nn.functional.silu(self.feature_proj(self.feature_norm(x)) + block_hidden)

        page_context = torch.zeros_like(hidden)
        contextual_hidden = torch.zeros_like(hidden)
        for page_id in torch.unique(page_ids, sorted=False):
            mask = page_ids == page_id
            if not bool(mask.any()):
                continue
            group_hidden = hidden[mask]
            group_query = query_hidden[mask]
            group_tokens = self.token_norm(group_hidden + group_query)
            group_tokens_batch = group_tokens.unsqueeze(0)
            page_query = self.query_norm(group_query.mean(dim=0, keepdim=True)).unsqueeze(0)
            cross_summary, _ = self.cross_attn(
                page_query,
                group_tokens_batch,
                group_tokens_batch,
                need_weights=False,
            )
            cross_summary = cross_summary.squeeze(0)
            cross_context = cross_summary.expand(int(mask.sum().item()), -1)
            attn_input = group_tokens + cross_context
            self_context, _ = self.self_attn(
                attn_input.unsqueeze(0),
                attn_input.unsqueeze(0),
                attn_input.unsqueeze(0),
                need_weights=False,
            )
            mixed = group_hidden + self_context.squeeze(0)
            mixed = mixed + self.ffn(mixed)
            page_context[mask] = cross_context
            contextual_hidden[mask] = mixed

        residual_input = torch.cat(
            [
                x,
                hidden,
                query_hidden,
                page_context,
                contextual_hidden,
                contextual_hidden - page_context,
            ],
            dim=-1,
        )
        residual = self.residual_proj(residual_input).reshape(-1)
        return anchor + residual


def _flatten_module_parameter_vector(module: nn.Module) -> torch.Tensor | None:
    flat_params: list[torch.Tensor] = []
    for param in module.parameters():
        flat_params.append(param.detach().float().reshape(-1))
    if not flat_params:
        return None
    return torch.cat(flat_params, dim=0)


def _initialize_exact_block_anchor_weights(module: nn.Module, cfg: DMAConfig) -> None:
    anchor_proj = getattr(module, "anchor_proj", module)
    if not isinstance(anchor_proj, nn.Linear):
        return
    with torch.no_grad():
        anchor_proj.weight.zero_()
        anchor_proj.bias.zero_()
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["lexical_score_log"]] = float(
            getattr(cfg, "exact_block_init_lexical_weight", 0.35)
        )
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["lexical_overlap_ratio"]] = float(
            getattr(cfg, "exact_block_init_overlap_weight", 0.15)
        )
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["coarse_block_score"]] = float(
            getattr(cfg, "exact_block_init_coarse_weight", 0.75)
        )
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["page_score"]] = float(
            getattr(cfg, "exact_block_init_page_weight", 0.15)
        )
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["coarse_focus_match"]] = float(
            getattr(cfg, "exact_block_init_focus_match_weight", 0.75)
        )
        anchor_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["coarse_focus_l1"]] = float(
            getattr(cfg, "exact_block_init_focus_l1_weight", -0.5)
        )


def score_dma_exact_block_features(
    model: nn.Module,
    feature_tensor: torch.Tensor,
    page_id_tensor: torch.Tensor | None = None,
    query_sketch_tensor: torch.Tensor | None = None,
    block_sketch_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    scorer = resolve_exact_block_scorer(model)
    if scorer is None:
        return None
    if int(feature_tensor.shape[0]) == 0:
        return feature_tensor.new_empty((0,))
    ref_param = _resolve_runtime_module_param(scorer)
    runtime_device = feature_tensor.device if ref_param is None else ref_param.device
    runtime_dtype = feature_tensor.dtype if ref_param is None else ref_param.dtype
    runtime_features = feature_tensor.to(device=runtime_device, dtype=runtime_dtype)
    runtime_page_ids = None
    if page_id_tensor is not None:
        runtime_page_ids = page_id_tensor.to(device=runtime_device, dtype=torch.long)
    runtime_query_sketch = None
    if query_sketch_tensor is not None:
        runtime_query_sketch = query_sketch_tensor.to(device=runtime_device, dtype=runtime_dtype)
    runtime_block_sketch = None
    if block_sketch_tensor is not None:
        runtime_block_sketch = block_sketch_tensor.to(device=runtime_device, dtype=runtime_dtype)

    call_kwargs: dict[str, torch.Tensor] = {}
    if runtime_page_ids is not None:
        call_kwargs["page_ids"] = runtime_page_ids
    if runtime_query_sketch is not None:
        call_kwargs["query_sketch"] = runtime_query_sketch
    if runtime_block_sketch is not None:
        call_kwargs["block_sketch"] = runtime_block_sketch
    if call_kwargs:
        candidate_kwargs = [call_kwargs]
        if "page_ids" in call_kwargs:
            candidate_kwargs.append({"page_ids": call_kwargs["page_ids"]})
        if "query_sketch" in call_kwargs and "block_sketch" in call_kwargs:
            candidate_kwargs.append(
                {
                    "query_sketch": call_kwargs["query_sketch"],
                    "block_sketch": call_kwargs["block_sketch"],
                }
            )
        seen: set[tuple[str, ...]] = set()
        for kwargs in candidate_kwargs:
            key = tuple(sorted(kwargs.keys()))
            if key in seen:
                continue
            seen.add(key)
            try:
                return scorer(runtime_features, **kwargs).reshape(-1)
            except TypeError:
                continue
    return scorer(runtime_features).reshape(-1)


def _collect_dma_training_tensor_stats(
    model: nn.Module,
    *,
    num_images: int,
    tensor_key: str,
    attr_name: str = "_dma_train_image_logits",
) -> torch.Tensor | None:
    owner = _find_dma_stats_owner(model)
    if owner is None or num_images <= 0:
        return None

    raw_logits = getattr(owner, attr_name, None)
    if not isinstance(raw_logits, list) or len(raw_logits) == 0:
        return None

    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    ref_tensor: torch.Tensor | None = None
    for item in raw_logits:
        if not isinstance(item, dict):
            continue
        image_index = item.get("image_index")
        value = item.get(tensor_key)
        if isinstance(image_index, int) and torch.is_tensor(value):
            grouped[image_index].append(value)
            if ref_tensor is None:
                ref_tensor = value

    if ref_tensor is None:
        return None

    collected: list[torch.Tensor] = []
    for image_index in range(int(num_images)):
        tensors = grouped.get(image_index, [])
        if tensors:
            collected.append(torch.stack(tensors).mean(dim=0))
        else:
            collected.append(torch.zeros_like(ref_tensor))

    if not collected:
        return None
    return torch.stack(collected)


def collect_dma_training_page_logits(model: nn.Module, num_images: int) -> torch.Tensor | None:
    aligned_logits = _collect_dma_training_tensor_stats(
        model,
        num_images=num_images,
        tensor_key="logit",
        attr_name="_dma_page_readout_logits",
    )
    if aligned_logits is not None:
        return aligned_logits
    return _collect_dma_training_tensor_stats(model, num_images=num_images, tensor_key="logit")


def collect_dma_training_block_logits(model: nn.Module, num_images: int) -> torch.Tensor | None:
    aligned_block_logits = _collect_dma_training_tensor_stats(
        model,
        num_images=num_images,
        tensor_key="block_logits",
        attr_name="_dma_train_block_logits",
    )
    if aligned_block_logits is not None:
        return aligned_block_logits
    return _collect_dma_training_tensor_stats(model, num_images=num_images, tensor_key="block_logits")


def collect_dma_training_selected_ratios(model: nn.Module, num_images: int) -> torch.Tensor | None:
    owner = _find_dma_stats_owner(model)
    if owner is None or num_images <= 0:
        return None

    raw_stats = getattr(owner, "_dma_page_stats", None)
    if not isinstance(raw_stats, list) or len(raw_stats) == 0:
        return None

    grouped: dict[int, list[float]] = defaultdict(list)
    for item in raw_stats:
        if not isinstance(item, dict):
            continue
        image_index = item.get("image_index")
        selected_ratio = item.get("selected_ratio")
        if isinstance(image_index, int):
            try:
                grouped[image_index].append(float(selected_ratio))
            except (TypeError, ValueError):
                continue

    ratios = []
    for image_index in range(int(num_images)):
        image_ratios = grouped.get(image_index, [])
        if image_ratios:
            ratios.append(sum(image_ratios) / len(image_ratios))
        else:
            ratios.append(0.0)
    return torch.tensor(ratios, dtype=torch.float32)


def _collect_dma_retention_train_stats(owner: nn.Module) -> dict[int, dict[str, torch.Tensor]]:
    raw_logits = getattr(owner, "_dma_train_image_logits", None)
    if not isinstance(raw_logits, list) or len(raw_logits) == 0:
        return {}

    grouped_logits: dict[int, list[torch.Tensor]] = defaultdict(list)
    grouped_block_logits: dict[int, list[torch.Tensor]] = defaultdict(list)
    for item in raw_logits:
        if not isinstance(item, dict):
            continue
        image_index = item.get("image_index")
        page_logit = item.get("logit")
        block_logits = item.get("block_logits")
        if not isinstance(image_index, int) or not torch.is_tensor(page_logit):
            continue
        grouped_logits[image_index].append(page_logit.detach())
        if torch.is_tensor(block_logits):
            grouped_block_logits[image_index].append(block_logits.detach())

    raw_block_logits = getattr(owner, "_dma_train_block_logits", None)
    if isinstance(raw_block_logits, list) and len(raw_block_logits) > 0:
        grouped_block_logits = defaultdict(list)
        for item in raw_block_logits:
            if not isinstance(item, dict):
                continue
            image_index = item.get("image_index")
            block_logits = item.get("block_logits")
            if isinstance(image_index, int) and torch.is_tensor(block_logits):
                grouped_block_logits[image_index].append(block_logits.detach())

    train_stats: dict[int, dict[str, torch.Tensor]] = {}
    for image_index, page_logits in grouped_logits.items():
        stat: dict[str, torch.Tensor] = {
            "page_logit": torch.stack(page_logits).mean(dim=0),
        }
        block_tensors = grouped_block_logits.get(image_index, [])
        if block_tensors:
            stat["block_logits"] = torch.stack(block_tensors).mean(dim=0)
        train_stats[image_index] = stat
    return train_stats


def _resolve_retention_page_budget_ratio(owner: nn.Module, page_stats: list[dict[str, Any]]) -> float:
    override = getattr(owner, "_dma_retained_page_budget_ratio", 0.0)
    try:
        override_value = float(override)
    except (TypeError, ValueError):
        override_value = 0.0
    if override_value > 0.0:
        return override_value
    return sum(float(x["selected_ratio"]) for x in page_stats) / len(page_stats)


def _resolve_retention_block_topk(owner: nn.Module) -> int:
    raw_value = getattr(owner, "_dma_retained_block_topk", 2)
    try:
        block_topk = int(raw_value)
    except (TypeError, ValueError):
        block_topk = 2
    return max(1, block_topk)


def _normalize_retention_block_feature_mode(raw_value: Any) -> str:
    normalized = str(raw_value or "legacy").strip().lower()
    if normalized not in {"legacy", "sharp", "mixed"}:
        return "legacy"
    return normalized


def _resolve_retention_block_feature_mode(module: nn.Module, owner: nn.Module | None = None) -> str:
    raw_value = getattr(module, "dma_retention_block_feature_mode", None)
    if raw_value is None and owner is not None:
        raw_value = getattr(owner, "_dma_retention_block_feature_mode", None)
    return _normalize_retention_block_feature_mode(raw_value)


def _build_page_retention_feature_tensor(
    module: nn.Module,
    *,
    selected_ratio: float,
    selected_score_mean: float,
    evidence_score_mean: float,
    dominant_block_score_mean: float,
    dominant_block_mass: float,
) -> torch.Tensor | None:
    proj = getattr(module, "dma_retention_page_proj", None)
    if proj is None:
        return None
    ref_param = _resolve_runtime_module_param(proj)
    if ref_param is None:
        return None
    return torch.tensor(
        [
            [
                float(selected_ratio),
                float(selected_score_mean),
                float(evidence_score_mean),
                float(dominant_block_score_mean),
                float(dominant_block_mass),
            ]
        ],
        device=ref_param.device,
        dtype=ref_param.dtype,
    )


def _build_block_retention_feature_tensor(
    module: nn.Module,
    *,
    block_score_mean: torch.Tensor,
    block_score_mass: torch.Tensor,
    block_score_max: torch.Tensor,
    block_score_topk_mean: torch.Tensor,
    block_token_ratio: torch.Tensor,
) -> torch.Tensor | None:
    proj = getattr(module, "dma_retention_block_proj", None)
    if proj is None:
        return None
    ref_param = _resolve_runtime_module_param(proj)
    if ref_param is None:
        return None
    feature_mode = _resolve_retention_block_feature_mode(module)
    if feature_mode == "sharp":
        feature_tensors = [block_score_max, block_score_topk_mean, block_token_ratio]
    elif feature_mode == "mixed":
        feature_tensors = [block_score_mean, block_score_mass, block_token_ratio, block_score_max, block_score_topk_mean]
    else:
        feature_tensors = [block_score_mean, block_score_mass, block_token_ratio]
    return torch.stack([x.to(device=ref_param.device, dtype=ref_param.dtype) for x in feature_tensors], dim=-1)


def _aggregate_block_statistics(
    *,
    block_indices: torch.Tensor,
    token_scores: torch.Tensor,
    num_blocks: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_scores = token_scores.clamp_min(0.0)
    block_count = torch.zeros((num_blocks,), device=token_scores.device, dtype=token_scores.dtype)
    block_count.scatter_add_(0, block_indices, torch.ones_like(token_scores))
    block_token_ratio = block_count / block_count.sum().clamp_min(1.0)

    block_score_mass = torch.zeros((num_blocks,), device=token_scores.device, dtype=token_scores.dtype)
    block_score_mass.scatter_add_(0, block_indices, positive_scores)
    block_score_mean = block_score_mass / block_count.clamp_min(1.0)
    block_score_max = torch.zeros((num_blocks,), device=token_scores.device, dtype=token_scores.dtype)
    block_score_topk_mean = torch.zeros((num_blocks,), device=token_scores.device, dtype=token_scores.dtype)
    for block_index in range(int(num_blocks)):
        mask = block_indices == block_index
        if not bool(mask.any()):
            continue
        block_scores = positive_scores[mask]
        block_score_max[block_index] = block_scores.max()
        topk = max(1, min(int(block_scores.numel()), max(1, int(block_scores.numel() * 0.25))))
        block_score_topk_mean[block_index] = torch.topk(block_scores, k=topk).values.mean()
    return block_score_mean, block_score_mass, block_score_max, block_score_topk_mean, block_token_ratio, block_count


def _serialize_coarse_block_id(page_id: int | None, block_index: int) -> str:
    row_bin = int(block_index) // 3
    col_bin = int(block_index) % 3
    prefix = f"p{int(page_id)}" if page_id is not None else "p?"
    return f"{prefix}_g{row_bin}{col_bin}"


def _token_to_coarse_block_index(layout_features: torch.Tensor) -> torch.Tensor:
    row_centered = layout_features[:, 1]
    col_centered = layout_features[:, 2]
    row_bin = torch.clamp(torch.floor((row_centered + 1.0) * 1.5).to(torch.long), min=0, max=2)
    col_bin = torch.clamp(torch.floor((col_centered + 1.0) * 1.5).to(torch.long), min=0, max=2)
    return row_bin * 3 + col_bin


def _compute_training_block_logits(module: nn.Module, token_scores: torch.Tensor, num_tokens: int) -> torch.Tensor | None:
    image_index = getattr(module, "_dma_current_image_index", None)
    if not isinstance(image_index, int):
        return None

    layout_splits = getattr(module, "_dma_layout_feature_splits", None)
    if not isinstance(layout_splits, list) or not (0 <= image_index < len(layout_splits)):
        return None
    layout_features = layout_splits[image_index]
    if not isinstance(layout_features, torch.Tensor) or int(layout_features.shape[0]) != int(num_tokens):
        return None

    coarse_block_idx = _token_to_coarse_block_index(layout_features.to(device=token_scores.device))
    block_score_mean, block_score_mass, block_score_max, block_score_topk_mean, block_token_ratio, _ = _aggregate_block_statistics(
        block_indices=coarse_block_idx,
        token_scores=token_scores,
    )
    block_features = _build_block_retention_feature_tensor(
        module,
        block_score_mean=block_score_mean,
        block_score_mass=block_score_mass,
        block_score_max=block_score_max,
        block_score_topk_mean=block_score_topk_mean,
        block_token_ratio=block_token_ratio,
    )
    feature_mode = _resolve_retention_block_feature_mode(module)
    fallback_scores = block_score_mean if feature_mode in {"legacy", "mixed"} else block_score_topk_mean
    if block_features is None:
        return fallback_scores
    block_proj = getattr(module, "dma_retention_block_proj", None)
    if block_proj is None:
        return fallback_scores
    return block_proj(block_features).reshape(-1)


def _compute_training_page_and_block_logits(
    module: nn.Module,
    *,
    score_basis: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_index = getattr(module, "_dma_current_image_index", None)
    if not isinstance(image_index, int):
        return None, None
    if int(score_basis.shape[0]) != 1:
        return None, None

    token_scores = score_basis[0].amax(dim=0)
    top_token_count = max(1, min(int(num_tokens), max(1, int(num_tokens * 0.25))))
    top_token_mean = torch.topk(token_scores, k=top_token_count).values.mean()
    block_logits = _compute_training_block_logits(module, token_scores=token_scores, num_tokens=num_tokens)
    if block_logits is None:
        return top_token_mean, None
    block_logit = block_logits.max()
    return 0.5 * top_token_mean + 0.5 * block_logit, block_logits


def _record_dma_training_page_logit(module: nn.Module, score_basis: torch.Tensor, num_tokens: int) -> None:
    owner_ref = getattr(module, "_dma_owner_ref", None)
    owner = owner_ref() if callable(owner_ref) else None
    image_index = getattr(module, "_dma_current_image_index", None)
    if not isinstance(owner, nn.Module) or not isinstance(image_index, int):
        return
    if not _should_record_dma_retention_stats(module, owner):
        return

    page_logit, block_logits = _compute_training_page_and_block_logits(
        module,
        score_basis=score_basis,
        num_tokens=num_tokens,
    )
    if page_logit is None:
        return

    if not hasattr(owner, "_dma_train_image_logits") or not isinstance(owner._dma_train_image_logits, list):
        owner._dma_train_image_logits = []
    owner._dma_train_image_logits.append(
        {
            "layer_idx": int(getattr(module, "_dma_layer_idx", -1)),
            "image_index": int(image_index),
            "logit": page_logit,
            "block_logits": block_logits,
        }
    )


def summarize_dma_page_retention(model: nn.Module, input_page_ids: list[int] | None) -> dict[str, Any]:
    owner = _find_dma_stats_owner(model)
    if owner is None or not input_page_ids:
        return {}

    raw_stats = getattr(owner, "_dma_page_stats", None)
    if not isinstance(raw_stats, list) or len(raw_stats) == 0:
        return {}

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_stats:
        if not isinstance(item, dict):
            continue
        image_index = item.get("image_index")
        if isinstance(image_index, int):
            grouped[image_index].append(item)
    train_stats = _collect_dma_retention_train_stats(owner)
    retained_block_topk = _resolve_retention_block_topk(owner)

    page_stats: list[dict[str, Any]] = []
    retained_block_ids: list[str] = []
    summary_gate_weighted_sum = 0.0
    summary_gate_lt05_weighted_sum = 0.0
    summary_gate_weight = 0.0
    for image_index, page_id in enumerate(input_page_ids):
        items = grouped.get(image_index, [])
        if not items:
            page_stats.append(
                {
                    "page_id": int(page_id),
                    "page_index": int(image_index),
                    "page_score": 0.0,
                    "page_focus": 0.0,
                    "selected_ratio": 0.0,
                    "selected_tokens_mean": 0.0,
                    "soft_gate_mean": None,
                    "soft_gate_lt_05_ratio": None,
                    "top_block_ids": [],
                    "num_layers": 0,
                }
            )
            continue

        selected_ratio = sum(float(x["selected_ratio"]) for x in items) / len(items)
        selected_tokens_mean = sum(float(x["unique_selected_tokens"]) for x in items) / len(items)
        page_focus = sum(float(x.get("dominant_block_mass", 0.0)) for x in items) / len(items)
        gate_weighted_sum = 0.0
        gate_lt05_weighted_sum = 0.0
        gate_weight = 0.0
        for item in items:
            gate_mean = item.get("soft_gate_mean")
            gate_lt05_ratio = item.get("soft_gate_lt_05_ratio")
            if gate_mean is None or gate_lt05_ratio is None:
                continue
            try:
                token_weight = float(item.get("num_tokens", 0))
                gate_weighted_sum += float(gate_mean) * token_weight
                gate_lt05_weighted_sum += float(gate_lt05_ratio) * token_weight
                gate_weight += token_weight
            except (TypeError, ValueError):
                continue
        page_soft_gate_mean = (gate_weighted_sum / gate_weight) if gate_weight > 0.0 else None
        page_soft_gate_lt05_ratio = (gate_lt05_weighted_sum / gate_weight) if gate_weight > 0.0 else None
        if gate_weight > 0.0:
            summary_gate_weighted_sum += gate_weighted_sum
            summary_gate_lt05_weighted_sum += gate_lt05_weighted_sum
            summary_gate_weight += gate_weight

        page_score: float
        ranked_block_ids: list[str]
        coarse_block_scores: list[float]
        train_stat = train_stats.get(image_index)
        retention_logits = [
            float(x["retention_logit"])
            for x in items
            if x.get("retention_logit") is not None
        ]
        if retention_logits:
            page_score = sum(retention_logits) / len(retention_logits)
            ranked_block_ids = []
            coarse_block_scores = []
            if train_stat is not None:
                block_logits = train_stat.get("block_logits")
                if isinstance(block_logits, torch.Tensor) and int(block_logits.numel()) > 0:
                    page_focus = float(block_logits.max().item())
                    coarse_block_scores = [float(x) for x in block_logits.detach().cpu().tolist()]
                    for block_index in torch.argsort(block_logits, descending=True).tolist():
                        ranked_block_ids.append(_serialize_coarse_block_id(int(page_id), int(block_index)))
                        if len(ranked_block_ids) >= retained_block_topk:
                            break
            if not ranked_block_ids:
                block_score_accumulator: dict[int, list[float]] = defaultdict(list)
                for item in items:
                    for block_index, block_score in zip(item.get("top_block_indices", []), item.get("top_block_scores", [])):
                        block_score_accumulator[int(block_index)].append(float(block_score))
                coarse_block_scores = [
                    float(sum(block_score_accumulator.get(block_index, [0.0])) / len(block_score_accumulator.get(block_index, [1.0])))
                    for block_index in range(9)
                ]
                ranked_block_ids = [
                    _serialize_coarse_block_id(int(page_id), int(block_index))
                    for block_index, _ in sorted(
                        ((block_index, sum(scores) / len(scores)) for block_index, scores in block_score_accumulator.items()),
                        key=lambda x: (-float(x[1]), int(x[0])),
                    )[:retained_block_topk]
                ]
        elif train_stat is not None:
            page_score = float(train_stat["page_logit"].item())
            ranked_block_ids = []
            coarse_block_scores = []
            block_logits = train_stat.get("block_logits")
            if isinstance(block_logits, torch.Tensor) and int(block_logits.numel()) > 0:
                page_focus = float(block_logits.max().item())
                coarse_block_scores = [float(x) for x in block_logits.detach().cpu().tolist()]
                for block_index in torch.argsort(block_logits, descending=True).tolist():
                    ranked_block_ids.append(_serialize_coarse_block_id(int(page_id), int(block_index)))
                    if len(ranked_block_ids) >= retained_block_topk:
                        break
        else:
            score_key = "evidence_score_mean"
            if not any(score_key in x for x in items):
                score_key = "dominant_block_score_mean"
            if not any(score_key in x for x in items):
                score_key = "selected_score_mean" if any("selected_score_mean" in x for x in items) else "selected_dt_mean"
            page_score = sum(float(x[score_key]) for x in items) / len(items)
            block_score_accumulator: dict[int, list[float]] = defaultdict(list)
            for item in items:
                for block_index, block_score in zip(item.get("top_block_indices", []), item.get("top_block_scores", [])):
                    block_score_accumulator[int(block_index)].append(float(block_score))
            coarse_block_scores = [
                float(sum(block_score_accumulator.get(block_index, [0.0])) / len(block_score_accumulator.get(block_index, [1.0])))
                for block_index in range(9)
            ]
            ranked_block_ids = [
                _serialize_coarse_block_id(int(page_id), int(block_index))
                for block_index, _ in sorted(
                    ((block_index, sum(scores) / len(scores)) for block_index, scores in block_score_accumulator.items()),
                    key=lambda x: (-float(x[1]), int(x[0])),
                )[:retained_block_topk]
            ]
        page_stats.append(
            {
                "page_id": int(page_id),
                "page_index": int(image_index),
                "page_score": float(page_score),
                "page_focus": float(page_focus),
                "selected_ratio": float(selected_ratio),
                "selected_tokens_mean": float(selected_tokens_mean),
                "soft_gate_mean": (None if page_soft_gate_mean is None else float(page_soft_gate_mean)),
                "soft_gate_lt_05_ratio": (
                    None if page_soft_gate_lt05_ratio is None else float(page_soft_gate_lt05_ratio)
                ),
                "top_block_ids": ranked_block_ids,
                "coarse_block_scores": coarse_block_scores,
                "num_layers": int(len(items)),
            }
        )

    page_budget_ratio = _resolve_retention_page_budget_ratio(owner, page_stats)
    num_retained_pages = max(1, min(len(page_stats), int(round(len(page_stats) * page_budget_ratio))))
    ranked = sorted(
        page_stats,
        key=lambda x: (-float(x["page_score"]), -float(x.get("page_focus", 0.0)), -float(x["selected_ratio"]), int(x["page_index"])),
    )
    retained_pages = ranked[:num_retained_pages]
    retained_page_ids = [int(x["page_id"]) for x in retained_pages]
    for page in retained_pages:
        retained_block_ids.extend(str(block_id) for block_id in page.get("top_block_ids", []))

    summary = {
        "retained_page_ids": retained_page_ids,
        "retained_block_ids": sorted(set(retained_block_ids)),
        "retained_page_count": int(num_retained_pages),
        "retained_page_budget_ratio": float(page_budget_ratio),
        "soft_gate_mean": (summary_gate_weighted_sum / summary_gate_weight) if summary_gate_weight > 0.0 else None,
        "soft_gate_lt_05_ratio": (
            summary_gate_lt05_weighted_sum / summary_gate_weight if summary_gate_weight > 0.0 else None
        ),
        "retained_page_stats": page_stats,
    }
    owner._dma_last_page_summary = summary
    return summary


def _record_dma_page_stat(
    module: nn.Module,
    *,
    topk_idx: torch.Tensor,
    dt_sel: torch.Tensor,
    score_sel: torch.Tensor | None,
    num_tokens: int,
    full_gate_probs: torch.Tensor | None = None,
) -> None:
    owner_ref = getattr(module, "_dma_owner_ref", None)
    owner = owner_ref() if callable(owner_ref) else None
    image_index = getattr(module, "_dma_current_image_index", None)
    if not isinstance(owner, nn.Module) or not isinstance(image_index, int):
        return
    if not _should_record_dma_retention_stats(module, owner):
        return

    if not hasattr(owner, "_dma_page_stats") or not isinstance(owner._dma_page_stats, list):
        owner._dma_page_stats = []

    unique_selected_tokens = int(torch.unique(topk_idx.detach().reshape(-1)).numel())
    selected_dt_mean = float(dt_sel.detach().float().mean().item())
    score_tensor = dt_sel if score_sel is None else score_sel
    selected_score_mean = float(score_tensor.detach().float().mean().item())
    max_selected_score = float(score_tensor.detach().float().max().item())
    soft_gate_mean = None
    soft_gate_lt_05_ratio = None
    if full_gate_probs is not None:
        gate_tensor = full_gate_probs.detach().float()
        soft_gate_mean = float(gate_tensor.mean().item())
        soft_gate_lt_05_ratio = float((gate_tensor < 0.5).float().mean().item())
    evidence_score_mean = max_selected_score
    dominant_block_score_mean = selected_score_mean
    dominant_block_mass = 0.0
    block_rank_scores_tensor: torch.Tensor | None = None
    top_block_indices: list[int] = []
    top_block_scores: list[float] = []

    flat_topk = topk_idx.detach().reshape(-1).to(dtype=torch.long)
    flat_scores = score_tensor.detach().reshape(-1).float()
    if flat_topk.numel() > 0:
        unique_tokens, inverse = torch.unique(flat_topk, return_inverse=True, sorted=True)
        token_score_max = torch.full(
            (int(unique_tokens.numel()),),
            fill_value=torch.finfo(flat_scores.dtype).min,
            device=flat_scores.device,
            dtype=flat_scores.dtype,
        )
        token_score_max.scatter_reduce_(0, inverse, flat_scores, reduce="amax", include_self=True)
        top_token_count = max(1, min(int(unique_tokens.numel()), max(1, int(unique_tokens.numel() * 0.25))))
        evidence_score_mean = float(torch.topk(token_score_max, k=top_token_count).values.mean().item())

        layout_splits = getattr(module, "_dma_layout_feature_splits", None)
        if isinstance(layout_splits, list) and 0 <= image_index < len(layout_splits):
            layout_features = layout_splits[image_index]
            if isinstance(layout_features, torch.Tensor) and int(layout_features.shape[0]) == int(num_tokens):
                coarse_block_idx = _token_to_coarse_block_index(layout_features.to(device=unique_tokens.device))
                selected_block_idx = coarse_block_idx[unique_tokens]
                block_score_mean, block_score_mass, block_score_max, block_score_topk_mean, block_token_ratio, block_count = _aggregate_block_statistics(
                    block_indices=selected_block_idx,
                    token_scores=token_score_max,
                )
                block_features = _build_block_retention_feature_tensor(
                    module,
                    block_score_mean=block_score_mean,
                    block_score_mass=block_score_mass,
                    block_score_max=block_score_max,
                    block_score_topk_mean=block_score_topk_mean,
                    block_token_ratio=block_token_ratio,
                )
                feature_mode = _resolve_retention_block_feature_mode(module, owner)
                block_rank_scores = block_score_mean if feature_mode in {"legacy", "mixed"} else block_score_topk_mean
                block_proj = getattr(module, "dma_retention_block_proj", None)
                if block_features is not None and block_proj is not None:
                    block_rank_scores = block_proj(block_features).reshape(-1)
                block_rank_scores_tensor = block_rank_scores
                dominant_block_score_mean = float(block_rank_scores.max().item())
                dominant_block_mass = float(block_score_mass.max().item())
                ranked_blocks = torch.argsort(block_rank_scores, descending=True)
                retained_block_topk = _resolve_retention_block_topk(owner)
                for block_index in ranked_blocks.tolist():
                    if float(block_count[block_index].item()) <= 0.0:
                        continue
                    top_block_indices.append(int(block_index))
                    top_block_scores.append(float(block_rank_scores[block_index].item()))
                    if len(top_block_indices) >= retained_block_topk:
                        break

    if block_rank_scores_tensor is not None:
        if not hasattr(owner, "_dma_train_block_logits") or not isinstance(owner._dma_train_block_logits, list):
            owner._dma_train_block_logits = []
        owner._dma_train_block_logits.append(
            {
                "layer_idx": int(getattr(module, "_dma_layer_idx", -1)),
                "image_index": int(image_index),
                "block_logits": block_rank_scores_tensor,
            }
        )
        _update_dma_block_feedback_cache(
            owner,
            image_index=int(image_index),
            layer_idx=int(getattr(module, "_dma_layer_idx", -1)),
            block_logits=block_rank_scores_tensor,
        )

    retention_logit = None
    retention_features = _build_page_retention_feature_tensor(
        module,
        selected_ratio=float(unique_selected_tokens / max(1, int(num_tokens))),
        selected_score_mean=float(selected_score_mean),
        evidence_score_mean=float(evidence_score_mean),
        dominant_block_score_mean=float(dominant_block_score_mean),
        dominant_block_mass=float(dominant_block_mass),
    )
    if retention_features is not None:
        retention_proj = getattr(module, "dma_retention_page_proj", None)
        if retention_proj is not None:
            retention_logit_tensor = retention_proj(retention_features).reshape(())
            retention_logit = float(retention_logit_tensor.detach().item())
            if not hasattr(owner, "_dma_page_readout_logits") or not isinstance(owner._dma_page_readout_logits, list):
                owner._dma_page_readout_logits = []
            owner._dma_page_readout_logits.append(
                {
                    "layer_idx": int(getattr(module, "_dma_layer_idx", -1)),
                    "image_index": int(image_index),
                    "logit": retention_logit_tensor,
                }
            )

    owner._dma_page_stats.append(
        {
            "layer_idx": int(getattr(module, "_dma_layer_idx", -1)),
            "image_index": int(image_index),
            "num_tokens": int(num_tokens),
            "unique_selected_tokens": int(unique_selected_tokens),
            "selected_ratio": float(unique_selected_tokens / max(1, int(num_tokens))),
            "selected_dt_mean": float(selected_dt_mean),
            "selected_score_mean": float(selected_score_mean),
            "max_selected_score": float(max_selected_score),
            "soft_gate_mean": (None if soft_gate_mean is None else float(soft_gate_mean)),
            "soft_gate_lt_05_ratio": (
                None if soft_gate_lt_05_ratio is None else float(soft_gate_lt_05_ratio)
            ),
            "evidence_score_mean": float(evidence_score_mean),
            "dominant_block_score_mean": float(dominant_block_score_mean),
            "dominant_block_mass": float(dominant_block_mass),
            "retention_logit": None if retention_logit is None else float(retention_logit),
            "top_block_indices": top_block_indices,
            "top_block_scores": top_block_scores,
        }
    )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    复制 KV head 以匹配 Q head（适配 GQA/MQA）。
    输入: (B, H_kv, S, D)
    输出: (B, H_q,  S, D)，其中 H_q = H_kv * n_rep
    """

    if n_rep == 1:
        return hidden_states
    b, hk, s, d = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(b, hk, n_rep, s, d)
    return expanded.reshape(b, hk * n_rep, s, d)


def _normalize_layout_axis(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 1:
        return torch.zeros((1,), device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, steps=int(length), device=device, dtype=dtype)


def _build_layout_feature_splits(
    *,
    grid_thw: torch.Tensor,
    merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """
    为每个视觉 chunk 生成与 token 顺序严格对齐的 layout 特征。

    当前特征：
    - `t_centered`：时间维归一化坐标（图片恒为 0）
    - `row_centered / col_centered`：页内 2D 中心化坐标
    - `|row_centered| / |col_centered|`：离中心的轴向距离
    - `radial`：二维半径，刻画更粗粒度的块位置
    """

    layout_splits: list[torch.Tensor] = []
    for num_frames, height, width in grid_thw.detach().cpu().tolist():
        num_frames = int(num_frames)
        height = int(height)
        width = int(width)
        if num_frames <= 0 or height <= 0 or width <= 0:
            continue

        merged_h = max(1, height // int(merge_size))
        merged_w = max(1, width // int(merge_size))
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(int(merge_size), device=device)
        intra_col = torch.arange(int(merge_size), device=device)

        row_idx = block_rows[:, None, None, None] * int(merge_size) + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * int(merge_size) + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, int(merge_size), int(merge_size)).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, int(merge_size), int(merge_size)).reshape(-1)

        row_axis = _normalize_layout_axis(height, device=device, dtype=dtype)
        col_axis = _normalize_layout_axis(width, device=device, dtype=dtype)
        row_centered = row_axis[row_idx]
        col_centered = col_axis[col_idx]
        radial = torch.sqrt((row_centered.square() + col_centered.square()).clamp_min(0.0)) / 1.41421356237

        if num_frames == 1:
            t_values = torch.zeros((1,), device=device, dtype=dtype)
        else:
            t_values = torch.linspace(-1.0, 1.0, steps=num_frames, device=device, dtype=dtype)

        for frame_idx in range(num_frames):
            t_centered = torch.full_like(row_centered, fill_value=float(t_values[frame_idx].item()))
            features = torch.stack(
                (
                    t_centered,
                    row_centered,
                    col_centered,
                    row_centered.abs(),
                    col_centered.abs(),
                    radial,
                ),
                dim=-1,
            )
            layout_splits.append(features)

    return layout_splits


def _resolve_vision_chunk_lengths(
    *,
    cu_seqlens: torch.Tensor,
    seq_length: int,
    layout_splits: list[torch.Tensor] | None,
) -> list[int]:
    """
    解析视觉 attention 的逐页 token 长度。

    正常情况下优先使用上游 `cu_seqlens`。
    但当前环境下，推理时偶发出现 `cu_seqlens` 差分全 0 的回归，
    会直接导致 `torch.split` 崩溃。此时回退到与 `grid_thw`
    同源构造的 `layout_splits` 长度，保证按页切分仍与视觉 token 对齐。
    """

    lengths = [int(x) for x in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()]
    if lengths and sum(lengths) == int(seq_length) and all(x > 0 for x in lengths):
        return lengths

    if isinstance(layout_splits, list):
        fallback = [int(split.shape[0]) for split in layout_splits if isinstance(split, torch.Tensor) and int(split.shape[0]) > 0]
        if fallback and sum(fallback) == int(seq_length):
            return fallback

    raise ValueError(
        "无法解析视觉 chunk 长度："
        f" seq_length={int(seq_length)},"
        f" cu_seqlens={cu_seqlens.detach().cpu().tolist()},"
        f" layout_split_count={0 if layout_splits is None else len(layout_splits)}"
    )


def _resolve_vision_rotary_pos_emb_fn(attn: nn.Module) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """按 attention 模块来源解析视觉 RoPE 函数，便于 Qwen2/Qwen3 共享视觉 DMA patch。"""

    module_name = attn.__class__.__module__
    if "qwen2_vl" in module_name:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

        return apply_rotary_pos_emb_vision
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    return apply_rotary_pos_emb_vision


def _compute_dma_score_bias(module: nn.Module, v_flat: torch.Tensor) -> torch.Tensor | None:
    """
    视觉侧可选的 page-conditioned bias。

    设计动机：
    - 原始 dt 只看单 token 的 value；
    - 这里额外引入“页内全局摘要 + token 内容”的联合打分；
    - 仍然只是给 DMA 的稀疏选择加偏置，不改变其 top-k 稀疏范式。
    """

    score_terms: list[torch.Tensor] = []

    if bool(getattr(module, "dma_use_page_bias", False)):
        token_proj = getattr(module, "dma_page_token_proj", None)
        context_proj = getattr(module, "dma_page_context_proj", None)
        gate = getattr(module, "dma_page_gate", None)
        if token_proj is not None and context_proj is not None and gate is not None:
            token_input = v_flat.to(dtype=token_proj.weight.dtype)
            token_bias = token_proj(token_input)  # (B, K, H)
            page_summary = token_input.mean(dim=1)  # (B, H*D)
            context_bias = context_proj(page_summary)[:, None, :]  # (B, 1, H)
            page_gate = gate(page_summary).to(dtype=token_bias.dtype, device=token_bias.device)  # (H,)
            score_terms.append((torch.tanh(token_bias + context_bias) * page_gate[None, None, :]).transpose(1, 2))

    if bool(getattr(module, "dma_use_layout_bias", False)):
        layout_proj = getattr(module, "dma_layout_proj", None)
        layout_gate = getattr(module, "dma_layout_gate", None)
        image_index = getattr(module, "_dma_current_image_index", None)
        layout_splits = getattr(module, "_dma_layout_feature_splits", None)
        if (
            layout_proj is not None
            and layout_gate is not None
            and isinstance(image_index, int)
            and isinstance(layout_splits, list)
            and 0 <= image_index < len(layout_splits)
        ):
            layout_features = layout_splits[image_index]
            if isinstance(layout_features, torch.Tensor) and int(layout_features.shape[0]) == int(v_flat.shape[1]):
                layout_input = layout_features.to(device=v_flat.device, dtype=layout_proj.weight.dtype)
                layout_bias = layout_proj(layout_input).transpose(0, 1).unsqueeze(0)  # (1, H, K)
                layout_gate_value = layout_gate(layout_input).to(dtype=layout_bias.dtype, device=layout_bias.device)  # (H,)
                layout_bias = torch.tanh(layout_bias) * layout_gate_value[None, :, None]
                if int(v_flat.shape[0]) > 1:
                    layout_bias = layout_bias.expand(int(v_flat.shape[0]), -1, -1)
                score_terms.append(layout_bias)

    if bool(getattr(module, "dma_use_query_routing", False)):
        token_proj = getattr(module, "dma_query_token_proj", None)
        context_proj = getattr(module, "dma_query_context_proj", None)
        gate = getattr(module, "dma_query_gate", None)
        owner_ref = getattr(module, "_dma_owner_ref", None)
        owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
        query_sketch = None if owner is None else getattr(owner, "_dma_query_routing_sketch", None)
        if (
            token_proj is not None
            and context_proj is not None
            and gate is not None
            and isinstance(query_sketch, torch.Tensor)
            and int(query_sketch.ndim) == 2
            and int(query_sketch.shape[-1]) == int(context_proj.in_features)
        ):
            token_input = v_flat.to(dtype=token_proj.weight.dtype)
            token_bias = token_proj(token_input)  # (B, K, H)
            query_input = query_sketch.to(device=token_bias.device, dtype=context_proj.weight.dtype)
            if int(query_input.shape[0]) == 1 and int(token_bias.shape[0]) > 1:
                query_input = query_input.expand(int(token_bias.shape[0]), -1)
            elif int(query_input.shape[0]) != int(token_bias.shape[0]):
                query_input = query_input[: int(token_bias.shape[0])]
            if int(query_input.shape[0]) == int(token_bias.shape[0]):
                context_bias = context_proj(query_input)[:, None, :]  # (B, 1, H)
                query_gate = gate(query_input).to(dtype=token_bias.dtype, device=token_bias.device)  # (H,)
                score_terms.append(
                    (torch.tanh(token_bias + context_bias.to(dtype=token_bias.dtype)) * query_gate[None, None, :]).transpose(1, 2)
                )

    if bool(getattr(module, "dma_use_block_feedback", False)):
        token_proj = getattr(module, "dma_block_token_proj", None)
        context_proj = getattr(module, "dma_block_context_proj", None)
        gate = getattr(module, "dma_block_gate", None)
        owner_ref = getattr(module, "_dma_owner_ref", None)
        owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
        image_index = getattr(module, "_dma_current_image_index", None)
        layout_splits = getattr(module, "_dma_layout_feature_splits", None)
        layer_idx = int(getattr(module, "_dma_layer_idx", -1))
        cache = None if owner is None else getattr(owner, "_dma_block_feedback_cache", None)
        start_layer = 0 if owner is None else int(getattr(owner, "_dma_block_feedback_start_layer", 1))
        if (
            token_proj is not None
            and context_proj is not None
            and gate is not None
            and isinstance(owner, nn.Module)
            and isinstance(image_index, int)
            and isinstance(layout_splits, list)
            and 0 <= image_index < len(layout_splits)
            and isinstance(cache, dict)
            and layer_idx >= start_layer
        ):
            cache_item = cache.get(int(image_index))
            cached_block_logits = cache_item.get("block_logits") if isinstance(cache_item, dict) else None
            layout_features = layout_splits[image_index]
            if (
                isinstance(cached_block_logits, torch.Tensor)
                and int(cached_block_logits.ndim) == 1
                and int(cached_block_logits.numel()) > 0
                and isinstance(layout_features, torch.Tensor)
                and int(layout_features.shape[0]) == int(v_flat.shape[1])
            ):
                token_input = v_flat.to(dtype=token_proj.weight.dtype)
                token_bias = token_proj(token_input)  # (B, K, H)
                coarse_block_idx = _token_to_coarse_block_index(layout_features.to(device=cached_block_logits.device))
                coarse_block_idx = coarse_block_idx.clamp(min=0, max=max(0, int(cached_block_logits.numel()) - 1))
                block_prior = cached_block_logits.to(device=token_bias.device, dtype=context_proj.weight.dtype)
                token_block_prior = block_prior.index_select(0, coarse_block_idx).unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
                if int(token_bias.shape[0]) > 1:
                    token_block_prior = token_block_prior.expand(int(token_bias.shape[0]), -1, -1)
                context_bias = context_proj(token_block_prior)  # (B, K, H)
                block_gate = gate(token_block_prior).to(dtype=token_bias.dtype, device=token_bias.device)  # (H,)
                score_terms.append(
                    (torch.tanh(token_bias + context_bias.to(dtype=token_bias.dtype)) * block_gate[None, None, :]).transpose(1, 2)
                )

    if not score_terms:
        return None

    score_bias = score_terms[0]
    for term in score_terms[1:]:
        score_bias = score_bias + term.to(dtype=score_bias.dtype, device=score_bias.device)
    return score_bias


def _compute_dma_score_bias_batch(
    module: nn.Module,
    v_flat: torch.Tensor,
    *,
    image_indices: list[int],
) -> torch.Tensor | None:
    """批量视觉快路径的 DMA bias 计算。

    原 `_compute_dma_score_bias` 依赖 `_dma_current_image_index`，一次只处理一页。
    这里只覆盖当前主线实际使用的 page/layout/query routing；block feedback 仍走旧逐页路径，
    避免把跨层 cache 语义改坏。
    """

    if bool(getattr(module, "dma_use_block_feedback", False)):
        return None

    score_terms: list[torch.Tensor] = []
    batch_size = int(v_flat.shape[0])

    if bool(getattr(module, "dma_use_page_bias", False)):
        token_proj = getattr(module, "dma_page_token_proj", None)
        context_proj = getattr(module, "dma_page_context_proj", None)
        gate = getattr(module, "dma_page_gate", None)
        if token_proj is not None and context_proj is not None and gate is not None:
            token_input = v_flat.to(dtype=token_proj.weight.dtype)
            token_bias = token_proj(token_input)  # (B, K, H)
            page_summary = token_input.mean(dim=1)  # (B, H*D)
            context_bias = context_proj(page_summary)[:, None, :]  # (B, 1, H)
            page_gate = gate(page_summary).to(dtype=token_bias.dtype, device=token_bias.device)  # (H,)
            score_terms.append((torch.tanh(token_bias + context_bias) * page_gate[None, None, :]).transpose(1, 2))

    if bool(getattr(module, "dma_use_layout_bias", False)):
        layout_proj = getattr(module, "dma_layout_proj", None)
        layout_gate = getattr(module, "dma_layout_gate", None)
        layout_splits = getattr(module, "_dma_layout_feature_splits", None)
        layout_inputs: list[torch.Tensor] = []
        if layout_proj is not None and layout_gate is not None and isinstance(layout_splits, list):
            for image_index in image_indices:
                if not (0 <= int(image_index) < len(layout_splits)):
                    layout_inputs = []
                    break
                layout_features = layout_splits[int(image_index)]
                if not (
                    isinstance(layout_features, torch.Tensor)
                    and int(layout_features.shape[0]) == int(v_flat.shape[1])
                ):
                    layout_inputs = []
                    break
                layout_inputs.append(layout_features.to(device=v_flat.device, dtype=layout_proj.weight.dtype))
        if len(layout_inputs) == batch_size:
            layout_batch = torch.stack(layout_inputs, dim=0)  # (B, K, 6)
            layout_bias = layout_proj(layout_batch).permute(0, 2, 1)  # (B, H, K)
            layout_gate_value = layout_gate(layout_batch.reshape(-1, layout_batch.shape[-1])).to(
                dtype=layout_bias.dtype,
                device=layout_bias.device,
            )
            score_terms.append(torch.tanh(layout_bias) * layout_gate_value[None, :, None])

    if bool(getattr(module, "dma_use_query_routing", False)):
        token_proj = getattr(module, "dma_query_token_proj", None)
        context_proj = getattr(module, "dma_query_context_proj", None)
        gate = getattr(module, "dma_query_gate", None)
        owner_ref = getattr(module, "_dma_owner_ref", None)
        owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
        query_sketch = None if owner is None else getattr(owner, "_dma_query_routing_sketch", None)
        if (
            token_proj is not None
            and context_proj is not None
            and gate is not None
            and isinstance(query_sketch, torch.Tensor)
            and int(query_sketch.ndim) == 2
            and int(query_sketch.shape[-1]) == int(context_proj.in_features)
        ):
            token_input = v_flat.to(dtype=token_proj.weight.dtype)
            token_bias = token_proj(token_input)  # (B, K, H)
            query_input = query_sketch.to(device=token_bias.device, dtype=context_proj.weight.dtype)
            if int(query_input.shape[0]) == 1 and batch_size > 1:
                query_input = query_input.expand(batch_size, -1)
            elif int(query_input.shape[0]) != batch_size:
                query_input = query_input[:batch_size]
            if int(query_input.shape[0]) == batch_size:
                context_bias = context_proj(query_input)[:, None, :]  # (B, 1, H)
                query_gate = gate(query_input).to(dtype=token_bias.dtype, device=token_bias.device)  # (H,)
                score_terms.append(
                    (torch.tanh(token_bias + context_bias.to(dtype=token_bias.dtype)) * query_gate[None, None, :]).transpose(
                        1, 2
                    )
                )

    if not score_terms:
        return None

    score_bias = score_terms[0]
    for term in score_terms[1:]:
        score_bias = score_bias + term.to(dtype=score_bias.dtype, device=score_bias.device)
    return score_bias


def _dma_attention_forward_full_batch(
    *,
    module: nn.Module,
    query: torch.Tensor,  # (B, Hq, Q, D)
    key: torch.Tensor,  # (B, Hkv, K, D)
    value: torch.Tensor,  # (B, Hkv, K, D)
    image_indices: list[int],
    scaling: float,
    dropout: float,
    keep_window_size: int,
) -> torch.Tensor:
    """等长多页的批量 DMA attention。

    当 `w >= K` 时等价于全量 DMA attention；当 `w < K` 时等价于逐页旧路径的
    shared top-k 稀疏 attention，但把多页放到 batch 维度一起跑。
    """

    key_states = _repeat_kv(key, int(getattr(module, "num_key_value_groups", 1)))
    value_states = _repeat_kv(value, int(getattr(module, "num_key_value_groups", 1)))

    bsz, num_heads, k_len, head_dim = key_states.shape
    w = min(max(1, int(keep_window_size)), int(k_len))
    v_flat = value_states.transpose(1, 2).reshape(bsz, k_len, num_heads * head_dim)
    dt_input = v_flat.to(dtype=module.dma_dt_proj.weight.dtype)
    dt_raw = module.dma_dt_proj(dt_input)
    a = module.dma_gate(dt_raw).to(dtype=dt_raw.dtype, device=dt_raw.device)
    dt = torch.exp(torch.relu(dt_raw) * a[None, None, :]).transpose(1, 2)
    score_bias = _compute_dma_score_bias_batch(module, v_flat, image_indices=image_indices)
    score_basis = dt if score_bias is None else (dt + score_bias.to(dtype=dt.dtype, device=dt.device))
    if bool(getattr(module, "dma_soft_gate_enable", False)):
        gate_logits = dt_raw.transpose(1, 2) * a[None, :, None]
        if score_bias is not None:
            gate_logits = gate_logits + score_bias.to(dtype=gate_logits.dtype, device=gate_logits.device)
        gate_probs = torch.sigmoid(gate_logits)
        selected_idx = torch.topk(gate_probs, k=w, dim=-1, sorted=False).indices
        selected_gate_probs = torch.gather(gate_probs, dim=2, index=selected_idx)

        owner_ref = getattr(module, "_dma_owner_ref", None)
        owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
        if isinstance(owner, nn.Module) and _should_record_dma_retention_stats(module, owner):
            previous_image_index = getattr(module, "_dma_current_image_index", None)
            try:
                for batch_index, image_index in enumerate(image_indices):
                    module._dma_current_image_index = int(image_index)
                    _record_dma_training_page_logit(
                        module,
                        score_basis=gate_probs[batch_index : batch_index + 1],
                        num_tokens=k_len,
                    )
                    _record_dma_page_stat(
                        module,
                        topk_idx=selected_idx[batch_index],
                        dt_sel=selected_gate_probs[batch_index],
                        score_sel=selected_gate_probs[batch_index],
                        num_tokens=k_len,
                        full_gate_probs=gate_probs[batch_index],
                    )
            finally:
                module._dma_current_image_index = previous_image_index

        attn_bias = torch.log(gate_probs.clamp_min(1e-6))[:, :, None, :].to(device=query.device, dtype=query.dtype)
        out_full = F.scaled_dot_product_attention(
            query,
            key_states.to(dtype=query.dtype),
            value_states.to(dtype=query.dtype),
            attn_mask=attn_bias,
            dropout_p=float(dropout) if module.training else 0.0,
            is_causal=False,
            scale=float(scaling),
        )
        return out_full.transpose(1, 2).contiguous()

    if w >= k_len:
        selected_idx = torch.arange(k_len, device=score_basis.device, dtype=torch.long).view(1, 1, k_len)
        selected_idx = selected_idx.expand(bsz, num_heads, k_len)
        selected_key_states = key_states
        selected_value_states = value_states
        selected_dt = dt
        selected_score_basis = score_basis
    else:
        selected_idx = torch.topk(score_basis, k=w, dim=-1, sorted=False).indices  # (B, H, w)
        idx_k = selected_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_key_states = torch.gather(key_states, dim=2, index=idx_k)
        selected_value_states = torch.gather(value_states, dim=2, index=idx_k)
        selected_dt = torch.gather(dt, dim=2, index=selected_idx)
        selected_score_basis = torch.gather(score_basis, dim=2, index=selected_idx)

    owner_ref = getattr(module, "_dma_owner_ref", None)
    owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
    if isinstance(owner, nn.Module) and _should_record_dma_retention_stats(module, owner):
        previous_image_index = getattr(module, "_dma_current_image_index", None)
        try:
            for batch_index, image_index in enumerate(image_indices):
                module._dma_current_image_index = int(image_index)
                _record_dma_training_page_logit(
                    module,
                    score_basis=score_basis[batch_index : batch_index + 1],
                    num_tokens=k_len,
                )
                _record_dma_page_stat(
                    module,
                    topk_idx=selected_idx[batch_index],
                    dt_sel=selected_dt[batch_index],
                    score_sel=selected_score_basis[batch_index],
                    num_tokens=k_len,
                )
        finally:
            module._dma_current_image_index = previous_image_index

    selection_bias_only = bool(getattr(module, "dma_selection_bias_only", False))
    attn_bias = None if selection_bias_only else selected_score_basis[:, :, None, :].to(device=query.device, dtype=query.dtype)
    out_full = F.scaled_dot_product_attention(
        query,
        selected_key_states.to(dtype=query.dtype),
        selected_value_states.to(dtype=query.dtype),
        attn_mask=attn_bias,
        dropout_p=float(dropout) if module.training else 0.0,
        is_causal=False,
        scale=float(scaling),
    )
    return out_full.transpose(1, 2).contiguous()


def _dma_attention_forward(
    *,
    module: nn.Module,
    query: torch.Tensor,  # (B, Hq, Q, D)
    key: torch.Tensor,  # (B, Hkv, K, D)
    value: torch.Tensor,  # (B, Hkv, K, D)
    attention_mask: Optional[torch.Tensor],  # (B, 1, Q, K) additive mask，0 或 -inf
    scaling: float,
    dropout: float,
    keep_window_size: int,
    query_block_size: int,
) -> torch.Tensor:
    """
    DMA 前向（不返回 attn_weights）。

    关键步骤与 docs/local/dynamic_mask_attention.md Listing1 一致：
    1) 由 value 表示计算 dt: dt = exp(relu(W_dt(v)) * A)
    2) 用 (dt + attention_mask) 做 top-k 选择（每 head、每 query 保留 w 个 key 索引）
    3) 仅在被选索引上计算 attention，logit = q·k*scaling + dt + mask
    """

    if keep_window_size <= 0:
        raise ValueError(f"keep_window_size 必须为正数，当前={keep_window_size}")
    if query_block_size <= 0:
        raise ValueError(f"query_block_size 必须为正数，当前={query_block_size}")

    # GQA/MQA：把 kv heads repeat 到 q heads
    key_states = _repeat_kv(key, int(getattr(module, "num_key_value_groups", 1)))
    value_states = _repeat_kv(value, int(getattr(module, "num_key_value_groups", 1)))

    bsz, num_heads, k_len, head_dim = key_states.shape
    q_len = query.shape[2]
    w = min(int(keep_window_size), int(k_len))

    # (B, K, H*D) -> (B, K, H) -> (B, H, K)
    v_flat = value_states.transpose(1, 2).reshape(bsz, k_len, num_heads * head_dim)
    dt_input = v_flat.to(dtype=module.dma_dt_proj.weight.dtype)
    dt_raw = module.dma_dt_proj(dt_input)  # (B, K, H)

    a = module.dma_gate(dt_raw).to(dtype=dt_raw.dtype, device=dt_raw.device)  # (H,)
    dt = torch.exp(torch.relu(dt_raw) * a[None, None, :]).transpose(1, 2)  # (B, H, K)
    score_bias = _compute_dma_score_bias(module, v_flat)
    score_basis = dt if score_bias is None else (dt + score_bias.to(dtype=dt.dtype, device=dt.device))
    if attention_mask is None:
        _record_dma_training_page_logit(module, score_basis=score_basis, num_tokens=k_len)
    if bool(getattr(module, "dma_soft_gate_enable", False)):
        gate_logits = dt_raw.transpose(1, 2) * a[None, :, None]
        if score_bias is not None:
            gate_logits = gate_logits + score_bias.to(dtype=gate_logits.dtype, device=gate_logits.device)
        gate_probs = torch.sigmoid(gate_logits)
        if attention_mask is None:
            _record_dma_training_page_logit(module, score_basis=gate_probs, num_tokens=k_len)
            diag_topk_idx = torch.topk(gate_probs, k=w, dim=-1, sorted=False).indices
            diag_gate_probs = torch.gather(gate_probs, dim=2, index=diag_topk_idx)
            _record_dma_page_stat(
                module,
                topk_idx=diag_topk_idx[0],
                dt_sel=diag_gate_probs[0],
                score_sel=diag_gate_probs[0],
                num_tokens=k_len,
                full_gate_probs=gate_probs[0],
            )
        attn_bias = torch.log(gate_probs.clamp_min(1e-6))[:, :, None, :].to(device=query.device, dtype=query.dtype)
        if attention_mask is not None:
            mask_bias = attention_mask.expand(-1, num_heads, -1, -1).to(device=query.device, dtype=query.dtype)
            attn_bias = attn_bias + mask_bias
        out_full = F.scaled_dot_product_attention(
            query,
            key_states.to(dtype=query.dtype),
            value_states.to(dtype=query.dtype),
            attn_mask=attn_bias,
            dropout_p=float(dropout) if module.training else 0.0,
            is_causal=False,
            scale=float(scaling),
        )
        return out_full.transpose(1, 2).contiguous()

    if attention_mask is None and w >= k_len:
        # 页内视觉 token 数很小时，top-k 会退化成全量 attention。
        # 此时保留 DMA bias/统计，但用 SDPA 避免 Python topk/gather/einsum 慢路径。
        all_idx = torch.arange(k_len, device=score_basis.device, dtype=torch.long).view(1, 1, k_len)
        all_idx = all_idx.expand(bsz, num_heads, k_len)
        _record_dma_page_stat(
            module,
            topk_idx=all_idx[0],
            dt_sel=dt[0],
            score_sel=score_basis[0],
            num_tokens=k_len,
        )
        attn_bias = score_basis[:, :, None, :].to(device=query.device, dtype=query.dtype)
        out_full = F.scaled_dot_product_attention(
            query,
            key_states.to(dtype=query.dtype),
            value_states.to(dtype=query.dtype),
            attn_mask=attn_bias,
            dropout_p=float(dropout) if module.training else 0.0,
            is_causal=False,
            scale=float(scaling),
        )
        return out_full.transpose(1, 2).contiguous()

    out = torch.empty((bsz, num_heads, q_len, head_dim), device=query.device, dtype=query.dtype)
    shared_k_sel: torch.Tensor | None = None
    shared_v_sel: torch.Tensor | None = None
    shared_dt_sel: torch.Tensor | None = None
    shared_score_sel: torch.Tensor | None = None
    if attention_mask is None:
        # 非 causal 且无额外 mask 时，dt 只依赖 key/value，不依赖 query；
        # 因此每个 head 的 top-k 对所有 query 完全相同，可以只选一次并整块复用。
        shared_topk_idx = torch.topk(score_basis, k=w, dim=-1, sorted=False).indices  # (B, H, w)
        idx_k = shared_topk_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # (B, H, w, D)
        shared_k_sel = torch.gather(key_states, dim=2, index=idx_k)
        shared_v_sel = torch.gather(value_states, dim=2, index=idx_k)
        shared_dt_sel = torch.gather(dt, dim=2, index=shared_topk_idx)
        shared_score_sel = torch.gather(score_basis, dim=2, index=shared_topk_idx)
        _record_dma_page_stat(
            module,
            topk_idx=shared_topk_idx[0],
            dt_sel=shared_dt_sel[0],
            score_sel=shared_score_sel[0],
            num_tokens=k_len,
        )

    # 逐块处理 query，避免显存峰值与 Q*K 成正比
    for q0 in range(0, q_len, int(query_block_size)):
        q1 = min(q_len, q0 + int(query_block_size))
        qb = q1 - q0

        q_blk = query[:, :, q0:q1, :]  # (B, H, Qb, D)

        # 用于 top-k 的分数：dt + mask（mask=-inf 的位置不会被优先选中）
        if attention_mask is None:
            assert shared_k_sel is not None
            assert shared_v_sel is not None
            assert shared_score_sel is not None
            logits = torch.einsum("bhqd,bhwd->bhqw", q_blk, shared_k_sel.to(dtype=q_blk.dtype)) * float(scaling)
            # 无 mask 的视觉分支也要与有 mask 分支保持同口径，把完整 score_basis 注入 logits。
            # 否则 page/layout/query/block routing 只影响离散 top-k 选择，梯度无法回到这些模块。
            if not bool(getattr(module, "dma_selection_bias_only", False)):
                logits = logits + shared_score_sel[:, :, None, :].to(dtype=logits.dtype)
            attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=q_blk.dtype)
            attn = torch.dropout(attn, p=float(dropout), train=module.training)
            out_blk = torch.einsum("bhqw,bhwd->bhqd", attn, shared_v_sel.to(dtype=q_blk.dtype))
            out[:, :, q0:q1, :] = out_blk
            continue
        else:
            mask_blk = attention_mask[:, :, q0:q1, :k_len]  # (B, 1, Qb, K)
            # 广播到 (B, H, Qb, K)
            sel_scores = score_basis[:, :, None, :].to(q_blk.dtype) + mask_blk.to(q_blk.dtype)

        topk_idx = torch.topk(sel_scores, k=w, dim=-1, sorted=False).indices  # (B, H, Qb, w)

        # gather K/V/dt/mask 到被选中的 w 个位置
        key_exp = key_states.unsqueeze(2).expand(-1, -1, qb, -1, -1)  # (B, H, Qb, K, D)
        val_exp = value_states.unsqueeze(2).expand(-1, -1, qb, -1, -1)  # (B, H, Qb, K, D)
        idx_k = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)  # (B, H, Qb, w, D)
        k_sel = torch.gather(key_exp, dim=3, index=idx_k)  # (B, H, Qb, w, D)
        v_sel = torch.gather(val_exp, dim=3, index=idx_k)  # (B, H, Qb, w, D)

        score_sel = torch.gather(score_basis.unsqueeze(2).expand(-1, -1, qb, -1), dim=3, index=topk_idx)  # (B,H,Qb,w)

        # attention logits：q·k*scaling + dt + mask
        logits = (q_blk.unsqueeze(-2) * k_sel).sum(dim=-1) * float(scaling)
        if not bool(getattr(module, "dma_selection_bias_only", False)):
            logits = logits + score_sel

        if attention_mask is not None:
            # 把 mask 也按 topk_idx gather，确保被 mask 的位置 logit=-inf（softmax 后权重为 0）
            mask_full = attention_mask[:, :, q0:q1, :k_len].expand(-1, num_heads, -1, -1)  # (B,H,Qb,K)
            mask_sel = torch.gather(mask_full, dim=3, index=topk_idx)  # (B,H,Qb,w)
            logits = logits + mask_sel.to(dtype=logits.dtype)

        attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=q_blk.dtype)
        attn = torch.dropout(attn, p=float(dropout), train=module.training)

        out_blk = (attn.unsqueeze(-1) * v_sel).sum(dim=-2)  # (B, H, Qb, D)
        out[:, :, q0:q1, :] = out_blk

    return out.transpose(1, 2).contiguous()  # (B, Q, H, D)


def _patch_qwen3vl_text_attention_instance(attn: nn.Module, cfg: DMAConfig) -> None:
    """
    给单个 Qwen3VLTextAttention 实例打补丁：
    - 增加 DMA 参数（dma_dt_proj / dma_gate）
    - 替换 forward 为 DMA 版本
    """

    if hasattr(attn, "_dma_patched") and bool(getattr(attn, "_dma_patched")):
        return

    # 这些属性在 Qwen3VLTextAttention 上存在：config / head_dim / scaling / num_key_value_groups ...
    hidden_size = int(attn.config.hidden_size)
    num_heads = int(attn.config.num_attention_heads)
    ref_param = next(attn.parameters(), None)

    # W_dt（等价于论文中的 Δ 的线性实现）：(H*D)->H
    attn.dma_dt_proj = nn.Linear(hidden_size, num_heads, bias=True)
    attn.dma_gate = DMAGate(num_heads=num_heads, init=float(cfg.a_init))
    _align_module_like(attn.dma_dt_proj, ref_param)
    _align_module_like(attn.dma_gate, ref_param)
    attn.dma_soft_gate_enable = bool(getattr(cfg, "soft_gate_enable", False))
    if attn.dma_soft_gate_enable and attn.dma_dt_proj.bias is not None:
        with torch.no_grad():
            attn.dma_dt_proj.bias.fill_(float(getattr(cfg, "soft_gate_bias_init", 2.0)))

    attn.dma_keep_window_size = int(cfg.keep_window_size)
    attn.dma_query_block_size = int(cfg.query_block_size)
    attn.dma_selection_bias_only = bool(getattr(cfg, "selection_bias_only", False))

    # 绑定实例方法：保持与 Transformers 原 forward 签名兼容
    def _dma_text_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # 与原实现保持一致：Rotary
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb  # 局部导入避免循环

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = _dma_attention_forward(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            scaling=float(self.scaling),
            dropout=0.0 if (not self.training) else float(self.attention_dropout),
            keep_window_size=int(self.dma_keep_window_size),
            query_block_size=int(self.dma_query_block_size),
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # 与 sdpa 行为对齐：不强制返回权重，避免显存爆炸
        return attn_output, None

    attn.forward = MethodType(_dma_text_forward, attn)
    attn._dma_patched = True


def _patch_qwen3vl_vision_attention_instance(attn: nn.Module, cfg: DMAConfig) -> None:
    """
    可选：给 Qwen3VLVisionAttention 实例打补丁（默认不开启）。
    说明：视觉 attention 非 causal，按 docs/local/dynamic_mask_attention.md Listing1 的 mask 逻辑会导致每个 query 共享同一组 top-k key，
    这在多模态场景未必合理，因此默认仅提供“可跑通”的探索开关。
    """

    if hasattr(attn, "_dma_patched") and bool(getattr(attn, "_dma_patched")):
        return

    dim = int(attn.dim)
    num_heads = int(attn.num_heads)
    ref_param = next(attn.parameters(), None)

    attn.dma_dt_proj = nn.Linear(dim, num_heads, bias=True)
    attn.dma_gate = DMAGate(num_heads=num_heads, init=float(cfg.a_init))
    _align_module_like(attn.dma_dt_proj, ref_param)
    _align_module_like(attn.dma_gate, ref_param)
    attn.dma_soft_gate_enable = bool(getattr(cfg, "soft_gate_enable", False))
    if attn.dma_soft_gate_enable and attn.dma_dt_proj.bias is not None:
        with torch.no_grad():
            attn.dma_dt_proj.bias.fill_(float(getattr(cfg, "soft_gate_bias_init", 2.0)))
    attn.dma_keep_window_size = int(cfg.keep_window_size)
    attn.dma_query_block_size = int(cfg.query_block_size)
    attn.dma_selection_bias_only = bool(getattr(cfg, "selection_bias_only", False))
    attn.dma_use_page_bias = bool(cfg.vision_page_bias_enable)
    if attn.dma_use_page_bias:
        attn.dma_page_token_proj = nn.Linear(dim, num_heads, bias=False)
        attn.dma_page_context_proj = nn.Linear(dim, num_heads, bias=True)
        attn.dma_page_gate = DMAGate(num_heads=num_heads, init=float(cfg.vision_page_bias_init))
        _align_module_like(attn.dma_page_token_proj, ref_param)
        _align_module_like(attn.dma_page_context_proj, ref_param)
        _align_module_like(attn.dma_page_gate, ref_param)
    attn.dma_use_layout_bias = bool(cfg.vision_layout_bias_enable)
    if attn.dma_use_layout_bias:
        attn.dma_layout_proj = nn.Linear(6, num_heads, bias=True)
        attn.dma_layout_gate = DMAGate(num_heads=num_heads, init=float(cfg.vision_layout_bias_init))
        _align_module_like(attn.dma_layout_proj, ref_param)
        _align_module_like(attn.dma_layout_gate, ref_param)
    attn.dma_use_query_routing = bool(cfg.vision_query_routing_enable)
    if attn.dma_use_query_routing:
        query_dim = int(getattr(attn, "_dma_query_routing_dim", dim))
        attn.dma_query_token_proj = nn.Linear(dim, num_heads, bias=False)
        attn.dma_query_context_proj = nn.Linear(query_dim, num_heads, bias=True)
        attn.dma_query_gate = DMAGate(num_heads=num_heads, init=float(cfg.vision_query_routing_init))
        _align_module_like(attn.dma_query_token_proj, ref_param)
        _align_module_like(attn.dma_query_context_proj, ref_param)
        _align_module_like(attn.dma_query_gate, ref_param)
    attn.dma_use_block_feedback = bool(cfg.vision_block_feedback_enable)
    if attn.dma_use_block_feedback:
        attn.dma_block_token_proj = nn.Linear(dim, num_heads, bias=False)
        attn.dma_block_context_proj = nn.Linear(1, num_heads, bias=True)
        attn.dma_block_gate = DMAGate(num_heads=num_heads, init=float(cfg.vision_block_feedback_init))
        _align_module_like(attn.dma_block_token_proj, ref_param)
        _align_module_like(attn.dma_block_context_proj, ref_param)
        _align_module_like(attn.dma_block_gate, ref_param)
    attn.dma_use_retention_readout = bool(cfg.retention_readout_enable)
    if attn.dma_use_retention_readout:
        block_feature_mode = _normalize_retention_block_feature_mode(
            getattr(cfg, "retention_block_feature_mode", "legacy")
        )
        block_feature_dim = 5 if block_feature_mode == "mixed" else 3
        attn.dma_retention_page_proj = nn.Linear(5, 1, bias=True)
        attn.dma_retention_block_proj = nn.Linear(block_feature_dim, 1, bias=True)
        attn.dma_retention_block_feature_mode = block_feature_mode
        # 这两个读出头参数量极小；若直接随主干走 bf16，1e-4 量级更新很容易被精度台阶吞掉。
        # 训练与推理都固定保留为 fp32，只对齐 device。
        attn.dma_retention_page_proj.to(device=ref_param.device, dtype=torch.float32)
        attn.dma_retention_block_proj.to(device=ref_param.device, dtype=torch.float32)
        attn.dma_retention_page_proj._dma_keep_fp32_modules_to_save = True
        attn.dma_retention_block_proj._dma_keep_fp32_modules_to_save = True
        with torch.no_grad():
            attn.dma_retention_page_proj.weight.zero_()
            attn.dma_retention_page_proj.bias.zero_()
            attn.dma_retention_page_proj.weight[0, 2] = 0.5
            attn.dma_retention_page_proj.weight[0, 3] = 0.5
            attn.dma_retention_block_proj.weight.zero_()
            attn.dma_retention_block_proj.bias.zero_()
            # 初始保持与旧逻辑一致：默认按 coarse block 平均热点分数排序，再让训练去修正。
            attn.dma_retention_block_proj.weight[0, 0] = 1.0

    def _dma_vision_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 基于原实现：先做 qkv + rotary，再按 chunk 处理
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        apply_rotary_pos_emb_vision = _resolve_vision_rotary_pos_emb_fn(self)

        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # (1, H, S, D)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = _resolve_vision_chunk_lengths(
            cu_seqlens=cu_seqlens,
            seq_length=int(seq_length),
            layout_splits=getattr(self, "_dma_layout_feature_splits", None),
        )
        if (
            lengths
            and len(set(int(x) for x in lengths)) == 1
            and not bool(getattr(self, "dma_use_block_feedback", False))
        ):
            # 多页同尺寸时，原逐页 Python 循环会成为 throughput_extreme 的主要瓶颈。
            # 这里把等长页堆到 batch 维度，数学上仍是“每页独立 attention + DMA bias”。
            chunk_len = int(lengths[0])
            num_images = int(len(lengths))
            head_dim = int(query_states.shape[-1])
            q_batch = (
                query_states.reshape(1, self.num_heads, num_images, chunk_len, head_dim)
                .squeeze(0)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            k_batch = (
                key_states.reshape(1, self.num_heads, num_images, chunk_len, head_dim)
                .squeeze(0)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            v_batch = (
                value_states.reshape(1, self.num_heads, num_images, chunk_len, head_dim)
                .squeeze(0)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            self._dma_current_image_index = None
            o_batch = _dma_attention_forward_full_batch(
                module=self,
                query=q_batch,
                key=k_batch,
                value=v_batch,
                image_indices=list(range(num_images)),
                scaling=float(self.scaling),
                dropout=0.0 if (not self.training) else float(self.attention_dropout),
                keep_window_size=int(self.dma_keep_window_size),
            )
            attn_output = o_batch.permute(2, 0, 1, 3).reshape(1, self.num_heads, seq_length, head_dim).contiguous()
            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            return self.proj(attn_output)

        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = []
        for image_index, (q, k, v) in enumerate(zip(*splits)):
            # 非 causal：attention_mask=None
            self._dma_current_image_index = int(image_index)
            o = _dma_attention_forward(
                module=self,
                query=q,
                key=k,
                value=v,
                attention_mask=None,
                scaling=float(self.scaling),
                dropout=0.0 if (not self.training) else float(self.attention_dropout),
                keep_window_size=int(self.dma_keep_window_size),
                query_block_size=int(self.dma_query_block_size),
            )
            # o: (1, Q, H, D) -> (1, H, Q, D)
            attn_outputs.append(o.transpose(1, 2))
        self._dma_current_image_index = None

        attn_output = torch.cat(attn_outputs, dim=2)  # (1, H, S, D)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    attn.forward = MethodType(_dma_vision_forward, attn)
    attn._dma_patched = True


def _patch_qwen3vl_vision_model_instance(vision_model: nn.Module) -> None:
    if hasattr(vision_model, "_dma_layout_patched") and bool(getattr(vision_model, "_dma_layout_patched")):
        return

    original_forward = vision_model.forward

    def _dma_vision_model_forward(self: nn.Module, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        if self.training and torch.is_floating_point(hidden_states) and not hidden_states.requires_grad:
            # 视觉 patch/embed 往往被冻结；在 reentrant gradient checkpointing 下，
            # 若输入不带梯度，整块视觉 adapter 会被静默截成“无梯度前向”。
            # 这里显式给 checkpoint 入口补一个 requires_grad，保证视觉侧 LoRA/DMA 可训练。
            hidden_states = hidden_states.detach().requires_grad_(True)
        layout_splits = _build_layout_feature_splits(
            grid_thw=grid_thw,
            merge_size=int(getattr(self, "spatial_merge_size", 1)),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        attn_modules: list[nn.Module] = []
        for blk in getattr(self, "blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            attn._dma_layout_feature_splits = layout_splits
            attn_modules.append(attn)
        if attn_modules:
            owner_ref = getattr(attn_modules[0], "_dma_owner_ref", None)
            owner = owner_ref() if callable(owner_ref) else None
            if isinstance(owner, nn.Module):
                clear_dma_block_feedback_cache(owner)
        try:
            return original_forward(hidden_states, grid_thw, **kwargs)
        finally:
            for attn in attn_modules:
                attn._dma_layout_feature_splits = None
                attn._dma_current_image_index = None

    vision_model.forward = MethodType(_dma_vision_model_forward, vision_model)
    vision_model._dma_layout_patched = True


def apply_dma_to_qwen3vl_model(model: nn.Module, cfg: DMAConfig) -> None:
    """
    对一个 Qwen3-VL Transformers 模型实例进行 DMA 接入（in-place）。
    """

    if not cfg.enable:
        return

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLVisionAttention, Qwen3VLVisionModel

    qwen2_text_attention_cls: type[nn.Module] | tuple[()] = ()
    qwen2_vision_attention_cls: type[nn.Module] | tuple[()] = ()
    qwen2_vision_model_cls: type[nn.Module] | tuple[()] = ()
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLAttention,
            Qwen2VisionTransformerPretrainedModel,
            VisionAttention,
        )

        qwen2_text_attention_cls = Qwen2VLAttention
        qwen2_vision_attention_cls = VisionAttention
        qwen2_vision_model_cls = Qwen2VisionTransformerPretrainedModel
    except Exception:
        # 某些环境可能没有 Qwen2-VL 类；保持既有 Qwen3-VL 路径可用。
        pass

    model._dma_stats_owner_marker = True
    model._dma_page_stats = []
    model._dma_last_page_summary = None
    model._dma_query_routing_enable = bool(getattr(cfg, "vision_query_routing_enable", False))
    model._dma_query_routing_runtime_ablate = bool(getattr(cfg, "vision_query_routing_runtime_ablate", False))
    model._dma_query_routing_sketch = None
    model._dma_block_feedback_enable = bool(getattr(cfg, "vision_block_feedback_enable", False))
    model._dma_block_feedback_cache = {}
    model._dma_block_feedback_start_layer = max(1, int(getattr(cfg, "vision_block_feedback_start_layer", 1)))
    model._dma_retained_page_budget_ratio = float(getattr(cfg, "retained_page_budget_ratio", 0.0))
    model._dma_retained_block_topk = max(1, int(getattr(cfg, "retained_block_topk", 2)))
    model._dma_retention_block_feature_mode = _normalize_retention_block_feature_mode(
        getattr(cfg, "retention_block_feature_mode", "legacy")
    )
    model._dma_retention_record_layer_idx = None
    if bool(cfg.apply_to_vision) and bool(getattr(cfg, "exact_block_readout_enable", False)):
        ref_param = next(model.parameters(), None)
        scorer_type = str(getattr(cfg, "exact_block_scorer_type", "page_interaction") or "page_interaction").strip().lower()
        if scorer_type in {"vlm_cross_encoder", "vlm_rerank"}:
            model.dma_exact_block_proj = None
        elif scorer_type == "linear":
            model.dma_exact_block_proj = nn.Linear(EXACT_BLOCK_FEATURE_DIM, 1, bias=True)
        elif scorer_type == "query_interaction":
            model.dma_exact_block_proj = DMAExactBlockQueryInteractionScorer(
                feature_dim=EXACT_BLOCK_FEATURE_DIM,
                text_sketch_dim=EXACT_BLOCK_TEXT_SKETCH_DIM,
                hidden_dim=int(getattr(cfg, "exact_block_scorer_hidden_dim", 32)),
            )
        elif scorer_type == "cross_block_attention":
            model.dma_exact_block_proj = DMAExactBlockCrossBlockAttentionScorer(
                feature_dim=EXACT_BLOCK_FEATURE_DIM,
                text_sketch_dim=EXACT_BLOCK_TEXT_SKETCH_DIM,
                hidden_dim=int(getattr(cfg, "exact_block_scorer_hidden_dim", 32)),
            )
        else:
            model.dma_exact_block_proj = DMAExactBlockPageInteractionScorer(
                feature_dim=EXACT_BLOCK_FEATURE_DIM,
                hidden_dim=int(getattr(cfg, "exact_block_scorer_hidden_dim", 32)),
            )
        if model.dma_exact_block_proj is not None and ref_param is not None:
            model.dma_exact_block_proj.to(device=ref_param.device, dtype=torch.float32)
        if model.dma_exact_block_proj is not None:
            model.dma_exact_block_proj._dma_keep_fp32_modules_to_save = True
            _initialize_exact_block_anchor_weights(model.dma_exact_block_proj, cfg)
            init_param_vector = _flatten_module_parameter_vector(model.dma_exact_block_proj)
            if init_param_vector is not None:
                # 用于训练诊断：记录 scorer 初始化参数，方便观测“头到底有没有真的学到东西”。不随 checkpoint 持久化。
                model.dma_exact_block_proj.register_buffer(
                    "_dma_init_param_vector",
                    init_param_vector.detach().clone(),
                    persistent=False,
                )
    vision_layer_idx = 0
    for m in model.modules():
        if cfg.apply_to_text and isinstance(m, Qwen3VLTextAttention):
            _patch_qwen3vl_text_attention_instance(m, cfg)
        if cfg.apply_to_text and qwen2_text_attention_cls and isinstance(m, qwen2_text_attention_cls):
            raise NotImplementedError("当前 DMA 只支持 Qwen2-VL 视觉侧 patch；文本侧 Qwen2-VL DMA 尚未接入。")
        if cfg.apply_to_vision and isinstance(m, (Qwen3VLVisionAttention, qwen2_vision_attention_cls)):
            m._dma_owner_ref = weakref.ref(model)
            m._dma_layer_idx = int(vision_layer_idx)
            try:
                input_embeddings = model.get_input_embeddings()
                m._dma_query_routing_dim = int(input_embeddings.weight.shape[1])
            except Exception:
                m._dma_query_routing_dim = int(getattr(m, "dim", 0))
            vision_layer_idx += 1
            _patch_qwen3vl_vision_attention_instance(m, cfg)
        if cfg.apply_to_vision and isinstance(m, (Qwen3VLVisionModel, qwen2_vision_model_cls)):
            _patch_qwen3vl_vision_model_instance(m)
    if bool(cfg.apply_to_vision) and bool(cfg.retention_readout_enable) and vision_layer_idx > 0:
        model._dma_retention_record_layer_idx = int(vision_layer_idx - 1)
        _freeze_non_record_retention_readout_modules(model)
