from __future__ import annotations

"""
外接 Cross-Attn Adapter（论文实验用）。

设计目标：
- 不改动 Transformers 源码，通过 monkey-patch 挂到 Qwen3-VL 的 decoder layer 上；
- 使用 Qwen3VLTextModel.forward 里已经提供的 `visual_pos_masks` 作为“视觉 token 位置”信号；
- 训练时与 LoRA 并行训练；推理（generate）时支持 KV cache：prefill 阶段缓存每层的视觉记忆，decode 阶段复用。
"""

from dataclasses import dataclass
import math
from types import MethodType
from typing import Optional
import weakref

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CrossAttnConfig:
    enable: bool = False
    # 每隔多少层插一次 adapter（1=每层都插）
    every_n_layers: int = 1
    # 只在 [start_layer, end_layer) 范围内生效；end_layer=None 表示到最后一层
    start_layer: int = 0
    end_layer: int | None = None
    # query 分块大小（避免 T*V 过大导致峰值显存）
    query_block_size: int = 128
    # adapter 输出的残差缩放（可训练标量）
    residual_scale_init: float = 0.0
    # question-conditioned evidence memory：仅保留 top-k 视觉记忆；0/null 表示关闭
    memory_topk: int | None = None


def cross_modules_to_save() -> list[str]:
    """PEFT（LoRA）场景：把 cross-attn adapter 的新增模块随 adapter 保存。"""

    return ["cross_attn_adapter", "cross_attn_norm"]


class CrossAttnAdapter(nn.Module):
    """
    标准 cross-attention：
    - Q 来自文本 token（非视觉位置）
    - K/V 来自视觉 token（visual_pos_masks 标记的位置）
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, residual_scale_init: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size 必须能被 num_heads 整除：{hidden_size=} {num_heads=}")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # 用文本摘要给视觉记忆打分，形成最小版 question-conditioned evidence memory。
        self.memory_query_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.memory_key_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 训练稳定性：用可学习 residual scale，默认 0（一开始等价于不插模块）
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _select_question_conditioned_memory(
        self,
        *,
        text_states: torch.Tensor,  # (B, T, H)
        vision_states: torch.Tensor,  # (B, V, H)
        vision_mask: Optional[torch.Tensor],  # (B, V) True=有效
        memory_topk: int | None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if memory_topk is None or int(memory_topk) <= 0:
            return vision_states, vision_mask

        bsz, _, hidden = vision_states.shape
        valid_mask = vision_mask
        if valid_mask is None:
            valid_mask = torch.ones(
                vision_states.shape[:2],
                device=vision_states.device,
                dtype=torch.bool,
            )
        if int(valid_mask.sum().item()) == 0:
            return vision_states[:, :0, :], valid_mask[:, :0]

        keep = min(int(memory_topk), int(vision_states.shape[1]))
        if keep <= 0 or keep >= int(vision_states.shape[1]):
            return vision_states, valid_mask

        query_summary = self.memory_query_proj(text_states).mean(dim=1)  # (B, H)
        memory_keys = self.memory_key_proj(vision_states)  # (B, V, H)
        scores = torch.einsum("bh,bvh->bv", query_summary, memory_keys) / math.sqrt(float(hidden))
        scores = scores.masked_fill(~valid_mask, -float("inf"))

        selected_states = torch.zeros(
            (bsz, keep, hidden),
            device=vision_states.device,
            dtype=vision_states.dtype,
        )
        selected_mask = torch.zeros((bsz, keep), device=vision_states.device, dtype=torch.bool)

        for b in range(bsz):
            valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue
            keep_b = min(keep, int(valid_idx.numel()))
            top_idx = torch.topk(scores[b, valid_idx], k=keep_b, dim=-1).indices
            chosen = valid_idx[top_idx]
            chosen_states = vision_states[b, chosen, :]
            # 保留 hard top-k，同时用可训练分数做软门控，避免完全退化成静态 gather。
            chosen_gate = torch.sigmoid(scores[b, chosen]).to(dtype=vision_states.dtype).unsqueeze(-1)
            selected_states[b, :keep_b, :] = chosen_states * chosen_gate
            selected_mask[b, :keep_b] = True

        return selected_states, selected_mask

    def forward(
        self,
        text_states: torch.Tensor,  # (B, T, H)
        vision_states: torch.Tensor,  # (B, V, H)
        vision_mask: Optional[torch.Tensor],  # (B, V) True=有效
        query_block_size: int,
        memory_topk: int | None = None,
    ) -> torch.Tensor:
        vision_states, vision_mask = self._select_question_conditioned_memory(
            text_states=text_states,
            vision_states=vision_states,
            vision_mask=vision_mask,
            memory_topk=memory_topk,
        )
        bsz, t_len, _ = text_states.shape
        _, v_len, _ = vision_states.shape
        if v_len == 0 or t_len == 0:
            return torch.zeros_like(text_states)

        q = self.q_proj(text_states)
        k = self.k_proj(vision_states)
        v = self.v_proj(vision_states)

        q = q.view(bsz, t_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,V,D)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,V,D)

        out = torch.empty((bsz, self.num_heads, t_len, self.head_dim), device=q.device, dtype=q.dtype)

        qbs = max(1, int(query_block_size))
        for t0 in range(0, t_len, qbs):
            t1 = min(t_len, t0 + qbs)
            q_blk = q[:, :, t0:t1, :]  # (B,H,Tb,D)

            logits = torch.matmul(q_blk, k.transpose(-2, -1)) * float(self.scaling)  # (B,H,Tb,V)
            if vision_mask is not None:
                # mask: True 有效，False -> -inf
                logits = logits.masked_fill((~vision_mask)[:, None, None, :], -float("inf"))

            attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            attn = torch.dropout(attn, p=self.dropout, train=self.training)
            out[:, :, t0:t1, :] = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, t_len, self.hidden_size)
        out = self.o_proj(out)
        return out * self.residual_scale


def _gather_padded(
    hidden_states: torch.Tensor,  # (B,S,H)
    pos_mask: torch.Tensor,  # (B,S) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将每个 batch 的若干位置 gather 成 padding 后的张量：
    - memory: (B, Vmax, H)
    - mask:   (B, Vmax) True=有效
    """

    bsz, _, hidden = hidden_states.shape
    lengths = [int(pos_mask[b].sum().item()) for b in range(bsz)]
    vmax = max(lengths) if lengths else 0
    mem = torch.zeros((bsz, vmax, hidden), device=hidden_states.device, dtype=hidden_states.dtype)
    msk = torch.zeros((bsz, vmax), device=hidden_states.device, dtype=torch.bool)
    for b in range(bsz):
        vb = lengths[b]
        if vb == 0:
            continue
        idx = pos_mask[b].nonzero(as_tuple=False).squeeze(1)
        mem[b, :vb, :] = hidden_states[b, idx, :]
        msk[b, :vb] = True
    return mem, msk


def apply_cross_attn_adapter_to_qwen3vl_model(model: nn.Module, cfg: CrossAttnConfig) -> None:
    """
    对 Qwen3-VL Transformers 模型实例进行 cross-attn adapter 接入（in-place）。

    兼容输入：
    - Qwen3VLForConditionalGeneration（有 .model）
    - Qwen3VLModel（有 .language_model）
    """

    if not cfg.enable:
        return

    base = getattr(model, "model", model)
    language_model = getattr(base, "language_model", None)
    if language_model is None:
        raise TypeError("找不到 language_model，无法接入 cross-attn adapter（请传入 Qwen3VLForConditionalGeneration 或 Qwen3VLModel）。")

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRMSNorm

    if hasattr(language_model, "_cross_attn_patched") and bool(getattr(language_model, "_cross_attn_patched")):
        return

    language_model._cross_attn_cfg = cfg

    # 让 decoder layer 能读取到 visual_pos_masks（不改原 forward 的循环）
    orig_lm_forward = language_model.forward

    def _lm_forward_with_ctx(self, *args, visual_pos_masks=None, **kwargs):
        # 将 visual_pos_masks 暂存到 language_model 上，供每层 adapter 读取
        self._cross_visual_pos_masks = visual_pos_masks
        return orig_lm_forward(*args, visual_pos_masks=visual_pos_masks, **kwargs)

    language_model.forward = MethodType(_lm_forward_with_ctx, language_model)

    # 给每一层挂 adapter
    num_layers = len(getattr(language_model, "layers"))
    end_layer = int(cfg.end_layer) if cfg.end_layer is not None else num_layers

    for layer_idx, layer in enumerate(language_model.layers):
        if not isinstance(layer, Qwen3VLTextDecoderLayer):
            continue

        layer._cross_layer_idx = int(layer_idx)
        layer._cross_parent_ref = weakref.ref(language_model)

        if hasattr(layer, "_cross_attn_patched") and bool(getattr(layer, "_cross_attn_patched")):
            continue

        # 选择是否插在该层
        if layer_idx < int(cfg.start_layer) or layer_idx >= end_layer:
            continue
        if int(cfg.every_n_layers) <= 0:
            raise ValueError(f"every_n_layers 必须为正数，当前={cfg.every_n_layers}")
        if (layer_idx - int(cfg.start_layer)) % int(cfg.every_n_layers) != 0:
            continue

        hidden_size = int(layer.hidden_size)
        num_heads = int(layer.self_attn.config.num_attention_heads)

        layer.cross_attn_norm = Qwen3VLTextRMSNorm(hidden_size, eps=layer.self_attn.config.rms_norm_eps)
        layer.cross_attn_adapter = CrossAttnAdapter(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            residual_scale_init=float(cfg.residual_scale_init),
        )
        ref_param = layer.input_layernorm.weight
        layer.cross_attn_norm.to(device=ref_param.device, dtype=ref_param.dtype)
        layer.cross_attn_adapter.to(device=ref_param.device, dtype=ref_param.dtype)

        def _layer_forward_with_cross(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            # 复刻 Qwen3VLTextDecoderLayer.forward 的结构，并把 cross-attn 插在 self-attn 之前，
            # 使其影响 Q/K/V 的投影与 KV cache（generate 场景更一致）。
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            parent_ref = getattr(self, "_cross_parent_ref", None)
            parent = parent_ref() if callable(parent_ref) else None
            visual_pos_masks = getattr(parent, "_cross_visual_pos_masks", None) if parent is not None else None
            if visual_pos_masks is not None:
                visual_pos_masks = visual_pos_masks.to(device=hidden_states.device, dtype=torch.bool)

                layer_id = int(getattr(self, "_cross_layer_idx", 0))
                cache_key = "_cross_vision_memory"
                vision_mem = None
                vision_msk = None

                if past_key_values is not None and hasattr(past_key_values, cache_key):
                    mem_list = getattr(past_key_values, cache_key)
                    if isinstance(mem_list, list) and layer_id < len(mem_list) and mem_list[layer_id] is not None:
                        vision_mem, vision_msk = mem_list[layer_id]

                # prefill：本轮含视觉位置，则从“归一化后的 hidden_states”抽取记忆并缓存
                if vision_mem is None and bool(visual_pos_masks.any().item()):
                    vision_mem, vision_msk = _gather_padded(hidden_states, visual_pos_masks)
                    if past_key_values is not None:
                        if not hasattr(past_key_values, cache_key):
                            setattr(past_key_values, cache_key, [None for _ in range(num_layers)])
                        mem_list = getattr(past_key_values, cache_key)
                        mem_list[layer_id] = (vision_mem.detach() if not self.training else vision_mem, vision_msk)

                if vision_mem is not None and vision_msk is not None and int(vision_msk.sum().item()) > 0:
                    # 只更新文本位置（非视觉 token）
                    text_mask = ~visual_pos_masks
                    if int(text_mask.sum().item()) > 0:
                        hidden2 = hidden_states.clone()
                        bsz = hidden_states.shape[0]
                        for b in range(bsz):
                            t_idx = text_mask[b].nonzero(as_tuple=False).squeeze(1)
                            if t_idx.numel() == 0:
                                continue
                            v_valid = int(vision_msk[b].sum().item())
                            if v_valid == 0:
                                continue
                            v_states = vision_mem[b : b + 1, :v_valid, :]
                            v_mask = vision_msk[b : b + 1, :v_valid]

                            t_states = hidden2[b : b + 1, t_idx, :]
                            t_states = self.cross_attn_norm(t_states)
                            delta = self.cross_attn_adapter(
                                t_states,
                                v_states,
                                v_mask,
                                int(getattr(parent, "_cross_attn_cfg").query_block_size),
                                getattr(parent, "_cross_attn_cfg").memory_topk,
                            )
                            hidden2[b : b + 1, t_idx, :] = hidden2[b : b + 1, t_idx, :] + delta
                        hidden_states = hidden2

            # Self Attention（使用已注入 cross-attn 的 hidden_states）
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        layer.forward = MethodType(_layer_forward_with_cross, layer)
        layer._cross_attn_patched = True

    language_model._cross_attn_patched = True
