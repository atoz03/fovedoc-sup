from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask

from eccv26.model.cross_attn import CrossAttnConfig, apply_cross_attn_adapter_to_qwen3vl_model, cross_modules_to_save
from eccv26.model.dma import (
    DMAConfig,
    apply_dma_to_qwen3vl_model,
    clear_dma_query_routing_sketch,
    dma_modules_to_save,
    reset_dma_page_stats,
    set_dma_query_routing_sketch,
    summarize_dma_page_retention,
)
from eccv26.model.visual_token_pruning import (
    prune_visual_tokens_by_scores_with_metadata,
    prune_visual_tokens_random_with_metadata,
    prune_visual_tokens_spatial_downsample_with_metadata,
    prune_visual_tokens_with_metadata,
)
from eccv26.utils.text import extract_choice_letter


@dataclass(frozen=True)
class GenerateConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float = 1.0
    assistant_prefill_text: str | None = None


@dataclass(frozen=True)
class VisualTokenPruningConfig:
    enable: bool = False
    keep_ratio: float = 1.0
    min_tokens_per_page: int = 16
    record_score_stats: bool = False
    score_mode: str = "cosine"
    attention_prefill_layers: int = 2
    spatial_stride: int = 2
    random_seed: int = 42


def build_qwen3vl_messages(
    *,
    images: list,
    user_text: str,
    system_text: str,
    assistant_text: str | None = None,
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": ([{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user_text}]),
        },
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return messages


def prepare_qwen3vl_inputs(
    *,
    processor: Any,
    model: nn.Module,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    image_inputs = [item["image"] for item in messages[1]["content"] if item.get("type") == "image"]
    video_inputs = None
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
    except Exception:
        video_inputs = None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def score_qwen3vl_assistant_text(
    *,
    model: nn.Module,
    processor: Any,
    images: list,
    user_text: str,
    system_text: str,
    assistant_text: str,
) -> torch.Tensor:
    prompt_messages = build_qwen3vl_messages(
        images=images,
        user_text=user_text,
        system_text=system_text,
        assistant_text=None,
    )
    full_messages = build_qwen3vl_messages(
        images=images,
        user_text=user_text,
        system_text=system_text,
        assistant_text=assistant_text,
    )
    prompt_inputs = prepare_qwen3vl_inputs(
        processor=processor,
        model=model,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    full_inputs = prepare_qwen3vl_inputs(
        processor=processor,
        model=model,
        messages=full_messages,
        add_generation_prompt=False,
    )

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    input_ids = full_inputs["input_ids"]
    if int(input_ids.shape[1]) <= prompt_len:
        return input_ids.new_full((), float("-inf"), dtype=torch.float32)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    set_dma_query_routing_sketch(
        model,
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs.get("attention_mask"),
        image_token_id=image_token_id,
    )
    try:
        outputs = model(**full_inputs)
    finally:
        clear_dma_query_routing_sketch(model)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_pos = torch.arange(1, int(input_ids.shape[1]), device=input_ids.device)
    target_mask = target_pos >= prompt_len
    if not bool(target_mask.any()):
        return logits.new_full((), float("-inf"))
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    masked = token_log_probs[:, target_mask]
    if int(masked.numel()) == 0:
        return logits.new_full((), float("-inf"))
    return masked.mean()


def score_qwen3vl_block_relevance(
    *,
    model: nn.Module,
    processor: Any,
    images: list,
    user_text: str,
    system_text: str,
    positive_text: str = "是",
    negative_text: str = "否",
) -> torch.Tensor:
    positive_score = score_qwen3vl_assistant_text(
        model=model,
        processor=processor,
        images=images,
        user_text=user_text,
        system_text=system_text,
        assistant_text=positive_text,
    )
    negative_score = score_qwen3vl_assistant_text(
        model=model,
        processor=processor,
        images=images,
        user_text=user_text,
        system_text=system_text,
        assistant_text=negative_text,
    )
    return positive_score - negative_score


def _load_local_peft_adapter_config(model_path: str) -> dict[str, Any] | None:
    cfg_path = Path(model_path) / "adapter_config.json"
    if not cfg_path.is_file():
        return None
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _validate_adapter_runtime_modules(
    *,
    model_path: str,
    adapter_cfg: dict[str, Any],
    dma: DMAConfig | None,
    cross_attn: CrossAttnConfig | None,
) -> None:
    modules_to_save = set(adapter_cfg.get("modules_to_save") or [])
    dma_page_bias_modules = {"dma_page_token_proj", "dma_page_context_proj", "dma_page_gate"}
    dma_layout_bias_modules = {"dma_layout_proj", "dma_layout_gate"}
    dma_query_routing_modules = {"dma_query_token_proj", "dma_query_context_proj", "dma_query_gate"}
    dma_block_feedback_modules = {"dma_block_token_proj", "dma_block_context_proj", "dma_block_gate"}
    dma_retention_readout_modules = {"dma_retention_page_proj", "dma_retention_block_proj"}
    dma_exact_block_modules = {"dma_exact_block_proj"}

    if any(name in modules_to_save for name in dma_modules_to_save()):
        if dma is None or not dma.enable:
            raise ValueError(
                f"{model_path} 是包含 DMA 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true，并设置与训练一致的 DMA 超参数。"
            )
    if any(name in modules_to_save for name in dma_page_bias_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.vision_page_bias_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA page-bias 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.vision_page_bias_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )
    if any(name in modules_to_save for name in dma_layout_bias_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.vision_layout_bias_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA layout-bias 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.vision_layout_bias_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )
    if any(name in modules_to_save for name in dma_query_routing_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.vision_query_routing_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA query-routing 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.vision_query_routing_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )
    if any(name in modules_to_save for name in dma_block_feedback_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.vision_block_feedback_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA block-feedback 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.vision_block_feedback_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )
    if any(name in modules_to_save for name in dma_retention_readout_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.retention_readout_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA summary-readout 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.retention_readout_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )
    if any(name in modules_to_save for name in dma_exact_block_modules):
        if dma is None or not dma.enable or not dma.apply_to_vision or not dma.exact_block_readout_enable:
            raise ValueError(
                f"{model_path} 是包含 Vision-DMA exact-block readout 模块的 PEFT adapter；评测时请显式传入 "
                "dma.enable=true、dma.apply_to_vision=true、dma.exact_block_readout_enable=true，"
                "并设置与训练一致的 DMA 配置。"
            )

    if any(name in modules_to_save for name in cross_modules_to_save()):
        if cross_attn is None or not cross_attn.enable:
            raise ValueError(
                f"{model_path} 是包含 cross-attn adapter 模块的 PEFT adapter；评测时请显式传入 "
                "cross_attn.enable=true，并设置与训练一致的 cross-attn 配置。"
            )


def _align_peft_modules_to_save_devices(model: nn.Module) -> None:
    """
    `modules_to_save` 在 `device_map=auto` 下不会总是自动跟随基座模块迁移设备。
    这里将活动 adapter 的拷贝模块显式对齐到原模块所在 device/dtype，避免推理时 CPU/CUDA 冲突。
    """

    for module in model.modules():
        modules_to_save = getattr(module, "modules_to_save", None)
        original_module = getattr(module, "original_module", None)
        active_adapters = getattr(module, "active_adapters", None)
        if not isinstance(modules_to_save, nn.ModuleDict) or original_module is None or not active_adapters:
            continue

        adapter_name = active_adapters[0]
        if adapter_name not in modules_to_save:
            continue

        try:
            ref_param = next(original_module.parameters())
        except StopIteration:
            continue

        target_kwargs: dict[str, Any] = {"device": ref_param.device}
        keep_fp32 = bool(getattr(original_module, "_dma_keep_fp32_modules_to_save", False))
        if ref_param.is_floating_point() and not keep_fp32:
            target_kwargs["dtype"] = ref_param.dtype
        modules_to_save[adapter_name].to(**target_kwargs)


def _should_force_single_gpu_load(device_map: str | dict[str, Any]) -> bool:
    """
    Qwen3-VL 在多卡 + device_map=auto 的推理路径下，存在 image placeholder mask 与
    image features 对齐异常（会触发 tokens/features mismatch）。
    默认启用单卡安全模式；如确需保留 auto 分片，可设置环境变量
    `ECCV26_QWEN3VL_ALLOW_AUTO_SHARD=1` 显式关闭该兜底。
    """

    if device_map != "auto":
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() <= 1:
        return True

    allow_auto_shard = str(os.getenv("ECCV26_QWEN3VL_ALLOW_AUTO_SHARD", "")).strip().lower()
    return allow_auto_shard not in {"1", "true", "yes", "on"}


class Qwen3VL:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "auto",
        device_map: str | dict[str, Any] = "auto",
        attn_implementation: str | None = None,
        dma: DMAConfig | None = None,
        cross_attn: CrossAttnConfig | None = None,
        visual_token_pruning: VisualTokenPruningConfig | None = None,
    ) -> None:
        adapter_cfg = _load_local_peft_adapter_config(model_path)
        base_model_path = model_path
        force_single_gpu_load = _should_force_single_gpu_load(device_map)
        if adapter_cfg is not None:
            _validate_adapter_runtime_modules(
                model_path=model_path,
                adapter_cfg=adapter_cfg,
                dma=dma,
                cross_attn=cross_attn,
            )
            base_model_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
            if not base_model_path:
                raise ValueError(f"{model_path} 的 adapter_config.json 缺少 base_model_name_or_path，无法加载基座模型。")

        if force_single_gpu_load and device_map == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(
                "[Qwen3VL] 检测到多卡 + device_map=auto。已自动切换到单卡安全加载，"
                "以规避 image token/feature 对齐异常。"
                "如需显式保留 auto 分片，请设置 ECCV26_QWEN3VL_ALLOW_AUTO_SHARD=1。",
                flush=True,
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        torch_dtype: Any
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"未知 dtype: {dtype}")

        effective_device_map = None if force_single_gpu_load else device_map
        kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": effective_device_map}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **kwargs)
        if dma is not None:
            apply_dma_to_qwen3vl_model(self.model, dma)
        if cross_attn is not None:
            apply_cross_attn_adapter_to_qwen3vl_model(self.model, cross_attn)
        if adapter_cfg is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, model_path)
            _align_peft_modules_to_save_devices(self.model)
        if force_single_gpu_load:
            self.model.to("cuda:0")
        self.model.eval()

        # 便于统计视觉 token 数
        self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._vision_start_token_id = int(self.model.config.vision_start_token_id)
        self._visual_token_pruning = visual_token_pruning or VisualTokenPruningConfig()

    def _build_messages(
        self,
        *,
        images: list,
        user_text: str,
        system_text: str,
        assistant_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return build_qwen3vl_messages(
            images=images,
            user_text=user_text,
            system_text=system_text,
            assistant_text=assistant_text,
        )

    def _prepare_inputs(
        self,
        *,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        return prepare_qwen3vl_inputs(
            processor=self.processor,
            model=self.model,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    def _resolve_language_backbone(self) -> nn.Module:
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if hasattr(base_model, "model"):
            return base_model.model
        return base_model

    def _resolve_generation_model(self) -> nn.Module:
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return base_model

    def _build_query_embedding(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        embedding_module = self.model.get_input_embeddings()
        token_embeddings = embedding_module(input_ids.to(device=embedding_module.weight.device, dtype=torch.long))
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool, device=token_embeddings.device)
        if attention_mask is not None:
            valid_mask = valid_mask & attention_mask.to(device=token_embeddings.device).bool()
        valid_mask = valid_mask & input_ids.to(device=token_embeddings.device).ne(int(self._image_token_id))
        if not bool(valid_mask.any()):
            raise ValueError("当前样本没有可用于 visual token pruning 的文本 query token。")
        mask_f = valid_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        pooled = (token_embeddings * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def _build_visual_prompt_state(
        self,
        *,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if inputs.get("pixel_values_videos") is not None:
            raise NotImplementedError("当前 visual token pruning POC 只支持图片输入，不支持视频。")

        generation_model = self._resolve_generation_model()
        language_backbone = self._resolve_language_backbone()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        image_embeds_pages, deepstack_visual_embeds = generation_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_pages, dim=0).to(device=input_ids.device, dtype=generation_model.dtype)
        full_position_ids, _ = language_backbone.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        inputs_embeds = generation_model.get_input_embeddings()(input_ids)
        image_mask = input_ids == int(self._image_token_id)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask_expanded,
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        return {
            "generation_model": generation_model,
            "language_backbone": language_backbone,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "image_embeds": image_embeds,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "position_ids": full_position_ids,
            "inputs_embeds": inputs_embeds,
            "image_mask": image_mask,
            "visual_pos_masks": image_mask,
        }

    def _build_query_text_mask(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("attention-based visual pruning 目前只支持 batch_size=1。")

        base_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask is not None:
            base_mask = base_mask & attention_mask.bool()
        valid_mask = base_mask & input_ids.ne(int(self._image_token_id))

        image_positions = torch.nonzero(
            base_mask[0] & input_ids[0].eq(int(self._image_token_id)),
            as_tuple=False,
        ).squeeze(1)
        if int(image_positions.numel()) > 0:
            query_only_mask = valid_mask.clone()
            query_only_mask[:, : int(image_positions.max().item()) + 1] = False
            if bool(query_only_mask.any()):
                return query_only_mask

        if not bool(valid_mask.any()):
            raise ValueError("当前样本没有可用于 attention scoring 的文本 query token。")
        return valid_mask

    def _build_attention_prefill_scores(
        self,
        *,
        prompt_state: dict[str, Any],
        prefill_layers: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        generation_model = prompt_state["generation_model"]
        language_backbone = prompt_state["language_backbone"]
        text_model = language_backbone.language_model
        hidden_states = prompt_state["inputs_embeds"]
        attention_mask = prompt_state["attention_mask"]
        position_ids = prompt_state["position_ids"]
        visual_pos_masks = prompt_state["visual_pos_masks"]
        deepstack_visual_embeds = prompt_state["deepstack_visual_embeds"]
        input_ids = prompt_state["input_ids"]
        image_mask = prompt_state["image_mask"]

        layer_count = min(int(prefill_layers), len(text_model.layers))
        if layer_count <= 0:
            raise ValueError(f"attention_prefill_layers 必须 >= 1，当前={prefill_layers}")

        query_text_mask = self._build_query_text_mask(input_ids=input_ids, attention_mask=attention_mask)
        visual_positions = torch.nonzero(image_mask[0], as_tuple=False).squeeze(1)
        query_positions = torch.nonzero(query_text_mask[0], as_tuple=False).squeeze(1)
        if int(visual_positions.numel()) != int(prompt_state["image_embeds"].shape[0]):
            raise ValueError(
                "attention scoring 的 visual placeholder 数与 image_embeds 数不一致，无法安全对齐。"
            )
        if int(query_positions.numel()) == 0:
            raise ValueError("attention scoring 未找到 query text token。")

        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        text_position_ids = position_ids[0]
        causal_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
        original_attn_impl = getattr(text_model.config, "_attn_implementation", "sdpa")
        layer_scores: list[torch.Tensor] = []
        try:
            # 诊断/浅层打分阶段强制切 eager，这样可以稳定拿到完整 attention weights。
            text_model.config._attn_implementation = "eager"
            for layer_idx in range(layer_count):
                decoder_layer = text_model.layers[layer_idx]

                residual = hidden_states
                attn_hidden_states = decoder_layer.input_layernorm(hidden_states)
                attn_output, attn_weights = decoder_layer.self_attn(
                    hidden_states=attn_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                if attn_weights is None:
                    raise RuntimeError("attention scoring 未拿到 attn_weights。")

                text_to_visual = attn_weights[:, :, query_positions, :]
                text_to_visual = text_to_visual.index_select(dim=-1, index=visual_positions)
                layer_scores.append(text_to_visual.mean(dim=(0, 1, 2)).detach().to(dtype=torch.float32))

                hidden_states = residual + attn_output
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                hidden_states = decoder_layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    hidden_states = text_model._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )
        finally:
            text_model.config._attn_implementation = original_attn_impl

        pooled_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
        debug = {
            "score_source": "attention_prefill",
            "attention_prefill_layers": int(layer_count),
            "query_text_token_count": int(query_positions.numel()),
            "visual_token_count": int(visual_positions.numel()),
        }
        return pooled_scores.to(device=prompt_state["image_embeds"].device), debug

    def _select_deepstack_visual_embeds(
        self,
        *,
        deepstack_visual_embeds: list[torch.Tensor],
        page_keep_indices: torch.Tensor,
        page_keep_counts: torch.Tensor,
        split_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        selected_per_layer: list[torch.Tensor] = []
        split_sizes_list = [int(x) for x in split_sizes.tolist()]
        keep_counts_list = [int(x) for x in page_keep_counts.tolist()]

        for layer_embeds in deepstack_visual_embeds:
            page_chunks = torch.split(layer_embeds, split_sizes_list, dim=0)
            kept_pages: list[torch.Tensor] = []
            for page_idx, page_chunk in enumerate(page_chunks):
                keep_count = keep_counts_list[page_idx]
                keep_idx = page_keep_indices[page_idx, :keep_count].to(device=page_chunk.device, dtype=torch.long)
                kept_pages.append(page_chunk.index_select(0, keep_idx))
            selected_per_layer.append(torch.cat(kept_pages, dim=0))
        return selected_per_layer

    def _build_pruned_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pruned_grid_thw: torch.Tensor,
        page_keep_indices: torch.Tensor,
        page_keep_counts: torch.Tensor,
        split_sizes: torch.Tensor,
        position_strategy: str = "select_existing",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_backbone = self._resolve_language_backbone()
        strategy = str(position_strategy).strip().lower()
        if strategy == "recompute_from_pruned_grid":
            keep_counts_list = [int(x) for x in page_keep_counts.tolist()]
            split_sizes_list = [int(x) for x in split_sizes.tolist()]
            if int(pruned_grid_thw.shape[0]) != len(keep_counts_list):
                raise ValueError("pruned_grid_thw 页数与 page_keep_counts 不一致，无法重算位置。")

            sample_ids = input_ids[0]
            attn_ids = attention_mask[0].bool()
            valid_token_indices = torch.nonzero(attn_ids, as_tuple=False).squeeze(1)
            valid_token_ids = sample_ids.index_select(0, valid_token_indices)

            vision_start_positions = torch.nonzero(
                valid_token_ids == int(self._vision_start_token_id),
                as_tuple=False,
            ).squeeze(1)
            if int(vision_start_positions.numel()) != int(image_grid_thw.shape[0]):
                raise ValueError(
                    "visual token pruning 只支持每个图片块都以标准 <|vision_start|> ... <|image_pad|> 结构编码。"
                )

            rebuilt_valid_ids: list[torch.Tensor] = []
            cursor = 0
            for page_idx, vision_start_pos in enumerate(vision_start_positions.tolist()):
                span_start = int(vision_start_pos) + 1
                keep_count = keep_counts_list[page_idx]
                page_prefix = valid_token_ids[cursor:span_start]
                page_tokens = torch.full(
                    (keep_count,),
                    fill_value=int(self._image_token_id),
                    dtype=valid_token_ids.dtype,
                    device=valid_token_ids.device,
                )
                rebuilt_valid_ids.extend([page_prefix, page_tokens])
                cursor = span_start + split_sizes_list[page_idx]
            rebuilt_valid_ids.append(valid_token_ids[cursor:])
            rebuilt_sequence = torch.cat(rebuilt_valid_ids, dim=0).unsqueeze(0)
            rebuilt_attention_mask = attention_mask.new_ones(
                rebuilt_sequence.shape,
                dtype=attention_mask.dtype,
            )
            rebuilt_position_ids, rope_deltas = language_backbone.get_rope_index(
                input_ids=rebuilt_sequence,
                image_grid_thw=pruned_grid_thw,
                video_grid_thw=None,
                attention_mask=rebuilt_attention_mask,
            )
            return rebuilt_sequence, rebuilt_attention_mask, rebuilt_position_ids

        full_position_ids, rope_deltas = language_backbone.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        keep_counts_list = [int(x) for x in page_keep_counts.tolist()]
        split_sizes_list = [int(x) for x in split_sizes.tolist()]

        sample_ids = input_ids[0]
        attn_ids = attention_mask[0].bool()
        valid_token_indices = torch.nonzero(attn_ids, as_tuple=False).squeeze(1)
        valid_token_ids = sample_ids.index_select(0, valid_token_indices)

        vision_start_positions = torch.nonzero(valid_token_ids == int(self._vision_start_token_id), as_tuple=False).squeeze(1)
        if int(vision_start_positions.numel()) != int(image_grid_thw.shape[0]):
            raise ValueError(
                "visual token pruning 只支持每个图片块都以标准 <|vision_start|> ... <|image_pad|> 结构编码。"
            )

        keep_mask_valid = torch.ones_like(valid_token_ids, dtype=torch.bool)
        for page_idx, vision_start_pos in enumerate(vision_start_positions.tolist()):
            span_start = int(vision_start_pos) + 1
            page_token_count = split_sizes_list[page_idx]
            page_positions = valid_token_indices[span_start : span_start + page_token_count]
            if int(page_positions.numel()) != page_token_count:
                raise ValueError("按页解析 visual token span 时遇到长度不一致，无法安全裁剪。")
            keep_mask_valid[span_start : span_start + page_token_count] = False
            keep_count = keep_counts_list[page_idx]
            keep_idx = page_keep_indices[page_idx, :keep_count].to(device=page_positions.device, dtype=torch.long)
            keep_mask_valid[span_start + keep_idx] = True

        keep_indices = valid_token_indices[keep_mask_valid]
        pruned_input_ids = input_ids.index_select(1, keep_indices)
        pruned_attention_mask = attention_mask.index_select(1, keep_indices)
        pruned_position_ids = full_position_ids.index_select(2, keep_indices)
        return pruned_input_ids, pruned_attention_mask, pruned_position_ids

    def _sample_next_token(
        self,
        *,
        logits: torch.Tensor,
        gen: GenerateConfig,
    ) -> torch.Tensor:
        next_token_logits = logits[:, -1, :]
        repetition_penalty = float(getattr(gen, "repetition_penalty", 1.0))
        if repetition_penalty != 1.0:
            next_token_logits = next_token_logits / repetition_penalty
        if not bool(gen.do_sample):
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

        temperature = max(float(gen.temperature), 1e-6)
        next_token_logits = next_token_logits / temperature
        if float(gen.top_p) < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_mask = cumulative_probs > float(gen.top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            filtered_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            next_token_logits = torch.full_like(next_token_logits, float("-inf"))
            next_token_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_logits)
        probs = F.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generate_with_visual_token_pruning(
        self,
        *,
        inputs: dict[str, Any],
        gen: GenerateConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        pruning_cfg = self._visual_token_pruning
        prompt_state = self._build_visual_prompt_state(inputs=inputs)
        generation_model = prompt_state["generation_model"]
        language_backbone = prompt_state["language_backbone"]
        input_ids = prompt_state["input_ids"]
        attention_mask = prompt_state["attention_mask"]
        image_grid_thw = prompt_state["image_grid_thw"]
        image_embeds = prompt_state["image_embeds"]
        deepstack_visual_embeds = prompt_state["deepstack_visual_embeds"]

        score_mode = str(getattr(pruning_cfg, "score_mode", "cosine") or "cosine").strip().lower()
        score_debug_extra: dict[str, Any] | None = None
        if score_mode == "attention_prefill":
            token_scores = None
            if float(pruning_cfg.keep_ratio) < 0.999999 or bool(pruning_cfg.record_score_stats):
                token_scores, score_debug_extra = self._build_attention_prefill_scores(
                    prompt_state=prompt_state,
                    prefill_layers=int(getattr(pruning_cfg, "attention_prefill_layers", 2)),
                )
            pruning_result = prune_visual_tokens_by_scores_with_metadata(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                token_scores=token_scores,
                keep_ratio=float(pruning_cfg.keep_ratio),
                min_tokens_per_page=int(pruning_cfg.min_tokens_per_page),
                spatial_merge_size=int(generation_model.visual.spatial_merge_size),
                record_score_stats=bool(pruning_cfg.record_score_stats),
                score_source="attention_prefill",
            )
        elif score_mode == "cosine":
            query_embeds = self._build_query_embedding(input_ids=input_ids, attention_mask=attention_mask).to(
                device=image_embeds.device,
                dtype=image_embeds.dtype,
            )
            pruning_result = prune_visual_tokens_with_metadata(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                query_embeds=query_embeds,
                keep_ratio=float(pruning_cfg.keep_ratio),
                min_tokens_per_page=int(pruning_cfg.min_tokens_per_page),
                spatial_merge_size=int(generation_model.visual.spatial_merge_size),
                record_score_stats=bool(pruning_cfg.record_score_stats),
            )
        elif score_mode == "random":
            pruning_result = prune_visual_tokens_random_with_metadata(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                keep_ratio=float(pruning_cfg.keep_ratio),
                min_tokens_per_page=int(pruning_cfg.min_tokens_per_page),
                spatial_merge_size=int(generation_model.visual.spatial_merge_size),
                record_score_stats=bool(pruning_cfg.record_score_stats),
                random_seed=int(getattr(pruning_cfg, "random_seed", 42)),
            )
        elif score_mode == "spatial_downsample":
            pruning_result = prune_visual_tokens_spatial_downsample_with_metadata(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                keep_ratio=float(pruning_cfg.keep_ratio),
                min_tokens_per_page=int(pruning_cfg.min_tokens_per_page),
                spatial_merge_size=int(generation_model.visual.spatial_merge_size),
                stride=int(getattr(pruning_cfg, "spatial_stride", 2)),
            )
        else:
            raise ValueError(f"不支持的 visual token pruning score_mode: {score_mode}")

        pruned_input_ids, pruned_attention_mask, pruned_position_ids = self._build_pruned_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            pruned_grid_thw=pruning_result.pruned_grid_thw,
            page_keep_indices=pruning_result.page_keep_indices,
            page_keep_counts=pruning_result.page_keep_counts,
            split_sizes=pruning_result.split_sizes,
            position_strategy=str(getattr(pruning_result, "position_strategy", "select_existing")),
        )
        inputs_embeds = generation_model.get_input_embeddings()(pruned_input_ids)
        image_mask = pruned_input_ids == int(self._image_token_id)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask_expanded,
            pruning_result.pruned_image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        pruned_deepstack_visual_embeds = self._select_deepstack_visual_embeds(
            deepstack_visual_embeds=deepstack_visual_embeds,
            page_keep_indices=pruning_result.page_keep_indices,
            page_keep_counts=pruning_result.page_keep_counts,
            split_sizes=pruning_result.split_sizes,
        )
        visual_pos_masks = image_mask

        batch_size, prompt_len = pruned_input_ids.shape
        cache_position = torch.arange(prompt_len, device=pruned_input_ids.device, dtype=torch.long)
        outputs = language_backbone.language_model(
            input_ids=None,
            attention_mask=pruned_attention_mask,
            position_ids=pruned_position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=pruned_deepstack_visual_embeds,
        )
        logits = generation_model.lm_head(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values
        language_backbone.rope_deltas = generation_model.model.rope_deltas
        generated_ids: list[torch.Tensor] = []
        current_attention_mask = pruned_attention_mask

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_token_ids = {int(eos_token_id)}
        else:
            eos_token_ids = {int(x) for x in eos_token_id}

        next_token = self._sample_next_token(logits=logits, gen=gen)
        for step_idx in range(int(gen.max_new_tokens)):
            generated_ids.append(next_token)
            if eos_token_ids and int(next_token[0, 0].item()) in eos_token_ids:
                break
            current_attention_mask = torch.cat(
                [current_attention_mask, current_attention_mask.new_ones((batch_size, 1))],
                dim=1,
            )
            cache_position = cache_position[-1:] + 1
            next_position_ids = torch.full(
                (3, batch_size, 1),
                fill_value=int(pruned_position_ids.max().item()) + 1 + step_idx,
                device=pruned_position_ids.device,
                dtype=pruned_position_ids.dtype,
            )
            outputs = generation_model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                cache_position=cache_position,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token = self._sample_next_token(logits=logits, gen=gen)

        gen_ids = torch.cat(generated_ids, dim=1) if generated_ids else pruned_input_ids.new_empty((batch_size, 0))
        debug_info = {
            "visual_tokens_before": int(input_ids.eq(int(self._image_token_id)).sum().item()),
            "visual_tokens_after": int(pruning_result.pruned_image_embeds.shape[0]),
            "input_len_before": int(input_ids.shape[1]),
            "input_len_after": int(pruned_input_ids.shape[1]),
            "page_keep_indices": pruning_result.page_keep_indices.detach().cpu().tolist(),
            "page_keep_counts": pruning_result.page_keep_counts.detach().cpu().tolist(),
            "pruned_grid_thw": pruning_result.pruned_grid_thw.detach().cpu().tolist(),
        }
        if pruning_result.score_debug is not None or score_debug_extra is not None:
            score_debug = dict(pruning_result.score_debug or {})
            if score_debug_extra is not None:
                score_debug.update(score_debug_extra)
            debug_info["pruning_score_debug"] = score_debug
        return gen_ids, debug_info

    @torch.inference_mode()
    def _score_assistant_text(
        self,
        *,
        images: list,
        user_text: str,
        system_text: str,
        assistant_text: str,
    ) -> float:
        return float(
            score_qwen3vl_assistant_text(
                model=self.model,
                processor=self.processor,
                images=images,
                user_text=user_text,
                system_text=system_text,
                assistant_text=assistant_text,
            )
            .detach()
            .item()
        )

    @torch.inference_mode()
    def score_block_relevance_one(
        self,
        *,
        images: list,
        user_text: str,
        system_text: str,
        positive_text: str = "是",
        negative_text: str = "否",
    ) -> float:
        return float(
            score_qwen3vl_block_relevance(
                model=self.model,
                processor=self.processor,
                images=images,
                user_text=user_text,
                system_text=system_text,
                positive_text=positive_text,
                negative_text=negative_text,
            )
            .detach()
            .item()
        )

    @torch.inference_mode()
    def generate_one(
        self,
        *,
        images: list,
        input_page_ids: list[int] | None = None,
        user_text: str,
        system_text: str,
        gen: GenerateConfig,
        record_dma_stats: bool = True,
    ) -> dict[str, Any]:
        messages = self._build_messages(
            images=images,
            user_text=user_text,
            system_text=system_text,
            assistant_text=getattr(gen, "assistant_prefill_text", None),
        )
        assistant_prefill_text = getattr(gen, "assistant_prefill_text", None)
        inputs = self._prepare_inputs(
            messages=messages,
            add_generation_prompt=assistant_prefill_text is None,
        )

        input_ids = inputs["input_ids"]
        visual_tokens = int((input_ids == self._image_token_id).sum().item())
        previous_stats_disabled = bool(getattr(self.model, "_dma_retention_stats_disabled", False))
        self.model._dma_retention_stats_disabled = not bool(record_dma_stats)
        if input_page_ids and record_dma_stats:
            reset_dma_page_stats(self.model)

        set_dma_query_routing_sketch(
            self.model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_token_id=self._image_token_id,
        )
        try:
            pruning_enabled = bool(self._visual_token_pruning.enable)
            if pruning_enabled:
                gen_ids, pruning_debug = self._generate_with_visual_token_pruning(inputs=inputs, gen=gen)
                out_ids = torch.cat([inputs["input_ids"], gen_ids.to(inputs["input_ids"].device)], dim=1)
            else:
                pruning_debug = None
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=gen.do_sample,
                    temperature=gen.temperature if gen.do_sample else None,
                    top_p=gen.top_p if gen.do_sample else None,
                    repetition_penalty=getattr(gen, "repetition_penalty", 1.0),
                )
        finally:
            clear_dma_query_routing_sketch(self.model)
            self.model._dma_retention_stats_disabled = previous_stats_disabled

        # 只解码新生成部分
        if pruning_enabled:
            gen_ids = out_ids[:, input_ids.shape[1] :]
        else:
            gen_ids = out_ids[:, input_ids.shape[1] :]
        pred = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        out = {
            "pred": pred,
            "visual_tokens": int(pruning_debug["visual_tokens_after"]) if pruning_enabled else visual_tokens,
            "visual_tokens_before": visual_tokens,
            "visual_tokens_after": (
                int(pruning_debug["visual_tokens_after"]) if pruning_enabled else visual_tokens
            ),
            "input_len": int(pruning_debug["input_len_after"]) if pruning_enabled else int(input_ids.shape[1]),
            "input_len_before": int(input_ids.shape[1]),
            "input_len_after": (
                int(pruning_debug["input_len_after"]) if pruning_enabled else int(input_ids.shape[1])
            ),
            "gen_len": int(gen_ids.shape[1]),
        }
        if pruning_enabled and pruning_debug is not None:
            out.update(
                {
                    "page_keep_indices": pruning_debug["page_keep_indices"],
                    "page_keep_counts": pruning_debug["page_keep_counts"],
                    "pruned_grid_thw": pruning_debug["pruned_grid_thw"],
                }
            )
            if pruning_debug.get("pruning_score_debug") is not None:
                out["pruning_score_debug"] = pruning_debug["pruning_score_debug"]
        if input_page_ids and record_dma_stats:
            out.update(summarize_dma_page_retention(self.model, input_page_ids=input_page_ids))
        return out

    @torch.inference_mode()
    def prefill_one(
        self,
        *,
        images: list,
        input_page_ids: list[int] | None = None,
        user_text: str,
        system_text: str,
    ) -> dict[str, Any]:
        """只执行一次 prompt prefill，用于获取 DMA 页保留统计，不生成答案 token。"""
        messages = self._build_messages(
            images=images,
            user_text=user_text,
            system_text=system_text,
            assistant_text=None,
        )
        inputs = self._prepare_inputs(messages=messages, add_generation_prompt=True)
        input_ids = inputs["input_ids"]
        visual_tokens = int((input_ids == self._image_token_id).sum().item())
        if input_page_ids:
            reset_dma_page_stats(self.model)

        set_dma_query_routing_sketch(
            self.model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_token_id=self._image_token_id,
        )
        try:
            _ = self.model(**inputs, use_cache=False)
        finally:
            clear_dma_query_routing_sketch(self.model)

        out = {
            "pred": "",
            "visual_tokens": visual_tokens,
            "input_len": int(input_ids.shape[1]),
            "gen_len": 0,
            "prefill_only": True,
        }
        if input_page_ids:
            out.update(summarize_dma_page_retention(self.model, input_page_ids=input_page_ids))
        return out

    def postprocess_multiple_choice(self, pred: str) -> str:
        letter = extract_choice_letter(pred)
        return letter or pred
