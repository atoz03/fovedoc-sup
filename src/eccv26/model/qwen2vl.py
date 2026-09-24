from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from eccv26.model.dma import (
    DMAConfig,
    apply_dma_to_qwen3vl_model,
    clear_dma_query_routing_sketch,
    reset_dma_page_stats,
    set_dma_query_routing_sketch,
    summarize_dma_page_retention,
)
from eccv26.model.qwen3vl import build_qwen3vl_messages, prepare_qwen3vl_inputs


@dataclass(frozen=True)
class GenerateConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float = 1.0
    assistant_prefill_text: str | None = None


class Qwen2VL:
    """B1 第二 backbone 的轻量 wrapper，先覆盖视觉 DMA prefill / page-retention smoke。"""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "auto",
        device_map: str | dict[str, Any] | None = "auto",
        attn_implementation: str | None = None,
        dma: DMAConfig | None = None,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

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

        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "local_files_only": True,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if dma is not None:
            apply_dma_to_qwen3vl_model(self.model, dma)
        self.model.eval()

        self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

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

    @torch.inference_mode()
    def prefill_one(
        self,
        *,
        images: list,
        input_page_ids: list[int] | None = None,
        user_text: str,
        system_text: str,
    ) -> dict[str, Any]:
        """只执行一次 prompt prefill，用于验证 Qwen2-VL 视觉 DMA 页保留链路。"""
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

    @torch.inference_mode()
    def generate_one(
        self,
        *,
        images: list,
        input_page_ids: list[int] | None = None,
        user_text: str,
        system_text: str,
        gen: GenerateConfig,
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
        if input_page_ids:
            reset_dma_page_stats(self.model)

        set_dma_query_routing_sketch(
            self.model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_token_id=self._image_token_id,
        )
        try:
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

        gen_ids = out_ids[:, input_ids.shape[1] :]
        pred = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        out = {
            "pred": pred,
            "visual_tokens": visual_tokens,
            "input_len": int(input_ids.shape[1]),
            "gen_len": int(gen_ids.shape[1]),
        }
        if input_page_ids:
            out.update(summarize_dma_page_retention(self.model, input_page_ids=input_page_ids))
        return out
