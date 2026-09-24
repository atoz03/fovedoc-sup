"""A minimal, family-agnostic VLM reader for the generalization grid.

`Qwen3VL` carries the whole DMA/pruning/block-scoring apparatus, which the retrieval-reading
gap experiment does not need — it needs one thing: hand a model N page images plus a text
prompt and read back a string. Replicating Qwen3VL for every backbone would import risk with
no benefit, so this wrapper implements only that contract, on top of the stock
`AutoProcessor` + chat-template path shared by Qwen2.5-VL, Qwen3-VL, LLaVA-OneVision and
Idefics3.

The `generate_one` signature matches `Qwen3VL.generate_one` for the arguments the DMR harness
actually passes, so the harness can swap readers behind `--reader-impl hf` and keep the arms,
prompts and scoring identical across families. DMA/pruning-specific keys are absent from the
returned dict by design: this reader has no such machinery to report on.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "auto": "auto",
}


def _resolve_dtype(dtype: Any) -> Any:
    if not isinstance(dtype, str):
        return dtype
    key = dtype.lower()
    if key in _DTYPES:
        return _DTYPES[key]
    resolved = getattr(torch, key, None)
    if resolved is None:
        raise ValueError(f"unknown dtype: {dtype!r}")
    return resolved


class HFVLM:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: Any = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        max_image_pixels: int | None = None,
        processor_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        torch_dtype = _resolve_dtype(dtype)

        # Per-family tiling knobs. Idefics3 emits 3036 tokens/page with its default image
        # splitting (48k for a 16-page prompt) and 184 with `do_image_splitting=False`, so the
        # grid needs this to run a uniform 16-page budget across families.
        processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        processor_kwargs.update(dict(processor_overrides or {}))
        if max_image_pixels:
            # Qwen-style processors accept a pixel cap; others ignore it.
            processor_kwargs["max_pixels"] = int(max_image_pixels)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        except TypeError:
            processor_kwargs.pop("max_pixels", None)
            self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self.processor_kwargs = processor_kwargs

        model_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = str(attn_implementation)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()

        tok = getattr(self.processor, "tokenizer", None)
        self.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_messages(
        self, *, images: list, user_text: str, system_text: str | None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": user_text})
        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        messages.append({"role": "user", "content": content})
        return messages

    @torch.inference_mode()
    def generate_one(
        self,
        *,
        images: list,
        user_text: str,
        system_text: str | None = None,
        gen: Any,
        input_page_ids: list[int] | None = None,
        record_dma_stats: bool = False,
    ) -> dict[str, Any]:
        del input_page_ids, record_dma_stats  # no DMA machinery in this reader

        messages = self._build_messages(images=images, user_text=user_text, system_text=system_text)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=list(images), return_tensors="pt", padding=True
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])

        do_sample = bool(getattr(gen, "do_sample", False))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(getattr(gen, "max_new_tokens", 64)),
            "do_sample": do_sample,
            "repetition_penalty": float(getattr(gen, "repetition_penalty", 1.0)),
            "pad_token_id": self.pad_token_id,
        }
        if do_sample:
            # Passing temperature/top_p with do_sample=False makes transformers warn and ignore them.
            gen_kwargs["temperature"] = float(getattr(gen, "temperature", 1.0))
            gen_kwargs["top_p"] = float(getattr(gen, "top_p", 1.0))

        t0 = time.time()
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        latency = time.time() - t0

        new_ids = out_ids[:, input_len:]
        pred = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return {
            "pred": pred,
            "input_len": input_len,
            "gen_len": int(new_ids.shape[1]),
            "latency": latency,
            "visual_tokens": None,
        }

    def postprocess_multiple_choice(self, pred: str) -> str:
        return pred
