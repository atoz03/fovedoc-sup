#!/usr/bin/env python3
"""DMR Step-0 去险实验：在 oracle gold-page 条件下，测一个显式"文档记忆"scratchpad
（从 gold 页 lexical 抽取的 per-page 事实文本）能否帮 reader 做跨页关联。

两个对照臂，输入图像完全相同（仅 gold evidence 页），唯一差别是 user_text 是否附带
lexical memory scratchpad：
  - pages        : gold 页图像 + 原始问题（reader 必须从原始视觉页做关联）
  - pages_memory : gold 页图像 + 从同一批 gold 页 lexical 抽取的 per-page 证据条（DMR 记忆）

关键诊断：V-MQAR（两页关联）下 pages_memory 相对 pages 的提升。V-NIAH（单页）作对照，
记忆带来的增益应当更小。模型/口径沿用 eval config（默认指向已训练 best_val），保证 DMA
flag 匹配——尽管 DMA 已证伪 inert，这里只把它当成"代表性 reader"。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from eccv26.data.benchmarks import LoadSpec, iter_samples
from eccv26.eval import (
    _build_user_text,
    _resolve_document_json_path,
    _score_one,
    _system_prompt_for_dataset,
)
from eccv26.model.cross_attn import CrossAttnConfig
from eccv26.model.dma import (
    DMAConfig,
    DMAExactBlockCrossBlockAttentionScorer,
    DMAExactBlockQueryInteractionScorer,
)
from eccv26.model.qwen3vl import GenerateConfig, Qwen3VL
from eccv26.utils.exact_block import (
    EXACT_BLOCK_FEATURE_DIM,
    EXACT_BLOCK_TEXT_SKETCH_DIM,
    build_exact_block_candidate_rows,
)
from eccv26.utils.image import resize_to_max_pixels
from eccv26.utils.io import dump_json, ensure_dir
from eccv26.utils.memory_controls import (
    CROP_SCALES,
    MEMORY_PREFIX_STYLES,
    MEMORY_SELECTORS,
    VISUAL_MEMORY_MODES,
    apply_memory_selector,
    crop_blocks,
    draw_block_highlights,
    lexical_char_budget,
    localized_blocks_from_facts,
    memory_prefix_text,
    text_memory_user_text,
    visual_memory_user_text,
)


def _page_id_to_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _collect_gold_page_ids(meta: dict[str, Any]) -> list[int]:
    raw = meta.get("evidence")
    if not isinstance(raw, list):
        return []
    ids: list[int] = []
    for item in raw:
        if isinstance(item, dict):
            pid = _page_id_to_int(item.get("page_id"))
            if pid is not None:
                ids.append(pid)
    return sorted(set(ids))


def _build_page_subset(sample, target_page_ids: set[int]) -> tuple[list[Any], list[int]]:
    """从输入页中挑出 target_page_ids 对应的页图像（保持输入顺序）。"""
    input_page_ids = [_page_id_to_int(v) for v in (sample.meta.get("_input_page_ids") or [])]
    if len(input_page_ids) < len(sample.images):
        return [], []
    imgs: list[Any] = []
    pids: list[int] = []
    for image, pid in zip(sample.images, input_page_ids):
        if pid is not None and pid in target_page_ids:
            imgs.append(image)
            pids.append(pid)
    return imgs, pids


def _load_retrieved_pages_map(paths: list[str], top_k: int) -> dict[str, list[int]]:
    """sample_id -> top-k ranked_page_ids（来自 ColQwen2 等外部检索预测）。"""
    out: dict[str, list[int]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = str(r.get("sample_id") or r.get("id") or "").strip()
                ranked = [p for p in (_page_id_to_int(x) for x in (r.get("ranked_page_ids") or [])) if p is not None]
                if sid and ranked:
                    out[sid] = ranked[: max(1, int(top_k))]
    return out


def _load_selector(ckpt_path: str, scorer_type: str, hidden_dim: int, device):
    if scorer_type == "cross_block_attention":
        scorer = DMAExactBlockCrossBlockAttentionScorer(EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_TEXT_SKETCH_DIM, hidden_dim)
    else:
        scorer = DMAExactBlockQueryInteractionScorer(EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_TEXT_SKETCH_DIM, hidden_dim)
    scorer.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    scorer.to(device).eval()
    return scorer


def _selector_scores(selector, rows, device) -> list[float]:
    feats = torch.tensor([list(r.get("features") or []) for r in rows], dtype=torch.float32, device=device)
    pids = torch.tensor([int(r["page_id"]) for r in rows], dtype=torch.long, device=device)
    qsk = torch.tensor([list(r.get("query_sketch") or [0.0] * EXACT_BLOCK_TEXT_SKETCH_DIM) for r in rows],
                       dtype=torch.float32, device=device)
    bsk = torch.tensor([list(r.get("block_sketch") or [0.0] * EXACT_BLOCK_TEXT_SKETCH_DIM) for r in rows],
                       dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = selector(feats, page_ids=pids, query_sketch=qsk, block_sketch=bsk)
    return logits.detach().cpu().tolist()


def _build_memory_scratchpad(
    *,
    sample,
    gold_page_ids: list[int],
    benchmark_root: str,
    topk: int,
    max_chars_per_block: int,
    selector=None,
    selector_device=None,
    union_selector=None,
    document_root: str | None = None,
    strict_document_root: bool = False,
    memory_selector: str = "lexical",
) -> tuple[str, list[dict[str, Any]], str]:
    """从页内 block 抽 top-k 证据块组成 per-page 记忆文本。
    selector 为 None 时用 lexical 打分（naive memory）；否则用训练好的 selector 打分。
    union_selector 提供时：memory = (lexical top-k) ∪ (selector top-k)，去重，budget≤2k —
    用来测 selector 的 gold-recall 是否作为对 lexical 的补充信号有用（而非替换）。
    memory_selector（selection 对照，见 eccv26.utils.memory_controls）：
      lexical = 论文原路径；random = 同页面同 block 数、与问题无关的伪随机 top-k；
      random_chars = 同上但按 lexical 实际打包字符数配对；all = 不选择，全部 block 按阅读序打包（忽略 topk）。"""
    document_json_path = _resolve_document_json_path(sample.meta, benchmark_root)
    block_source = "pdf_text_layer"
    if document_root:
        # OCR-parity control: same pages, blocks recovered from the rendered images instead of
        # the born-digital PDF text layer. A silent per-document fallback would contaminate the
        # OCR arm invisibly, so the source is recorded per sample and can be made fatal.
        doc_id = str(sample.meta.get("document_id") or "").strip()
        ocr_path = Path(document_root) / f"{doc_id}.json" if doc_id else None
        if ocr_path is not None and ocr_path.exists():
            document_json_path = str(ocr_path)
            block_source = "ocr"
        elif strict_document_root:
            raise FileNotFoundError(
                f"--require-memory-document-root: no OCR store for document_id={doc_id!r} "
                f"under {document_root}"
            )
        else:
            block_source = "pdf_text_layer_fallback"
    if document_json_path is None:
        return "", [], block_source
    rows = build_exact_block_candidate_rows(
        document_json_path=document_json_path,
        source_page_ids=gold_page_ids,
        query_texts=[sample.question, sample.context],
    )
    rows = [r for r in rows if str(r.get("text") or "").strip()]
    if not rows:
        return "", [], block_source
    if union_selector is not None:
        # union 模式：分别取 lexical top-k 与 selector top-k，去重合并
        lex = [float(r.get("lexical_score") or 0.0) for r in rows]
        sel = _selector_scores(union_selector, rows, selector_device)
        for r, ls, ss in zip(rows, lex, sel):
            r["_lex_score"] = float(ls)
            r["_sel_score"] = float(ss)  # for reporting; page-sort uses this
        k = max(1, int(topk))
        top_lex = sorted(rows, key=lambda r: r["_lex_score"], reverse=True)[:k]
        top_sel = sorted(rows, key=lambda r: r["_sel_score"], reverse=True)[:k]
        seen: set[tuple[int, str]] = set()
        selected = []
        for r in top_lex + top_sel:
            key = (int(r["page_id"]), str(r.get("block_id")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(r)
    elif selector is None and memory_selector != "lexical":
        # selection 对照：random / all。seed 取 document_id + 问题原文，保证跨 shard、跨重跑可复现。
        seed_key = f"{sample.meta.get('document_id') or ''}\x1f{sample.question or ''}"
        # random_chars 的字符预算 = lexical 臂在同一样本、同一截断规则下的实际打包字符数
        budget = (lexical_char_budget(rows, topk=topk, max_chars_per_block=max_chars_per_block)
                  if memory_selector == "random_chars" else None)
        selected = apply_memory_selector(rows, selector=memory_selector, topk=topk, seed_key=seed_key,
                                         char_budget=budget, max_chars_per_block=max_chars_per_block)
    else:
        if selector is not None:
            scores = _selector_scores(selector, rows, selector_device)
        else:
            scores = [float(r.get("lexical_score") or 0.0) for r in rows]
        for r, sc in zip(rows, scores):
            r["_sel_score"] = float(sc)
        rows.sort(key=lambda r: r["_sel_score"], reverse=True)
        selected = rows[: max(1, int(topk))]
    # 按 gold 页顺序稳定排版，便于 reader 做跨页对齐
    selected.sort(key=lambda r: (gold_page_ids.index(int(r["page_id"])) if int(r["page_id"]) in gold_page_ids else 0,
                                 -r["_sel_score"]))
    facts: list[dict[str, Any]] = []
    lines: list[str] = []
    for r in selected:
        text = str(r.get("text") or "").strip().replace("\n", " ")
        if max_chars_per_block > 0 and len(text) > max_chars_per_block:
            text = text[:max_chars_per_block].rstrip() + "..."
        pid = int(r["page_id"])
        lines.append(f"[第{pid}页] {text}")
        facts.append({"page_id": pid, "block_id": r.get("block_id"),
                      "sel_score": float(r["_sel_score"]),
                      "lexical_score": float(r.get("lexical_score") or 0.0), "text": text,
                      # 定位信息：红框 / 裁剪臂据此在页面图上找到与文本臂完全相同的 block
                      "bbox": list(r["bbox"]) if r.get("bbox") is not None else None,
                      "page_width": r.get("page_width"), "page_height": r.get("page_height"),
                      "order_index": r.get("order_index")})
    return "\n".join(lines), facts, block_source


# 论文各臂使用的原前缀（keyword 版）；neutral 版与视觉臂前缀见 eccv26.utils.memory_controls。
_MEMORY_PREFIX = memory_prefix_text("keyword")


def _assoc_scratchpad_from_facts(facts: list[dict[str, Any]], max_groups: int = 0) -> str:
    """把已选中的记忆 facts 按页码分组渲染成"证据组"结构（关联式排版）。
    内容与 flat scratchpad 完全相同，只改变版式：每个不同页码 = 一组带标号的证据，
    强制 reader 把"分散在不同页的证据"显式配对再组合，攻击 join 失败。"""
    by_page: dict[int, list[str]] = {}
    order: list[int] = []
    for f in facts:
        pid = int(f.get("page_id"))
        txt = str(f.get("text") or "").strip().replace("\n", " ")
        if not txt:
            continue
        if pid not in by_page:
            by_page[pid] = []
            order.append(pid)
        by_page[pid].append(txt)
    if max_groups and len(order) > max_groups:
        order = order[:max_groups]
    blocks: list[str] = []
    for gi, pid in enumerate(order, start=1):
        body = "\n".join(f"  - {t}" for t in by_page[pid])
        blocks.append(f"【证据组{gi} · 第{pid}页】\n{body}")
    return "\n".join(blocks)


def _mode_user_text(*, mode: str, base_user_text: str, scratchpad: str,
                    memory_prefix_style: str = "keyword") -> str:
    if mode == "pages":
        return base_user_text
    if mode == "pages_memory":
        # keyword 版与原实现逐字节相同；neutral 版只去掉"按关键词抽取的候选证据"这一相关性暗示
        return text_memory_user_text(base_user_text=base_user_text, scratchpad=scratchpad,
                                     style=memory_prefix_style)
    if mode == "pages_memory_assoc":
        # 关联式记忆：证据已按页分组（每组来自不同页）；显式要求"逐组读取→跨组组合"。
        if scratchpad.strip() == "":
            return base_user_text
        return (
            "下面给出按【页码分组】的文档记忆，每一组来自不同的页面，"
            "每组可能只包含最终答案的一部分。这个问题通常需要把【多个不同证据组】的信息组合起来：\n"
            "请先确认每一组各自提供了什么信息，再把不同组的信息合并成一个完整答案，只回答最短答案。\n"
            f"{scratchpad}\n\n"
            f"{base_user_text}"
        )
    if mode == "pages_memory_selfask":
        # 主动跨页关联：先按页抽取各自相关事实，再显式合并成最终答案。
        if scratchpad.strip() == "":
            return base_user_text
        return (
            f"{_MEMORY_PREFIX}回答问题。这个问题需要跨页关联多条证据，请严格按以下步骤：\n"
            "1) 先逐条列出回答该问题所需的、来自不同页码的关键事实（标注页码）；\n"
            "2) 再把这些事实组合起来推理；\n"
            "3) 最后另起一行，以 `Final answer:` 开头，只写最短的最终答案。\n"
            f"{scratchpad}\n\n"
            f"{base_user_text}"
        )
    raise ValueError(f"未知模式：{mode}")


_2CALL_EXTRACT = (
    "下面给出从相关页面抽取的候选证据条（每条标注页码）。这个问题通常需要把分布在不同页的"
    "多条信息组合起来。第一步：请只做【抽取】，不要给最终答案——逐条列出回答该问题所需的关键"
    "事实，每条标注它来自第几页、对应问题的哪一部分（格式：`第X页 → <这一部分的具体值>`）。"
)
_2CALL_COMBINE = (
    "这是你上一步从文档记忆中抽取的分页事实：\n{facts}\n\n"
    "现在第二步：把上面各页的事实组合成一个完整答案，只输出最短的最终答案，不要解释。"
)


def _generate_two_call(*, model, images, gold_pids, base_user_text, system_text,
                       scratchpad, gen_extract, gen_combine):
    """B2 两段式：call-1 从 memory 逐页抽取子答案，call-2 组合成最终最短答案。
    返回 (final_pred, intermediate_extract, out2)。"""
    if scratchpad.strip() == "":
        out = model.generate_one(images=images, input_page_ids=gold_pids,
                                 user_text=base_user_text, system_text=system_text, gen=gen_combine)
        return str(out.get("pred") or ""), "", out
    user1 = f"{_2CALL_EXTRACT}\n{scratchpad}\n\n{base_user_text}"
    out1 = model.generate_one(images=images, input_page_ids=gold_pids,
                              user_text=user1, system_text=system_text, gen=gen_extract)
    extract = str(out1.get("pred") or "").strip()
    user2 = _2CALL_COMBINE.format(facts=extract) + "\n\n" + base_user_text
    out2 = model.generate_one(images=images, input_page_ids=gold_pids,
                              user_text=user2, system_text=system_text, gen=gen_combine)
    return str(out2.get("pred") or ""), extract, out2


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(pred: str) -> str:
    """去掉 Thinking 模型的推理块，只保留最终答案。
    处理两种情况：完整 <think>...</think> 包裹，或模板已开 <think>、输出以 </think> 收尾。"""
    text = str(pred or "")
    stripped = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in stripped:  # 模板预置了开标签，取最后一个闭标签之后的内容
        stripped = stripped.rsplit("</think>", 1)[-1]
    stripped = stripped.strip()
    return stripped or text.strip()


def _extract_final_answer(pred: str) -> str:
    """从 selfask 链式输出里抽 `Final answer:` 之后的最短答案；无标记则退回原文。"""
    text = str(pred or "")
    marker = None
    for cand in ("Final answer:", "final answer:", "Final Answer:", "最终答案：", "最终答案:"):
        idx = text.rfind(cand)
        if idx >= 0:
            marker = idx + len(cand)
            break
    if marker is None:
        return text
    tail = text[marker:].strip()
    return tail.splitlines()[0].strip() if tail else text


def _load_model(cfg, reader_impl: str = "qwen3vl"):
    if reader_impl == "hf":
        # Family-agnostic reader for the generalization grid (LLaVA-OneVision, Idefics3,
        # Qwen2.5-VL, ...). No DMA/pruning, which those arms do not use anyway.
        from eccv26.model.hf_vlm import HFVLM

        overrides = getattr(cfg.model, "processor_overrides", None)
        return HFVLM(
            str(cfg.model.path),
            dtype=str(cfg.model.dtype),
            device_map=cfg.model.device_map,
            attn_implementation=getattr(cfg.model, "attn_implementation", None),
            trust_remote_code=bool(getattr(cfg.model, "trust_remote_code", False)),
            processor_overrides=OmegaConf.to_container(overrides, resolve=True) if overrides else None,
        )
    dma_cfg = DMAConfig(**OmegaConf.to_container(cfg.dma, resolve=True)) if "dma" in cfg else DMAConfig()
    cross_cfg = CrossAttnConfig(**OmegaConf.to_container(cfg.cross_attn, resolve=True)) if "cross_attn" in cfg else CrossAttnConfig()
    return Qwen3VL(
        str(cfg.model.path),
        dtype=str(cfg.model.dtype),
        device_map=cfg.model.device_map,
        attn_implementation=cfg.model.attn_implementation,
        dma=dma_cfg,
        cross_attn=cross_cfg,
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scorable = [r for r in records if r.get("scoring", {}).get("scorable") and r["scoring"].get("score") is not None]
    scores = [float(r["scoring"]["score"]) for r in scorable]
    return {
        "num_samples": len(records),
        "num_scorable": len(scores),
        "score_mean": (sum(scores) / len(scores)) if scores else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DMR Step-0 oracle scratchpad 去险。")
    p.add_argument("--eval-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--benchmark-root", default=None)
    p.add_argument("--datasets", default="V-MQAR,V-NIAH")
    p.add_argument("--split", default="all")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--reader-impl", choices=("qwen3vl", "hf"), default="qwen3vl",
                   help="hf = family-agnostic AutoProcessor reader for the generalization grid "
                        "(no DMA/pruning; selector arms are unavailable)")
    p.add_argument("--memory-document-root", default=None,
                   help="OCR-parity control: directory of <document_id>.json whose blocks replace the "
                        "PDF-text-layer blocks when packing memory (images/retrieval unchanged)")
    p.add_argument("--require-memory-document-root", action="store_true",
                   help="fail instead of silently falling back to the PDF text layer for documents "
                        "missing from --memory-document-root (keeps the OCR arm 100%% OCR)")
    p.add_argument("--max-pages", type=int, default=120,
                   help="MMLongBench_DOC only: pages rendered per document. The 120 default is the "
                        "loader's historical cap; it silently excludes the 8 longest documents "
                        "(151-468 pages) from any run needing a complete page set, which are exactly "
                        "the documents where retrieval is hardest")
    p.add_argument("--memory-topk", type=int, default=8)
    p.add_argument("--memory-max-chars-per-block", type=int, default=300,
                   help="每条记忆的字符上限；0 = 不截断（全量打包臂使用）。")
    p.add_argument("--memory-selector", default="lexical", choices=list(MEMORY_SELECTORS),
                   help="selection 对照：lexical=论文原路径；random=同页面同 block 数、与问题无关的伪随机 top-k；"
                        "random_chars=同上但按 lexical 实际打包字符数配对；"
                        "all=不选择，全部 block 按阅读序全量打包（忽略 --memory-topk）。")
    p.add_argument("--memory-prefix-style", default="keyword", choices=list(MEMORY_PREFIX_STYLES),
                   help="记忆前缀措辞：keyword=论文原文（'按关键词抽取的候选证据条'）；"
                        "neutral=去掉相关性暗示（'抽取的文本条'）。视觉臂的前缀同步切换。")
    p.add_argument("--crop-scale", default="reader", choices=list(CROP_SCALES),
                   help="pages_crops 的裁剪分辨率：reader=与 reader 看到的整页同像素密度（纯定位）；"
                        "native=渲染原始分辨率（定位 + 分辨率）。")
    p.add_argument("--image-max-pixels", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--modes", default="pages,pages_memory",
                   help="逗号分隔：pages,pages_memory,pages_memory_selfask,pages_highlight,pages_crops")
    p.add_argument("--page-source", default="gold", choices=["gold", "retrieved"],
                   help="页来源：gold=oracle 证据页；retrieved=外部检索 top-k（带 distractor）。")
    p.add_argument("--retrieved-pages-jsonl", default=None,
                   help="逗号分隔的检索预测 jsonl（含 sample_id + ranked_page_ids），page-source=retrieved 时必填。")
    p.add_argument("--retrieved-top-k", type=int, default=8)
    p.add_argument("--memory-oracle-filter", action="store_true",
                   help="完美 selector 上界：图像保持 retrieved，但 memory 只从 (retrieved ∩ gold) 页抽取，"
                        "用来测 denoise memory 的 headroom。")
    p.add_argument("--selector-ckpt", default=None,
                   help="训练好的 DMR memory selector .pt；提供后 memory 用 selector 打分排序（否则用 lexical）。")
    p.add_argument("--memory-union-ckpt", default=None,
                   help="union 记忆：memory = lexical top-k ∪ selector(此 ckpt) top-k（去重，budget≤2k）；"
                        "测 selector gold-recall 作为对 lexical 的补充是否有用。")
    p.add_argument("--selector-scorer", default="cross_block_attention",
                   choices=["cross_block_attention", "query_interaction"])
    p.add_argument("--selector-hidden-dim", type=int, default=32)
    p.add_argument("--selfask-max-new-tokens", type=int, default=384,
                   help="selfask 链式输出需要更多 token。")
    p.add_argument("--strip-think", action="store_true",
                   help="Thinking 模型：判分前剥离 <think>…</think> 推理块，只留最终答案。")
    p.add_argument("--only-ids-file", default=None,
                   help="只跑这个文件里列出的 sample id（每行一个），用于补跑截断样本。")
    p.add_argument("--num-shards", type=int, default=1,
                   help="把样本按 (idx %% num_shards) 切片，配合不同 GPU/output-dir 并行；之后合并。")
    p.add_argument("--shard-idx", type=int, default=0, help="本进程负责的 shard 编号 [0, num_shards)。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.eval_config)
    benchmark_root = str(args.benchmark_root or cfg.data.benchmark_root)
    memory_document_root = str(args.memory_document_root) if args.memory_document_root else None
    image_max_pixels = int(args.image_max_pixels or cfg.data.image_max_pixels)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    if str(args.memory_selector) != "lexical" and (args.selector_ckpt or args.memory_union_ckpt):
        raise SystemExit("--memory-selector random/all 与 --selector-ckpt / --memory-union-ckpt 互斥")
    if any(m in VISUAL_MEMORY_MODES for m in modes) and str(args.memory_selector) == "all":
        raise SystemExit("视觉定位臂（pages_highlight/pages_crops）需要有限的 block 集合，不能与 --memory-selector all 组合")
    only_ids = None
    if args.only_ids_file:
        only_ids = {l.strip() for l in open(args.only_ids_file, encoding="utf-8") if l.strip()}
        print(f"[dmr_derisk] only-ids filter: {len(only_ids)} ids", flush=True)
    out_dir = Path(ensure_dir(args.output_dir))

    retrieved_map: dict[str, list[int]] = {}
    if str(args.page_source) == "retrieved":
        if not args.retrieved_pages_jsonl:
            raise SystemExit("page-source=retrieved 需要 --retrieved-pages-jsonl")
        paths = [p.strip() for p in str(args.retrieved_pages_jsonl).split(",") if p.strip()]
        retrieved_map = _load_retrieved_pages_map(paths, int(args.retrieved_top_k))
        print(f"[dmr_derisk] loaded retrieved pages for {len(retrieved_map)} samples (top_k={args.retrieved_top_k})", flush=True)

    model = _load_model(cfg, reader_impl=str(args.reader_impl))

    selector = None
    selector_device = None
    if args.selector_ckpt:
        try:
            selector_device = next(model.model.parameters()).device
        except Exception:
            selector_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        selector = _load_selector(args.selector_ckpt, args.selector_scorer, int(args.selector_hidden_dim), selector_device)
        print(f"[dmr_derisk] loaded memory selector: {args.selector_ckpt} ({args.selector_scorer}) on {selector_device}", flush=True)

    union_selector = None
    if args.memory_union_ckpt:
        if selector_device is None:
            try:
                selector_device = next(model.model.parameters()).device
            except Exception:
                selector_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        union_selector = _load_selector(args.memory_union_ckpt, args.selector_scorer, int(args.selector_hidden_dim), selector_device)
        print(f"[dmr_derisk] loaded UNION selector: {args.memory_union_ckpt} (memory = lexical∪selector)", flush=True)

    def _make_gen(max_new_tokens: int) -> GenerateConfig:
        return GenerateConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(cfg.model.do_sample),
            temperature=float(cfg.model.temperature),
            top_p=float(cfg.model.top_p),
            repetition_penalty=float(getattr(cfg.model, "repetition_penalty", 1.0)),
        )

    gen_by_mode = {
        m: _make_gen(args.selfask_max_new_tokens if (m.endswith("selfask") or m == "pages_memory_2call")
                     else args.max_new_tokens)
        for m in modes
    }
    # two-call: call-1 (extract) gets the long budget above; call-2 (combine) stays short.
    gen_by_mode["_combine"] = _make_gen(args.max_new_tokens)
    dump_json(out_dir / "run_config.json", {
        "eval_config": str(args.eval_config),
        "model_path": str(cfg.model.path),
        "benchmark_root": benchmark_root,
        "datasets": datasets,
        "split": str(args.split),
        "max_samples": int(args.max_samples),
        "modes": modes,
        "memory_topk": int(args.memory_topk),
        "memory_max_chars_per_block": int(args.memory_max_chars_per_block),
        "memory_selector": str(args.memory_selector),
        "memory_prefix_style": str(args.memory_prefix_style),
        "crop_scale": str(args.crop_scale),
        "image_max_pixels": image_max_pixels,
        "max_new_tokens": int(args.max_new_tokens),
        "oracle": "gold_evidence_pages",
        "memory_source": "lexical_block_packing_on_gold_pages",
        "memory_document_root": memory_document_root,
        "memory_block_source": "ocr" if memory_document_root else "pdf_text_layer",
    })

    records_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for dataset in datasets:
        spec = LoadSpec(benchmark_root=benchmark_root, dataset=dataset,
                        split=str(args.split), max_samples=int(args.max_samples),
                        max_pages=int(args.max_pages))
        ds_dir = ensure_dir(out_dir / dataset.lower().replace("-", ""))
        writers = {m: (Path(ds_dir) / f"predictions_{m}.jsonl").open("w", encoding="utf-8") for m in modes}
        try:
            for idx, sample in enumerate(iter_samples(spec), start=1):
                if int(args.num_shards) > 1 and ((idx - 1) % int(args.num_shards)) != int(args.shard_idx):
                    continue
                if only_ids is not None and str(sample.sample_id) not in only_ids:
                    continue
                source_kind = None if sample.meta.get("_source_kind") is None else str(sample.meta.get("_source_kind"))
                gold_pids_all = _collect_gold_page_ids(sample.meta)
                if str(args.page_source) == "gold":
                    target_ids = set(gold_pids_all)
                    skip_reason0 = "no_gold_page_subset"
                else:
                    ranked = retrieved_map.get(str(sample.sample_id), [])
                    target_ids = set(ranked)
                    skip_reason0 = "no_retrieved_pages"
                sel_imgs, sel_pids = _build_page_subset(sample, target_ids)
                skip_reason = None if sel_imgs else skip_reason0
                # 检索召回：top-k 是否覆盖 gold 证据页（全召回）
                gold_set = set(gold_pids_all)
                full_recall = bool(gold_set) and gold_set.issubset(set(sel_pids))
                partial_recall = len(gold_set & set(sel_pids))
                resized = []
                scratchpad, facts, block_source = "", [], None
                if not skip_reason:
                    for im in sel_imgs:
                        rim, _ = resize_to_max_pixels(im, image_max_pixels)
                        resized.append(rim)
                    memory_pages = sel_pids
                    if args.memory_oracle_filter:
                        memory_pages = [p for p in sel_pids if p in gold_set]  # 完美 selector：只留 gold∩retrieved
                    scratchpad, facts, block_source = ("", [], None) if not memory_pages else _build_memory_scratchpad(
                        sample=sample, gold_page_ids=memory_pages, benchmark_root=benchmark_root,
                        topk=int(args.memory_topk), max_chars_per_block=int(args.memory_max_chars_per_block),
                        selector=selector, selector_device=selector_device, union_selector=union_selector,
                        document_root=memory_document_root,
                        strict_document_root=bool(args.require_memory_document_root),
                        memory_selector=str(args.memory_selector),
                    )
                gold_pids = sel_pids  # 下游 generate 用选中的页（图像始终是 retrieved）
                scratchpad_assoc = _assoc_scratchpad_from_facts(facts)
                base_user_text = _build_user_text(sample.context, sample.question, source_kind)
                system_text = _system_prompt_for_dataset(str(sample.dataset), source_kind)
                for mode in modes:
                    intermediate = None
                    n_visual_extra = 0
                    if skip_reason:
                        pred, latency = "", None
                        out: dict[str, Any] = {"visual_tokens": None, "input_len": None}
                    elif mode == "pages_memory_2call":
                        t0 = time.time()
                        pred, intermediate, out = _generate_two_call(
                            model=model, images=resized, gold_pids=gold_pids,
                            base_user_text=base_user_text, system_text=system_text, scratchpad=scratchpad,
                            gen_extract=gen_by_mode[mode], gen_combine=gen_by_mode.get("_combine", gen_by_mode[mode]))
                        latency = time.time() - t0
                    elif mode in VISUAL_MEMORY_MODES:
                        # localization 对照：与文本臂完全相同的 block 集合，但不给文本，只在页面图上
                        # 画红框（pages_highlight）或把区域裁成小图附在页面之后（pages_crops）。
                        blocks = localized_blocks_from_facts(facts)
                        if mode == "pages_highlight":
                            mode_images = draw_block_highlights(resized, gold_pids, blocks)
                            n_visual_extra = 0
                        else:
                            crops = crop_blocks(sel_imgs, gold_pids, blocks, scale=str(args.crop_scale),
                                                reader_max_pixels=image_max_pixels)
                            mode_images = list(resized) + crops
                            n_visual_extra = len(crops)
                        user_text = visual_memory_user_text(mode=mode, base_user_text=base_user_text,
                                                            blocks=blocks, style=str(args.memory_prefix_style))
                        t0 = time.time()
                        out = model.generate_one(images=mode_images, input_page_ids=gold_pids,
                                                 user_text=user_text, system_text=system_text, gen=gen_by_mode[mode])
                        latency = time.time() - t0
                        pred = str(out.get("pred") or "")
                    else:
                        mode_scratchpad = scratchpad_assoc if mode == "pages_memory_assoc" else scratchpad
                        user_text = _mode_user_text(mode=mode, base_user_text=base_user_text, scratchpad=mode_scratchpad,
                                                    memory_prefix_style=str(args.memory_prefix_style))
                        t0 = time.time()
                        out = model.generate_one(images=resized, input_page_ids=gold_pids,
                                                 user_text=user_text, system_text=system_text, gen=gen_by_mode[mode])
                        latency = time.time() - t0
                        pred = str(out.get("pred") or "")
                    pred_raw = pred
                    if args.strip_think and pred:
                        pred = _strip_think(pred)
                    scoring_pred = _extract_final_answer(pred) if mode.endswith("selfask") else pred
                    scoring = _score_one(str(sample.dataset), scoring_pred, sample.answers)
                    rec = {
                        "dataset": str(sample.dataset), "id": str(sample.sample_id), "mode": mode,
                        "question": sample.question, "answers": sample.answers,
                        "pred": pred, "pred_raw": pred_raw, "scoring_pred": scoring_pred,
                        "scoring": scoring, "latency_s": latency,
                        "page_source": str(args.page_source),
                        "selected_page_ids": gold_pids, "num_selected_pages": len(gold_pids),
                        "gold_evidence_page_ids": gold_pids_all,
                        "full_recall": full_recall, "partial_recall_count": partial_recall,
                        "memory_num_facts": len(facts), "memory_has_text": bool(scratchpad.strip()),
                        # provenance + packed size: lets the OCR arm be asserted 100%% OCR and the
                        # two stores be checked for block-budget parity rather than assumed equal.
                        "memory_block_source": block_source,
                        "memory_packed_chars": len(scratchpad),
                        "memory_facts": facts if mode in ("pages_memory", "pages_memory_assoc", *VISUAL_MEMORY_MODES) else None,
                        # selection / localization 对照的出处，便于合并后逐行断言各臂确实按预期构造
                        "memory_selector": str(args.memory_selector),
                        "memory_prefix_style": str(args.memory_prefix_style),
                        "visual_memory_images": n_visual_extra if mode in VISUAL_MEMORY_MODES else None,
                        "intermediate_extract": intermediate,
                        "visual_tokens": out.get("visual_tokens"), "skip_reason": skip_reason,
                    }
                    records_by_key[(dataset, mode)].append(rec)
                    writers[mode].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    writers[mode].flush()
                print(f"[dmr_derisk] {dataset} {idx}/{args.max_samples} done", flush=True)
        finally:
            for w in writers.values():
                w.close()

    metrics: dict[str, Any] = {"overall": {}, "by_dataset": {}}
    overall_acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, mode), records in sorted(records_by_key.items()):
        metrics["by_dataset"].setdefault(dataset, {})[mode] = _summarize(records)
        overall_acc[mode].extend(records)
    metrics["overall"] = {m: _summarize(recs) for m, recs in overall_acc.items()}
    # 各 mode score 与相对 pages baseline / 相对 pages_memory 的 delta
    deltas: dict[str, Any] = {}
    for dataset in {d for d, _ in records_by_key}:
        bd = metrics["by_dataset"].get(dataset, {})
        scores = {m: (bd.get(m) or {}).get("score_mean") for m in modes}
        base = scores.get("pages")
        mem = scores.get("pages_memory")
        row: dict[str, Any] = {"scores": scores}
        if base is not None:
            row["delta_vs_pages"] = {m: (None if s is None else round(s - base, 4)) for m, s in scores.items()}
        if mem is not None and "pages_memory_selfask" in scores and scores["pages_memory_selfask"] is not None:
            row["selfask_minus_memory"] = round(scores["pages_memory_selfask"] - mem, 4)
        deltas[dataset] = row
    metrics["deltas"] = deltas
    dump_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
