"""记忆臂的"选择 / 定位"对照构件（selection & localization controls）。

背景（审稿关切）：memory 臂并不是"把同样的证据从像素换成文本"。它先按问题对 16 页的全部 block
做词面打分，选出 top-16，每条截到 300 字符，再连同页码塞进 prompt。于是 memory − images 的增益
至少有三个来源无法区分：
  (1) recognition ：VLM 从像素识别文字的能力不足；
  (2) localization：VLM 读得出文字，但在 16 页里定位不到答案；
  (3) selection   ：按问题条件化的 block 选择本身就是第二级细粒度检索器，把答案区域直接递到模型面前。

本模块提供把三者拆开所需的最小构件，全部复用 harness 已有的候选 block 行（含 bbox 与页尺寸）：
  - 选择变体 `apply_memory_selector`：
      random       —— 同页面、同 block 数预算、与问题无关的可复现伪随机 top-k（只去掉"按问题选"）；
      random_chars —— 同上，但预算改为字符数：词面打分偏爱长段落，按 block 数配对的 random 只有
                      lexical 约 1/3 的字符量，故再给一个按 lexical 实际打包字符数配对的版本；
      all          —— 不做选择，16 页全部 block 按阅读序全量打包（去掉"选"与"预算"）。
  - 视觉定位变体：把 lexical top-k 的 bbox 以红框画在页面图上（`draw_block_highlights`），
    或按 bbox 裁成小图附在页面之后（`crop_blocks`），**不给任何文本**。它们与文本记忆臂使用
    完全相同的 block 集合，因此"给文本"与"只指位置"严格配对。
  - 与文本记忆前缀逐句对应的提示语（`memory_prefix_text` / `visual_memory_user_text`），
    使各臂的措辞差异只剩"文本条 / 红框 / 裁剪图"，并可选去掉"按关键词"这一相关性暗示。

全部为纯函数、不依赖模型与 GPU，便于单测与 CPU 干跑。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

MEMORY_SELECTORS: tuple[str, ...] = ("lexical", "random", "random_chars", "all")
VISUAL_MEMORY_MODES: tuple[str, ...] = ("pages_highlight", "pages_crops")
CROP_SCALES: tuple[str, ...] = ("reader", "native")
MEMORY_PREFIX_STYLES: tuple[str, ...] = ("keyword", "neutral")

# 红框颜色与线宽：线宽按页面宽度的比例取整，保证在 reader 的 ~430 px 宽页面上仍有 2–3 px 可见。
HIGHLIGHT_COLOR: tuple[int, int, int] = (220, 0, 0)
HIGHLIGHT_WIDTH_RATIO: float = 0.006
HIGHLIGHT_MIN_WIDTH: int = 2

# 与 harness 的 `_MEMORY_PREFIX` 逐句对应。keyword 版是论文各臂实际使用的原文；neutral 版只去掉
# "按关键词抽取的候选证据"这一相关性暗示（改为"抽取的文本条"），其余措辞不变。
_TEXT_PREFIX: dict[str, str] = {
    "keyword": "下面先给出从相关页面中按关键词抽取的候选证据条（每条标注所在页码），"
               "这是用于跨页关联的文档记忆；请结合这些证据条与页面图像",
    "neutral": "下面先给出从相关页面中抽取的文本条（每条标注所在页码），"
               "这是用于跨页关联的文档记忆；请结合这些文本条与页面图像",
}
_HIGHLIGHT_PREFIX: dict[str, str] = {
    "keyword": "页面图像中已用红色方框标出从相关页面中按关键词抽取的候选证据区域（所在页码：{pages}），"
               "这是用于跨页关联的文档记忆；请结合这些红框区域与页面图像",
    "neutral": "页面图像中已用红色方框标出从相关页面中抽取的文本区域（所在页码：{pages}），"
               "这是用于跨页关联的文档记忆；请结合这些红框区域与页面图像",
}
_CROPS_PREFIX: dict[str, str] = {
    "keyword": "页面图像之后另附 {n} 张从相关页面中按关键词裁剪出的候选证据区域图（依次来自：{pages}），"
               "这是用于跨页关联的文档记忆；请结合这些区域图与页面图像",
    "neutral": "页面图像之后另附 {n} 张从相关页面中裁剪出的文本区域图（依次来自：{pages}），"
               "这是用于跨页关联的文档记忆；请结合这些区域图与页面图像",
}
_ANSWER_SUFFIX = "，只回答问题所需的最短答案。"


# --------------------------------------------------------------------------------------
# 选择变体
# --------------------------------------------------------------------------------------
def stable_random_score(seed_key: str, row: Mapping[str, Any]) -> float:
    """与问题内容无关的可复现伪随机分数 ∈ [0, 1)。

    seed_key 由调用方给定（建议 document_id + 问题原文），使同一样本在任何 shard / 任何重跑下
    选出同一组 block；分数不读取 lexical_score，因此选择与问题的词面相关性完全无关。
    """
    parts = [
        str(seed_key),
        str(row.get("page_id", "")),
        str(row.get("block_id", "")),
        str(row.get("order_index", "")),
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def packed_text_length(row: Mapping[str, Any], max_chars_per_block: int) -> int:
    """一行打包后的字符数（与 harness 的截断规则一致：超过上限则截到上限并加 "..."）。"""
    text = str(row.get("text") or "").strip().replace("\n", " ")
    cap = int(max_chars_per_block)
    if cap > 0 and len(text) > cap:
        return len(text[:cap].rstrip()) + 3
    return len(text)


def lexical_char_budget(rows: Sequence[Mapping[str, Any]], *, topk: int, max_chars_per_block: int) -> int:
    """lexical top-k 在同一截断规则下实际打包的字符总数，作为 random_chars 的字符预算。"""
    ordered = sorted(rows, key=lambda r: float(r.get("lexical_score") or 0.0), reverse=True)
    return sum(packed_text_length(r, max_chars_per_block) for r in ordered[: max(1, int(topk))])


def apply_memory_selector(
    rows: list[dict[str, Any]],
    *,
    selector: str,
    topk: int,
    seed_key: str,
    char_budget: int | None = None,
    max_chars_per_block: int = 0,
) -> list[dict[str, Any]]:
    """按 `selector` 给每行写入 `_sel_score` 并返回入选行（未按页排序，排序交给 harness）。

    - random      ：伪随机分数降序取 top-k（block 数与 lexical 臂相同）；
    - random_chars：伪随机顺序逐块累加，直到打包字符数达到 `char_budget`（通常取 lexical 臂在同一
                    样本上的实际打包字符数），块数可多于 topk；
    - all         ：全部行入选，`_sel_score = -order_index`，这样 harness 现有的"页内按 _sel_score 降序"
                    排序恰好还原为页内阅读序，无需改动排序逻辑；
    - lexical 不在此处处理（harness 原路径保持逐字节不变），传入即报错，避免静默双实现。
    """
    if selector not in MEMORY_SELECTORS:
        raise ValueError(f"未知 memory selector: {selector!r}（可选 {MEMORY_SELECTORS}）")
    if selector == "lexical":
        raise ValueError("lexical 选择由 harness 原路径负责，不应调用 apply_memory_selector")
    if selector in ("random", "random_chars"):
        for r in rows:
            r["_sel_score"] = stable_random_score(seed_key, r)
        ordered = sorted(rows, key=lambda r: r["_sel_score"], reverse=True)
        if selector == "random":
            return ordered[: max(1, int(topk))]
        if char_budget is None or int(char_budget) <= 0:
            raise ValueError("random_chars 需要正的 char_budget")
        picked: list[dict[str, Any]] = []
        total = 0
        for r in ordered:
            picked.append(r)
            total += packed_text_length(r, max_chars_per_block)
            if total >= int(char_budget):
                break
        return picked
    # all：不选择、不限预算
    for r in rows:
        r["_sel_score"] = -float(r.get("order_index") or 0)
    return list(rows)


# --------------------------------------------------------------------------------------
# 视觉定位变体
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizedBlock:
    """一个可在页面图上定位的 block：bbox 以 (page_width, page_height) 为参考坐标系。"""

    page_id: int
    bbox: tuple[float, float, float, float]
    page_width: float
    page_height: float


def localized_blocks_from_facts(facts: Sequence[Mapping[str, Any]]) -> list[LocalizedBlock]:
    """从打包后的 memory facts 提取可定位 block，顺序与打包顺序一致；缺 bbox / 页尺寸的条目跳过。"""
    out: list[LocalizedBlock] = []
    for f in facts:
        bbox = f.get("bbox")
        pw, ph = f.get("page_width"), f.get("page_height")
        if bbox is None or pw is None or ph is None:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            pw_f, ph_f = float(pw), float(ph)
        except (TypeError, ValueError):
            continue
        if pw_f <= 0 or ph_f <= 0 or x1 <= x0 or y1 <= y0:
            continue
        out.append(LocalizedBlock(page_id=int(f["page_id"]), bbox=(x0, y0, x1, y1), page_width=pw_f, page_height=ph_f))
    return out


def scale_bbox_to_image(block: LocalizedBlock, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """把 block 的 bbox 从其参考坐标系等比映射到给定图像的像素坐标（已裁到图像边界内）。"""
    img_w, img_h = image_size
    sx = float(img_w) / block.page_width
    sy = float(img_h) / block.page_height
    x0, y0, x1, y1 = block.bbox
    left = max(0, int(math.floor(x0 * sx)))
    top = max(0, int(math.floor(y0 * sy)))
    right = min(img_w, int(math.ceil(x1 * sx)))
    bottom = min(img_h, int(math.ceil(y1 * sy)))
    return left, top, right, bottom


def draw_block_highlights(
    page_images: Sequence[Image.Image],
    page_ids: Sequence[int],
    blocks: Sequence[LocalizedBlock],
) -> list[Image.Image]:
    """在页面图副本上为 blocks 画红框；没有候选块的页面原样返回同一对象。

    输入通常是已按 reader 像素预算缩放后的页面，因此红框臂与 images 臂的图像张数、顺序、分辨率
    完全一致，唯一差别是红框像素本身。
    """
    if len(page_images) != len(page_ids):
        raise ValueError(f"page_images 与 page_ids 长度不一致：{len(page_images)} vs {len(page_ids)}")
    by_page: dict[int, list[LocalizedBlock]] = {}
    for b in blocks:
        by_page.setdefault(int(b.page_id), []).append(b)
    out: list[Image.Image] = []
    for img, pid in zip(page_images, page_ids):
        todo = by_page.get(int(pid))
        if not todo:
            out.append(img)
            continue
        canvas = img.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        width = max(HIGHLIGHT_MIN_WIDTH, int(round(canvas.size[0] * HIGHLIGHT_WIDTH_RATIO)))
        for b in todo:
            left, top, right, bottom = scale_bbox_to_image(b, canvas.size)
            if right <= left or bottom <= top:
                continue
            draw.rectangle((left, top, right - 1, bottom - 1), outline=HIGHLIGHT_COLOR, width=width)
        out.append(canvas)
    return out


def reader_scale_for_page(image_size: tuple[int, int], max_pixels: int | None) -> float:
    """reader 对该页施加的等比缩放系数（与 `resize_to_max_pixels` 的规则一致；不放大）。"""
    if max_pixels is None:
        return 1.0
    w, h = image_size
    if w * h <= max_pixels:
        return 1.0
    return (float(max_pixels) / float(w * h)) ** 0.5


def crop_blocks(
    page_images_native: Sequence[Image.Image],
    page_ids: Sequence[int],
    blocks: Sequence[LocalizedBlock],
    *,
    scale: str,
    reader_max_pixels: int | None,
    expand_ratio: float = 0.02,
) -> list[Image.Image]:
    """按 blocks 顺序从原始分辨率页面图裁剪候选区域。

    scale="reader"：裁剪图按该页在 reader 中的缩放比缩小，使其像素密度与 reader 看到的整页一致，
                    于是裁剪臂相对 images 臂只多了"位置"，没有分辨率红利（纯 localization 对照）；
    scale="native"：保留渲染原始分辨率（localization + 分辨率），单张不超过 reader_max_pixels。
    找不到对应页或 bbox 退化的 block 被跳过；返回顺序与 blocks 一致。
    """
    if scale not in CROP_SCALES:
        raise ValueError(f"未知 crop scale: {scale!r}（可选 {CROP_SCALES}）")
    if len(page_images_native) != len(page_ids):
        raise ValueError(f"page_images_native 与 page_ids 长度不一致：{len(page_images_native)} vs {len(page_ids)}")
    page_by_id = {int(pid): img for img, pid in zip(page_images_native, page_ids)}
    crops: list[Image.Image] = []
    for b in blocks:
        page = page_by_id.get(int(b.page_id))
        if page is None:
            continue
        page = page.convert("RGB")
        left, top, right, bottom = scale_bbox_to_image(b, page.size)
        # 与既有裁剪工具一致的少量外扩，避免把字符边缘切掉
        ex = max(2, int(round((right - left) * max(0.0, float(expand_ratio)))))
        ey = max(2, int(round((bottom - top) * max(0.0, float(expand_ratio)))))
        left, top = max(0, left - ex), max(0, top - ey)
        right, bottom = min(page.size[0], right + ex), min(page.size[1], bottom + ey)
        if right <= left or bottom <= top:
            continue
        crop = page.crop((left, top, right, bottom))
        if scale == "reader":
            s = reader_scale_for_page(page.size, reader_max_pixels)
        else:
            s = reader_scale_for_page(crop.size, reader_max_pixels)
        if s < 1.0:
            new_w = max(1, int(crop.size[0] * s))
            new_h = max(1, int(crop.size[1] * s))
            crop = crop.resize((new_w, new_h), resample=Image.BICUBIC)
        crops.append(crop)
    return crops


# --------------------------------------------------------------------------------------
# 提示语
# --------------------------------------------------------------------------------------
def _fmt_pages(page_ids: Sequence[int]) -> str:
    return "、".join(f"第{int(p)}页" for p in page_ids) if page_ids else "无"


def memory_prefix_text(style: str) -> str:
    """文本记忆臂的前缀（不含末尾的作答要求）。keyword 为论文原文，neutral 去掉相关性暗示。"""
    if style not in MEMORY_PREFIX_STYLES:
        raise ValueError(f"未知 memory prefix style: {style!r}（可选 {MEMORY_PREFIX_STYLES}）")
    return _TEXT_PREFIX[style]


def text_memory_user_text(*, base_user_text: str, scratchpad: str, style: str) -> str:
    """文本记忆臂完整 user_text；scratchpad 为空时退化为 base（与 harness 原行为一致）。"""
    if scratchpad.strip() == "":
        return base_user_text
    return f"{memory_prefix_text(style)}{_ANSWER_SUFFIX}\n{scratchpad}\n\n{base_user_text}"


def visual_memory_user_text(
    *,
    mode: str,
    base_user_text: str,
    blocks: Sequence[LocalizedBlock],
    style: str,
) -> str:
    """红框 / 裁剪臂的 user_text：措辞与文本记忆前缀逐句对应，只把"证据条"换成"红框区域 / 区域图"。

    没有可定位 block 时退化为 base_user_text（与文本臂 scratchpad 为空时的处理一致）。
    """
    if mode not in VISUAL_MEMORY_MODES:
        raise ValueError(f"未知视觉记忆模式: {mode!r}（可选 {VISUAL_MEMORY_MODES}）")
    if style not in MEMORY_PREFIX_STYLES:
        raise ValueError(f"未知 memory prefix style: {style!r}（可选 {MEMORY_PREFIX_STYLES}）")
    if not blocks:
        return base_user_text
    if mode == "pages_highlight":
        pages = sorted({int(b.page_id) for b in blocks})
        prefix = _HIGHLIGHT_PREFIX[style].format(pages=_fmt_pages(pages))
    else:
        pages = [int(b.page_id) for b in blocks]
        prefix = _CROPS_PREFIX[style].format(n=len(blocks), pages=_fmt_pages(pages))
    return f"{prefix}{_ANSWER_SUFFIX}\n\n{base_user_text}"
