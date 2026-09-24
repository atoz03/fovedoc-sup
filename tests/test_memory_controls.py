"""memory_controls 单测：选择变体、红框 / 裁剪定位、提示语，全部离线、无模型。"""
from __future__ import annotations

from PIL import Image

from eccv26.utils.memory_controls import (
    HIGHLIGHT_COLOR,
    LocalizedBlock,
    apply_memory_selector,
    crop_blocks,
    lexical_char_budget,
    packed_text_length,
    draw_block_highlights,
    localized_blocks_from_facts,
    memory_prefix_text,
    reader_scale_for_page,
    scale_bbox_to_image,
    stable_random_score,
    text_memory_user_text,
    visual_memory_user_text,
)


def _rows(n_pages: int = 3, per_page: int = 5) -> list[dict]:
    rows = []
    for p in range(1, n_pages + 1):
        for i in range(1, per_page + 1):
            rows.append({
                "page_id": p, "block_id": f"p{p}_b{i}", "order_index": i - 1,
                # 词面分故意与阅读序反相关，便于区分"按分选"与"按序排"
                "lexical_score": float(per_page - i), "text": f"page {p} block {i}",
                "bbox": (10.0 * i, 20.0 * i, 10.0 * i + 50.0, 20.0 * i + 15.0),
                "page_width": 200.0, "page_height": 300.0,
            })
    return rows


def test_random_selector_is_reproducible_and_query_independent() -> None:
    a = apply_memory_selector(_rows(), selector="random", topk=4, seed_key="doc\x1fq")
    b = apply_memory_selector(_rows(), selector="random", topk=4, seed_key="doc\x1fq")
    assert [r["block_id"] for r in a] == [r["block_id"] for r in b]
    assert len(a) == 4
    # 分数不来自 lexical_score：同一 seed 下 lexical 最高的块不应恒在首位
    scores = {r["block_id"]: stable_random_score("doc\x1fq", r) for r in _rows()}
    assert all(0.0 <= v < 1.0 for v in scores.values())
    top_by_lex = sorted(_rows(), key=lambda r: -r["lexical_score"])[:4]
    assert [r["block_id"] for r in a] != [r["block_id"] for r in top_by_lex]


def test_random_selector_changes_with_seed() -> None:
    a = apply_memory_selector(_rows(), selector="random", topk=4, seed_key="doc\x1fq1")
    b = apply_memory_selector(_rows(), selector="random", topk=4, seed_key="doc\x1fq2")
    assert {r["block_id"] for r in a} != {r["block_id"] for r in b}


def test_all_selector_keeps_every_row_in_reading_order_after_harness_sort() -> None:
    rows = _rows()
    selected = apply_memory_selector(rows, selector="all", topk=16, seed_key="x")
    assert len(selected) == len(rows)
    page_order = [1, 2, 3]
    # 复现 harness 的排序规则：(页序, -_sel_score)
    selected.sort(key=lambda r: (page_order.index(int(r["page_id"])), -r["_sel_score"]))
    assert [r["block_id"] for r in selected] == [r["block_id"] for r in rows]


def test_lexical_is_rejected_here() -> None:
    try:
        apply_memory_selector(_rows(), selector="lexical", topk=4, seed_key="x")
    except ValueError:
        return
    raise AssertionError("lexical 应由 harness 原路径处理")


def test_localized_blocks_skip_rows_without_geometry() -> None:
    facts = [
        {"page_id": 1, "bbox": [0, 0, 10, 10], "page_width": 100, "page_height": 100},
        {"page_id": 1, "bbox": None, "page_width": 100, "page_height": 100},
        {"page_id": 2, "bbox": [5, 5, 5, 20], "page_width": 100, "page_height": 100},  # 退化 bbox
    ]
    blocks = localized_blocks_from_facts(facts)
    assert len(blocks) == 1 and blocks[0].page_id == 1


def test_highlight_draws_only_on_pages_with_blocks_and_keeps_geometry() -> None:
    pages = [Image.new("RGB", (100, 150), "white") for _ in range(3)]
    block = LocalizedBlock(page_id=2, bbox=(20.0, 30.0, 60.0, 90.0), page_width=200.0, page_height=300.0)
    out = draw_block_highlights(pages, [1, 2, 3], [block])
    assert len(out) == 3 and out[0] is pages[0] and out[2] is pages[2]
    assert out[1].size == pages[1].size
    # bbox 等比映射到 100×150：x 10..30，y 15..45；左边框像素应为红色，框内中心保持白色
    left, top, right, bottom = scale_bbox_to_image(block, out[1].size)
    assert (left, top, right, bottom) == (10, 15, 30, 45)
    assert out[1].getpixel((left, (top + bottom) // 2)) == HIGHLIGHT_COLOR
    assert out[1].getpixel(((left + right) // 2, (top + bottom) // 2)) == (255, 255, 255)
    assert out[0].getpixel((10, 30)) == (255, 255, 255)


def test_crop_reader_scale_matches_page_downscale() -> None:
    # 原始页 400×600（24 万像素）；reader 预算 6 万像素 → 缩放系数 0.5
    page = Image.new("RGB", (400, 600), "white")
    for x in range(100, 200):
        for y in range(150, 210):
            page.putpixel((x, y), (0, 0, 0))
    block = LocalizedBlock(page_id=7, bbox=(100.0, 150.0, 200.0, 210.0), page_width=400.0, page_height=600.0)
    assert reader_scale_for_page(page.size, 60_000) == 0.5
    crops = crop_blocks([page], [7], [block], scale="reader", reader_max_pixels=60_000)
    assert len(crops) == 1
    w, h = crops[0].size
    # 外扩 2% 后约 104×64，再乘 0.5
    assert 48 <= w <= 56 and 28 <= h <= 36
    # 裁剪图中心应为黑色区域
    assert crops[0].getpixel((w // 2, h // 2)) == (0, 0, 0)
    native = crop_blocks([page], [7], [block], scale="native", reader_max_pixels=60_000)
    assert native[0].size[0] >= 100 and native[0].size[1] >= 60


def test_crop_skips_unknown_pages_and_keeps_order() -> None:
    page = Image.new("RGB", (100, 100), "white")
    b1 = LocalizedBlock(page_id=1, bbox=(0.0, 0.0, 10.0, 10.0), page_width=100.0, page_height=100.0)
    b_missing = LocalizedBlock(page_id=9, bbox=(0.0, 0.0, 10.0, 10.0), page_width=100.0, page_height=100.0)
    b2 = LocalizedBlock(page_id=1, bbox=(50.0, 50.0, 90.0, 90.0), page_width=100.0, page_height=100.0)
    crops = crop_blocks([page], [1], [b1, b_missing, b2], scale="native", reader_max_pixels=None)
    assert len(crops) == 2 and crops[1].size[0] > crops[0].size[0]


def test_text_prefix_keyword_matches_paper_wording_and_neutral_drops_relevance_hint() -> None:
    kw = memory_prefix_text("keyword")
    ne = memory_prefix_text("neutral")
    assert "按关键词" in kw and "候选证据" in kw
    assert "按关键词" not in ne and "候选证据" not in ne
    full = text_memory_user_text(base_user_text="Q?\nAnswer:", scratchpad="[第3页] foo", style="keyword")
    assert full == f"{kw}，只回答问题所需的最短答案。\n[第3页] foo\n\nQ?\nAnswer:"
    assert text_memory_user_text(base_user_text="Q?", scratchpad="  ", style="neutral") == "Q?"


def test_visual_prompts_name_pages_and_counts() -> None:
    blocks = [
        LocalizedBlock(page_id=5, bbox=(0.0, 0.0, 1.0, 1.0), page_width=10.0, page_height=10.0),
        LocalizedBlock(page_id=2, bbox=(0.0, 0.0, 1.0, 1.0), page_width=10.0, page_height=10.0),
        LocalizedBlock(page_id=5, bbox=(1.0, 1.0, 2.0, 2.0), page_width=10.0, page_height=10.0),
    ]
    hl = visual_memory_user_text(mode="pages_highlight", base_user_text="Q", blocks=blocks, style="neutral")
    assert "第2页、第5页" in hl and hl.endswith("\n\nQ") and "按关键词" not in hl
    cr = visual_memory_user_text(mode="pages_crops", base_user_text="Q", blocks=blocks, style="keyword")
    assert "3 张" in cr and "第5页、第2页、第5页" in cr and "按关键词" in cr
    assert visual_memory_user_text(mode="pages_crops", base_user_text="Q", blocks=[], style="keyword") == "Q"


def test_random_chars_matches_lexical_character_budget() -> None:
    rows = _rows()
    for r in rows:
        r["text"] = "x" * (5 * int(r["block_id"].split("_b")[1]))  # 5..25 字符
    budget = lexical_char_budget(rows, topk=4, max_chars_per_block=300)
    assert budget == sum(packed_text_length(r, 300) for r in sorted(rows, key=lambda r: -r["lexical_score"])[:4])
    picked = apply_memory_selector(rows, selector="random_chars", topk=4, seed_key="s", char_budget=budget,
                                   max_chars_per_block=300)
    total = sum(packed_text_length(r, 300) for r in picked)
    assert total >= budget
    assert total - packed_text_length(picked[-1], 300) < budget  # 去掉最后一块就不够
    try:
        apply_memory_selector(rows, selector="random_chars", topk=4, seed_key="s")
    except ValueError:
        return
    raise AssertionError("缺 char_budget 应报错")


def test_packed_text_length_follows_harness_truncation() -> None:
    assert packed_text_length({"text": "abc"}, 300) == 3
    assert packed_text_length({"text": "a" * 400}, 300) == 303
    assert packed_text_length({"text": "a" * 400}, 0) == 400
    assert packed_text_length({"text": "a b\nc"}, 300) == 5
