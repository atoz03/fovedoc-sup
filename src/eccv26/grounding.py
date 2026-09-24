from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

_ALLOWED_JUDGE_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_WORD_HYPHEN_RE = re.compile(r"(?<=\w)\s*[-‐‑‒–—]\s*(?=\w)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_grounding_training_phase(training_phase: str | None) -> str:
    phase = str(training_phase or "final").strip().lower()
    if phase in {"", "grounded_json", "default"}:
        return "final"
    if phase not in {"plan", "final", "pref"}:
        raise ValueError(f"不支持的 grounded 训练阶段: {training_phase}")
    return phase


def should_use_grounded_output(*, source_kind: str | None, structured_output_mode: str | None) -> bool:
    if str(structured_output_mode or "").strip().lower() != "grounded_json":
        return False
    return str(source_kind or "").strip().lower() != "throughput"


def apply_grounded_output_format(
    *,
    system_text: str,
    user_text: str,
    training_phase: str | None = None,
) -> tuple[str, str]:
    phase = normalize_grounding_training_phase(training_phase)
    grounded_system = (
        f"{system_text.strip()}\n\n"
        "你必须输出一个严格合法的 JSON 对象，不要输出 Markdown，不要输出解释，不要输出额外前后缀。"
        " `claims` 是数组，每项至少包含 `claim_id` 和 `text`。"
        " `evidence` 是数组，每项至少包含 `evidence_id` 和 `supports`。"
        " `supports` 是被该证据支持的 claim_id 列表。"
    )
    if phase == "plan":
        grounded_system += " 当前阶段是 evidence-plan，只学习 claim 与 evidence_id 对齐，不输出最终答案。"
        grounded_user = (
            f"{user_text.strip()}\n\n"
            "请严格输出如下 JSON 结构：\n"
            "{\n"
            '  "claims": [{"claim_id": "c1", "text": "..."}],\n'
            '  "evidence": [{"evidence_id": "e1", "supports": ["c1"]}]\n'
            "}\n"
            "要求：\n"
            "1. 只输出 claim 与 evidence_id 对齐，不输出最终答案；\n"
            "2. `evidence_id` 必须来自候选证据集合；\n"
            "3. 如果无法确定，就返回空数组，但 JSON 结构仍必须完整。\n"
        )
    else:
        grounded_system += (
            " 当前阶段允许输出最终答案。"
            " 若无法稳定给出 `page_id/block_id/quote`，优先保证 `evidence_id` 与 `supports` 正确；"
            " 评测侧会按 `evidence_id` 做确定性 citation 映射。"
            " 输出必须简洁，禁止重复同一条 claim 或 evidence。"
            " `result` 必须优先写成直接回答问题的最短答案。"
            " 若答案是编号、日期、标题、专有名词、数值、百分比、面积、单位短语或其它短字符串，"
            " 必须尽量从证据中原样复制，保留大小写、连字符、空格和标点，不要自行改写。"
        )
        grounded_user = (
            f"{user_text.strip()}\n\n"
            "请严格输出如下 JSON 结构：\n"
            "{\n"
            '  "result": "... 或结构化对象 ...",\n'
            '  "claims": [{"claim_id": "c1", "text": "..."}],\n'
            '  "evidence": [\n'
            '    {"evidence_id": "e1", "supports": ["c1"]}\n'
            "  ]\n"
            "}\n"
            "要求：\n"
            "1. `evidence` 至少给出 `evidence_id` 与 `supports`；\n"
            "2. 如果已经能回答问题，就不要把 `claims` 和 `evidence` 都输出为空数组；\n"
            "3. 若你能稳定给出 citation，也可额外补充 `page_id/block_id/quote`；\n"
            "4. `result` 优先输出直接回答问题的最短答案；若是短字符串，尽量原样复制证据中的 span；\n"
            "5. 如果无法确定，就返回空数组，但 JSON 结构仍必须完整。\n"
        )
    return grounded_system, grounded_user


def build_grounded_output_prefill(*, training_phase: str | None = None) -> str:
    phase = normalize_grounding_training_phase(training_phase)
    if phase == "plan":
        return '{"claims":'
    return '{"result":'


def _page_id_to_int(value: Any) -> int | None:
    try:
        page_id = int(value)
    except (TypeError, ValueError):
        return None
    return page_id if page_id > 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00ad", "")
    text = _WORD_HYPHEN_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_compact_text(value: Any) -> str:
    text = _normalize_text(value).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _text_contains_quote(container: Any, quote: Any) -> bool:
    normalized_container = _normalize_text(container).lower()
    normalized_quote = _normalize_text(quote).lower()
    if normalized_container == "" or normalized_quote == "":
        return False
    if normalized_quote in normalized_container:
        return True
    compact_container = _normalize_compact_text(container)
    compact_quote = _normalize_compact_text(quote)
    return compact_quote != "" and compact_quote in compact_container


def _snippet_around_quote(container: Any, quote: Any, *, fallback: int = 220) -> str:
    text = _normalize_text(container)
    normalized_quote = _normalize_text(quote)
    if text == "":
        return ""
    if normalized_quote == "":
        return text[:fallback]
    lower_text = text.lower()
    lower_quote = normalized_quote.lower()
    index = lower_text.find(lower_quote)
    if index < 0:
        compact_text = _normalize_compact_text(text)
        compact_quote = _normalize_compact_text(normalized_quote)
        if compact_quote == "" or compact_quote not in compact_text:
            return text[:fallback]
        return text[:fallback]
    start = max(0, index - fallback // 3)
    end = min(len(text), index + len(normalized_quote) + fallback // 2)
    return text[start:end]


def _extract_json_candidate(text: str) -> str | None:
    stripped = str(text or "").strip()
    if stripped == "":
        return None
    fence_match = _JSON_FENCE_RE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    depth = 0
    start = None
    for index, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return stripped[start : index + 1]
    return None


def _extract_json_prefix(text: str) -> str | None:
    stripped = str(text or "").strip()
    if stripped == "":
        return None
    fence_match = _JSON_FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start = stripped.find("{")
    if start < 0:
        return None
    return stripped[start:].strip()


def _repair_json_candidate(candidate: str) -> str:
    buf: list[str] = []
    closers: list[str] = []
    in_string = False
    escaped = False
    for ch in str(candidate or ""):
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if in_string:
            if ch == "\\":
                buf.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
            buf.append(ch)
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
            continue
        if ch == "{":
            closers.append("}")
            buf.append(ch)
            continue
        if ch == "[":
            closers.append("]")
            buf.append(ch)
            continue
        if ch in "}]":
            if closers and ch == closers[-1]:
                closers.pop()
                buf.append(ch)
            continue
        buf.append(ch)

    repaired = "".join(buf).rstrip()
    if escaped and repaired.endswith("\\"):
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    while repaired.endswith(",") or repaired.endswith(":"):
        repaired = repaired[:-1].rstrip()
    while closers:
        repaired += closers.pop()
    return re.sub(r",\s*([}\]])", r"\1", repaired).strip()


def _load_relaxed_json_payload(pred_text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = _extract_json_prefix(pred_text)
    if candidate is None:
        return None, None
    for cut in range(len(candidate), 1, -1):
        repaired = _repair_json_candidate(candidate[:cut])
        if repaired == "":
            continue
        try:
            payload = json.loads(repaired)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, repaired
    return None, None


def _stringify_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_structured_prediction(pred_text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(pred_text)
    if candidate is None:
        payload, candidate = _load_relaxed_json_payload(pred_text)
        if payload is None:
            return {
                "valid": False,
                "error": "missing_json_object",
                "result": None,
                "result_text": "",
                "claims": [],
                "evidence": [],
                "raw_text": pred_text,
                "normalized_text": None,
            }
    else:
        try:
            payload = json.loads(candidate)
        except Exception:
            payload, candidate = _load_relaxed_json_payload(pred_text)
            if payload is None:
                return {
                    "valid": False,
                    "error": "json_decode_error",
                    "result": None,
                    "result_text": "",
                    "claims": [],
                    "evidence": [],
                    "raw_text": pred_text,
                    "normalized_text": None,
                }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "error": "json_root_not_object",
            "result": None,
            "result_text": "",
            "claims": [],
            "evidence": [],
            "raw_text": pred_text,
            "normalized_text": None,
        }

    normalized_claims: list[dict[str, Any]] = []
    raw_claims = payload.get("claims")
    if isinstance(raw_claims, list):
        for index, raw_claim in enumerate(raw_claims, start=1):
            if isinstance(raw_claim, dict):
                claim_id = str(raw_claim.get("claim_id") or f"c{index}").strip() or f"c{index}"
                text = _normalize_text(raw_claim.get("text"))
            else:
                claim_id = f"c{index}"
                text = _normalize_text(raw_claim)
            if text == "":
                continue
            normalized_claims.append({"claim_id": claim_id, "text": text})

    normalized_evidence: list[dict[str, Any]] = []
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list):
        raw_evidence = payload.get("evidence_refs")
    if isinstance(raw_evidence, list):
        for index, raw_item in enumerate(raw_evidence, start=1):
            item = raw_item if isinstance(raw_item, dict) else {}
            evidence_id = str(item.get("evidence_id") or f"e{index}").strip() or f"e{index}"
            page_id = _page_id_to_int(item.get("page_id"))
            block_id = _normalize_text(item.get("block_id")) or None
            quote = _normalize_text(item.get("quote"))
            raw_supports = item.get("supports")
            supports: list[str] = []
            if isinstance(raw_supports, list):
                for value in raw_supports:
                    support_id = _normalize_text(value)
                    if support_id != "":
                        supports.append(support_id)
            normalized_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "page_id": page_id,
                    "block_id": block_id,
                    "quote": quote,
                    "supports": supports,
                }
            )

    result = payload.get("result")
    normalized_payload: dict[str, Any] = {
        "result": result if result is not None else "",
        "claims": normalized_claims,
        "evidence": normalized_evidence,
    }
    normalized_text = json.dumps(normalized_payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "valid": True,
        "error": None,
        "result": result,
        "result_text": _stringify_json_value(result),
        "claims": normalized_claims,
        "evidence": normalized_evidence,
        "raw_text": pred_text,
        "normalized_text": normalized_text,
    }


def normalize_structured_prediction_text(pred_text: str) -> str:
    parsed = parse_structured_prediction(pred_text)
    normalized_text = parsed.get("normalized_text")
    if not parsed.get("valid") or not isinstance(normalized_text, str) or normalized_text.strip() == "":
        return pred_text
    return normalized_text


def replace_structured_prediction_result(pred_text: str, *, new_result: Any) -> str:
    parsed = parse_structured_prediction(pred_text)
    if not parsed.get("valid"):
        return pred_text
    payload = {
        "result": new_result,
        "claims": list(parsed.get("claims") or []),
        "evidence": list(parsed.get("evidence") or []),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=4096)
def _load_document_bundle(document_json_path: str) -> dict[str, Any]:
    path = Path(document_json_path)
    if not path.exists():
        return {"pages": {}, "page_texts": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"pages": {}, "page_texts": {}}
    raw_pages = data.get("pages")
    if not isinstance(raw_pages, list):
        return {"pages": {}, "page_texts": {}}

    pages: dict[int, dict[str, Any]] = {}
    page_texts: dict[int, str] = {}
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page_id = _page_id_to_int(raw_page.get("page_id"))
        if page_id is None:
            continue
        page_text = _normalize_text(raw_page.get("page_text"))
        page_texts[page_id] = page_text
        raw_blocks = raw_page.get("blocks")
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        blocks: dict[str, dict[str, Any]] = {}
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            block_id = _normalize_text(raw_block.get("block_id"))
            if block_id == "":
                continue
            block_text = _normalize_text(raw_block.get("text"))
            blocks[block_id] = {
                "block_id": block_id,
                "page_id": page_id,
                "text": block_text,
            }
        pages[page_id] = {"page_id": page_id, "page_text": page_text, "blocks": blocks}
    return {"pages": pages, "page_texts": page_texts}


def _resolve_document_json_path(meta: dict[str, Any], benchmark_root: str | None) -> str | None:
    raw_path = meta.get("document_json")
    if isinstance(raw_path, str) and raw_path.strip() != "":
        path = Path(raw_path)
        if not path.is_absolute() and benchmark_root is not None:
            path = Path(str(benchmark_root)) / str(path)
        return str(path)
    document_id = _normalize_text(meta.get("document_id"))
    if benchmark_root is None or document_id == "":
        return None
    return str(Path(str(benchmark_root)) / "data" / "documents" / f"{document_id}.json")


def build_gold_grounding_package(
    *,
    meta: dict[str, Any],
    answers: list[str] | None,
    benchmark_root: str | None,
) -> dict[str, Any]:
    grounding_meta = meta.get("grounding")
    if not isinstance(grounding_meta, dict):
        grounding_meta = {}

    gold_result: Any = grounding_meta.get("gold_result")
    if gold_result is None:
        if answers is not None and len(answers) == 1:
            gold_result = answers[0]
        elif answers is not None and len(answers) > 1:
            gold_result = list(answers)
        elif meta.get("ground_truth") is not None:
            gold_result = meta.get("ground_truth")

    gold_atomic_facts: list[dict[str, Any]] = []
    raw_gold_facts = grounding_meta.get("gold_atomic_facts")
    if isinstance(raw_gold_facts, list):
        for index, raw_fact in enumerate(raw_gold_facts, start=1):
            fact = raw_fact if isinstance(raw_fact, dict) else {}
            fact_id = str(fact.get("fact_id") or f"f{index}").strip() or f"f{index}"
            text = _normalize_text(fact.get("text") or fact.get("fact") or raw_fact)
            if text == "":
                continue
            gold_atomic_facts.append({"fact_id": fact_id, "text": text})

    gold_evidence_units: list[dict[str, Any]] = []
    raw_gold_evidence = grounding_meta.get("gold_evidence_units")
    if not isinstance(raw_gold_evidence, list):
        raw_gold_evidence = meta.get("evidence")
    if not isinstance(raw_gold_evidence, list):
        raw_gold_evidence = []
    for index, raw_item in enumerate(raw_gold_evidence, start=1):
        item = raw_item if isinstance(raw_item, dict) else {}
        fact_id = str(item.get("fact_id") or f"f{index}").strip() or f"f{index}"
        evidence_id = str(item.get("evidence_id") or f"e{index}").strip() or f"e{index}"
        page_id = _page_id_to_int(item.get("page_id"))
        block_id = _normalize_text(item.get("block_id")) or None
        text_excerpt = _normalize_text(item.get("text_excerpt") or item.get("quote") or item.get("fact"))
        text_excerpt_raw = str(item.get("text_excerpt") or item.get("quote") or item.get("fact") or "").strip()
        gold_evidence_units.append(
            {
                "evidence_id": evidence_id,
                "fact_id": fact_id,
                "page_id": page_id,
                "block_id": block_id,
                "text_excerpt": text_excerpt,
                "text_excerpt_raw": text_excerpt_raw,
                "supports": [fact_id],
            }
        )
        if not any(fact["fact_id"] == fact_id for fact in gold_atomic_facts):
            fact_text = text_excerpt or f"evidence fact {index}"
            gold_atomic_facts.append({"fact_id": fact_id, "text": fact_text})

    document_json_path = _resolve_document_json_path(meta, benchmark_root)
    return {
        "available": len(gold_evidence_units) > 0,
        "gold_result": gold_result,
        "gold_result_text": _stringify_json_value(gold_result),
        "gold_atomic_facts": gold_atomic_facts,
        "gold_evidence_units": gold_evidence_units,
        "document_json_path": document_json_path,
    }


def canonicalize_structured_evidence(
    *,
    evidence_items: list[dict[str, Any]],
    gold_package: dict[str, Any],
) -> list[dict[str, Any]]:
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_unit in enumerate(gold_package.get("gold_evidence_units") or [], start=1):
        if not isinstance(raw_unit, dict):
            continue
        evidence_id = str(raw_unit.get("evidence_id") or f"e{index}").strip() or f"e{index}"
        canonical_by_id[evidence_id] = {
            "evidence_id": evidence_id,
            "page_id": _page_id_to_int(raw_unit.get("page_id")),
            "block_id": _normalize_text(raw_unit.get("block_id")) or None,
            "quote": _normalize_text(raw_unit.get("text_excerpt")),
            "supports": list(raw_unit.get("supports") or []),
        }

    canonicalized: list[dict[str, Any]] = []
    for index, raw_item in enumerate(evidence_items, start=1):
        item = raw_item if isinstance(raw_item, dict) else {}
        evidence_id = str(item.get("evidence_id") or f"e{index}").strip() or f"e{index}"
        raw_supports = item.get("supports")
        supports: list[str] = []
        if isinstance(raw_supports, list):
            for value in raw_supports:
                support_id = _normalize_text(value)
                if support_id != "":
                    supports.append(support_id)

        canonical = canonical_by_id.get(evidence_id) or {}
        page_id = _page_id_to_int(item.get("page_id"))
        if page_id is None:
            page_id = _page_id_to_int(canonical.get("page_id"))
        block_id = _normalize_text(item.get("block_id")) or _normalize_text(canonical.get("block_id")) or None
        quote = _normalize_text(item.get("quote"))
        if quote == "":
            quote = _normalize_text(canonical.get("quote"))
        if not supports:
            canonical_supports = canonical.get("supports")
            if isinstance(canonical_supports, list):
                supports = [_normalize_text(value) for value in canonical_supports if _normalize_text(value) != ""]

        canonicalized.append(
            {
                "evidence_id": evidence_id,
                "page_id": page_id,
                "block_id": block_id,
                "quote": quote,
                "supports": supports,
            }
        )
    return canonicalized


def _validate_one_evidence(
    *,
    evidence_item: dict[str, Any],
    gold_package: dict[str, Any],
) -> dict[str, Any]:
    page_id = _page_id_to_int(evidence_item.get("page_id"))
    block_id = _normalize_text(evidence_item.get("block_id")) or None
    quote = _normalize_text(evidence_item.get("quote"))
    evidence_id = str(evidence_item.get("evidence_id") or "").strip() or "unknown"
    supports = evidence_item.get("supports")
    if not isinstance(supports, list):
        supports = []

    document_bundle = {"pages": {}, "page_texts": {}}
    document_json_path = gold_package.get("document_json_path")
    if isinstance(document_json_path, str) and document_json_path.strip() != "":
        document_bundle = _load_document_bundle(document_json_path)

    pages = document_bundle.get("pages", {})
    page_data = pages.get(page_id) if page_id is not None else None
    exact_block_text = None
    if isinstance(page_data, dict) and block_id is not None:
        block_data = page_data.get("blocks", {}).get(block_id)
        if isinstance(block_data, dict):
            exact_block_text = block_data.get("text")

    if page_id is not None and block_id is not None and _text_contains_quote(exact_block_text, quote):
        return {
            "evidence_id": evidence_id,
            "page_id": page_id,
            "block_id": block_id,
            "quote": quote,
            "supports": supports,
            "validation_score": 1.0,
            "validation_label": "exact_block_quote",
            "matched_page_id": page_id,
            "matched_block_id": block_id,
            "matched_snippet": _snippet_around_quote(exact_block_text, quote),
            "matched_snippet_raw": str(exact_block_text or "").strip(),
        }

    if page_id is not None and isinstance(page_data, dict):
        page_text = page_data.get("page_text") or ""
        if _text_contains_quote(page_text, quote):
            return {
                "evidence_id": evidence_id,
                "page_id": page_id,
                "block_id": block_id,
                "quote": quote,
                "supports": supports,
                "validation_score": 0.5,
                "validation_label": "page_quote_only",
                "matched_page_id": page_id,
                "matched_block_id": None,
                "matched_snippet": _snippet_around_quote(page_text, quote),
                "matched_snippet_raw": str(page_text or "").strip(),
            }
        for candidate_block_id, block_data in page_data.get("blocks", {}).items():
            if _text_contains_quote(block_data.get("text"), quote):
                return {
                    "evidence_id": evidence_id,
                    "page_id": page_id,
                    "block_id": block_id,
                    "quote": quote,
                    "supports": supports,
                    "validation_score": 0.5,
                    "validation_label": "page_quote_only",
                    "matched_page_id": page_id,
                    "matched_block_id": candidate_block_id,
                    "matched_snippet": _snippet_around_quote(block_data.get("text"), quote),
                    "matched_snippet_raw": str(block_data.get("text") or "").strip(),
                }

    for unit in gold_package.get("gold_evidence_units", []):
        if not isinstance(unit, dict):
            continue
        unit_page_id = _page_id_to_int(unit.get("page_id"))
        unit_block_id = _normalize_text(unit.get("block_id")) or None
        text_excerpt = _normalize_text(unit.get("text_excerpt"))
        if page_id is not None and block_id is not None:
            if unit_page_id == page_id and unit_block_id == block_id and _text_contains_quote(text_excerpt, quote):
                return {
                    "evidence_id": evidence_id,
                    "page_id": page_id,
                    "block_id": block_id,
                    "quote": quote,
                    "supports": supports,
                    "validation_score": 1.0,
                    "validation_label": "exact_block_quote",
                    "matched_page_id": page_id,
                    "matched_block_id": block_id,
                    "matched_snippet": text_excerpt,
                    "matched_snippet_raw": str(unit.get("text_excerpt_raw") or text_excerpt).strip(),
                }
        if page_id is not None and unit_page_id == page_id and _text_contains_quote(text_excerpt, quote):
            return {
                "evidence_id": evidence_id,
                "page_id": page_id,
                "block_id": block_id,
                "quote": quote,
                "supports": supports,
                "validation_score": 0.5,
                "validation_label": "page_quote_only",
                "matched_page_id": page_id,
                "matched_block_id": unit_block_id,
                "matched_snippet": text_excerpt,
                "matched_snippet_raw": str(unit.get("text_excerpt_raw") or text_excerpt).strip(),
            }

    return {
        "evidence_id": evidence_id,
        "page_id": page_id,
        "block_id": block_id,
        "quote": quote,
        "supports": supports,
        "validation_score": 0.0,
        "validation_label": "invalid",
        "matched_page_id": None,
        "matched_block_id": None,
        "matched_snippet": "",
        "matched_snippet_raw": "",
    }


def compute_grounding_hard_eval(
    *,
    meta: dict[str, Any],
    answers: list[str] | None,
    pred_text: str,
    benchmark_root: str | None,
) -> dict[str, Any]:
    gold_package = build_gold_grounding_package(meta=meta, answers=answers, benchmark_root=benchmark_root)
    parsed_prediction = parse_structured_prediction(pred_text)
    available = bool(gold_package.get("available"))
    fallback_result_text = _normalize_text(pred_text)
    pred_result: Any = parsed_prediction.get("result")
    result_text = _normalize_text(parsed_prediction.get("result_text"))
    if result_text == "":
        result_text = fallback_result_text
    if pred_result is None and fallback_result_text != "":
        pred_result = fallback_result_text

    pred_evidence = canonicalize_structured_evidence(
        evidence_items=list(parsed_prediction.get("evidence") or []),
        gold_package=gold_package,
    )
    verified_evidence: list[dict[str, Any]] = []
    citation_score: float | None = None
    exact_count = 0
    page_only_count = 0
    invalid_count = 0

    if available:
        for evidence_item in pred_evidence:
            verified = _validate_one_evidence(evidence_item=evidence_item, gold_package=gold_package)
            verified_evidence.append(verified)
            label = verified.get("validation_label")
            if label == "exact_block_quote":
                exact_count += 1
            elif label == "page_quote_only":
                page_only_count += 1
            else:
                invalid_count += 1
        if verified_evidence:
            citation_score = sum(float(item["validation_score"]) for item in verified_evidence) / len(verified_evidence)
        else:
            citation_score = 0.0

    evidence_count = len(verified_evidence)
    return {
        "available": available,
        "structured_output_valid": bool(parsed_prediction.get("valid")),
        "parse_error": parsed_prediction.get("error"),
        "raw_prediction_text": pred_text,
        "result_text": result_text,
        "pred_result": pred_result,
        "pred_claims": list(parsed_prediction.get("claims", [])),
        "pred_evidence": pred_evidence,
        "gold_package": gold_package,
        "verified_evidence": verified_evidence,
        "citation_score": citation_score,
        "exact_citation_rate": None if evidence_count == 0 else exact_count / evidence_count,
        "page_only_citation_rate": None if evidence_count == 0 else page_only_count / evidence_count,
        "invalid_citation_rate": None if evidence_count == 0 else invalid_count / evidence_count,
        "invalid_evidence_summary": {
            "num_pred_evidence": evidence_count,
            "exact_block_quote": exact_count,
            "page_quote_only": page_only_count,
            "invalid": invalid_count,
        },
    }


def build_grounding_judge_messages(
    *,
    question: str | None,
    context: str | None,
    grounding_hard_eval: dict[str, Any],
) -> tuple[str, str]:
    gold_package = grounding_hard_eval.get("gold_package") or {}
    pred_claims = grounding_hard_eval.get("pred_claims") or []
    verified_evidence = grounding_hard_eval.get("verified_evidence") or []
    invalid_summary = grounding_hard_eval.get("invalid_evidence_summary") or {}
    required_output_schema = {
        "result_score": 0.0,
        "gold_fact_scores": [{"fact_id": "f1", "score": 0.0, "reason": "..."}],
        "claim_scores": [{"claim_id": "c1", "score": 0.0, "reason": "..."}],
        "summary": "...",
    }
    scoring_rules = {
        "result_score": "最终结果与 gold result 的语义一致程度；允许部分正确。",
        "gold_fact_scores": "每条 gold fact 被 verified evidence 支撑的程度。",
        "claim_scores": "每条 pred claim 被 verified evidence 支撑的程度。",
    }
    system_payload = {
        "role": "strict long-document grounded benchmark judge",
        "instructions": [
            "只能基于给定的 gold result、gold facts、以及已经通过硬校验整理后的证据包打分。",
            "不要使用外部常识，不要脑补文档中未提供的信息。",
            "所有分数只能取 0、0.25、0.5、0.75、1.0 五档。",
            "必须输出严格 JSON，对每个 gold fact 和每个 pred claim 分别打分。",
        ],
        "required_output_schema": required_output_schema,
        "scoring_rules": scoring_rules,
    }
    system_text = (
        "你是一个严格的 long-document grounded benchmark judge。\n"
        "下面是固定评分规则与输出要求，你必须始终遵守：\n"
        f"{json.dumps(system_payload, ensure_ascii=False, indent=2)}"
    )
    user_payload = {
        "query": question,
        "context": context,
        "gold_result": gold_package.get("gold_result"),
        "gold_result_text": gold_package.get("gold_result_text"),
        "gold_atomic_facts": gold_package.get("gold_atomic_facts"),
        "pred_result": grounding_hard_eval.get("pred_result"),
        "pred_result_text": grounding_hard_eval.get("result_text"),
        "raw_prediction_text": grounding_hard_eval.get("raw_prediction_text"),
        "pred_claims": pred_claims,
        "verified_evidence": verified_evidence,
        "invalid_evidence_summary": invalid_summary,
    }
    user_text = (
        "下面是当前样本的评分输入数据（JSON）：\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return system_text, user_text


def _snap_judge_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(_ALLOWED_JUDGE_SCORES, key=lambda candidate: abs(candidate - score))


def parse_grounding_judge_response(
    judge_text: str,
    *,
    expected_fact_ids: list[str],
    expected_claim_ids: list[str],
) -> dict[str, Any]:
    candidate = _extract_json_candidate(judge_text)
    if candidate is None:
        return {
            "valid": False,
            "error": "missing_json_object",
            "result_score": 0.0,
            "fact_coverage_score": 0.0,
            "claim_grounding_score": 0.0,
            "gold_fact_scores": [],
            "claim_scores": [],
            "summary": "",
        }
    try:
        payload = json.loads(candidate)
    except Exception as exc:
        return {
            "valid": False,
            "error": f"json_decode_error:{type(exc).__name__}",
            "result_score": 0.0,
            "fact_coverage_score": 0.0,
            "claim_grounding_score": 0.0,
            "gold_fact_scores": [],
            "claim_scores": [],
            "summary": "",
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "error": "json_root_not_object",
            "result_score": 0.0,
            "fact_coverage_score": 0.0,
            "claim_grounding_score": 0.0,
            "gold_fact_scores": [],
            "claim_scores": [],
            "summary": "",
        }

    result_score = _snap_judge_score(payload.get("result_score"))

    fact_score_map: dict[str, dict[str, Any]] = {}
    raw_fact_scores = payload.get("gold_fact_scores")
    if isinstance(raw_fact_scores, list):
        for raw_item in raw_fact_scores:
            if not isinstance(raw_item, dict):
                continue
            fact_id = _normalize_text(raw_item.get("fact_id"))
            if fact_id == "":
                continue
            fact_score_map[fact_id] = {
                "fact_id": fact_id,
                "score": _snap_judge_score(raw_item.get("score")),
                "reason": _normalize_text(raw_item.get("reason")),
            }
    normalized_fact_scores: list[dict[str, Any]] = []
    for fact_id in expected_fact_ids:
        normalized_fact_scores.append(
            fact_score_map.get(
                fact_id,
                {"fact_id": fact_id, "score": 0.0, "reason": "judge_missing_fact_score"},
            )
        )

    claim_score_map: dict[str, dict[str, Any]] = {}
    raw_claim_scores = payload.get("claim_scores")
    if isinstance(raw_claim_scores, list):
        for raw_item in raw_claim_scores:
            if not isinstance(raw_item, dict):
                continue
            claim_id = _normalize_text(raw_item.get("claim_id"))
            if claim_id == "":
                continue
            claim_score_map[claim_id] = {
                "claim_id": claim_id,
                "score": _snap_judge_score(raw_item.get("score")),
                "reason": _normalize_text(raw_item.get("reason")),
            }
    normalized_claim_scores: list[dict[str, Any]] = []
    for claim_id in expected_claim_ids:
        normalized_claim_scores.append(
            claim_score_map.get(
                claim_id,
                {"claim_id": claim_id, "score": 0.0, "reason": "judge_missing_claim_score"},
            )
        )

    fact_coverage_score = (
        0.0
        if len(normalized_fact_scores) == 0
        else sum(float(item["score"]) for item in normalized_fact_scores) / len(normalized_fact_scores)
    )
    claim_grounding_score = (
        0.0
        if len(normalized_claim_scores) == 0
        else sum(float(item["score"]) for item in normalized_claim_scores) / len(normalized_claim_scores)
    )
    return {
        "valid": True,
        "error": None,
        "result_score": result_score,
        "fact_coverage_score": fact_coverage_score,
        "claim_grounding_score": claim_grounding_score,
        "gold_fact_scores": normalized_fact_scores,
        "claim_scores": normalized_claim_scores,
        "summary": _normalize_text(payload.get("summary")),
    }


def combine_grounding_scores(
    *,
    grounding_hard_eval: dict[str, Any],
    judge_eval: dict[str, Any],
) -> dict[str, Any]:
    citation_score = float(grounding_hard_eval.get("citation_score") or 0.0)
    result_score = float(judge_eval.get("result_score") or 0.0)
    fact_coverage_score = float(judge_eval.get("fact_coverage_score") or 0.0)
    claim_grounding_score = float(judge_eval.get("claim_grounding_score") or 0.0)
    judge_score = 0.4 * result_score + 0.4 * fact_coverage_score + 0.2 * claim_grounding_score
    grounded_score = 0.2 * citation_score + 0.8 * judge_score
    strict_grounded_success = 1.0 if min(citation_score, result_score, fact_coverage_score, claim_grounding_score) >= 1.0 else 0.0
    return {
        "available": bool(grounding_hard_eval.get("available")),
        "structured_output_valid": bool(grounding_hard_eval.get("structured_output_valid")),
        "citation_score": citation_score,
        "judge_valid": bool(judge_eval.get("valid")),
        "judge_score": judge_score,
        "result_score": result_score,
        "fact_coverage_score": fact_coverage_score,
        "claim_grounding_score": claim_grounding_score,
        "grounded_score": grounded_score,
        "strict_grounded_success": strict_grounded_success,
        "judge_eval": judge_eval,
    }


def init_grounding_aggregate() -> dict[str, float | int]:
    return {
        "num_available": 0,
        "json_valid_count": 0,
        "citation_score_sum": 0.0,
        "exact_citation_rate_sum": 0.0,
        "page_only_citation_rate_sum": 0.0,
        "judge_count": 0,
        "judge_score_sum": 0.0,
        "result_score_sum": 0.0,
        "fact_coverage_score_sum": 0.0,
        "claim_grounding_score_sum": 0.0,
        "grounded_score_sum": 0.0,
        "strict_grounded_success_sum": 0.0,
    }


def update_grounding_aggregate(aggregate: dict[str, float | int], grounding_eval: dict[str, Any] | None) -> None:
    if not isinstance(grounding_eval, dict):
        return
    if not grounding_eval.get("available"):
        return
    aggregate["num_available"] += 1
    if grounding_eval.get("structured_output_valid"):
        aggregate["json_valid_count"] += 1
    citation_score = grounding_eval.get("citation_score")
    if citation_score is not None:
        aggregate["citation_score_sum"] += float(citation_score)
    exact_citation_rate = grounding_eval.get("exact_citation_rate")
    if exact_citation_rate is not None:
        aggregate["exact_citation_rate_sum"] += float(exact_citation_rate)
    page_only_citation_rate = grounding_eval.get("page_only_citation_rate")
    if page_only_citation_rate is not None:
        aggregate["page_only_citation_rate_sum"] += float(page_only_citation_rate)
    if grounding_eval.get("judge_valid"):
        aggregate["judge_count"] += 1
        aggregate["judge_score_sum"] += float(grounding_eval.get("judge_score") or 0.0)
        aggregate["result_score_sum"] += float(grounding_eval.get("result_score") or 0.0)
        aggregate["fact_coverage_score_sum"] += float(grounding_eval.get("fact_coverage_score") or 0.0)
        aggregate["claim_grounding_score_sum"] += float(grounding_eval.get("claim_grounding_score") or 0.0)
        aggregate["grounded_score_sum"] += float(grounding_eval.get("grounded_score") or 0.0)
        aggregate["strict_grounded_success_sum"] += float(grounding_eval.get("strict_grounded_success") or 0.0)


def finalize_grounding_aggregate(aggregate: dict[str, float | int]) -> dict[str, Any]:
    num_available = int(aggregate["num_available"])
    judge_count = int(aggregate["judge_count"])
    metrics: dict[str, Any] = {
        "grounding_num_samples": num_available,
        "grounding_json_valid_rate": None,
        "grounding_citation_score_mean": None,
        "grounding_exact_citation_rate_mean": None,
        "grounding_page_only_citation_rate_mean": None,
        "grounding_judge_num_samples": judge_count,
        "grounding_judge_score_mean": None,
        "grounding_result_score_mean": None,
        "grounding_fact_coverage_score_mean": None,
        "grounding_claim_grounding_score_mean": None,
        "grounding_score_mean": None,
        "grounding_strict_success_rate": None,
    }
    if num_available > 0:
        metrics["grounding_json_valid_rate"] = float(aggregate["json_valid_count"]) / num_available
        metrics["grounding_citation_score_mean"] = float(aggregate["citation_score_sum"]) / num_available
        metrics["grounding_exact_citation_rate_mean"] = float(aggregate["exact_citation_rate_sum"]) / num_available
        metrics["grounding_page_only_citation_rate_mean"] = float(aggregate["page_only_citation_rate_sum"]) / num_available
    if judge_count > 0:
        metrics["grounding_judge_score_mean"] = float(aggregate["judge_score_sum"]) / judge_count
        metrics["grounding_result_score_mean"] = float(aggregate["result_score_sum"]) / judge_count
        metrics["grounding_fact_coverage_score_mean"] = float(aggregate["fact_coverage_score_sum"]) / judge_count
        metrics["grounding_claim_grounding_score_mean"] = float(aggregate["claim_grounding_score_sum"]) / judge_count
        metrics["grounding_score_mean"] = float(aggregate["grounded_score_sum"]) / judge_count
        metrics["grounding_strict_success_rate"] = float(aggregate["strict_grounded_success_sum"]) / judge_count
    return metrics
