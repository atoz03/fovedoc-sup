from __future__ import annotations

import re
import string
from typing import Iterable


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_answer(s: str) -> str:
    """
    轻量归一化：用于 VQA/EM 等基线指标。
    - 小写
    - 去标点（转空格）
    - 去多余空格
    - 去英文冠词
    """
    s = s.lower().translate(_PUNCT_TABLE)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def extract_choice_letter(text: str) -> str | None:
    """
    从生成文本中抽取多选题选项字母（A/B/C/D/E）。
    仅用于基线评测，尽量鲁棒但不做过度启发式。
    """
    if text is None:
        return None
    m = re.search(r"\b([A-E])\b", text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"answer\s*[:：]?\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def first_non_empty(strings: Iterable[str]) -> str | None:
    for s in strings:
        if s is None:
            continue
        ss = str(s).strip()
        if ss != "":
            return ss
    return None

