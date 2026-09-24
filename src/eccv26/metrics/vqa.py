from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from eccv26.utils.text import normalize_answer


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float


def vqa_soft_accuracy(pred: str, answers: Sequence[str]) -> float:
    """
    轻量 soft accuracy：
      acc = min(1, matches / len(answers))
    其中 matches 是参考答案里与预测一致的数量（做轻量 normalize）。

    这个仓库里的 FoveDoc / V-NIAH / V-MQAR 是 single-answer 设计，
    因此 exact match 应该记 1.0，而不是固定除以 3。
    """
    if pred is None:
        return 0.0
    pred_n = normalize_answer(pred)
    if pred_n == "":
        return 0.0
    if len(answers) == 0:
        return 0.0
    matches = 0
    for a in answers:
        if normalize_answer(a) == pred_n:
            matches += 1
    return min(1.0, matches / float(len(answers)))


def exact_match(pred: str, answers: Sequence[str]) -> float:
    if pred is None:
        return 0.0
    pred_n = normalize_answer(pred)
    return 1.0 if any(normalize_answer(a) == pred_n for a in answers) else 0.0


def relaxed_numeric_match(pred: str, answers: Sequence[str], rel_tol: float = 0.01, abs_tol: float = 1e-4) -> float:
    """
    简单数值容差匹配（用于 ChartQA 等可能是数值答案的数据集）。
    - 若预测/答案都能解析成 float，则按容差匹配
    - 否则回退到 exact match
    """
    if pred is None:
        return 0.0
    pred_n = normalize_answer(pred)
    try:
        pv = float(pred_n)
    except Exception:
        return exact_match(pred, answers)

    for a in answers:
        try:
            av = float(normalize_answer(a))
        except Exception:
            continue
        if abs(pv - av) <= max(abs_tol, rel_tol * max(1.0, abs(av))):
            return 1.0
    return 0.0


def anls_score(pred: str, answers: Sequence[str], threshold: float = 0.5) -> float:
    """
    Doc/slide VQA 常用 ANLS：按归一化编辑距离计分，距离过大记 0。
    这里取多个参考答案中的最大分数。
    """
    if pred is None:
        return 0.0
    pred_n = normalize_answer(pred)
    if pred_n == "":
        return 0.0
    best = 0.0
    for answer in answers:
        answer_n = normalize_answer(answer)
        if answer_n == "":
            continue
        distance = _levenshtein_distance(pred_n, answer_n)
        norm = distance / max(len(pred_n), len(answer_n), 1)
        score = 1.0 - norm if norm < threshold else 0.0
        best = max(best, score)
    return best


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]
