from __future__ import annotations

import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eccv26.metrics.vqa import vqa_soft_accuracy


class TestVQAMetrics(unittest.TestCase):
    def test_single_answer_exact_match_scores_one(self) -> None:
        self.assertEqual(vqa_soft_accuracy("23", ["23"]), 1.0)

    def test_multi_answer_scales_by_reference_count(self) -> None:
        self.assertAlmostEqual(vqa_soft_accuracy("foo", ["foo", "foo", "bar"]), 2.0 / 3.0)

    def test_empty_answers_scores_zero(self) -> None:
        self.assertEqual(vqa_soft_accuracy("foo", []), 0.0)


if __name__ == "__main__":
    unittest.main()
