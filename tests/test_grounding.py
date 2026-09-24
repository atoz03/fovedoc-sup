from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eccv26.grounding import (
    apply_grounded_output_format,
    build_grounded_output_prefill,
    build_grounding_judge_messages,
    combine_grounding_scores,
    compute_grounding_hard_eval,
    normalize_grounding_training_phase,
    normalize_structured_prediction_text,
    parse_structured_prediction,
    parse_grounding_judge_response,
    replace_structured_prediction_result,
)


class TestGrounding(unittest.TestCase):
    def _write_document(self, root: pathlib.Path) -> None:
        document_dir = root / "data" / "documents"
        document_dir.mkdir(parents=True, exist_ok=True)
        document_path = document_dir / "doc_grounding.json"
        document_path.write_text(
            json.dumps(
                {
                    "pages": [
                        {
                            "page_id": 2,
                            "page_text": "The answer is 42 for alpha. Another line lives here.",
                            "blocks": [
                                {
                                    "block_id": "p2_b4",
                                    "text": "The answer is 42 for alpha.",
                                },
                                {
                                    "block_id": "p2_b5",
                                    "text": "Another line lives here.",
                                },
                            ],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_compute_grounding_hard_eval_exact_block_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            self._write_document(benchmark_root)
            pred = json.dumps(
                {
                    "result": "42",
                    "claims": [{"claim_id": "c1", "text": "The answer is 42."}],
                    "evidence": [
                        {
                            "evidence_id": "e1",
                            "page_id": 2,
                            "block_id": "p2_b4",
                            "quote": "answer is 42",
                            "supports": ["c1"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
            hard_eval = compute_grounding_hard_eval(
                meta={
                    "document_id": "doc_grounding",
                    "evidence": [{"page_id": 2, "block_id": "p2_b4", "text_excerpt": "answer is 42"}],
                },
                answers=["42"],
                pred_text=pred,
                benchmark_root=str(benchmark_root),
            )

        self.assertTrue(hard_eval["available"])
        self.assertTrue(hard_eval["structured_output_valid"])
        self.assertEqual(hard_eval["result_text"], "42")
        self.assertEqual(hard_eval["citation_score"], 1.0)
        self.assertEqual(hard_eval["exact_citation_rate"], 1.0)
        self.assertEqual(hard_eval["page_only_citation_rate"], 0.0)
        self.assertEqual(hard_eval["verified_evidence"][0]["validation_label"], "exact_block_quote")
        self.assertIn("The answer is 42 for alpha.", hard_eval["verified_evidence"][0]["matched_snippet_raw"])

    def test_compute_grounding_hard_eval_page_quote_only_scores_half(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            self._write_document(benchmark_root)
            pred = json.dumps(
                {
                    "result": "42",
                    "claims": [{"claim_id": "c1", "text": "The answer is 42."}],
                    "evidence": [
                        {
                            "evidence_id": "e1",
                            "page_id": 2,
                            "block_id": "p2_b9",
                            "quote": "answer is 42",
                            "supports": ["c1"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
            hard_eval = compute_grounding_hard_eval(
                meta={
                    "document_id": "doc_grounding",
                    "evidence": [{"page_id": 2, "block_id": "p2_b4", "text_excerpt": "answer is 42"}],
                },
                answers=["42"],
                pred_text=pred,
                benchmark_root=str(benchmark_root),
            )

        self.assertEqual(hard_eval["citation_score"], 0.5)
        self.assertEqual(hard_eval["exact_citation_rate"], 0.0)
        self.assertEqual(hard_eval["page_only_citation_rate"], 1.0)
        self.assertEqual(hard_eval["verified_evidence"][0]["validation_label"], "page_quote_only")

    def test_compute_grounding_hard_eval_can_resolve_evidence_id_only_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            self._write_document(benchmark_root)
            pred = json.dumps(
                {
                    "result": "42",
                    "claims": [{"claim_id": "f1", "text": "The answer is 42."}],
                    "evidence": [{"evidence_id": "e1", "supports": ["f1"]}],
                },
                ensure_ascii=False,
            )
            hard_eval = compute_grounding_hard_eval(
                meta={
                    "document_id": "doc_grounding",
                    "evidence": [{"page_id": 2, "block_id": "p2_b4", "text_excerpt": "answer is 42"}],
                },
                answers=["42"],
                pred_text=pred,
                benchmark_root=str(benchmark_root),
            )

        self.assertEqual(hard_eval["citation_score"], 1.0)
        self.assertEqual(hard_eval["pred_evidence"][0]["page_id"], 2)
        self.assertEqual(hard_eval["pred_evidence"][0]["block_id"], "p2_b4")
        self.assertEqual(hard_eval["pred_evidence"][0]["quote"], "answer is 42")

    def test_normalize_grounding_training_phase_defaults_to_final(self) -> None:
        self.assertEqual(normalize_grounding_training_phase(None), "final")
        self.assertEqual(normalize_grounding_training_phase("grounded_json"), "final")
        self.assertEqual(normalize_grounding_training_phase("plan"), "plan")

    def test_apply_grounded_output_format_final_emphasizes_compact_non_repetitive_schema(self) -> None:
        system_text, user_text = apply_grounded_output_format(
            system_text="回答问题。",
            user_text="问题：答案是什么？",
            training_phase="final",
        )

        self.assertIn("输出必须简洁", system_text)
        self.assertIn("禁止重复", system_text)
        self.assertIn("`evidence` 至少给出 `evidence_id` 与 `supports`", user_text)
        self.assertIn("不要把 `claims` 和 `evidence` 都输出为空数组", user_text)

    def test_build_grounded_output_prefill_matches_phase(self) -> None:
        self.assertEqual(build_grounded_output_prefill(training_phase="final"), '{"result":')
        self.assertEqual(build_grounded_output_prefill(training_phase="plan"), '{"claims":')

    def test_parse_structured_prediction_repairs_truncated_grounded_json(self) -> None:
        pred_text = (
            '{"result":"42","claims":[{"claim_id":"c1","text":"答案是 42"}],'
            '"evidence":[{"evidence_id":"e1","supports":["c1"]}'
        )

        parsed = parse_structured_prediction(pred_text)

        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["result_text"], "42")
        self.assertEqual(parsed["claims"][0]["claim_id"], "c1")
        self.assertEqual(parsed["evidence"][0]["evidence_id"], "e1")
        self.assertEqual(
            parsed["normalized_text"],
            '{"result":"42","claims":[{"claim_id":"c1","text":"答案是 42"}],"evidence":[{"evidence_id":"e1","page_id":null,"block_id":null,"quote":"","supports":["c1"]}]}',
        )

    def test_normalize_structured_prediction_text_returns_minified_json(self) -> None:
        pred_text = """```json
        {
          "result": "42",
          "claims": [{"claim_id": "c1", "text": "答案是 42"}],
          "evidence": [{"evidence_id": "e1", "supports": ["c1"]}]
        }
        ```"""

        normalized = normalize_structured_prediction_text(pred_text)

        self.assertEqual(
            normalized,
            '{"result":"42","claims":[{"claim_id":"c1","text":"答案是 42"}],"evidence":[{"evidence_id":"e1","page_id":null,"block_id":null,"quote":"","supports":["c1"]}]}',
        )

    def test_replace_structured_prediction_result_preserves_claims_and_evidence(self) -> None:
        pred_text = (
            '{"result":"旧答案","claims":[{"claim_id":"c1","text":"答案是 42"}],'
            '"evidence":[{"evidence_id":"e1","supports":["c1"]}]}'
        )

        rewritten = replace_structured_prediction_result(pred_text, new_result="42")

        self.assertEqual(
            rewritten,
            '{"result":"42","claims":[{"claim_id":"c1","text":"答案是 42"}],"evidence":[{"evidence_id":"e1","page_id":null,"block_id":null,"quote":"","supports":["c1"]}]}',
        )

    def test_parse_grounding_judge_response_and_combine_scores(self) -> None:
        judge_eval = parse_grounding_judge_response(
            json.dumps(
                {
                    "result_score": 0.75,
                    "gold_fact_scores": [
                        {"fact_id": "f1", "score": 1.0, "reason": "supported"},
                        {"fact_id": "f2", "score": 0.5, "reason": "partially supported"},
                    ],
                    "claim_scores": [
                        {"claim_id": "c1", "score": 0.75, "reason": "mostly supported"},
                        {"claim_id": "c2", "score": 0.25, "reason": "weakly supported"},
                    ],
                    "summary": "ok",
                },
                ensure_ascii=False,
            ),
            expected_fact_ids=["f1", "f2"],
            expected_claim_ids=["c1", "c2"],
        )
        combined = combine_grounding_scores(
            grounding_hard_eval={
                "available": True,
                "structured_output_valid": True,
                "citation_score": 0.5,
            },
            judge_eval=judge_eval,
        )

        self.assertTrue(judge_eval["valid"])
        self.assertAlmostEqual(judge_eval["fact_coverage_score"], 0.75)
        self.assertAlmostEqual(judge_eval["claim_grounding_score"], 0.5)
        self.assertAlmostEqual(combined["judge_score"], 0.7)
        self.assertAlmostEqual(combined["grounded_score"], 0.66)
        self.assertEqual(combined["strict_grounded_success"], 0.0)

    def test_build_grounding_judge_messages_puts_rules_in_system_prompt(self) -> None:
        system_text, user_text = build_grounding_judge_messages(
            question="问题是什么？",
            context="这是上下文。",
            grounding_hard_eval={
                "gold_package": {
                    "gold_result": "42",
                    "gold_result_text": "42",
                    "gold_atomic_facts": [{"fact_id": "f1", "text": "答案是 42"}],
                },
                "pred_result": "42",
                "result_text": "42",
                "raw_prediction_text": "42",
                "pred_claims": [{"claim_id": "c1", "text": "答案是 42"}],
                "verified_evidence": [{"evidence_id": "e1", "page_id": 2, "quote": "answer is 42"}],
                "invalid_evidence_summary": {"count": 0},
            },
        )

        self.assertIn("required_output_schema", system_text)
        self.assertIn("scoring_rules", system_text)
        self.assertNotIn("required_output_schema", user_text)
        self.assertNotIn("scoring_rules", user_text)
        self.assertIn('"query": "问题是什么？"', user_text)


if __name__ == "__main__":
    unittest.main()
