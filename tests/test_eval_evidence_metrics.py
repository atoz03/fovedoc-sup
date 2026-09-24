from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

from PIL import Image
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eccv26.eval import (
    _build_exact_block_stage_inputs,
    _compute_evidence_eval,
    _finalize_evidence_aggregate,
    _init_evidence_aggregate,
    _sample_belongs_to_shard,
    _stable_random_block_score,
    _update_evidence_aggregate,
)
from eccv26.utils.exact_block import EXACT_BLOCK_FEATURE_DIM, EXACT_BLOCK_FEATURE_INDEX


class TestEvalEvidenceMetrics(unittest.TestCase):
    def test_sample_belongs_to_shard_uses_modulo_partition(self) -> None:
        kept = [idx for idx in range(10) if _sample_belongs_to_shard(idx, shard_index=1, num_shards=3)]
        self.assertEqual(kept, [1, 4, 7])

    def test_sample_belongs_to_shard_rejects_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            _sample_belongs_to_shard(0, shard_index=0, num_shards=0)
        with self.assertRaises(ValueError):
            _sample_belongs_to_shard(0, shard_index=2, num_shards=2)

    def test_fallback_to_input_pages_when_model_does_not_return_retrieval(self) -> None:
        evidence_eval = _compute_evidence_eval(
            {
                "_input_page_ids": [2, 4, 6],
                "evidence": [
                    {
                        "page_id": 4,
                        "block_id": "p4_b2",
                        "span_ids": [],
                        "text_excerpt": "needle",
                    }
                ]
            },
            num_input_pages=3,
            visual_tokens=1000,
            model_out={},
        )

        self.assertTrue(evidence_eval["available"])
        self.assertEqual(evidence_eval["mode"], "input_pages_fallback")
        self.assertEqual(evidence_eval["input_page_ids"], [2, 4, 6])
        self.assertEqual(evidence_eval["target_page_ids"], [4])
        self.assertEqual(evidence_eval["visible_target_page_ids"], [4])
        self.assertEqual(evidence_eval["visible_target_block_ids"], ["p4_b2"])
        self.assertEqual(evidence_eval["retained_page_ids"], [2, 4, 6])
        self.assertEqual(evidence_eval["evidence_pages_covered"], [4])
        self.assertEqual(evidence_eval["visible_pages_covered"], [4])
        self.assertEqual(evidence_eval["evidence_pages_missed"], [])
        self.assertEqual(evidence_eval["target_block_ids"], ["p4_b2"])
        self.assertEqual(evidence_eval["input_page_recall"], 1.0)
        self.assertEqual(evidence_eval["page_recall"], 1.0)
        self.assertEqual(evidence_eval["page_full_recall"], 1.0)
        self.assertEqual(evidence_eval["visible_page_recall"], 1.0)
        self.assertEqual(evidence_eval["visible_page_full_recall"], 1.0)
        self.assertIsNone(evidence_eval["block_hit"])

    def test_model_retained_pages_override_input_fallback(self) -> None:
        evidence_eval = _compute_evidence_eval(
            {
                "_input_page_ids": [4, 6, 9],
                "evidence": [
                    {"page_id": 4, "block_id": "p4_b1"},
                    {"page_id": 6, "block_id": "p6_b3"},
                    {"page_id": 6, "block_id": "p6_b7"},
                ]
            },
            num_input_pages=3,
            visual_tokens=200,
            model_out={
                "retained_page_ids": [6],
                "retained_block_ids": ["p6_b3"],
            },
        )

        self.assertTrue(evidence_eval["available"])
        self.assertEqual(evidence_eval["mode"], "model_retained_pages")
        self.assertEqual(evidence_eval["target_page_ids"], [4, 6])
        self.assertEqual(evidence_eval["visible_target_page_ids"], [4, 6])
        self.assertEqual(evidence_eval["retained_page_ids"], [6])
        self.assertEqual(evidence_eval["evidence_pages_covered"], [6])
        self.assertEqual(evidence_eval["visible_pages_covered"], [6])
        self.assertEqual(evidence_eval["evidence_pages_missed"], [4])
        self.assertEqual(evidence_eval["num_target_pages"], 2)
        self.assertEqual(evidence_eval["num_target_blocks"], 3)
        self.assertEqual(evidence_eval["input_page_recall"], 1.0)
        self.assertEqual(evidence_eval["page_recall"], 0.5)
        self.assertEqual(evidence_eval["page_full_recall"], 0.0)
        self.assertEqual(evidence_eval["visible_page_recall"], 0.5)
        self.assertEqual(evidence_eval["visible_page_full_recall"], 0.0)
        self.assertEqual(evidence_eval["block_hit"], 1.0)

    def test_exact_block_recall_uses_exact_retained_block_ids(self) -> None:
        evidence_eval = _compute_evidence_eval(
            {
                "_input_page_ids": [4, 6],
                "evidence": [
                    {"page_id": 4, "block_id": "p4_b1"},
                    {"page_id": 6, "block_id": "p6_b3"},
                    {"page_id": 6, "block_id": "p6_b7"},
                ],
            },
            num_input_pages=2,
            visual_tokens=200,
            model_out={
                "retained_page_ids": [4, 6],
                "exact_retained_block_ids": ["p6_b3", "p4_b1"],
            },
        )

        self.assertEqual(evidence_eval["exact_retained_block_ids"], ["p4_b1", "p6_b3"])
        self.assertEqual(evidence_eval["exact_blocks_covered"], ["p4_b1", "p6_b3"])
        self.assertEqual(evidence_eval["exact_blocks_missed"], ["p6_b7"])
        self.assertAlmostEqual(evidence_eval["exact_block_recall"], 2.0 / 3.0)
        self.assertEqual(evidence_eval["exact_block_full_recall"], 0.0)
        self.assertAlmostEqual(evidence_eval["visible_exact_block_recall"], 2.0 / 3.0)
        self.assertEqual(evidence_eval["visible_exact_block_full_recall"], 0.0)

    def test_bbox_based_coarse_mapping_supports_block_ids_beyond_nine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_bbox.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 300.0,
                                "height": 300.0,
                                "blocks": [
                                    {"block_id": "p1_b1", "bbox": [0.0, 0.0, 50.0, 50.0]},
                                    {"block_id": "p1_b122", "bbox": [220.0, 220.0, 290.0, 290.0]},
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            evidence_eval = _compute_evidence_eval(
                {
                    "_input_page_ids": [1],
                    "document_id": "doc_bbox",
                    "evidence": [
                        {"page_id": 1, "block_id": "p1_b122"},
                    ],
                },
                num_input_pages=1,
                visual_tokens=128,
                model_out={
                    "retained_page_ids": [1],
                    "retained_block_ids": ["p1_g22"],
                },
                benchmark_root=str(benchmark_root),
            )

        self.assertEqual(evidence_eval["target_coarse_block_ids"], ["p1_g22"])
        self.assertEqual(evidence_eval["visible_target_coarse_block_ids"], ["p1_g22"])
        self.assertEqual(evidence_eval["retained_coarse_block_ids"], ["p1_g22"])
        self.assertEqual(evidence_eval["block_hit"], 1.0)

    def test_build_exact_block_stage_inputs_selects_matching_block_and_builds_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_blocks.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 100.0,
                                "height": 100.0,
                                "page_text": "alpha beta gamma",
                                "blocks": [
                                    {
                                        "block_id": "p1_b1",
                                        "type": "text",
                                        "bbox": [0.0, 0.0, 40.0, 40.0],
                                        "text": "apple orange",
                                    },
                                    {
                                        "block_id": "p1_b2",
                                        "type": "text",
                                        "bbox": [50.0, 50.0, 95.0, 95.0],
                                        "text": "target answer token",
                                    },
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            page = Image.new("RGB", (200, 200), color=(255, 255, 255))
            packed_images, debug = _build_exact_block_stage_inputs(
                model=torch.nn.Module(),
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={"retained_page_ids": [1]},
                selector_type="lexical",
                source_page_mode="retained",
                block_topk=1,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=True,
            )

        self.assertTrue(debug["used"])
        self.assertEqual(debug["exact_retained_block_ids"], ["p1_b2"])
        self.assertEqual(len(packed_images), 1)
        self.assertGreater(packed_images[0].size[0], 0)
        self.assertGreater(packed_images[0].size[1], 0)

    def test_build_exact_block_stage_inputs_supports_model_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_blocks_model.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 100.0,
                                "height": 100.0,
                                "blocks": [
                                    {
                                        "block_id": "p1_b1",
                                        "type": "text",
                                        "bbox": [0.0, 0.0, 40.0, 40.0],
                                        "text": "apple orange",
                                    },
                                    {
                                        "block_id": "p1_b2",
                                        "type": "text",
                                        "bbox": [50.0, 50.0, 95.0, 95.0],
                                        "text": "target answer token",
                                    },
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            model = torch.nn.Module()
            model.dma_exact_block_proj = torch.nn.Linear(EXACT_BLOCK_FEATURE_DIM, 1, bias=True)
            with torch.no_grad():
                model.dma_exact_block_proj.weight.zero_()
                model.dma_exact_block_proj.bias.zero_()
                model.dma_exact_block_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["lexical_score_log"]] = 1.0
                model.dma_exact_block_proj.weight[0, EXACT_BLOCK_FEATURE_INDEX["coarse_block_score"]] = 0.5
            page = Image.new("RGB", (200, 200), color=(255, 255, 255))
            packed_images, debug = _build_exact_block_stage_inputs(
                model=model,
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks_model"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={
                    "retained_page_ids": [1],
                    "retained_page_stats": [
                        {
                            "page_id": 1,
                            "page_score": 1.0,
                            "page_focus": 1.2,
                            "selected_ratio": 0.25,
                            "coarse_block_scores": [0.1, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        }
                    ],
                },
                selector_type="model",
                source_page_mode="retained",
                block_topk=1,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=False,
            )

        self.assertTrue(debug["used"])
        self.assertEqual(debug["type"], "model")
        self.assertEqual(debug["exact_retained_block_ids"], ["p1_b2"])
        self.assertEqual(len(packed_images), 1)

    def test_build_exact_block_stage_inputs_passes_page_ids_to_custom_scorer(self) -> None:
        class _PageAwareScorer(torch.nn.Module):
            def forward(self, feature_tensor, *, page_ids=None):
                if page_ids is None:
                    raise AssertionError("page_ids should not be None")
                return feature_tensor[:, EXACT_BLOCK_FEATURE_INDEX["lexical_score_log"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_blocks_model_pageaware.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 100.0,
                                "height": 100.0,
                                "blocks": [
                                    {
                                        "block_id": "p1_b1",
                                        "type": "text",
                                        "bbox": [0.0, 0.0, 40.0, 40.0],
                                        "text": "apple orange",
                                    },
                                    {
                                        "block_id": "p1_b2",
                                        "type": "text",
                                        "bbox": [50.0, 50.0, 95.0, 95.0],
                                        "text": "target answer token",
                                    },
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            model = torch.nn.Module()
            model.dma_exact_block_proj = _PageAwareScorer()
            page = Image.new("RGB", (200, 200), color=(255, 255, 255))
            packed_images, debug = _build_exact_block_stage_inputs(
                model=model,
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks_model_pageaware"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={"retained_page_ids": [1]},
                selector_type="model",
                source_page_mode="retained",
                block_topk=1,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=False,
            )

        self.assertTrue(debug["used"])
        self.assertEqual(debug["exact_retained_block_ids"], ["p1_b2"])
        self.assertEqual(len(packed_images), 1)

    def test_build_exact_block_stage_inputs_supports_vlm_rerank_selector(self) -> None:
        class _FakeVlmReranker:
            def score_block_relevance_one(self, *, images, user_text: str, system_text: str) -> float:
                self.last_system_text = system_text
                self.last_image_count = len(images)
                return 10.0 if "候选证据块文本：\ntarget answer token" in user_text else -1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_blocks_vlm_rerank.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 100.0,
                                "height": 100.0,
                                "blocks": [
                                    {
                                        "block_id": "p1_b1",
                                        "type": "text",
                                        "bbox": [0.0, 0.0, 40.0, 40.0],
                                        "text": "apple orange",
                                    },
                                    {
                                        "block_id": "p1_b2",
                                        "type": "text",
                                        "bbox": [50.0, 50.0, 95.0, 95.0],
                                        "text": "target answer token",
                                    },
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            page = Image.new("RGB", (200, 200), color=(255, 255, 255))
            reranker = _FakeVlmReranker()
            packed_images, debug = _build_exact_block_stage_inputs(
                model=torch.nn.Module(),
                vlm_reranker=reranker,
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks_vlm_rerank"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={"retained_page_ids": [1]},
                selector_type="vlm_rerank",
                source_page_mode="retained",
                block_topk=1,
                vlm_rerank_topm=2,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=False,
            )

        self.assertTrue(debug["used"])
        self.assertEqual(debug["type"], "vlm_rerank")
        self.assertEqual(debug["exact_retained_block_ids"], ["p1_b2"])
        self.assertEqual(debug["vlm_rerank_topm"], 2)
        self.assertEqual(len(packed_images), 1)

    def test_build_exact_block_stage_inputs_supports_stable_random_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = pathlib.Path(tmpdir)
            document_dir = benchmark_root / "data" / "documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            document_json = document_dir / "doc_blocks_random.json"
            document_json.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "width": 100.0,
                                "height": 100.0,
                                "blocks": [
                                    {
                                        "block_id": "p1_b1",
                                        "type": "text",
                                        "bbox": [0.0, 0.0, 40.0, 40.0],
                                        "text": "apple orange",
                                    },
                                    {
                                        "block_id": "p1_b2",
                                        "type": "text",
                                        "bbox": [50.0, 50.0, 95.0, 95.0],
                                        "text": "target answer token",
                                    },
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            page = Image.new("RGB", (200, 200), color=(255, 255, 255))
            first_images, first_debug = _build_exact_block_stage_inputs(
                model=torch.nn.Module(),
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks_random"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={"retained_page_ids": [1]},
                selector_type="random",
                source_page_mode="retained",
                block_topk=1,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=False,
            )
            second_images, second_debug = _build_exact_block_stage_inputs(
                model=torch.nn.Module(),
                images=[page],
                input_page_ids=[1],
                meta={"document_id": "doc_blocks_random"},
                question="What is the target answer token?",
                context=None,
                benchmark_root=str(benchmark_root),
                model_out={"retained_page_ids": [1]},
                selector_type="random",
                source_page_mode="retained",
                block_topk=1,
                crop_expand_ratio=0.0,
                pack_max_blocks_per_canvas=4,
                pack_enable=False,
            )

        self.assertTrue(first_debug["used"])
        self.assertEqual(first_debug["type"], "random")
        self.assertEqual(first_debug["exact_retained_block_ids"], second_debug["exact_retained_block_ids"])
        self.assertEqual(len(first_images), 1)
        self.assertEqual(len(second_images), 1)

    def test_stable_random_block_score_is_reproducible(self) -> None:
        block = {"page_id": 1, "block_id": "p1_b2", "order_index": 2}
        score = _stable_random_block_score(
            document_json_path="/tmp/doc.json",
            question="question",
            context="context",
            block=block,
        )

        self.assertEqual(
            score,
            _stable_random_block_score(
                document_json_path="/tmp/doc.json",
                question="question",
                context="context",
                block=block,
            ),
        )
        self.assertNotEqual(
            score,
            _stable_random_block_score(
                document_json_path="/tmp/doc.json",
                question="question",
                context="context",
                block={"page_id": 1, "block_id": "p1_b3", "order_index": 3},
            ),
        )

    def test_input_page_recall_and_visible_page_recall_are_separated(self) -> None:
        evidence_eval = _compute_evidence_eval(
            {
                "_input_page_ids": [1, 2, 3],
                "evidence": [
                    {"page_id": 2, "block_id": "p2_b1"},
                    {"page_id": 5, "block_id": "p5_b9"},
                ],
            },
            num_input_pages=3,
            visual_tokens=256,
            model_out={
                "retained_page_ids": [2],
                "retained_block_ids": [],
            },
        )

        self.assertEqual(evidence_eval["target_page_ids"], [2, 5])
        self.assertEqual(evidence_eval["visible_target_page_ids"], [2])
        self.assertEqual(evidence_eval["input_page_recall"], 0.5)
        self.assertEqual(evidence_eval["page_recall"], 0.5)
        self.assertEqual(evidence_eval["visible_page_recall"], 1.0)
        self.assertEqual(evidence_eval["visible_page_full_recall"], 1.0)
        self.assertEqual(evidence_eval["visible_pages_covered"], [2])
        self.assertEqual(evidence_eval["visible_pages_missed"], [])

    def test_missing_evidence_returns_unavailable(self) -> None:
        evidence_eval = _compute_evidence_eval(
            {"_input_page_ids": [3, 5]},
            num_input_pages=2,
            visual_tokens=123,
            model_out={"retained_page_ids": [5]},
        )

        self.assertFalse(evidence_eval["available"])
        self.assertEqual(evidence_eval["mode"], "unavailable")
        self.assertEqual(evidence_eval["input_page_ids"], [3, 5])
        self.assertEqual(evidence_eval["target_page_ids"], [])
        self.assertEqual(evidence_eval["visible_target_page_ids"], [])
        self.assertEqual(evidence_eval["retained_page_ids"], [])
        self.assertIsNone(evidence_eval["input_page_recall"])
        self.assertIsNone(evidence_eval["page_recall"])
        self.assertIsNone(evidence_eval["page_full_recall"])
        self.assertIsNone(evidence_eval["visible_page_recall"])
        self.assertIsNone(evidence_eval["visible_page_full_recall"])

    def test_finalize_evidence_aggregate_uses_dataset_level_efficiency_formula(self) -> None:
        aggregate = _init_evidence_aggregate()
        _update_evidence_aggregate(
            aggregate,
            {
                "available": True,
                "input_page_recall": 1.0,
                "page_recall": 1.0,
                "page_full_recall": 1.0,
                "visible_page_recall": 1.0,
                "visible_page_full_recall": 1.0,
                "exact_block_recall": 1.0,
                "exact_block_full_recall": 1.0,
                "visible_exact_block_recall": 1.0,
                "visible_exact_block_full_recall": 1.0,
                "block_hit": 1.0,
            },
        )
        _update_evidence_aggregate(
            aggregate,
            {
                "available": True,
                "input_page_recall": 0.5,
                "page_recall": 0.5,
                "page_full_recall": 0.0,
                "visible_page_recall": 1.0,
                "visible_page_full_recall": 1.0,
                "exact_block_recall": 1.0,
                "exact_block_full_recall": 1.0,
                "visible_exact_block_recall": 1.0,
                "visible_exact_block_full_recall": 1.0,
                "block_hit": None,
            },
        )

        summary = _finalize_evidence_aggregate(aggregate, visual_tokens_mean=300.0)

        self.assertEqual(summary["evidence_num_samples"], 2)
        self.assertEqual(summary["evidence_input_page_recall_mean"], 0.75)
        self.assertEqual(summary["evidence_page_recall_mean"], 0.75)
        self.assertEqual(summary["evidence_page_full_recall_mean"], 0.5)
        self.assertEqual(summary["evidence_exact_block_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_exact_block_full_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_visible_num_samples"], 2)
        self.assertEqual(summary["evidence_visible_page_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_visible_page_full_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_visible_exact_block_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_visible_exact_block_full_recall_mean"], 1.0)
        self.assertEqual(summary["evidence_block_hit_rate"], 1.0)
        self.assertEqual(summary["evidence_efficiency"], 0.75 / 300.0)
        self.assertEqual(summary["evidence_visible_efficiency"], 1.0 / 300.0)

    def test_finalize_evidence_aggregate_is_safe_for_no_evidence(self) -> None:
        summary = _finalize_evidence_aggregate(_init_evidence_aggregate(), visual_tokens_mean=200.0)

        self.assertEqual(summary["evidence_num_samples"], 0)
        self.assertIsNone(summary["evidence_input_page_recall_mean"])
        self.assertIsNone(summary["evidence_page_recall_mean"])
        self.assertIsNone(summary["evidence_page_full_recall_mean"])
        self.assertIsNone(summary["evidence_exact_block_recall_mean"])
        self.assertIsNone(summary["evidence_exact_block_full_recall_mean"])
        self.assertEqual(summary["evidence_visible_num_samples"], 0)
        self.assertIsNone(summary["evidence_visible_page_recall_mean"])
        self.assertIsNone(summary["evidence_visible_page_full_recall_mean"])
        self.assertIsNone(summary["evidence_visible_exact_block_recall_mean"])
        self.assertIsNone(summary["evidence_visible_exact_block_full_recall_mean"])
        self.assertIsNone(summary["evidence_block_hit_rate"])
        self.assertIsNone(summary["evidence_efficiency"])
        self.assertIsNone(summary["evidence_visible_efficiency"])


if __name__ == "__main__":
    unittest.main()
