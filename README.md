# Found but Not Read — Supporting Materials

Supporting materials for the ICASSP 2027 submission
**"Found but Not Read: When Extracted Text Closes the Retrieval–Reading Gap in Document
Vision–Language Models"**.

Project page: https://atoz03.github.io/fovedoc-sup/

This repository contains:

- the **FoveDoc-Bench** annotations (2,346 questions over 1,173 born-digital documents, each with its answer and evidence pages, blocks and excerpts),
- the **ColQwen2 page rankings** used by every experiment,
- the **result tables** behind every number in the paper,
- the **per-question judged predictions** for every arm the paper reports (51,242 rows),
- the **code** for the paired protocol: retrieval, OCR and text-layer memory construction, reader evaluation, LLM judging, analysis and figures.

```
benchmark/
  fovedoc_bench/vniah_test.jsonl        1173 V-NIAH questions (one evidence block on one page)
  fovedoc_bench/vmqar_test.jsonl        1173 V-MQAR questions (two evidence blocks on distinct pages)
  documents.jsonl                       1173 source documents: source, licence class, origin URL, SHA-256
  retrieval/fovedoc_colqwen2_ranked.jsonl          ColQwen2 page ranking per FoveDoc-Bench question
  retrieval/mmlongbench_doc_colqwen2_ranked.jsonl  ColQwen2 page ranking + scores per MMLongBench-Doc question
results/
  tables/*.md                           analysis output for every paper number (see the map below)
  predictions/<experiment>/*.jsonl.gz   per-question predictions with judge verdicts
  evidence_audit.json                   audit of the verbatim cases shown in Figs. 1 and 4
figures/                                the five paper figures (PDF)
scripts/                                protocol, judging, analysis and figure scripts
src/eccv26/                             data loading, reader wrappers, memory packing, metrics
configs/dmr_*.yaml                      reader configurations (one per VLM)
tests/                                  unit tests (memory controls, metrics, grounding)
```

## Where each number in the paper comes from

| Paper element | Result table (`results/tables/`) | Per-question predictions (`results/predictions/`) |
|---|---|---|
| Table 1 (benchmarks, recall@16) | `dmr_external_retrieval_recall.md` | `benchmark/retrieval/` |
| Table 2 (images / +OCR / +PDF, Fig. 1 Step 2) | `dmr_ocr_parity.md` | `main_ocr_parity_qwen3vl2b/` |
| OCR fidelity (packed characters, blocks per page, 76% / 53% excerpt recovery) | `dmr_packed_chars_parity.md`, `dmr_ocr_fidelity.md`, `dmr_dose_response_{vmqar,vniah}.md` | — |
| Fig. 2 (six readers, first 400 samples per task) | `dmr_reader_family_grid.md` | `reader_family_grid/` |
| "Cheaper levers do not close it" | `dmr_lever_gaps.md`, `dmr_clean_main_table.md` | — |
| Localization controls (full text, crops 87% / 92%, red boxes) | `dmr_selection_controls.md` | `selection_controls/` |
| Table 3, Fig. 3 (MMLongBench-Doc by evidence modality and recall) | `dmr_external_validity.md`, `dmr_external_retrieval_recall.md` | `external_mmlongbench_doc/` |
| Fig. 5a,b (Control 1, recall withdrawn at a fixed 16-page budget) | `dmr_recall_sweep_{vmqar,vniah}.md` | `recall_sweep/` |
| Fig. 5c (Control 2, leak-free zero recall) | `dmr_zero_recall_leakage.md` | `recall_sweep/` |
| Fig. 5 (Control 3, foreign-document memory) | `dmr_foreign_memory.md`, `20260905_dmr_foreign_memory_results.md` | `foreign_memory/` |
| Scoring (GPT-5.4 vs GPT-5.6 Luna agreement, 98.8% over 1,963 predictions) | `dmr_judge_calibration.md` | — |

`dmr_clean_main_table.md` also reports learned-selector and associative-memory arms from an
earlier method that the paper does not use; only its stock-reader rows are relevant here.

Arm names in the prediction files:

| file prefix | arm |
|---|---|
| `pages` | images: the 16 retrieved page images only |
| `ocrmem` | images + 16 RapidOCR blocks selected lexically against the question (the paper's OCR memory) |
| `pdfmem` | images + 16 blocks from the born-digital PDF text layer |
| `pages_memory` | images + text memory; `memory_block_source` is `ocr` or `pdf_text_layer` (null in the oldest runs, where it was always the PDF text layer) |
| `lex16n` | `ocrmem` with the relevance hint removed from the prompt (neutral wording) |
| `alln` | images + all OCR blocks of the 16 pages in reading order, no selection |
| `rand16n`, `randcn` | images + 16 random OCR blocks / random blocks matched in characters to `lex16n` |
| `crops` | images + the `lex16n` blocks as image crops at the reader's page pixel density, no text |
| `hilite` | the 16 page images with red boxes around the `lex16n` blocks, no text |
| `*_drop{0,1,2}` | recall level: 0, 1 or 2 gold pages withdrawn and replaced by non-gold pages the retriever ranked below the top 16, so 16 pages are always shown |

## FoveDoc-Bench

Each line of `benchmark/fovedoc_bench/{vniah,vmqar}_test.jsonl` is one question:

```json
{
  "id": "gemini32_vmqar_resume_v2_20260314:vmqar:<document_id>:6-29",
  "document_id": "arxiv_recent_mix_local_039757d8a4a890dc",
  "benchmark": "V-MQAR",
  "task_type": "cross_page_association",
  "question": "…",
  "answers": ["…"],
  "images": ["render/<document_id>/page_0006.png", "…", "render/<document_id>/page_0029.png"],
  "meta": {
    "difficulty": {"pages": 24, "hop": 2, "noise_pages": 22},
    "evidence": [
      {"page_id": 16, "block_id": "p16_b24", "text_excerpt": "…"},
      {"page_id": 20, "block_id": "p20_b14", "text_excerpt": "…"}
    ]
  }
}
```

- The `a-b` suffix of `id` is the question's context: source pages `a` to `b` (16–24 pages). `images` lists them in order.
- Every page number (`page_id` in the evidence, the retrieval files and the predictions) is the page's number in the source PDF.
- `block_id` names a block of the source PDF's text layer (`p<page>_b<index>`), and `text_excerpt` is that block's text.
- `meta.split` is a leftover field from the builder's internal splits and does not matter here. All 2,346 questions are test questions, and all 1,173 documents are held out from the auxiliary training and validation QA pools.
- All documents are in English. 93 questions (and their answers) were generated in Chinese about English documents. We kept them as generated.

**Not included.** We do not redistribute the source PDFs, rendered page images or the full OCR and
text-layer block stores, because the documents belong to third parties. `benchmark/documents.jsonl`
gives each document's source family, licence class (`arxiv`, `official_public_document`,
`official_public_webpage_rendered_pdf`, `public-web`), origin URL (1,054 of 1,173) and the SHA-256 of
the exact PDF we used, so a re-acquired copy can be checked byte for byte.

## Per-question predictions

Each line of `results/predictions/**/*.jsonl.gz` is one (question, arm) reader call, trimmed to
the fields the analysis uses:

| field | meaning |
|---|---|
| `id`, `dataset`, `mode` | question id, benchmark, reader mode (`pages`, `pages_memory`, `pages_crops`, …) |
| `pred`, `answers` | reader output and gold answers |
| `judge` | `{verdict: correct / partial / incorrect, score, reason}`; strict accuracy counts only `correct` |
| `scoring` | string-match score (`vqa_soft` / `anls`), not used in the paper |
| `selected_page_ids`, `gold_evidence_page_ids`, `full_recall`, `partial_recall_count` | pages shown to the reader and recall against the gold pages |
| `memory_block_source`, `memory_selector`, `memory_prefix_style` | how the text memory was built |
| `memory_blocks`, `memory_num_facts`, `memory_packed_chars` | `[page_id, block_id]` of each packed block, block count, packed characters |

The packed memory text is dropped, since it copies document content; `memory_blocks` names each packed block.
MMLongBench-Doc rows also keep `question`. Their evidence-modality tags come from the benchmark's own TSV
(`MMLongBench_DOC.tsv` in VLMEvalKit's LMUData), which `scripts/dmr_external_analysis.py` joins on `id`.

Strict accuracy for one arm, for example:

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("results/predictions/main_ocr_parity_qwen3vl2b/ocrmem_vmqar.jsonl.gz", "rt")]
print(sum(r["judge"]["verdict"] == "correct" for r in rows) / len(rows))  # 0.287 (Table 2)
```

## Running the pipeline

Environment: Python 3.11, `uv sync` (pinned in `pyproject.toml` / `uv.lock`); `rapidocr-onnxruntime`,
`pymupdf` and `colpali-engine` are needed for OCR, text-layer extraction and retrieval. The scripts
run from the repository root with `PYTHONPATH=src` and call `.venv/bin/python`.

Model paths in `configs/dmr_*.yaml` are Hugging Face ids (`Qwen/Qwen3-VL-2B-Instruct`, …) and can be
replaced with local directories. The scripts read two data roots:

- `data/LMUData/`: VLMEvalKit's LMUData directory, holding `MMLongBench_DOC.tsv` and its images.
- `data/fovedoc_project/`: the FoveDoc-Bench build tree, laid out as our benchmark builder writes it:

  ```
  data/fovedoc_project/data/
    render/<document_id>/page_NNNN.png         source pages rendered at 120 dpi (NNNN = source page number)
    documents/<document_id>.json               the PDF's text-layer blocks per page (feeds the +PDF arm)
    manifests/final_benchmark/<task>/{train,val,test}.jsonl
  ```

  The benchmark files load into this layout as follows. The loader concatenates all three split files, so `train` and `val` are left empty:

  ```bash
  for t in vniah vmqar; do
    d=data/fovedoc_project/data/manifests/final_benchmark/$t; mkdir -p $d
    sed "s#\"render/#\"$PWD/data/fovedoc_project/data/render/#g" benchmark/fovedoc_bench/${t}_test.jsonl > $d/test.jsonl
    : > $d/train.jsonl; : > $d/val.jsonl
  done
  ```

The page renders and the text-layer store are derived from the third-party PDFs, so they are not
included (see above). The OCR arm needs only the rendered pages: `scripts/dmr_build_ocr_document_cache.py`
builds its block store from them. The released tables and per-question predictions let every
reported number be checked without re-running any model.

Stages, in the order the paper's numbers were produced (each script's header documents its options):

| stage | command |
|---|---|
| page retrieval (ColQwen2, top-16) | `scripts/export_fovedoc_page_retrieval_manifest.py`, `scripts/run_colqwen2_page_retrieval.py`; the output is already in `benchmark/retrieval/` |
| OCR block store (RapidOCR, CPU) | `scripts/dmr_build_ocr_document_cache.py` |
| Table 2: images / +OCR / +PDF | `bash scripts/run_dmr_clean_eval.sh 2b` (its `pages` and `pages_memory` arms are images and +PDF; its selector arms belong to an earlier method and are not used), then `bash scripts/run_dmr_ocr_parity_eval.sh 2b` (+OCR) |
| Fig. 2: six readers | `bash scripts/run_dmr_reader_family_grid.sh 400` |
| cheaper levers | `bash scripts/run_dmr_lever_gaps.sh all`, `scripts/run_dmr_b1_thinking.sh`, `scripts/run_dmr_c_train.sh` |
| localization controls | `bash scripts/run_dmr_selection_controls.sh` |
| MMLongBench-Doc | `bash scripts/run_dmr_external_validity.sh all` |
| Controls 1–2 (recall sweep, leakage) | `bash scripts/run_dmr_recall_sweep.sh all` |
| Control 3 (foreign memory) | `bash scripts/run_dmr_foreign_memory.sh` |
| LLM judging | `bash scripts/run_dmr_judging.sh <parity\|grid\|levers\|external\|sweep\|foreign\|controls>` |
| tables | `scripts/dmr_*_analysis.py`, `scripts/dmr_ocr_fidelity_report.py`, `scripts/dmr_zero_recall_leakage.py`, `scripts/dmr_judge_calibration.py` |
| figures 2, 3, 5 | `.venv/bin/python scripts/dmr_paper_figures.py --out figures` (reads `results/tables/` only) |

The judge runs through an OpenAI-compatible `/responses` endpoint: `export OPENAI_API_KEY=… OPENAI_BASE_URL=…`.
The grader is set by `DMR_JUDGE_MODEL` (default `gpt-5.4`) and `DMR_JUDGE_REASONING_EFFORT`. The recall-sweep,
foreign-memory and localization stages were graded with `DMR_JUDGE_MODEL=gpt-5.6-luna` at `medium` effort. The
header of `scripts/run_dmr_judging.sh` records the grader and effort for each stage. Every arm of a contrast is
graded in one session by one judge.
The judge prompt is in `scripts/judge_dmr_derisk_correctness.py`. The reader's system prompt is in
`src/eccv26/eval.py` (`_system_prompt_for_dataset`). Memory selection, packing and the memory prefixes are in
`scripts/run_dmr_oracle_scratchpad_derisk.py` and `src/eccv26/utils/memory_controls.py`.

Run the unit tests with `PYTHONPATH=src python -m pytest tests`.
