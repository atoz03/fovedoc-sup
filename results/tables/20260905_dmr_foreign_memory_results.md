# Foreign-memory control: the zero-recall residual is topical context, not format (2026-09-05)

> Closes the question left open by `20260903_dmr_recall_sweep_results.md`. Reader: stock
> Qwen3-VL-2B-Instruct. Pages: ColQwen2 top-16, degraded exactly as in the sweep. Memory: RapidOCR,
> 100% OCR, but drawn from a **different document**. All four files judged in one session,
> 0 error rows after repair.
> Table: `results/tables/dmr_foreign_memory.md`. Script: `scripts/dmr_foreign_memory_analysis.py`.

## The question

The recall sweep showed the memory advantage is dose-dependent in retrieval recall but does **not**
fall to zero: with no gold page retrieved at all, the memory arm was still ahead. The leakage
control removed one explanation (a stray copy of the answer string) and the residual survived at
+0.028 on both tasks. But "format" — text simply being easier for the reader to consume than
pixels — was never established, because leak-free memory is still text of the *same document*.
Topical relatedness explains the residual just as well and implies far less.

These two are separated only by memory from a **different** document: it can neither leak the
answer nor be topically related, so it isolates what a text memory is worth for its form alone.

## Design

Nothing changes but whose text is in the memory. `--memory-document-root` already redirects the
lookup, so no harness patch was needed: `scripts/dmr_make_foreign_memory_store.py` writes a
deranged OCR store where each `{doc_id}.json` holds another document's blocks with page ids
remapped so resolution still succeeds. 1173 documents, **0 self-maps, 0 same-family pairs**;
verified on 4800 real (doc, page) lookups with 0 missing page ids and 0 empty pages. The only text
shared with the document's own page is bare numerals ("10", "200"), maximum 13 characters.

Only the `pages_memory` arm was run: in mode `pages` the memory is never read, so the image arm is
bit-identical by construction, and generation on this harness is deterministic (verified: the
sweep's drop0 `pages` arm reproduced the parity run byte-identically on all 1963 ids). Re-running
it would have burned 8 GPUs to reproduce a file we already have.

## Result: format buys nothing

Zero recall, leak-free subsets (leakage detected exactly as in the sweep's control, ≥6-character
normalised gold string in the packed own-document memory):

| task | n | own − images | **foreign − images** | own − foreign |
|---|---|---|---|---|
| V-MQAR | 893 | +0.028 [+0.013,+0.043] p=6.1e-04 | **+0.000** [−0.013,+0.013] p=0.87 | +0.028 [+0.012,+0.044] p=9.8e-04 |
| V-NIAH | 898 | +0.028 [+0.009,+0.047] p=5.6e-03 | **−0.009** [−0.027,+0.009] p=0.40 | +0.037 [+0.017,+0.056] p=3.2e-04 |

**Foreign text is worth nothing at all.** On neither task does it differ from showing the images
alone. The decomposition is therefore clean: of the leak-free zero-recall residual, the
content-independent (format) component is ~0 and the whole of it is same-document topical context.

`own − images ≈ own − foreign` in both rows, which is the same statement read a second way.

**A text memory confers no content-independent advantage.** The earlier campaign claim — from the
OCR-parity fidelity bins, where the OCR arm beat the image arm even in the worst bin — that "part
of the memory benefit is format, not evidence delivery" is **falsified by direct control** and must
not be repeated. That argument had the same unmeasured leakage hole this control closes.

## Irrelevant text is inert, not distracting

At **full** recall, foreign memory is also indistinguishable from images: +0.019 (p=0.09) on
V-MQAR, −0.017 (p=0.21) on V-NIAH. A reader handed a page of unrelated prose beside the correct
images simply ignores it.

This sharpens the external-validity finding. On MMLongBench the memory arm was *actively harmful*
on chart and figure evidence (−0.046) — and that memory was of the **correct** pages. So what
damages the reader is not irrelevant text but relevant text about the wrong modality: a faithful
transcription of a chart's axis labels and stray numbers, competing with the figure that actually
answers the question.

## Judge provenance — and a dialect change that had to be measured

The foreign files were graded by `gpt-5.6-luna` at `reasoning.effort=medium` over `/responses` on a
new proxy. The sweep arms they are compared against were graded by the same model name at the same
(default) effort, but over `/chat/completions` on the **old** proxy, which no longer serves it.

Every contrast *within* the sweep cancels that. `own − foreign` does not — it is a between-file
contrast. So it was measured rather than assumed, by re-grading 250 rows of each own zero-recall
arm through the new endpoint and comparing with the stored verdicts:

| task | n | three-way agreement | strict (chat) | strict (responses) | Δ strict | discordant |
|---|---|---|---|---|---|---|
| V-MQAR | 248 | 0.948 | 0.0887 | 0.0887 | **+0.0000** | b=0, c=0 |
| V-NIAH | 249 | 0.976 | 0.1807 | 0.1847 | **+0.0040** | b=0, c=1 |

On the metric the paper reports — strict accuracy, `correct` only — the two dialects are
effectively identical: **one discordant pair in 497 rows**. On V-MQAR they agree exactly. The 0.948
three-way figure is disagreement on the `partial`/`incorrect` boundary, which strict accuracy does
not touch. The +0.028 / +0.037 contrasts are an order of magnitude larger and safe against this.

**And the residual sign is conservative.** `/responses` grades marginally *more* generously
(+0.004 on V-NIAH, +0.000 on V-MQAR), and it is the **foreign** arm that ran there while the own
arm was graded on `chat`. Any dialect bias therefore inflates the foreign arm and *shrinks*
`own − foreign` — so +0.028 / +0.037 are lower bounds with respect to this confound, not upper
ones.

> Note the general lesson: three-way agreement understates stability for a strict-accuracy paper,
> because it charges for disagreements on a class the metric discards. Report the discordant-pair
> count on the reported metric.

## Operational notes

- **Killing the judge worker does not stop the runner.** `pkill -f judge_dmr_derisk_correctness`
  killed the python process but not its parent `bash run_dmr_judging.sh`, which advanced to the
  next file and spawned a fresh worker. A stale and a new runner then raced over the same four
  output paths. Kill the parent first. No data was harmed here only because both runners carried
  an identical grader config.
- **Cloudflare rejects urllib.** The new proxy returns HTTP 403 `error code: 1010` to
  `User-Agent: Python-urllib/3.x` — before the request reaches the model — which is
  indistinguishable from a dead endpoint. The judge now sends an explicit UA.
- **Only the nested reasoning form works.** `{"reasoning":{"effort":...}}` is honoured;
  top-level `{"reasoning_effort":...}` returns 200 and is silently ignored.
- **`repair` used to leave the summary stale.** It spliced verdicts into the `.jsonl` and never
  refreshed `.summary.json`, so a fully repaired file still advertised its pre-repair `n_error`.
  Fixed via `dmr_rejudge_errors.py splice --summary`.

## Provenance

- Predictions: `code/runs/20260904/foreign_memory/eval/{vmqar_drop0,vmqar_drop2,vniah_drop0,vniah_drop1}/merged/`
- Verdicts: same paths under `judged/`, 3926/3926 judged, **0 error rows**, every summary recording
  `judge_model: gpt-5.6-luna`, `judge_api: responses`, `judge_reasoning_effort: medium`
- Reproduce: `bash scripts/run_dmr_foreign_memory.sh`, then
  `DMR_JUDGE_API=responses DMR_JUDGE_REASONING_EFFORT=medium bash scripts/run_dmr_judging.sh foreign`,
  then `repair`, then `scripts/dmr_foreign_memory_analysis.py`
