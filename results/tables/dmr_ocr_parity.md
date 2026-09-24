# OCR-parity control — does the retrieval–reading gap survive losing the free text layer?

Stock Qwen3-VL-2B-Instruct reading ColQwen2's top-16 retrieved pages (full-recall 0.998 V-MQAR /
1.000 V-NIAH, so retrieval is not the bottleneck). Three arms over the *same* pages and the
*same* 1173 samples per task, differing only in what the reader is given:

- **images** — the rendered pages as pixels;
- **OCR memory** — those same pixels **plus** top-16 text blocks recovered by RapidOCR (CPU) from them;
- **PDF text layer** — those same pixels **plus** top-16 blocks from the born-digital text layer.

**This is an addition, not a substitution.** All three arms carry the identical 16 page images
(`visual_tokens = 4032` in every arm); the memory arms only add packed text to the prompt. So the
comparison is not "text vs pixels" — it is "can the model use text it is handed, that it demonstrably
failed to read off an image already in its context". Do not describe it as replacing the images.

Judged by a single LLM grader in one session; `strict` counts only `verdict == "correct"`.

| task | images | **OCR memory** | PDF text layer |
|---|---|---|---|
| V-MQAR (multi-hop) | 0.153 | **0.287** | 0.432 |
| V-NIAH (single needle) | 0.350 | **0.506** | 0.691 |

Paired contrasts, McNemar with continuity correction, n=1173:

| contrast | task | Δ | 95% CI | p |
|---|---|---|---|---|
| **OCR memory − images** | V-MQAR | **+0.134** | [+0.109, +0.158] | 3.3e-26 |
| **OCR memory − images** | V-NIAH | **+0.157** | [+0.127, +0.187] | 2.8e-24 |
| OCR memory − PDF layer | V-MQAR | −0.145 | [−0.173, −0.116] | 4.6e-23 |
| OCR memory − PDF layer | V-NIAH | −0.185 | [−0.215, −0.155] | 6.9e-33 |
| PDF layer − images | V-MQAR | +0.279 | [+0.246, +0.311] | 6.1e-63 |
| PDF layer − images | V-NIAH | +0.342 | [+0.305, +0.379] | 2.6e-74 |

**Adding CPU-OCR text of the retrieved pages, on top of those same pages as images, improves
strict accuracy** by +0.134 and +0.157 at p < 1e-23. A CPU recogniser retains 48% (V-MQAR) and
46% (V-NIAH) of what a born-digital text layer buys, so roughly half the headline gap is
deployable today and roughly half depends on having a free text layer.

Three controls behind the number:

1. **Purity.** All 2346 OCR-arm records carry `memory_block_source: "ocr"`; a document missing
   from the OCR store is fatal, not silently backfilled from the PDF layer.
2. **Not a budget artefact.** The OCR arm packs *more* text than the PDF arm on 77.1% of samples
   (+894 chars mean), because the born-digital layer fragments into 53.4 blocks/page at a
   6-character median. At a fixed 16-block budget the OCR arm is better fed, not starved.
3. **No scanned stratum.** All 1173 documents are born-digital, so OCR has no stratum where it
   can win. This is a pure cost measured under conditions maximally favourable to the baseline.
