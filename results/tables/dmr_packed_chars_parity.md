# Block-budget parity between the two memory arms — Gate 4, settled

Both arms pack the top-16 blocks (16 *total* across all 16 retrieved pages, not 16 per page)
at a 300-character per-block cap. If the OCR store's blocks were systematically shorter or
more fragmented than PDF paragraphs, the OCR arm would simply be budget-starved and any
accuracy deficit would be confounded with how much text it got to see rather than with how
faithfully OCR recovered it.

Measured by replaying the harness's own `_build_memory_scratchpad` over both stores in a single
deterministic CPU pass (`scripts/dmr_dump_pdf_packed_chars.py`), so the two budgets are paired
per sample by construction. n=2340 samples with a non-empty memory, V-MQAR + V-NIAH, top-16.

| arm | packed chars mean | median | p10 | p90 | facts/sample |
|---|---|---|---|---|---|
| PDF text layer | 2934 | 3162 | 1126 | 4847 | 16.0 |
| OCR | 3826 | 4042 | 2429 | 4962 | 16.0 |

**OCR packs more than PDF on 1804/2340 samples (77.1%), mean paired delta +894 characters.**

So the confound runs the *opposite* way to the one the gate was written to catch: the OCR arm
is not starved, it is slightly better fed. A chars-matched variant is therefore unnecessary,
and any accuracy the OCR arm loses cannot be explained by having less text in the prompt.

The reason is visible in the per-block statistics: the born-digital PDF text layer is heavily
fragmented — 53.4 blocks per page at a **median of 6 characters** — so a fixed 16-block budget
buys the PDF arm a lot of scraps. RapidOCR's line-grouping produces 17.1 blocks per page at a
median of 23 characters, and 16 of those carry more text.

Note the p10 column: the PDF arm's worst decile packs 1126 characters against OCR's 2429. The
budget asymmetry is concentrated in exactly the samples where the text layer shatters worst.
