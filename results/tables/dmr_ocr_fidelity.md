# OCR fidelity of gold evidence — ceiling for the OCR-parity memory arm

Reference = born-digital PDF text layer (`data/documents`), candidate = RapidOCR over the
rendered page images (`code/runs/20260810/dmr_ocr_documents`). Tokenizer is the one the lexical block scorer uses.
*page recall* = fraction of gold-excerpt tokens present anywhere on the page; *best-block
recall* = the most any single packable block covers (what the memory can actually select).
`recoverable` = share of evidence items at or above 0.90 recall.

| task | n | src | page recall | page recoverable | best-block recall | block recoverable |
|---|---|---|---|---|---|---|
| vmqar | 2339 | PDF text layer | 1.000 | 1.000 | 1.000 | 1.000 |
| vmqar | 2339 | OCR | 0.825 | 0.624 | 0.762 | 0.529 |
| vniah | 1170 | PDF text layer | 1.000 | 1.000 | 1.000 | 1.000 |
| vniah | 1170 | OCR | 0.818 | 0.615 | 0.758 | 0.525 |

## By source family (OCR best-block recall)

| family | n | pdf best-block | ocr best-block | delta |
|---|---|---|---|---|
| wipo | 1057 | 1.000 | 0.613 | -0.387 |
| cloudflare | 1020 | 1.000 | 0.866 | -0.134 |
| registry | 525 | 1.000 | 0.777 | -0.223 |
| arxiv | 511 | 1.000 | 0.853 | -0.147 |
| faa | 217 | 1.000 | 0.790 | -0.210 |
| bis | 65 | 1.000 | 0.715 | -0.285 |
| long | 33 | 1.000 | 0.612 | -0.388 |
| eib | 30 | 1.000 | 0.552 | -0.448 |
| aws | 12 | 1.000 | 0.859 | -0.141 |
| corp | 9 | 1.000 | 0.655 | -0.345 |
| development | 6 | 1.000 | 0.779 | -0.221 |
| global | 6 | 1.000 | 0.703 | -0.297 |
| unctad | 6 | 1.000 | 0.830 | -0.170 |
| intl | 3 | 1.000 | 0.601 | -0.399 |
| lenovo | 3 | 1.000 | 0.857 | -0.143 |
| noaa | 3 | 1.000 | 0.619 | -0.381 |
| who | 3 | 1.000 | 1.000 | +0.000 |

## Block-budget parity

The two arms pack the same top-k blocks at the same per-block char cap, so if OCR blocks
are systematically shorter or more fragmented than PDF paragraphs the OCR arm is simply
budget-starved and fidelity is confounded with packing budget. These are the numbers that
decide whether a total-chars-matched variant is needed.

| src | blocks/page | chars/block (mean) | chars/block (median) | chars/page |
|---|---|---|---|---|
| PDF text layer | 53.4 | 53 | 6 | 2851 |
| OCR | 17.1 | 129 | 23 | 2192 |
