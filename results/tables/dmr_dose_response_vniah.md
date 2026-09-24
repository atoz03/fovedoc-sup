# Fidelity dose-response — does the OCR arm fail where OCR lost the evidence?

Dose axis = per-sample OCR best-block recall of the gold evidence, aggregated across a
sample's evidence items with `min` (a multi-hop sample is only answerable if
*every* item survived). Correctness metric = `judge`. n=1170 samples
common to the fidelity dump and all 3 arms.

| OCR fidelity | n | ocr | pdf | img |
|---|---|---|---|---|
| [0, .25) | 115 | 0.383 | 0.591 | 0.365 |
| [.25, .5) | 127 | 0.441 | 0.661 | 0.307 |
| [.5, .75) | 193 | 0.389 | 0.622 | 0.223 |
| [.75, 1) | 188 | 0.463 | 0.691 | 0.277 |
| 1.0 | 547 | 0.607 | 0.748 | 0.428 |
| **all** | **1170** | **0.508** | **0.693** | **0.350** |

## Paired delta, ocr − pdf, by fidelity bin

| OCR fidelity | n | delta | interpretation |
|---|---|---|---|
| [0, .25) | 115 | -0.209 | |
| [.25, .5) | 127 | -0.220 | |
| [.5, .75) | 193 | -0.233 | |
| [.75, 1) | 188 | -0.229 | |
| 1.0 | 547 | -0.141 | |
| **all** | **1170** | **-0.185** | |

A delta that shrinks toward zero as fidelity rises means the ocr arm's loss is the
recogniser, not the memory design. A flat delta means it is not.


## By source family

| family | n | mean OCR fidelity | ocr | pdf | img |
|---|---|---|---|---|---|
| wipo | 352 | 0.605 | 0.466 | 0.642 | 0.310 |
| cloudflare | 340 | 0.865 | 0.685 | 0.821 | 0.582 |
| registry | 175 | 0.787 | 0.423 | 0.589 | 0.200 |
| arxiv | 171 | 0.861 | 0.374 | 0.649 | 0.199 |
| faa | 72 | 0.770 | 0.500 | 0.694 | 0.264 |
| bis | 22 | 0.686 | 0.273 | 0.727 | 0.182 |
| long | 11 | 0.561 | 0.636 | 0.545 | 0.182 |
| eib | 10 | 0.558 | 0.300 | 0.700 | 0.200 |
| aws | 4 | 0.814 | 0.750 | 1.000 | 0.500 |
| corp | 3 | 0.607 | 0.000 | 0.333 | 0.667 |
| global | 2 | 0.700 | 0.500 | 1.000 | 0.000 |
| unctad | 2 | 0.745 | 0.500 | 0.500 | 0.500 |
| development | 2 | 0.731 | 0.500 | 1.000 | 0.500 |
| intl | 1 | 0.588 | 0.000 | 1.000 | 0.000 |
| who | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| noaa | 1 | 0.571 | 1.000 | 1.000 | 1.000 |
| lenovo | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
