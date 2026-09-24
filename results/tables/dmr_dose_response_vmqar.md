# Fidelity dose-response — does the OCR arm fail where OCR lost the evidence?

Dose axis = per-sample OCR best-block recall of the gold evidence, aggregated across a
sample's evidence items with `min` (a multi-hop sample is only answerable if
*every* item survived). Correctness metric = `judge`. n=1172 samples
common to the fidelity dump and all 3 arms.

| OCR fidelity | n | ocr | pdf | img |
|---|---|---|---|---|
| [0, .25) | 188 | 0.170 | 0.383 | 0.128 |
| [.25, .5) | 202 | 0.257 | 0.470 | 0.168 |
| [.5, .75) | 227 | 0.233 | 0.339 | 0.088 |
| [.75, 1) | 209 | 0.268 | 0.426 | 0.134 |
| 1.0 | 346 | 0.416 | 0.503 | 0.214 |
| **all** | **1172** | **0.288** | **0.433** | **0.154** |

## Paired delta, ocr − pdf, by fidelity bin

| OCR fidelity | n | delta | interpretation |
|---|---|---|---|
| [0, .25) | 188 | -0.213 | |
| [.25, .5) | 202 | -0.213 | |
| [.5, .75) | 227 | -0.106 | |
| [.75, 1) | 209 | -0.158 | |
| 1.0 | 346 | -0.087 | |
| **all** | **1172** | **-0.145** | |

A delta that shrinks toward zero as fidelity rises means the ocr arm's loss is the
recogniser, not the memory design. A flat delta means it is not.


## By source family

| family | n | mean OCR fidelity | ocr | pdf | img |
|---|---|---|---|---|---|
| wipo | 353 | 0.475 | 0.232 | 0.380 | 0.125 |
| cloudflare | 340 | 0.762 | 0.447 | 0.571 | 0.297 |
| registry | 175 | 0.675 | 0.200 | 0.297 | 0.057 |
| arxiv | 171 | 0.744 | 0.193 | 0.351 | 0.070 |
| faa | 73 | 0.697 | 0.288 | 0.507 | 0.110 |
| bis | 22 | 0.565 | 0.227 | 0.500 | 0.045 |
| long | 11 | 0.460 | 0.182 | 0.545 | 0.182 |
| eib | 10 | 0.401 | 0.100 | 0.500 | 0.000 |
| aws | 4 | 0.764 | 0.750 | 0.750 | 0.000 |
| corp | 3 | 0.602 | 0.000 | 0.333 | 0.000 |
| global | 2 | 0.700 | 0.500 | 0.500 | 0.000 |
| unctad | 2 | 0.745 | 0.500 | 0.000 | 0.500 |
| development | 2 | 0.731 | 0.000 | 0.500 | 0.500 |
| lenovo | 1 | 0.571 | 0.000 | 0.000 | 0.000 |
| who | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| noaa | 1 | 0.571 | 1.000 | 0.000 | 0.000 |
| intl | 1 | 0.588 | 0.000 | 1.000 | 0.000 |
