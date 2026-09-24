# Selection / localization controls — text, or a pointer to where to look?

Same 1173 samples per task, same 16 ColQwen2 pages as images, same OCR block store, stock Qwen3-VL-2B, every arm graded in one judge session. `lex16n` is the paper's OCR memory arm with the relevance hint removed from the prompt wording; `hilite`/`crops` show the reader exactly the blocks `lex16n` packs, but as red boxes / crops on the page images and no text.


## VMQAR (n=1173; not yet judged: cropsnat)

| arm | strict | Δ vs images | 95% CI | p |
|---|---|---|---|---|
| images | 0.154 | — | | |
| +OCR memory (paper wording) | 0.293 | +0.139 | [+0.114, +0.163] | 6.8e-26 |
| +lexical-16 (neutral wording) | 0.295 | +0.141 | [+0.116, +0.165] | 4.4e-26 |
| +random-16 blocks | 0.203 | +0.049 | [+0.029, +0.068] | 2.0e-06 |
| +random blocks, char-matched | 0.222 | +0.067 | [+0.047, +0.088] | 3.7e-10 |
| +all blocks, reading order | 0.333 | +0.179 | [+0.153, +0.205] | 4.8e-36 |
| red boxes on pages (no text) | 0.159 | +0.005 | [-0.014, +0.024] | 0.663 |
| crops at reader scale (no text) | 0.277 | +0.123 | [+0.098, +0.147] | 4.1e-21 |

**Key contrasts** (paired, McNemar):

| question | contrast | Δ | 95% CI | p |
|---|---|---|---|---|
| wording | ocrmem − lex16n | -0.002 | [-0.016, +0.012] | 0.905 |
| query-conditioning (block-matched) | lex16n − rand16n | +0.092 | [+0.068, +0.116] | 5.4e-13 |
| query-conditioning (char-matched) | lex16n − randcn | +0.073 | [+0.050, +0.097] | 2.2e-09 |
| selection vs all text | lex16n − alln | -0.038 | [-0.064, -0.012] | 0.005 |
| text vs pointer (boxes) | lex16n − hilite | +0.136 | [+0.111, +0.160] | 1.6e-24 |
| text vs pointer (crops) | lex16n − crops | +0.018 | [-0.007, +0.043] | 0.180 |
| pointer alone (boxes) | hilite − pages | +0.005 | [-0.014, +0.024] | 0.663 |
| pointer alone (crops) | crops − pages | +0.123 | [+0.098, +0.147] | 4.1e-21 |

Share of the `lex16n` gain recovered by `hilite` alone: **0.04** (+0.005 of +0.141).

Share of the `lex16n` gain recovered by `crops` alone: **0.87** (+0.123 of +0.141).

**Stratified by how many gold blocks the packed memory holds** (page-union token coverage ≥ 0.9) (all n=176, part n=411, none n=586; images-arm strict acc all=0.312, part=0.175, none=0.092):

| stratum | arm | Δ vs images | 95% CI | p |
|---|---|---|---|---|
| gold all packed | +OCR memory (paper wording) | +0.227 | [+0.149, +0.305] | 3.0e-07 |
| gold all packed | +lexical-16 (neutral wording) | +0.244 | [+0.170, +0.319] | 1.5e-08 |
| gold all packed | +random-16 blocks | +0.045 | [-0.013, +0.104] | 0.186 |
| gold all packed | +random blocks, char-matched | +0.080 | [+0.018, +0.141] | 0.022 |
| gold all packed | +all blocks, reading order | +0.148 | [+0.077, +0.218] | 1.6e-04 |
| gold all packed | red boxes on pages (no text) | -0.011 | [-0.066, +0.043] | 0.838 |
| gold all packed | crops at reader scale (no text) | +0.199 | [+0.127, +0.271] | 1.2e-06 |
| gold part packed | +OCR memory (paper wording) | +0.163 | [+0.119, +0.207] | 2.1e-11 |
| gold part packed | +lexical-16 (neutral wording) | +0.158 | [+0.113, +0.204] | 1.9e-10 |
| gold part packed | +random-16 blocks | +0.061 | [+0.025, +0.097] | 0.002 |
| gold part packed | +random blocks, char-matched | +0.068 | [+0.031, +0.105] | 6.1e-04 |
| gold part packed | +all blocks, reading order | +0.209 | [+0.164, +0.254] | 2.9e-16 |
| gold part packed | red boxes on pages (no text) | +0.017 | [-0.021, +0.055] | 0.457 |
| gold part packed | crops at reader scale (no text) | +0.134 | [+0.089, +0.178] | 3.0e-08 |
| gold none packed | +OCR memory (paper wording) | +0.096 | [+0.066, +0.125] | 1.2e-09 |
| gold none packed | +lexical-16 (neutral wording) | +0.097 | [+0.067, +0.127] | 1.2e-09 |
| gold none packed | +random-16 blocks | +0.041 | [+0.017, +0.065] | 0.001 |
| gold none packed | +random blocks, char-matched | +0.063 | [+0.038, +0.089] | 4.0e-06 |
| gold none packed | +all blocks, reading order | +0.167 | [+0.132, +0.202] | 5.6e-18 |
| gold none packed | red boxes on pages (no text) | +0.002 | [-0.020, +0.024] | 1.000 |
| gold none packed | crops at reader scale (no text) | +0.092 | [+0.062, +0.122] | 1.1e-08 |

## VNIAH (n=1173; not yet judged: cropsnat)

| arm | strict | Δ vs images | 95% CI | p |
|---|---|---|---|---|
| images | 0.346 | — | | |
| +OCR memory (paper wording) | 0.501 | +0.155 | [+0.127, +0.183] | 1.7e-24 |
| +lexical-16 (neutral wording) | 0.512 | +0.165 | [+0.138, +0.193] | 5.8e-28 |
| +random-16 blocks | 0.367 | +0.020 | [-0.004, +0.045] | 0.118 |
| +random blocks, char-matched | 0.400 | +0.054 | [+0.028, +0.080] | 8.5e-05 |
| +all blocks, reading order | 0.567 | +0.221 | [+0.191, +0.250] | 1.1e-40 |
| red boxes on pages (no text) | 0.316 | -0.030 | [-0.056, -0.004] | 0.031 |
| crops at reader scale (no text) | 0.498 | +0.152 | [+0.123, +0.181] | 8.1e-23 |

**Key contrasts** (paired, McNemar):

| question | contrast | Δ | 95% CI | p |
|---|---|---|---|---|
| wording | ocrmem − lex16n | -0.010 | [-0.025, +0.005] | 0.213 |
| query-conditioning (block-matched) | lex16n − rand16n | +0.145 | [+0.118, +0.172] | 8.0e-24 |
| query-conditioning (char-matched) | lex16n − randcn | +0.112 | [+0.085, +0.138] | 8.5e-16 |
| selection vs all text | lex16n − alln | -0.055 | [-0.084, -0.027] | 1.7e-04 |
| text vs pointer (boxes) | lex16n − hilite | +0.195 | [+0.167, +0.224] | 3.2e-35 |
| text vs pointer (crops) | lex16n − crops | +0.014 | [-0.013, +0.040] | 0.339 |
| pointer alone (boxes) | hilite − pages | -0.030 | [-0.056, -0.004] | 0.031 |
| pointer alone (crops) | crops − pages | +0.152 | [+0.123, +0.181] | 8.1e-23 |

Share of the `lex16n` gain recovered by `hilite` alone: **-0.18** (-0.030 of +0.165).

Share of the `lex16n` gain recovered by `crops` alone: **0.92** (+0.152 of +0.165).

**Stratified by how many gold blocks the packed memory holds** (page-union token coverage ≥ 0.9) (all n=432, part n=0, none n=741; images-arm strict acc all=0.491, none=0.262):

| stratum | arm | Δ vs images | 95% CI | p |
|---|---|---|---|---|
| gold all packed | +OCR memory (paper wording) | +0.229 | [+0.180, +0.279] | 1.5e-16 |
| gold all packed | +lexical-16 (neutral wording) | +0.255 | [+0.206, +0.303] | 1.1e-19 |
| gold all packed | +random-16 blocks | +0.023 | [-0.023, +0.069] | 0.373 |
| gold all packed | +random blocks, char-matched | +0.049 | [+0.001, +0.096] | 0.058 |
| gold all packed | +all blocks, reading order | +0.183 | [+0.134, +0.232] | 9.4e-12 |
| gold all packed | red boxes on pages (no text) | -0.035 | [-0.085, +0.015] | 0.203 |
| gold all packed | crops at reader scale (no text) | +0.192 | [+0.141, +0.244] | 9.8e-12 |
| gold none packed | +OCR memory (paper wording) | +0.112 | [+0.078, +0.146] | 4.5e-10 |
| gold none packed | +lexical-16 (neutral wording) | +0.113 | [+0.080, +0.146] | 1.2e-10 |
| gold none packed | +random-16 blocks | +0.019 | [-0.009, +0.047] | 0.223 |
| gold none packed | +random blocks, char-matched | +0.057 | [+0.026, +0.087] | 4.8e-04 |
| gold none packed | +all blocks, reading order | +0.243 | [+0.206, +0.280] | 1.2e-30 |
| gold none packed | red boxes on pages (no text) | -0.027 | [-0.057, +0.003] | 0.091 |
| gold none packed | crops at reader scale (no text) | +0.128 | [+0.094, +0.162] | 2.1e-12 |
