# Reader-family generalization — the retrieval–reading gap is not a Qwen3-VL quirk

Six stock readers from three model families, each given ColQwen2's top-16 retrieved pages two
ways: as page **images**, or as an extracted-text **memory** of those same pages. Held fixed
across families: retrieval, memory construction, prompts, block budget, greedy decoding, and the
evaluated samples (first 400 per task). Judged by one grader in one session; `strict` counts only
`verdict == "correct"`.

| reader | family | task | images | memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | Qwen3-VL | V-MQAR | 0.198 | 0.480 | **+0.282** | [+0.227, +0.338] | 2.8e-23 |
| Qwen3-VL-2B | Qwen3-VL | V-NIAH | 0.403 | 0.762 | **+0.360** | [+0.297, +0.423] | 1.3e-28 |
| Qwen3-VL-4B | Qwen3-VL | V-MQAR | 0.318 | 0.568 | **+0.250** | [+0.195, +0.305] | 2.1e-18 |
| Qwen3-VL-4B | Qwen3-VL | V-NIAH | 0.522 | 0.830 | **+0.307** | [+0.249, +0.366] | 1.9e-24 |
| Qwen3-VL-8B | Qwen3-VL | V-MQAR | 0.378 | 0.608 | **+0.230** | [+0.180, +0.280] | 9.7e-19 |
| Qwen3-VL-8B | Qwen3-VL | V-NIAH | 0.583 | 0.807 | **+0.225** | [+0.173, +0.277] | 4.1e-17 |
| Qwen2.5-VL-7B | Qwen2.5-VL | V-MQAR | 0.395 | 0.583 | **+0.188** | [+0.139, +0.236] | 5.8e-14 |
| Qwen2.5-VL-7B | Qwen2.5-VL | V-NIAH | 0.605 | 0.840 | **+0.235** | [+0.185, +0.285] | 7.6e-20 |
| Idefics3-8B | Idefics3 | V-MQAR | 0.115 | 0.535 | +0.420 | [+0.354, +0.486] | 7.9e-35 |
| Idefics3-8B | Idefics3 | V-NIAH | 0.200 | 0.812 | +0.613 | [+0.534, +0.691] | 2.6e-52 |
| LLaVA-OneVision-7B | LLaVA-OV | V-MQAR | 0.215 | 0.507 | +0.292 | [+0.234, +0.351] | 1.5e-22 |
| LLaVA-OneVision-7B | LLaVA-OV | V-NIAH | 0.383 | 0.767 | +0.385 | [+0.319, +0.451] | 8.2e-30 |

Paired within each model (McNemar, continuity-corrected, n=400). **Every reader, every task, at
p < 1e-13.** The gap is a property of current VLMs reading document pages, not of one backbone.

## Two things this table must not be read as

**Absolute scores are not comparable across families, and neither are the Δ magnitudes.** Per-page
image tokenisation differs by construction: at 512² these models cost 267 (Qwen3-VL), 346
(Qwen2.5-VL), 3036 (Idefics3) and 3708 (LLaVA-OV) tokens per page. Fitting 16 pages into context
therefore required per-family accommodation — Idefics3 runs with `do_image_splitting=False`
(3036 → 184 tokens/page) and LLaVA-OV at a 147456-pixel cap. Those settings degrade the *image*
arm specifically, which inflates their Δ. Idefics3's image score of 0.115/0.200 should be read as
"Idefics3 with splitting disabled", not as Idefics3's capability. The claim each row supports is
the **within-model** sign and significance, not a ranking.

**The gap narrows with scale, but does not close.** Qwen3-VL is the one family here where scale
varies at identical settings:

| Qwen3-VL | V-MQAR Δ | V-NIAH Δ |
|---|---|---|
| 2B | +0.282 | +0.360 |
| 4B | +0.250 | +0.307 |
| 8B | +0.230 | +0.225 |

Monotone decreasing on both tasks — a stronger visual reader needs the text memory less, which is
the obvious "will this vanish with scale?" objection. At 8B it is still +0.23 at p < 1e-16, so on
this evidence it shrinks rather than disappears. Three points is not a scaling law, and this does
not license extrapolation past 8B.
