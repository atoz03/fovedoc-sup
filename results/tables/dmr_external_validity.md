# External validity — does the memory gap transfer to MMLongBench-Doc?

Stock Qwen3-VL-2B, ColQwen2 top-16, OCR memory, n=1091 paired. Both arms see the *identical* retrieved page set (verified per-id), so the only difference is whether the extracted-text memory is in the prompt. Strict judge correctness, McNemar with continuity correction, single grader in one session.

## Headline

| stratum | n | images | +memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| all questions *(pooled — do not cite alone)* | 1091 | 0.180 | 0.193 | +0.014 | [-0.007, +0.035] | 0.235 |
| **answerable, gold-annotated** | 838 | 0.220 | 0.223 | +0.004 | [-0.022, +0.029] | 0.855 |
|   ⤷ full recall@16 | 706 | 0.227 | 0.242 | +0.016 | [-0.012, +0.043] | 0.32 |
|   ⤷ partial recall | 132 | 0.182 | 0.121 | -0.061 | [-0.123, +0.002] | 0.099 |
| unanswerable *(abstention)* | 244 | 0.041 | 0.090 | +0.049 | [+0.014, +0.085] | 0.0139 |

*The CI and the p-value come from different estimators — a Wald interval on the paired difference, and McNemar with a continuity correction, which is conservative. They can therefore disagree marginally (the 31–60 page row below is the case here). Where they do, the p-value is the one to trust.*

## By document length

Motivated by the retrieval curve, which is monotone in length (0.938 at ≤30 pages → 0.644 above 120).

| length | n | images | +memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| 0-30 pages | 430 | 0.221 | 0.251 | +0.030 | [-0.006, +0.067] | 0.137 |
| 31-60 pages | 190 | 0.242 | 0.184 | -0.058 | [-0.115, -0.001] | 0.0725 |
| 61-120 pages | 173 | 0.197 | 0.191 | -0.006 | [-0.052, +0.041] | 1 |
| >120 pages | 45 | 0.200 | 0.244 | +0.044 | [-0.061, +0.150] | 0.683 |

## By evidence source (the benchmark's own taxonomy)

Questions carry one or more evidence-source tags, so these strata overlap and the rows do not sum to the stratum n. The axis is pre-registered as the mechanism axis for a text-memory method; the rows are ordered by n, not by effect.

| evidence | n | images | +memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| Pure-text (Plain-text) | 288 | 0.198 | 0.250 | +0.052 | [+0.008, +0.096] | 0.0328 |
| Figure | 285 | 0.239 | 0.211 | -0.028 | [-0.074, +0.017] | 0.291 |
| Table | 214 | 0.150 | 0.168 | +0.019 | [-0.022, +0.060] | 0.502 |
| Chart | 177 | 0.147 | 0.107 | -0.040 | [-0.095, +0.016] | 0.23 |
| Generalized-text (Layout) | 116 | 0.267 | 0.276 | +0.009 | [-0.065, +0.082] | 1 |

No single row is significant, and they overlap, so they cannot carry a mechanism claim by themselves. The direct contrast can: restricted to the 414 **single-tag** questions, the per-question paired effect is +0.085 on Pure-text (n=129) against -0.046 on Chart+Figure (n=285) — a difference of **+0.131** [+0.046, +0.215], z = 3.03, **p = 0.0024**. That is the one test here that survives correction. Disclosure: the axis was pre-registered, but pooling Chart with Figure was chosen after seeing the per-stratum directions, so it is counted as exploratory below.

## Abstention: discrimination or response bias?

A memory arm that simply answers less often would gain on the unanswerable stratum for free. Signal-detection view: *hit* = abstains on an unanswerable question, *false alarm* = abstains on an answerable one. (Abstention is detected by surface pattern on `pred`, so read these as rates on a consistent rule, not as exact counts.)

| arm | hit rate (n=244) | false-alarm rate (n=838) | balanced acc. |
|---|---|---|---|
| images | 0.029 | 0.001 | 0.514 |
| +memory | 0.086 | 0.008 | 0.539 |

The hit rate roughly triples while the false-alarm rate moves by well under a point: of the 6 answerable questions where only the memory arm abstains, the image arm answered 0 correctly. So this is improved discrimination, not a bias shift — though both arms are poor at abstention in absolute terms.

Testing the abstention *behaviour* directly rather than through the judge — McNemar on the abstain indicator over the 244 unanswerable questions — gives b=18, c=4, p = 0.005578, i.e. the same direction somewhat more strongly than the correctness view. It is still not below the corrected threshold.

**This is not a replication of anything.** FoveDoc contains 1–2 unanswerable items in 1173, so V-MQAR and V-NIAH could not have measured abstention at all. This is a first observation, on one benchmark, at p just above a corrected threshold.

## Multiplicity

15 tests are reported, in two families. **Primary** (4 tests, α = 0.0125) are the strata named in `scripts/run_dmr_judging.sh` before any external verdict existed. **Exploratory** (11 tests, α = 0.0045) is everything else. Pooled over all 15, α = 0.0033.

Primary — **nothing survives α = 0.0125.**

Exploratory — survives α = 0.0045:
- **Pure-text vs Chart+Figure (single-tag only)**: Δ = +0.131, p = 0.00241

Directional only (p < 0.05 but above the corrected threshold) — these are hypotheses with a sign, not findings:
- unanswerable (abstention, not reading): Δ = +0.049, p = 0.0139
- evidence Pure-text (Plain-text): Δ = +0.052, p = 0.0328

