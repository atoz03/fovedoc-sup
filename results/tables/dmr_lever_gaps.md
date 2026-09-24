# Lever coverage — what actually closes the multi-hop join, under one judge

Every arm below is scored by a single grader in one session, paired on the full n=1173, McNemar
with continuity correction. This matters more than usual here: the effects are 1–6 points and the
judge-noise floor is ~1 point per 100, so the June figures — measured against a judge model that
no longer resolves — could not be reused.

Baselines: `naive4b` = stock Qwen3-VL-4B with lexical text memory, greedy. `naive2b` = the same
at 2B (the parity run's `pdfmem` arm).

## V-MQAR (multi-hop)

| lever | arm | base | Δ | 95% CI | p |
|---|---|---|---|---|---|
| A — 32 blocks instead of 16 | 0.561 | 0.546 | +0.014 | [−0.004, +0.033] | 0.16 |
| A — union selector | 0.552 | 0.546 | +0.005 | [−0.012, +0.023] | 0.63 |
| B1 — reasoning reader (vs greedy base) | 0.556 | 0.546 | +0.009 | [−0.014, +0.033] | 0.48 |
| *decode control* — naive memory, B1's sampling | 0.529 | 0.546 | −0.017 | [−0.032, −0.002] | 0.034 |
| **B1 — reasoning reader (vs decode-matched)** | 0.556 | 0.529 | **+0.026** | [+0.002, +0.050] | 0.037 |
| **B2 — self-ask, 2-call** | 0.503 | 0.546 | **−0.043** | [−0.066, −0.021] | 2.4e-4 |
| C — memory-join LoRA, 4B | 0.555 | 0.546 | +0.009 | [−0.012, +0.029] | 0.46 |
| **C — memory-join LoRA, 2B** | 0.492 | 0.432 | **+0.060** | [+0.034, +0.086] | 9.2e-6 |

## V-NIAH (single needle) — new coverage

| lever | arm | base | Δ | 95% CI | p |
|---|---|---|---|---|---|
| **B1 — reasoning reader** | 0.767 | 0.790 | **−0.023** | [−0.042, −0.004] | 0.022 |
| **C — memory-join LoRA, 2B** | 0.739 | 0.691 | **+0.048** | [+0.026, +0.070] | 3.1e-5 |

## What this says

**The decode control changes B1's story rather than confirming it.** Against its original greedy
baseline B1 gains only +0.009 (n.s.). But sampling at B1's own settings *costs* −0.017 on its own,
so once the baseline is decoding-matched the reasoning contribution is +0.026. The June number was
right in magnitude and wrong in attribution: it was crediting reasoning with a gain that a matched
baseline shows is partly cancelled decoding loss.

**B1 is join-specific; C is not.** B1 helps multi-hop (+0.026 decode-matched) and *hurts*
single-needle (−0.023) — a clean dissociation, and exactly what the missing V-NIAH runs were for.
C-LoRA-2B helps both (+0.060, +0.048), so it is a general reading gain rather than a join-closer,
and it does not survive scaling to 4B (+0.009, n.s.).

**Self-ask is harmful.** B2's 2-call decomposition is −0.043 against single-call. Its June write-up
reported n=981 because 192 rows were lost to a judge outage; re-judged over the full 1173 the
result is worse, not better.

**Multiplicity.** Ten tests are reported. At a Bonferroni-corrected α = 0.005 only three survive:
B2 (2.4e-4), C-2B on V-MQAR (9.2e-6) and C-2B on V-NIAH (3.1e-5). The decode-matched B1 effect
(p = 0.037), the decoding penalty (0.034) and B1's V-NIAH harm (0.022) are suggestive at
conventional α but do not survive correction — treat them as directional.

**The levers are small next to the thing they sit on.** The memory-vs-images gap is +0.13 to +0.28
(OCR-parity table) and +0.19 to +0.61 across reader families. The best lever here is +0.06. Whatever
is limiting these readers, it is not addressed by more blocks, a union selector, self-ask
decomposition, or a reasoning trace — and a 2B LoRA recovers more than any of them.
