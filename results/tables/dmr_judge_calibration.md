# Judge calibration — gpt-5.6-luna against gpt-5.4 on identical predictions

gpt-5.4 graded the parity, grid, lever and external tables. It is no longer served by any reachable proxy, so the recall sweep is graded by **gpt-5.6-luna**. The sweep's drop0 `pages` arm reproduced the parity condition exactly — the predictions are byte-identical on all 1963 ids — so the two judges can be compared directly, with the reader, the pages and the generated strings all held fixed.

| task | n | gpt-5.4 strict | gpt-5.6-luna strict | Δ | 95% CI | 3-way agree | strict agree | McNemar p |
|---|---|---|---|---|---|---|---|---|
| VMQAR | 938 | 0.140 | 0.142 | +0.002 | [-0.004, +0.008] | 0.954 | 0.991 | 0.724 |
| VNIAH | 1025 | 0.343 | 0.340 | -0.003 | [-0.010, +0.004] | 0.968 | 0.985 | 0.606 |
| **pooled** | **1963** | — | — | **-0.001** | [-0.005, +0.004] | — | **0.988** | 1 |

Discordant pairs: 11 where gpt-5.6-luna alone says correct, 12 where gpt-5.4 alone does.

**How to use this number.** It bounds the judge's contribution to a *level* of accuracy, which is why no sweep accuracy may be placed beside a parity accuracy. It does **not** propagate into the sweep's difference-in-differences: that contrast is computed within one judge, over arms and levels graded in a single session, so a constant judge offset cancels. What survives is the judge's random disagreement, and the strict-agreement rate above is the estimate of it.
