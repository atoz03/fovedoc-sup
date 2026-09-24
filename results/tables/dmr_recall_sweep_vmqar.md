# Controlled retrieval-degradation sweep — VMQAR

Same 938 samples at every level, page budget fixed at 16, only the number of gold pages present varies. Both arms see identical page sets within a level, so the comparison is paired twice over: across arms and across levels.

| recall | gold pages present | images | +memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| 1.00 | 2/2 | 0.142 | 0.267 | +0.125 | [+0.099, +0.151] | 4.54e-19 |
| 0.50 | 1/2 | 0.112 | 0.178 | +0.066 | [+0.045, +0.087] | 1.54e-09 |
| 0.00 | 0/2 | 0.055 | 0.085 | +0.030 | [+0.014, +0.045] | 0.000309 |

**Interaction (recall 1.00 vs 0.00), paired within sample:** the memory advantage changes by **+0.095** [+0.067, +0.123], p = 1.67e-11. This is the quantity that says whether the advantage is bought by the evidence being present rather than by the format of the prompt.

At zero recall the memory arm is +0.030 against the image arm. A non-negative value here means part of the memory benefit is format rather than evidence delivery, and survives the evidence being removed entirely.

