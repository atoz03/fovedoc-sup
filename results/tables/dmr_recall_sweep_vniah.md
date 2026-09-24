# Controlled retrieval-degradation sweep — VNIAH

Same 1025 samples at every level, page budget fixed at 16, only the number of gold pages present varies. Both arms see identical page sets within a level, so the comparison is paired twice over: across arms and across levels.

| recall | gold pages present | images | +memory | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| 1.00 | 1/1 | 0.340 | 0.490 | +0.149 | [+0.119, +0.180] | 1.22e-19 |
| 0.00 | 0/1 | 0.138 | 0.180 | +0.043 | [+0.022, +0.064] | 9.9e-05 |

**Interaction (recall 1.00 vs 0.00), paired within sample:** the memory advantage changes by **+0.106** [+0.073, +0.140], p = 5.88e-10. This is the quantity that says whether the advantage is bought by the evidence being present rather than by the format of the prompt.

At zero recall the memory arm is +0.043 against the image arm. A non-negative value here means part of the memory benefit is format rather than evidence delivery, and survives the evidence being removed entirely.

