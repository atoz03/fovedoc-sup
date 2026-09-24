# Foreign-memory control — is the zero-recall benefit format or topical context?

Same samples, same page images, same retrieved page ids, same lexical packing and the same 16-block budget at every level. The only variable is **whose text the memory contains**: the document's own retrieved pages (`own`), or a different document's (`foreign`, deranged store with 0 self-maps and 0 same-family pairs). All arms graded by gpt-5.6-luna.


## VMQAR

| recall | arm | strict | Δ vs images | 95% CI | p |
|---|---|---|---|---|---|
| 1.00 | images | 0.142 | — | | |
| 1.00 | own | 0.267 | **+0.125** | [+0.098, +0.152] | 4.54e-19 |
| 1.00 | foreign | 0.161 | **+0.019** | [-0.001, +0.040] | 0.0859 |
| 0.00 | images | 0.055 | — | | |
| 0.00 | own | 0.085 | **+0.030** | [+0.014, +0.045] | 0.000309 |
| 0.00 | foreign | 0.054 | **-0.001** | [-0.014, +0.012] | 1 |

**At zero recall** (neither memory can contain the answer): own is +0.030 over images, foreign is -0.001. Own − foreign = **+0.031** [+0.015, +0.047], p = 0.000267.

Foreign text reproduces **-4%** of the own-document zero-recall effect. The closer that is to 100%, the more the residual is pure format — text being easier for the reader to consume than pixels, regardless of what it says. The closer to 0%, the more it was the document's own non-gold pages supplying topical context, and the weaker the format claim becomes.


## VNIAH

| recall | arm | strict | Δ vs images | 95% CI | p |
|---|---|---|---|---|---|
| 1.00 | images | 0.340 | — | | |
| 1.00 | own | 0.490 | **+0.149** | [+0.117, +0.181] | 1.22e-19 |
| 1.00 | foreign | 0.324 | **-0.017** | [-0.041, +0.008] | 0.207 |
| 0.00 | images | 0.138 | — | | |
| 0.00 | own | 0.180 | **+0.043** | [+0.022, +0.064] | 9.9e-05 |
| 0.00 | foreign | 0.130 | **-0.008** | [-0.026, +0.010] | 0.456 |

**At zero recall** (neither memory can contain the answer): own is +0.043 over images, foreign is -0.008. Own − foreign = **+0.051** [+0.030, +0.072], p = 3.89e-06.

Foreign text reproduces **-18%** of the own-document zero-recall effect. The closer that is to 100%, the more the residual is pure format — text being easier for the reader to consume than pixels, regardless of what it says. The closer to 0%, the more it was the document's own non-gold pages supplying topical context, and the weaker the format claim becomes.
