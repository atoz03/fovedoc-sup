# Zero-recall leakage control — how much of the residual is a stray answer copy?

At zero recall every gold page has been swapped out of the retrieved set, so the memory is OCR text of pages that do not carry the evidence. It can still carry the *answer string*, because values recur elsewhere in a document and the lexical packer is explicitly searching for blocks that match the question. Leakage is detected by normalised string containment of the gold answer in the packed memory text; the contrast is then re-run on the leak-free subset, paired, McNemar with continuity correction.


## VMQAR — zero recall, n=938

Reported in the sweep: **+0.030** [+0.014, +0.045], p = 0.000309.

| min gold length | leaked | leak-free n | leak-free Δ | 95% CI | p | leaked-subset Δ | p |
|---|---|---|---|---|---|---|---|
| ≥4 chars | 55 (5.9%) | 883 | **+0.026** | [+0.011, +0.041] | 0.00133 | +0.091 | 0.182 |
| ≥6 chars | 45 (4.8%) | 893 | **+0.028** | [+0.013, +0.043] | 0.000607 | +0.067 | 0.45 |
| ≥8 chars | 38 (4.1%) | 900 | **+0.028** | [+0.013, +0.043] | 0.000607 | +0.079 | 0.45 |
| ≥12 chars | 31 (3.3%) | 907 | **+0.026** | [+0.011, +0.042] | 0.00114 | +0.129 | 0.221 |


## VNIAH — zero recall, n=1025

Reported in the sweep: **+0.043** [+0.022, +0.064], p = 9.9e-05.

| min gold length | leaked | leak-free n | leak-free Δ | 95% CI | p | leaked-subset Δ | p |
|---|---|---|---|---|---|---|---|
| ≥4 chars | 154 (15.0%) | 871 | **+0.025** | [+0.006, +0.044] | 0.0121 | +0.143 | 0.00359 |
| ≥6 chars | 127 (12.4%) | 898 | **+0.028** | [+0.009, +0.047] | 0.00558 | +0.150 | 0.00865 |
| ≥8 chars | 114 (11.1%) | 911 | **+0.027** | [+0.008, +0.047] | 0.00766 | +0.167 | 0.00494 |
| ≥12 chars | 76 (7.4%) | 949 | **+0.031** | [+0.010, +0.051] | 0.00407 | +0.197 | 0.00705 |


## Reading

The leak-free effect is **stable across every threshold** and remains significant, so the zero-recall residual is not an artefact of stray answer copies — something real survives removing the evidence. But the headline figure is inflated by leakage, materially so on V-NIAH (+0.043 → +0.028, about a third), and the leaked subset carries a large effect of its own, which is what leakage should look like if the model is actually reading it.

Quote the leak-free number. Two caveats on the detector: string containment **over-counts**, since a gold string appearing in the memory does not prove the model used it, and it **under-counts**, since a paraphrase or a differently-formatted number is missed. It is a control, not a measurement of what the model did.

What survives here is still not identified as *format*. Leak-free memory is text of the same document, so topical relatedness remains a live explanation. The foreign-memory control (`scripts/run_dmr_foreign_memory.sh`) is what separates those: foreign text cannot leak the answer and cannot be topically related, so it isolates format alone.
