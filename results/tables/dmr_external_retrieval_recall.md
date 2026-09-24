# MMLongBench-Doc retrieval recall — why the external result must be stratified

ColQwen2 page retrieval over all 135 MMLongBench-Doc documents (1091 QA, median 30 pages, max
468). Recall is computed over the **845 questions that carry at least one gold evidence page**;
the other 246 have an empty gold set, so `full_recall` is vacuously false for every one of them
and pooling would drag the rate down by construction.

"Has a gold page" is *not* the same set as "answerable", and the difference is small but real:
7 of MMLongBench's deliberately-unanswerable questions do carry gold pages (evidence of absence
points at a page), while 9 answerable questions carry none (whole-document counting questions
such as "how many line plots are shown in the paper?"). So the 845 = 838 clean answerable +
7 unanswerable. Restricted to the 838, full-recall@16 is 0.843 rather than 0.844 — the curve
below is unaffected, but the reading analysis uses the 838, since the other two groups measure
abstention and un-annotatable counting respectively.

| k | full-recall | any-hit | page-recall |
|---|---|---|---|
| 1 | 0.388 | 0.649 | 0.499 |
| 2 | 0.517 | 0.748 | 0.616 |
| 4 | 0.638 | 0.828 | 0.725 |
| 8 | 0.743 | 0.893 | 0.817 |
| 12 | 0.794 | 0.916 | 0.860 |
| **16** | **0.844** | 0.940 | 0.897 |
| 24 | 0.892 | 0.957 | 0.932 |
| 32 | 0.929 | 0.967 | 0.951 |

**Retrieval is not saturated here.** On FoveDoc, full-recall@16 is 0.998 (V-MQAR) and 1.000
(V-NIAH), which is what licenses the framing "retrieval is solved, so the remaining gap is pure
reading". Externally it is **0.844**, so 16% of evidence-bearing questions reach the reader
without their full evidence. Any external memory-vs-images delta therefore mixes a reading effect
with a retrieval shortfall, and must be reported **stratified by full-recall**, not pooled. It is:
see `dmr_external_validity.md`, where the partial-recall stratum is the one place the memory arm
is (directionally) *worse*.

## The shortfall is monotone in document length

| document length | n | full-recall@16 |
|---|---|---|
| ≤ 30 pages | 437 | 0.938 |
| 31–60 | 190 | 0.837 |
| 61–120 | 173 | 0.665 |
| > 120 | 45 | 0.644 |

This is also the retrospective justification for lifting the loader's 120-page render cap. Under
the cap, the `> 120` stratum — the worst-retrieval bucket — did not exist in the benchmark at all,
and full-recall@16 would have been measured only over documents where retrieval works best. The
external test would have looked more favourable than it is, on the one benchmark whose job is to
check that we are not flattering ourselves.
