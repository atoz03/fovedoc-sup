#!/usr/bin/env python3
"""Build a DERANGED OCR store: every document's memory file holds another document's text.

The recall sweep found that the memory advantage survives removing all gold evidence — +0.030
(V-MQAR) and +0.043 (V-NIAH), p < 1e-3 on ~1000 paired samples each. The write-up calls that a
*format* effect: text the model can read directly beats pixels it has to recognise, independent
of whether the text carries the answer.

That reading has a competing explanation the sweep cannot exclude. At zero recall the memory is
still OCR text of 16 pages **of the same document** — same topic, same entities, same vocabulary,
often the same section. It could be helping because it is text (format), or because it is
*related* text (topical context). Those imply very different things: the first says a recogniser
is worth running on anything, the second says the memory must at least be about the right
document.

This store separates them. Each document's memory is replaced by a different document's text,
with page ids remapped so the existing lookup still resolves. Everything else is untouched — same
images, same retrieved page ids, same lexical block packing, same budget — so the only variable
is *whose text it is*. No harness patch: `--memory-document-root` already redirects the store.

  own-document memory  -> evidence + topic + format
  foreign-doc memory   -> format only
  no memory            -> baseline

Read against the sweep's zero-recall arms:
  foreign ~= own-at-zero-recall  -> the residual is content-independent; format is the whole story
  foreign <  own-at-zero-recall  -> same-document context was doing the work; "format" overstates it
  foreign <  0                   -> irrelevant text actively distracts, as it does on charts/figures

  python scripts/dmr_make_foreign_memory_store.py \
      --src code/runs/20260810/dmr_ocr_documents \
      --out code/runs/20260904/foreign_ocr_documents
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(".")


def _family(doc_id: str) -> str:
    """Coarse corpus family, used to prefer a topically distant donor."""
    for fam in ("arxiv", "wipo", "cloudflare", "registry", "faa", "docvqa"):
        if doc_id.startswith(fam):
            return fam
    return "other"


def _derange(ids: list[str], fams: dict[str, str], seed: int) -> dict[str, str]:
    """Map each document to a donor that is not itself, preferring a different family.

    A plain shuffle would leave fixed points, and a fixed point is not a control — that document
    would silently keep its own memory and dilute the contrast.
    """
    rng = random.Random(seed)
    donors = list(ids)
    rng.shuffle(donors)
    mapping: dict[str, str] = {}
    for i, doc in enumerate(ids):
        mapping[doc] = donors[i]

    # Repair fixed points and same-family pairs by swapping with a later entry.
    order = list(ids)
    for _ in range(6):
        bad = [d for d in order
               if mapping[d] == d or fams[mapping[d]] == fams[d]]
        if not bad:
            break
        for d in bad:
            for e in order:
                if e == d:
                    continue
                a, b = mapping[d], mapping[e]
                # Swapping must not create a new violation on either side.
                if a == e or b == d:
                    continue
                if fams[b] == fams[d] or fams[a] == fams[e]:
                    continue
                mapping[d], mapping[e] = b, a
                break
    # Hard guarantee: no document may keep its own text.
    leftover = [d for d in order if mapping[d] == d]
    for d in leftover:
        for e in order:
            if e != d and mapping[e] != d and mapping[d] != e:
                mapping[d], mapping[e] = mapping[e], mapping[d]
                break
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="code/runs/20260810/dmr_ocr_documents")
    ap.add_argument("--out", default="code/runs/20260904/foreign_ocr_documents")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"no OCR store at {src}")
    docs: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        docs[str(d["document_id"])] = d
    ids = sorted(docs)
    fams = {i: _family(i) for i in ids}
    print(f"[foreign] {len(ids)} documents, families: {dict(Counter(fams.values()))}")

    mapping = _derange(ids, fams, args.seed)
    self_maps = sum(1 for d in ids if mapping[d] == d)
    same_fam = sum(1 for d in ids if fams[mapping[d]] == fams[d])
    if self_maps:
        raise SystemExit(f"derangement failed: {self_maps} documents kept their own text")
    print(f"[foreign] derangement ok: 0 self-maps, {same_fam} same-family pairs "
          f"({same_fam/len(ids):.1%} — unavoidable where a family dominates)")

    written = 0
    empty_donor = 0
    for doc_id in ids:
        recip, donor = docs[doc_id], docs[mapping[doc_id]]
        dpages = [p for p in donor["pages"] if (p.get("blocks") or [])]
        if not dpages:
            # A donor with no OCR text anywhere would silently produce an empty memory, which is
            # the `pages` arm, not the control. Fall back to any donor that has text.
            empty_donor += 1
            for alt in ids:
                cand = [p for p in docs[alt]["pages"] if (p.get("blocks") or [])]
                if alt != doc_id and cand:
                    donor, dpages = docs[alt], cand
                    mapping[doc_id] = alt
                    break
        pages = []
        for i, p in enumerate(recip["pages"]):
            src_page = dpages[i % len(dpages)]
            pid = p["page_id"]
            blocks = []
            for j, b in enumerate(src_page.get("blocks") or [], start=1):
                nb = dict(b)
                # Relabel so block ids stay consistent with the page they are served under;
                # downstream code parses `p<page>_b<n>` and a mismatch would be confusing in
                # provenance dumps even though nothing depends on it numerically.
                nb["block_id"] = f"p{pid}_b{j}"
                blocks.append(nb)
            pages.append({
                "page_id": pid,
                # Geometry stays the RECIPIENT's so nothing downstream that reasons about page
                # size is perturbed; only the text content is foreign.
                "width": p.get("width"), "height": p.get("height"),
                "image_path": p.get("image_path"),
                "page_text": src_page.get("page_text"),
                "blocks": blocks,
            })
        obj = {
            "schema_version": recip.get("schema_version", "1.0"),
            "document_id": doc_id,
            "meta": {**(recip.get("meta") or {}),
                     "block_source": "ocr_foreign_document",
                     "foreign_donor_document_id": mapping[doc_id],
                     "foreign_donor_family": fams[mapping[doc_id]],
                     "recipient_family": fams[doc_id]},
            "pages": pages,
        }
        (out / f"{doc_id}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        written += 1

    (out / "_derangement.json").write_text(
        json.dumps({"seed": args.seed, "mapping": mapping}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[foreign] wrote {written} documents to {out}")
    if empty_donor:
        print(f"[foreign] {empty_donor} donors had no OCR text and were reassigned")


if __name__ == "__main__":
    main()
