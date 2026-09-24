#!/usr/bin/env python3
"""selection / localization 对照的配对分析（scripts/run_dmr_selection_controls.sh 的收尾）。

输入：<judged-dir>/{pages,ocrmem,lex16n,rand16n,randcn,alln,hilite,crops[,cropsnat]}_{vmqar,vniah}.jsonl，
      同一 judge 会话下的逐样本 verdict（strict = verdict == "correct"）。
输出：Markdown 表（--out），含
  1. 每臂 strict 准确率与相对 images 臂的配对 Δ（Wald CI + McNemar 连续性校正 p）；
  2. 回答审稿关切的关键对比：
       措辞          ocrmem − lex16n          （"按关键词 / 候选证据"这一相关性暗示值多少）
       问题条件化    lex16n − rand16n / randcn （同预算下"按问题选"值多少）
       选择 vs 全量  lex16n − alln            （不选、不限预算的全文是否同样有效）
       文本 vs 指位  lex16n − hilite / crops  （同一组 block，给文本比只指位置多多少）
       指位本身      hilite / crops − pages   （只指位置能恢复多少）
     并给出"指位恢复的份额" = (hilite − pages) / (lex16n − pages)；
  3. 按"gold block 是否在 lexical 打包记忆内"分层（OCR 文本对 gold 摘录 token 覆盖 ≥ 0.9 视为在内），
     因为增益几乎只发生在 gold 在记忆内的样本上，这一层是 text-vs-pointer 的决定性对比。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from eccv26.utils.exact_block import tokenize_exact_block_text  # noqa: E402

TASKS = ("vmqar", "vniah")
DEFAULT_ARMS = ("pages", "ocrmem", "lex16n", "rand16n", "randcn", "alln", "hilite", "crops", "cropsnat")
ARM_LABEL = {
    "pages": "images", "ocrmem": "+OCR memory (paper wording)", "lex16n": "+lexical-16 (neutral wording)",
    "rand16n": "+random-16 blocks", "randcn": "+random blocks, char-matched", "alln": "+all blocks, reading order",
    "hilite": "red boxes on pages (no text)", "crops": "crops at reader scale (no text)",
    "cropsnat": "crops at native resolution (no text)",
}
CONTRASTS = (
    ("wording", "ocrmem", "lex16n"),
    ("query-conditioning (block-matched)", "lex16n", "rand16n"),
    ("query-conditioning (char-matched)", "lex16n", "randcn"),
    ("selection vs all text", "lex16n", "alln"),
    ("text vs pointer (boxes)", "lex16n", "hilite"),
    ("text vs pointer (crops)", "lex16n", "crops"),
    ("text vs pointer (native crops)", "lex16n", "cropsnat"),
    ("pointer alone (boxes)", "hilite", "pages"),
    ("pointer alone (crops)", "crops", "pages"),
    ("pointer alone (native crops)", "cropsnat", "pages"),
)
MANIFEST_ROOT = Path("data/fovedoc_project"
                     "/data/manifests/final_benchmark")


def load_judged(path: Path) -> dict[str, dict]:
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def strict(r: dict) -> bool:
    return (r.get("judge") or {}).get("verdict") == "correct"


def paired(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> tuple[float, float, float, int, int]:
    """mean(a) − mean(b)、Wald 95% 半宽、McNemar p、b、c。"""
    n = len(ids)
    bb = sum(1 for i in ids if strict(a[i]) and not strict(b[i]))
    cc = sum(1 for i in ids if strict(b[i]) and not strict(a[i]))
    d = (bb - cc) / n
    if bb + cc == 0:
        return d, 0.0, 1.0, bb, cc
    se = math.sqrt((bb + cc) - (bb - cc) ** 2 / n) / n
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return d, 1.96 * se, math.erfc(math.sqrt(chi / 2)), bb, cc


def load_evidence(manifest_root: Path) -> dict[str, list[dict]]:
    ev: dict[str, list[dict]] = {}
    for task_dir in ("vniah", "vmqar"):
        for split in ("train", "val", "test"):
            p = manifest_root / task_dir / f"{split}.jsonl"
            if not p.exists():
                continue
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                ev[str(r["id"])] = list((r.get("meta") or {}).get("evidence") or [])
    return ev


def covered(excerpt: str, text: str, thr: float = 0.9) -> bool:
    e = Counter(tokenize_exact_block_text(excerpt))
    if not e:
        return False
    t = Counter(tokenize_exact_block_text(text))
    return sum(min(v, t.get(k, 0)) for k, v in e.items()) / sum(e.values()) >= thr


def gold_hits(rec: dict, evidence: list[dict], thr: float = 0.9) -> tuple[int, int]:
    """返回 (被打包记忆覆盖的 gold 证据条数, gold 证据总数)。

    覆盖按“同页所有已打包 block 的 token 并集”计算（阈值 thr），而不是逐 block：manifest 的 text_excerpt 来自 PDF
    文本层，OCR block 的切分边界与之不同，逐 block 匹配会把“内容在记忆里、只是被切成两段”误判为不在。
    """
    facts = rec.get("memory_facts") or []
    by_page: dict[int, list[str]] = {}
    for f in facts:
        by_page.setdefault(int(f["page_id"]), []).append(str(f.get("text") or ""))
    n_hit = 0
    for e in evidence:
        page_text = "\n".join(by_page.get(int(e["page_id"]), []))
        if page_text and covered(str(e.get("text_excerpt") or ""), page_text, thr):
            n_hit += 1
    return n_hit, len(evidence)


def gold_stratum(rec: dict, evidence: list[dict], thr: float = 0.9) -> str:
    """all / part / none：gold 证据全部 / 部分 / 完全不在打包记忆内（V-MQAR 有 2 条 gold，part 与 none 行为不同）。"""
    n_hit, n_ev = gold_hits(rec, evidence, thr)
    if n_ev == 0 or n_hit == 0:
        return "none"
    return "all" if n_hit == n_ev else "part"


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--manifest-root", default=str(MANIFEST_ROOT))
    ap.add_argument("--coverage-thr", type=float, default=0.9,
                    help="gold 证据的 token 被同页已打包 block 并集覆盖的比例阈值（分层用）")
    args = ap.parse_args()
    jdir = Path(args.judged_dir)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    evidence = load_evidence(Path(args.manifest_root))

    lines = ["# Selection / localization controls — text, or a pointer to where to look?\n",
             "Same 1173 samples per task, same 16 ColQwen2 pages as images, same OCR block store, stock "
             "Qwen3-VL-2B, every arm graded in one judge session. `lex16n` is the paper's OCR memory arm "
             "with the relevance hint removed from the prompt wording; `hilite`/`crops` show the reader "
             "exactly the blocks `lex16n` packs, but as red boxes / crops on the page images and no text.\n"]
    for task in TASKS:
        data = {a: load_judged(jdir / f"{a}_{task}.jsonl") for a in arms if (jdir / f"{a}_{task}.jsonl").exists()}
        if "pages" not in data:
            lines.append(f"\n## {task.upper()}: no `pages_{task}.jsonl` judged yet\n")
            continue
        ids = sorted(set.intersection(*(set(d) for d in data.values())))
        missing = [a for a in arms if a not in data]
        lines.append(f"\n## {task.upper()} (n={len(ids)}"
                     + (f"; not yet judged: {', '.join(missing)}" if missing else "") + ")\n")
        lines.append("| arm | strict | Δ vs images | 95% CI | p |")
        lines.append("|---|---|---|---|---|")
        for a in arms:
            if a not in data:
                continue
            acc = sum(strict(data[a][i]) for i in ids) / len(ids)
            if a == "pages":
                lines.append(f"| {ARM_LABEL[a]} | {acc:.3f} | — | | |")
                continue
            d, hw, p, _, _ = paired(data[a], data["pages"], ids)
            lines.append(f"| {ARM_LABEL[a]} | {acc:.3f} | {d:+.3f} | [{d-hw:+.3f}, {d+hw:+.3f}] | {fmt_p(p)} |")

        lines.append("\n**Key contrasts** (paired, McNemar):\n")
        lines.append("| question | contrast | Δ | 95% CI | p |")
        lines.append("|---|---|---|---|---|")
        for label, a, b in CONTRASTS:
            if a in data and b in data:
                d, hw, p, _, _ = paired(data[a], data[b], ids)
                lines.append(f"| {label} | {a} − {b} | {d:+.3f} | [{d-hw:+.3f}, {d+hw:+.3f}] | {fmt_p(p)} |")
        if "lex16n" in data:
            base_gain, _, _, _, _ = paired(data["lex16n"], data["pages"], ids)
            for a in ("hilite", "crops", "cropsnat"):
                if a in data and base_gain != 0:
                    g, _, _, _, _ = paired(data[a], data["pages"], ids)
                    lines.append(f"\nShare of the `lex16n` gain recovered by `{a}` alone: **{g / base_gain:.2f}** "
                                 f"({g:+.3f} of {base_gain:+.3f}).")

        # 分层：gold 是否在 lexical 打包记忆内（以 lex16n 的 facts 为准；hilite / crops 用的是同一组 block）
        ref = data.get("lex16n") or data.get("ocrmem")
        if ref is not None and evidence:
            strata = {"all": [], "part": [], "none": []}
            for i in ids:
                strata[gold_stratum(ref[i], evidence.get(i, []), args.coverage_thr)].append(i)
            lines.append(f"\n**Stratified by how many gold blocks the packed memory holds** (page-union token coverage ≥ {args.coverage_thr}) "
                         f"(all n={len(strata['all'])}, part n={len(strata['part'])}, none n={len(strata['none'])}; "
                         f"images-arm strict acc "
                         + ", ".join(f"{k}={sum(strict(data['pages'][i]) for i in v)/len(v):.3f}" for k, v in strata.items() if v)
                         + "):\n")
            lines.append("| stratum | arm | Δ vs images | 95% CI | p |")
            lines.append("|---|---|---|---|---|")
            for name, sub in (("gold all packed", strata["all"]), ("gold part packed", strata["part"]), ("gold none packed", strata["none"])):
                if len(sub) < 20:
                    continue
                for a in ("ocrmem", "lex16n", "rand16n", "randcn", "alln", "hilite", "crops", "cropsnat"):
                    if a in data:
                        d, hw, p, _, _ = paired(data[a], data["pages"], sub)
                        lines.append(f"| {name} | {ARM_LABEL[a]} | {d:+.3f} | [{d-hw:+.3f}, {d+hw:+.3f}] | {fmt_p(p)} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
