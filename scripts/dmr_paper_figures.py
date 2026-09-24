#!/usr/bin/env python3
"""ICASSP 2027 论文（paper/main.tex；paper/ 是独立 Git 仓库）统计图的唯一生成入口。

设计原则：**脚本里不写死任何实验数值**。每个画出来的数字都从 `results/tables/dmr_*.md`
（以及一处 `docs/experiments/results/*.md`）的 markdown 表格里解析得到，表格由各自的分析脚本
重生成后，重跑本脚本即可让图件同步。这样图、表、正文三者共享同一份台账，不会各自漂移。

生成的图（写到 paper/figures/，由 main.tex 的 graphicspath 引用；Fig. 1 / Fig. 4 是 draw.io 示意图，
由 paper/figures/diagrams/ 的 build_diagrams.py + export_diagrams.py 生成为 fig1-design.pdf / fig4-controls.pdf）：
  fig2-readers.pdf   Fig. 2  六个 reader 的效应图 + Qwen3-VL 规模趋势        ← dmr_reader_family_grid.md
  fig3-external.pdf  Fig. 3  MMLongBench 检索召回曲线 + 按证据模态的效应   ← dmr_external_retrieval_recall.md,
                                                                           dmr_external_validity.md,
                                                                           paper/figures/diagrams/evidence_audit.json
  fig5-sweep.pdf     Fig. 5  受控召回退化 sweep（own/foreign/images）+ 零召回残差分解
                                                                         ← dmr_recall_sweep_*.md, dmr_foreign_memory.md,
                                                                           dmr_zero_recall_leakage.md,
                                                                           docs/experiments/results/20260905_dmr_foreign_memory_results.md
  archive/fig_design.pdf     原 Fig. 1（配对设计示意 + OCR-parity 柱状图），已由 draw.io 版取代，只作备份
                                                                         ← dmr_ocr_parity.md

Usage:
  .venv/bin/python scripts/dmr_paper_figures.py [--out paper/figures] [--png-preview /tmp/figprev]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
RES = ROOT / "results" / "tables"

TASK_NAME = {"vmqar": "V-MQAR", "vniah": "V-NIAH"}

# ---------------------------------------------------------------------------------------------
# 配色：按“实体”固定，不按图内顺序轮换。四个臂在所有图里颜色一致；任务差异用面板或标记形状表达。
#   images 基线用中性灰（它是参照，不是一个“系列”）；其余三色是 dataviz 调色板经验证的前三槽
#   （blue / orange / aqua，相邻对 CVD ΔE ≥ 9），并且每个系列都配了不同的标记形状作为二次编码。
# ---------------------------------------------------------------------------------------------
C_IMG = "#6b6a66"   # images 臂
C_MEM = "#2a78d6"   # own-document / OCR memory 臂，以及 memory − images 效应
C_PDF = "#eb6834"   # PDF text-layer memory 臂
C_FOR = "#1baf7a"   # foreign-document memory 臂
C_INK = "#0b0b0b"   # 两个 memory 臂之间的对比（own − foreign）
C_FILL = "#cde2fb"  # 蓝色 ramp 最浅一档，用于 images↔memory 之间的阴影
C_GRID = "#e6e5e1"

plt.rcParams.update({
    # Liberation Serif 是 Times 的度量兼容 TrueType 克隆；Nimbus Roman 是 CFF/OpenType，
    # 以 Type 42 嵌入时 poppler / PDF eXpress 会报 "font type mismatch"，故不用它。
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 6.8,
    "axes.titlesize": 7,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------------------------
# markdown 表格解析
# ---------------------------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _clean(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).replace("−", "-").strip()


def md_tables(path: Path) -> list[list[dict[str, str]]]:
    """把文件里所有 markdown 表格按出现顺序解析为 [table][row] -> {header: cell}。

    重复的表头（例如同一张表里出现两个 `p` 列）按出现顺序加后缀 `_2`、`_3`，不覆盖。
    """
    tables: list[list[dict[str, str]]] = []
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("|"):
            cells = [_clean(c) for c in line.strip("|").split("|")]
            if header is None:
                seen: dict[str, int] = {}
                header = []
                for h in cells:
                    seen[h] = seen.get(h, 0) + 1
                    header.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
            elif all(set(c) <= set("-: ") for c in cells):
                continue
            else:
                rows.append(dict(zip(header, cells)))
        elif header is not None:
            tables.append(rows)
            header, rows = None, []
    if header is not None:
        tables.append(rows)
    return tables


def num(cell: str) -> float:
    m = _NUM.search(_clean(cell))
    if not m:
        raise ValueError(f"no number in {cell!r}")
    return float(m.group())


def ci(cell: str) -> tuple[float, float]:
    vals = _NUM.findall(_clean(cell))
    if len(vals) < 2:
        raise ValueError(f"no interval in {cell!r}")
    return float(vals[0]), float(vals[1])


def _find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for r in rows:
        if all(match[k] in r.get(k, "") for k in match):
            return r
    raise KeyError(f"no row matching {match}")


# ---------------------------------------------------------------------------------------------
# 各数据源
# ---------------------------------------------------------------------------------------------
def parity() -> dict[str, dict[str, float]]:
    """dmr_ocr_parity.md 首表：task -> {images, ocr, pdf}。"""
    tables = md_tables(TAB / "dmr_ocr_parity.md")
    t = tables[0]
    out = {}
    for r in t:
        task = "V-MQAR" if "MQAR" in r["task"] else "V-NIAH"
        out[task] = {"images": num(r["images"]), "ocr": num(r["OCR memory"]),
                     "pdf": num(r["PDF text layer"])}
        contrast = _find(tables[1], contrast="OCR memory - images", task=task)
        out[task]["delta_ocr"] = num(contrast["Δ"])
    return out


def reader_grid() -> list[dict[str, str]]:
    return md_tables(TAB / "dmr_reader_family_grid.md")[0]


def external_recall() -> list[dict[str, str]]:
    return md_tables(TAB / "dmr_external_retrieval_recall.md")[0]


def external_evidence() -> tuple[list[dict[str, str]], dict[str, float]]:
    """按证据模态的效应表，加上正文里的单标签对比（+0.085 vs -0.046，差 +0.131）。"""
    path = TAB / "dmr_external_validity.md"
    tables = md_tables(path)
    evidence = next(t for t in tables if "evidence" in t[0])
    text = path.read_text(encoding="utf-8").replace("−", "-")
    m = re.search(
        r"effect is \*?\*?([-+]\d\.\d+)\*?\*? on Pure-text \(n=(\d+)\) against "
        r"\*?\*?([-+]\d\.\d+)\*?\*? on Chart\+Figure \(n=(\d+)\)[^*]*\*\*([-+]\d\.\d+)\*\* "
        r"\[([-+]\d\.\d+), ([-+]\d\.\d+)\], z = [\d.]+, \*\*p = ([\d.e-]+)\*\*",
        text,
    )
    if not m:
        raise SystemExit("[figs] single-tag contrast sentence not found in dmr_external_validity.md")
    single = {"text": float(m[1]), "text_n": int(m[2]), "chartfig": float(m[3]),
              "chartfig_n": int(m[4]), "diff": float(m[5]), "lo": float(m[6]),
              "hi": float(m[7]), "p": float(m[8])}
    return evidence, single


def sweep(task: str) -> list[dict[str, str]]:
    return md_tables(TAB / f"dmr_recall_sweep_{task}.md")[0]


def foreign(task: str) -> list[dict[str, str]]:
    t = md_tables(TAB / "dmr_foreign_memory.md")
    return t[0] if task == "vmqar" else t[1]


def leakage(task: str) -> list[dict[str, str]]:
    t = md_tables(TAB / "dmr_zero_recall_leakage.md")
    return t[0] if task == "vmqar" else t[1]


def foreign_leakfree(task: str) -> dict[str, tuple[float, float, float]]:
    """结果文档里的去泄漏三对比：contrast -> (Δ, lo, hi)。"""
    rows = md_tables(RES / "20260905_dmr_foreign_memory_results.md")
    table = next(t for t in rows if "own - foreign" in t[0])
    r = _find(table, task=TASK_NAME[task])
    out = {}
    for key in ("own - images", "foreign - images", "own - foreign"):
        v = _NUM.findall(r[key])
        out[key] = (float(v[0]), float(v[1]), float(v[2]))
    out["n"] = (num(r["n"]), 0.0, 0.0)
    return out


# ---------------------------------------------------------------------------------------------
# 绘图小工具
# ---------------------------------------------------------------------------------------------
def _grid(ax, axis="x"):
    ax.grid(True, axis=axis, color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)


def _panel_title(ax, text):
    ax.set_title(text, pad=3)


def _save(fig, out_dir: Path, name: str, preview: Path | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}.pdf"
    fig.savefig(fp, bbox_inches="tight", pad_inches=0.01)
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"-> {fp}")


# ---------------------------------------------------------------------------------------------
# Fig. 1：配对设计 + 主结果
# ---------------------------------------------------------------------------------------------
def fig_design(out_dir: Path, preview: Path | None):
    par = parity()
    fig = plt.figure(figsize=(3.39, 2.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.15], hspace=0.45)

    # (a) 示意图：两臂共享同一批检索页图像，memory 臂只是“加”文本
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#f4f3f0", ec="#52514e", ls="-", fs=6.0, color=C_INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc=fc, ec=ec, lw=0.6, ls=ls, mutation_aspect=0.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                linespacing=1.1)

    def arrow(p, q, color="#52514e", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                     color=color, ls=ls, shrinkA=0, shrinkB=0))

    box(0.00, 0.60, 0.13, 0.32, "question")
    box(0.18, 0.60, 0.40, 0.32, "ColQwen2 retrieval\ntop-16 pages, full recall $\\geq$0.998",
        fs=5.8)
    box(0.82, 0.30, 0.18, 0.40, "VLM\nreader")
    box(0.18, 0.02, 0.48, 0.34, "OCR text of the same 16 pages\n(16 lexically packed blocks)",
        fc="#e8f1fc", ec=C_MEM, ls="--", color=C_MEM)
    arrow((0.13, 0.76), (0.18, 0.76))
    arrow((0.58, 0.76), (0.82, 0.62))
    ax.text(0.73, 0.80, "16 page images: both arms", ha="center", va="bottom", fontsize=5.6,
            color=C_IMG)
    arrow((0.31, 0.60), (0.31, 0.36), color=C_MEM)
    ax.text(0.325, 0.48, "RapidOCR, CPU", ha="left", va="center", fontsize=5.4, color=C_MEM)
    arrow((0.66, 0.19), (0.82, 0.36), color=C_MEM, ls="--")
    ax.text(0.69, 0.11, "memory arm only", ha="left", va="top", fontsize=5.6, color=C_MEM)
    ax.text(0.0, 1.06, "(a) Paired design: text is added, pixels are never removed",
            fontsize=7, fontweight="bold", va="bottom", transform=ax.transAxes)

    # (b) OCR-parity 主结果
    ax = fig.add_subplot(gs[1])
    tasks = ["V-MQAR", "V-NIAH"]
    arms = [("images", "images", C_IMG), ("ocr", "+ OCR memory", C_MEM),
            ("pdf", "+ PDF text layer", C_PDF)]
    w = 0.24
    for j, (key, label, color) in enumerate(arms):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(tasks))]
        ys = [par[t][key] for t in tasks]
        ax.bar(xs, ys, width=w, color=color, label=label, lw=0)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=5.8)
    for i, t in enumerate(tasks):
        d_ocr = par[t]["delta_ocr"]
        ax.annotate("", xy=(i, par[t]["ocr"] + 0.075), xytext=(i - w - 0.02, par[t]["ocr"] + 0.075),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_MEM, mutation_scale=5))
        ax.text(i - (w + 0.02) / 2, par[t]["ocr"] + 0.09, f"{d_ocr:+.3f}", ha="center",
                va="bottom", fontsize=5.8, color=C_MEM)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("strict accuracy")
    _grid(ax, "y")
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.0, borderpad=0.2,
              labelspacing=0.25)
    _panel_title(ax, "(b) Qwen3-VL-2B, n = 1173 per task, same retrieved pages")
    _save(fig, out_dir / "archive", "fig_design", preview)  # 正文 Fig. 1 已换成 draw.io 版，这里只刷新备份


# ---------------------------------------------------------------------------------------------
# Fig. 2：reader 家族森林图 + Qwen3-VL 规模趋势
# ---------------------------------------------------------------------------------------------
READER_ORDER = ["Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen2.5-VL-7B",
                "LLaVA-OneVision-7B", "Idefics3-8B"]
READER_LABEL = {"LLaVA-OneVision-7B": "LLaVA-OV-7B$^\\dagger$", "Idefics3-8B": "Idefics3-8B$^\\dagger$"}


def fig_readers(out_dir: Path, preview: Path | None):
    """(a) 六个 reader 的 images → memory 哑铃图，两任务各占一行；(b) Qwen3-VL 规模趋势。单栏宽。"""
    from matplotlib.lines import Line2D

    rows = reader_grid()
    fig = plt.figure(figsize=(3.39, 2.12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.50, left=0.245, right=0.995,
                          top=0.905, bottom=0.165)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    tasks = [("V-MQAR", "o", 0.2), ("V-NIAH", "s", -0.2)]
    for i, reader in enumerate(READER_ORDER):
        y0 = len(READER_ORDER) - 1 - i
        for task, mk, off in tasks:
            r = _find(rows, reader=reader, task=task)
            a, b, d = num(r["images"]), num(r["memory"]), num(r["Δ"])
            y = y0 + off
            ax.plot([a, b], [y, y], color=C_FILL, lw=2.4, solid_capstyle="butt", zorder=1)
            ax.plot(a, y, mk, color=C_IMG, ms=3.0, zorder=2)
            ax.plot(b, y, mk, color=C_MEM, ms=3.0, zorder=3)
            ax.text(b + 0.02, y, f"{100*d:+.1f}", va="center", ha="left", fontsize=5.3, color=C_MEM)
    for y in range(len(READER_ORDER) - 1):
        # 粗线分隔 Qwen3-VL 家族与其余三个家族
        ax.axhline(y + 0.5, color=C_GRID if y != 2 else "0.6", lw=0.5 if y != 2 else 0.7)
    ax.set_yticks(range(len(READER_ORDER)))
    ax.set_yticklabels([READER_LABEL.get(r, r) for r in reversed(READER_ORDER)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(READER_ORDER) - 0.4 + 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("strict accuracy, first 400 samples per task")
    _grid(ax, "x")
    handles = [Line2D([], [], marker="o", color=C_IMG, ls="none", ms=3, label="images"),
               Line2D([], [], marker="o", color=C_MEM, ls="none", ms=3, label="+ text-layer memory"),
               Line2D([], [], marker="o", color="0.3", ls="none", ms=3, mfc="white", label="V-MQAR"),
               Line2D([], [], marker="s", color="0.3", ls="none", ms=3, mfc="white", label="V-NIAH")]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=5.4, handlelength=1.0,
              columnspacing=0.9, labelspacing=0.15, borderpad=0.1, handletextpad=0.4)
    _panel_title(ax, "(a) images → memory (Δ, pp), six readers")

    for task, mk, ls, side in [("V-MQAR", "o", "-", -1), ("V-NIAH", "s", "--", +1)]:
        xs, ds, los, his = [], [], [], []
        for size in ["2B", "4B", "8B"]:
            r = _find(rows, reader=f"Qwen3-VL-{size}", task=task)
            xs.append(int(size[:-1]))
            ds.append(num(r["Δ"]))
            lo, hi = ci(r["95% CI"])
            los.append(ds[-1] - lo)
            his.append(hi - ds[-1])
        bx.errorbar(xs, ds, yerr=[los, his], fmt=mk + ls, color=C_MEM, capsize=1.5,
                    elinewidth=0.8, capthick=0.6, label=task,
                    markerfacecolor=C_MEM if task == "V-MQAR" else "white")
        for x, dv, lo, hi in zip(xs, ds, los, his):
            # V-NIAH 标在误差棒上方，V-MQAR 标在下方，避免 8B 处两条曲线的标签相撞
            if side > 0:
                bx.text(x, dv + hi + 0.012, f"{100*dv:+.1f}", ha="center", va="bottom", fontsize=5.3, color=C_MEM)
            else:
                bx.text(x, dv - lo - 0.012, f"{100*dv:+.1f}", ha="center", va="top", fontsize=5.3, color=C_MEM)
    bx.set_xscale("log", base=2)
    bx.set_xticks([2, 4, 8])
    bx.set_xticklabels(["2B", "4B", "8B"])
    bx.minorticks_off()
    bx.set_xlim(1.55, 10.3)
    bx.set_ylim(0, 0.50)
    bx.set_xlabel("Qwen3-VL reader size")
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}"))
    bx.set_ylabel("Memory gain (pp)", labelpad=1)
    _grid(bx, "y")
    bx.legend(frameon=False, loc="lower left", handlelength=1.6, fontsize=5.4, borderpad=0.2, labelspacing=0.2)
    _panel_title(bx, "(b) Qwen3-VL scale")
    _save(fig, out_dir, "fig2-readers", preview)


# ---------------------------------------------------------------------------------------------
# Fig. 3：外部基准——检索未饱和 + 效应随证据模态翻转
# ---------------------------------------------------------------------------------------------
EVIDENCE_LABEL = {"Pure-text (Plain-text)": "Text", "Table": "Table",
                  "Generalized-text (Layout)": "Layout", "Figure": "Figure", "Chart": "Chart"}


def fig_external(out_dir: Path, preview: Path | None):
    rec = external_recall()
    evidence, single = external_evidence()
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.39, 1.6),
                               gridspec_kw=dict(width_ratios=[0.85, 1.2], wspace=0.62))

    ks = [int(num(r["k"])) for r in rec]
    fr = [num(r["full-recall"]) for r in rec]
    # 各预算的主任务召回直接复算自已有 ranked top-16 列表，不推断较小预算的 QA 正确率。
    audit = json.loads((ROOT / "results/evidence_audit.json").read_text())
    a.plot(ks, fr, "o-", color=C_MEM, label="MMLB-Doc", markersize=2.5)
    for task, label, color, marker in [("vniah", "V-NIAH", C_PDF, "s"), ("vmqar", "V-MQAR", C_FOR, "^")]:
        curve = audit["recall"][task]
        xx = [int(k) for k in curve]
        a.plot(xx, [curve[str(k)]["full"] for k in xx], marker+"--", color=color,
               label=label, markersize=2.5, linewidth=.8)
    k16 = fr[ks.index(16)]
    a.axvline(16, color="0.6", lw=0.5, ls=":")
    a.annotate(f"{100*k16:.1f}%", xy=(16, k16), xytext=(16, k16 - 0.28), fontsize=6, ha="center",
               arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    # 主任务 @16 的饱和值直接标在曲线终点旁，与 Table 1 一致
    sat = [100 * audit["recall"][t]["16"]["full"] for t in ("vniah", "vmqar")]
    a.text(38, 0.02, f"@16: V-NIAH {sat[0]:.0f}%\nV-MQAR {sat[1]:.1f}%", fontsize=4.8, color="0.3", ha="right",
           va="bottom", linespacing=1.1)
    a.set_xlim(0.8, 40)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 4, 16, 32])
    a.set_xticklabels(["1", "4", "16", "32"])
    a.minorticks_off()
    a.set_ylim(0, 1.08)
    a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    a.legend(frameon=False, loc="upper left", fontsize=4.8, handlelength=1.3, labelspacing=.25)
    a.set_xlabel("$k$ pages retrieved")
    a.set_ylabel("full recall@$k$")
    _grid(a, "y")
    _panel_title(a, "(a) Retrieval vs. page budget")

    order = ["Pure-text (Plain-text)", "Table", "Generalized-text (Layout)", "Figure", "Chart"]
    labels, ys = [], []
    right = 0.0
    y = len(order) + 2.0
    for name in order:
        r = _find(evidence, evidence=name)
        d, (lo, hi) = num(r["Δ"]), ci(r["95% CI"])
        b.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=C_MEM, capsize=1.5,
                   elinewidth=0.8, capthick=0.6)
        b.text(hi + 0.008, y, f"{100*d:+.1f}", va="center", ha="left", fontsize=5.2, color=C_MEM)
        right = max(right, hi)
        labels.append(f"{EVIDENCE_LABEL[name]} ({int(num(r['n']))})")
        ys.append(y)
        y -= 1
    # 单标签子集：只有点估计与二者之差的区间，故不画各自误差棒
    y -= 0.4
    for key, lab in [("text", "Text only"), ("chartfig", "Chart/Fig.")]:
        b.plot(single[key], y, marker="D", color=C_MEM, ms=3.2, ls="none")
        b.text(single[key] + 0.012, y, f"{100*single[key]:+.1f}", va="center", ha="left", fontsize=5.2, color=C_MEM)
        labels.append(f"{lab} ({single[key + '_n']})")
        ys.append(y)
        y -= 1
    # 双向箭头与差值标注放在最后一行之下，留出与 chart/figure 菱形不重叠的间距
    b.annotate("", xy=(single["chartfig"], ys[-1] - 0.62), xytext=(single["text"], ys[-1] - 0.62),
               arrowprops=dict(arrowstyle="<->", lw=0.6, color=C_INK, mutation_scale=5))
    b.text((single["text"] + single["chartfig"]) / 2, ys[-1] - 0.78,
           f"{100*single['diff']:+.1f} pp, $p$={single['p']:.4f}", ha="center", va="top", fontsize=5.6)
    b.axvline(0, color="0.55", lw=0.5)
    b.axhline(ys[-2] + 0.7, color="0.75", lw=0.4)
    b.set_yticks(ys)
    b.set_yticklabels(labels)
    b.set_ylim(ys[-1] - 1.85, ys[0] + 0.6)
    b.set_xlim(-0.14, max(0.14, right + 0.07))
    b.set_xticks([-0.1, 0, 0.1])
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}"))
    b.set_xlabel("Memory gain (pp)")
    _grid(b, "x")
    _panel_title(b, "(b) Effect by evidence modality")
    _save(fig, out_dir, "fig3-external", preview)


# ---------------------------------------------------------------------------------------------
# Fig. 4：召回退化 sweep + 零召回残差分解
# ---------------------------------------------------------------------------------------------
def fig_sweep(out_dir: Path, preview: Path | None):
    fig = plt.figure(figsize=(7.0, 1.68))
    # (a)(b) 之间靠得紧，(b)(c) 之间要给 (c) 的行标签留出空位，所以用两个 gridspec 而不是共用 wspace
    gs_ab = fig.add_gridspec(1, 2, left=0.055, right=0.545, wspace=0.32)
    gs_c = fig.add_gridspec(1, 1, left=0.745, right=0.99)
    axes = [fig.add_subplot(gs_ab[0]), fig.add_subplot(gs_ab[1]), fig.add_subplot(gs_c[0])]
    for k, (ax, task) in enumerate(zip(axes[:2], ["vmqar", "vniah"])):
        sw = sweep(task)
        fo = foreign(task)
        rs = [num(r["recall"]) for r in sw]
        img = [num(r["images"]) for r in sw]
        mem = [num(r["+memory"]) for r in sw]
        ax.fill_between(rs, img, mem, color=C_FILL, lw=0)
        ax.plot(rs, img, "o-", color=C_IMG, label="images")
        ax.plot(rs, mem, "s-", color=C_MEM, label="+ own-document memory")
        fr = {num(r["recall"]): num(r["strict"]) for r in fo if r["arm"] == "foreign"}
        xs = sorted(fr)
        ax.plot(xs, [fr[x] for x in xs], "^-.", color=C_FOR, label="+ foreign-document memory",
                markerfacecolor="white")
        for j, (r, i, m) in enumerate(zip(rs, img, mem)):
            ha = "left" if j == 0 else ("right" if j == len(rs) - 1 else "center")
            # 使用配对统计的 Δ 列，不能相减两个已舍入的 arm 均值。
            delta = num(sw[j]["Δ"])
            ax.text(r, m + 0.02, f"{100*delta:+.1f} pp", ha=ha, va="bottom", fontsize=5.7, color=C_MEM)
        text = (TAB / f"dmr_recall_sweep_{task}.md").read_text().replace("−", "-")
        n = re.search(r"Same (\d+) samples", text)
        # 交互项：full 与 zero recall 之间 memory 优势的 within-sample 变化，直接读台账句子
        inter = re.search(r"changes by \*\*([-+][\d.]+)\*\* \[([-+\d., ]+)\], p = (\d[\d.]*(?:e[-+]?\d+)?)", text)
        if not inter:
            raise SystemExit(f"[figs] interaction sentence not found in dmr_recall_sweep_{task}.md")
        lo_i, hi_i = (float(v) for v in inter[2].split(","))
        ax.text(0.98, 0.03, f"full → zero recall:\nΔ falls by {100*float(inter[1]):.1f} pp "
                            f"[{100*lo_i:.1f}, {100*hi_i:.1f}]\n$p$={float(inter[3]):.0e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=C_INK, linespacing=1.15)
        # foreign 臂在满召回处的对比，放在两条几乎重合的折线下方、右对齐，避开斜线
        f_full = next(r for r in fo if r["arm"] == "foreign" and num(r["recall"]) == 1.0)
        ax.set_xticks(rs)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, max(mem) + 0.12)
        # foreign 臂在满召回处的对比：沿自己那条折线的方向标在线上方的阴影带内，避开斜线、图例和交互项文字
        p0, p1 = ax.transData.transform((0.0, fr[0.0])), ax.transData.transform((1.0, fr[1.0]))
        angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        ax.annotate(f"foreign at full recall: {100*num(f_full['Δ vs images']):+.1f} pp, $p$={num(f_full['p']):.2f}",
                    xy=(0.5, (fr[0.0] + fr[1.0]) / 2), xytext=(0, 4), textcoords="offset points",
                    rotation=angle, rotation_mode="anchor", ha="center", va="bottom", fontsize=5.0, color=C_FOR)
        ax.set_xlabel("fraction of gold pages retrieved")
        if k == 0:
            ax.set_ylabel("strict accuracy")
        _grid(ax, "y")
        _panel_title(ax, f"({'ab'[k]}) {TASK_NAME[task]}, $n$={n.group(1) if n else '?'}")
    # 图例放在 (a) 的左上角：memory 折线在 recall≤0.5 处低于 0.2，上方是空的
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0.1,
                   labelspacing=0.2)

    # (c) 零召回残差分解
    ax = axes[2]
    y = 0.0
    ticks, labels = [], []
    for task in ["vmqar", "vniah"]:
        sw = sweep(task)
        zero = _find(sw, recall="0.00")
        leak = _find(leakage(task), **{"min gold length": "≥6"})
        lf = foreign_leakfree(task)
        n_all = re.search(r"Same (\d+) samples", (TAB / f"dmr_recall_sweep_{task}.md").read_text())
        n_lf = int(num(leak["leak-free n"]))
        leaked_share = re.search(r"\(([\d.]+)%\)", leak["leaked"])
        items = [
            ("own $-$ images, all", num(zero["Δ"]), *ci(zero["95% CI"]), C_MEM, "s"),
            ("own $-$ images", num(leak["leak-free Δ"]), *ci(leak["95% CI"]), C_MEM, "s"),
            ("foreign $-$ images", *lf["foreign - images"], C_FOR, "^"),
            ("own $-$ foreign", *lf["own - foreign"], C_INK, "D"),
        ]
        ax.text(-0.065, y + 0.6,
                f"{TASK_NAME[task]}: $n$={n_all.group(1) if n_all else '?'}, "
                f"leak-free $n$={n_lf} ({leaked_share[1] if leaked_share else '?'}% leaked)",
                fontsize=6.2, fontweight="bold", va="center", ha="left")
        for label, d, lo, hi, color, mk in items:
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt=mk, color=color, capsize=1.5,
                        elinewidth=0.8, capthick=0.6, ms=3.4,
                        markerfacecolor="white" if mk == "^" else color)
            ax.text(hi + 0.004, y, f"{100*d:+.1f}", va="center", ha="left", fontsize=5.7)
            ticks.append(y)
            labels.append(label)
            y -= 1
        y -= 1.2
    ax.axvline(0, color="0.55", lw=0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 1.5, 1.3)
    ax.set_xlim(-0.07, 0.09)
    ax.set_xticks([-0.05, 0, 0.05])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}"))
    ax.set_xlabel("Memory gain at zero recall (pp)")
    _grid(ax, "x")
    _panel_title(ax, "(c) What survives removing the evidence")
    _save(fig, out_dir, "fig5-sweep", preview)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--png-preview", default="")
    args = ap.parse_args()
    out = ROOT / args.out
    preview = Path(args.png_preview) if args.png_preview else None
    fig_design(out, preview)
    fig_readers(out, preview)
    fig_external(out, preview)
    fig_sweep(out, preview)


if __name__ == "__main__":
    main()
