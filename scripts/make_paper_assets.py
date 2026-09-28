"""Build every data figure, data table and number of the SoftwareX paper from finished SR4Rec runs.

Nothing in the paper is typed by hand. This script reads the run folders (metrics.csv, stats.csv,
predictions.csv, quality.csv, latency.csv, fingerprint.yaml, ...) and writes into paper/generated/:

  fig_d1_delta_by_size.pdf    D1: Rank-1 difference with bicubic by LR size bin, 95% CIs
  fig_d2_fidelity.pdf         D2: PSNR and SSIM vs Rank-1, one point per SR model
  fig_d1_qualitative.png      D1: before/after-SR panels (same drawing function as comparisons/)
  tab_d1_main.tex             D1: main results (Rank-1, Rank-5, F1, delta, CI, p, verdict)
  tab_d2_fidelity.tex         D2: PSNR, SSIM, training degradation and Rank-1 of every SR model
  tab_summary.tex             all demonstrations, one row each
  tab_latency.tex             CPU latency (from the D1 run)
  tab_validation.tex          correctness checks (Appendix C), from paper/validation.yaml
  tab_d1_by_size.tex          D1: delta and verdict per size bin (Appendix C)
  macros.tex                  numbers used in the text (\\newcommand definitions)

Usage (after the demo runs):
    python scripts/make_paper_assets.py --d1 runs/reproduce_d1_earvn \\
        --d1-fixed runs/reproduce_d1_earvn_fixed --d2 runs/reproduce_d2_lfw --d3 runs/reproduce_d3_cub
Every argument is optional; missing assets stay as placeholders in the PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sr4rec.data.sizes import SIZE_BIN_LABELS, SIZE_BINS  # noqa: E402
from sr4rec.report import style  # noqa: E402
from sr4rec.report.formatting import fmt_ci, fmt_delta, fmt_mean_sd, fmt_num, fmt_p  # noqa: E402
from sr4rec.report.style import plt  # noqa: E402
from sr4rec.sr.validated import lookup  # noqa: E402
from sr4rec.stats import INSUFFICIENT  # noqa: E402
from sr4rec.utils import backbone_display  # noqa: E402

LABEL = {"bicubic": "bicubic (baseline)", "hr": "HR (upper bound)"}
DEMO_NAMES = {"d1": "D1", "d1_fixed": "D1, fixed recognizer", "d2": "D2", "d3": "D3"}
MACRO_KEYS = {"d1": "DONE", "d1_fixed": "DONEFIXED", "d2": "DTWO", "d3": "DTHREE"}  # TeX names cannot hold digits


def tex(s) -> str:
    t = " ".join(str(s).replace("▲", "").replace("▼", "").replace("●", "").split())
    return (t.replace("\\", "\\textbackslash{}").replace("_", "\\_\\allowbreak{}").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#").replace("−", "$-$").replace("±", "$\\pm$").replace("×", "$\\times$")
            .replace("≥", "$\\geq$").replace("<", "$<$").replace("–", "--").replace("Δ", "$\\Delta$"))


def size_cell(verdict: str, delta) -> str:
    if verdict.startswith("▲"):
        tag = "gain"
    elif verdict.startswith("▼"):
        tag = "harm"
    elif verdict.startswith("●"):
        tag = "n.s."
    else:
        tag = tex(verdict)
    return tex(f"{tag} {fmt_delta(delta)}")


class RunData:
    """Read-only view of one finished run folder."""

    def __init__(self, path: Path):
        self.path = path
        self.cfg = yaml.safe_load((path / "sr4rec.yaml").read_text())
        self.fp = yaml.safe_load((path / "fingerprint.yaml").read_text())
        self.metrics = pd.read_csv(path / "metrics.csv")
        self.stats = pd.read_csv(path / "stats.csv")
        self.pred = pd.read_csv(path / "predictions.csv")
        self.quality = pd.read_csv(path / "quality.csv") if (path / "quality.csv").is_file() else None
        lat = path / "latency.csv"
        self.latency = pd.read_csv(lat) if lat.is_file() and lat.stat().st_size > 60 else None
        self.overall = self.metrics[self.metrics["size_bin"] == "all"]
        self.backbones = list(self.cfg["recognizer"]["backbones"])
        self.seeds = list(self.cfg["recognizer"]["seeds"])
        self.sr = [e["name"] for e in self.cfg["sr"]]
        self.methods = ["bicubic"] + self.sr + (["hr"] if self.cfg["mode"] == "synthetic" else [])

    def trained_for(self, method: str) -> str:
        if method in ("bicubic", "hr"):
            return "n/a"
        for s in self.fp["sr_models"]:
            if s["name"] == method and s["files"]:
                f = s["files"][0]
                info = lookup(f["sha256"], Path(f["path"]).name)
                return info["trained_for"] if info else "unknown"
        return "unknown"

    def per_seed(self, b: str, m: str, col: str) -> list[float]:
        return self.overall[(self.overall.backbone == b) & (self.overall.method == m)][col].astype(float).tolist()

    def mean(self, b: str, m: str, col: str) -> float:
        v = self.per_seed(b, m, col)
        return float(np.mean(v)) if v else float("nan")

    def stat(self, b: str, m: str, size_bin: str = "all") -> pd.Series | None:
        r = self.stats[(self.stats.backbone == b) & (self.stats.method == m) & (self.stats.size_bin == size_bin)]
        return r.iloc[0] if len(r) else None

    @property
    def hours(self) -> float:
        return float(self.fp.get("total_seconds", 0.0)) / 3600.0

    @property
    def device(self) -> str:
        env = self.fp["environment"]
        return env.get("gpu") or env.get("cpu") or "unknown"

    @property
    def n_test(self) -> int:
        p = self.pred
        return int(((p.backbone == self.backbones[0]) & (p.method == "bicubic") & (p.seed == self.seeds[0])).sum())

    @property
    def n_classes(self) -> int:
        return int(self.pred["label"].nunique())


# ----------------------------------------------------------------------------- tables


def _table(caption: str, label: str, cols: str, header: list[str], rows: list[list[str]],
           star: bool = False, pos: str = "t") -> str:
    env = "table*" if star else "table"
    out = [f"\\begin{{{env}}}[{pos}]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == ["MIDRULE"] else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


def _bb_prefix(run: RunData, b: str) -> list[str]:
    return [tex(backbone_display(b))] if len(run.backbones) > 1 else []


def tab_d1_main(d1: RunData) -> str:
    rows = []
    for b in d1.backbones:
        if rows:
            rows.append(["MIDRULE"])
        best = max(d1.mean(b, m, "rank1") for m in ["bicubic"] + d1.sr)
        for m in ["bicubic"] + d1.sr:
            acc = tex(fmt_mean_sd(d1.per_seed(b, m, "rank1")))
            if abs(d1.mean(b, m, "rank1") - best) < 1e-12:
                acc = f"\\textbf{{{acc}}}"
            cells = _bb_prefix(d1, b) + [tex(LABEL.get(m, m)), acc, tex(fmt_mean_sd(d1.per_seed(b, m, "rank5"))),
                                         tex(fmt_mean_sd(d1.per_seed(b, m, "f1")))]
            st = d1.stat(b, m)
            if st is None:
                cells += ["", "", "", "", ""]
            else:
                cells += [tex(fmt_delta(st["delta"])), tex(fmt_ci(st["ci_low"], st["ci_high"])), tex(fmt_p(st["p_holm"])),
                          tex(st["same_sign"]), tex(st["verdict"])]
            rows.append(cells)
    head = (["Backbone"] if len(d1.backbones) > 1 else []) + [
        "SR", "Rank-1 (\\%)", "Rank-5 (\\%)", "Macro F1 (\\%)", "$\\Delta$ (pp)", "95\\% CI (pp)", "$p$ (Holm)",
        "Same sign", "Verdict"]
    cols = ("l" if len(d1.backbones) > 1 else "") + "lrrrrlrrl"
    cap = (f"D1: EarVN1.0, native low-resolution images, protocol matched, "
           f"{tex(', '.join(backbone_display(b) for b in d1.backbones))}. Mean and sample standard deviation over "
           f"seeds {', '.join(map(str, d1.seeds))}; $\\Delta$, CI and $p$ compare each SR model with bicubic on the same "
           f"{d1.n_test:,} test images (paired bootstrap and sign-flip permutation test, Holm-corrected).")
    return _table(cap, "tab:d1", cols, head, rows, star=True)


def tab_d1_by_size(d1: RunData) -> str:
    rows = []
    for b in d1.backbones:
        for m in d1.sr:
            cells = _bb_prefix(d1, b) + [tex(m)]
            for sb in SIZE_BINS:
                st = d1.stat(b, m, sb)
                if st is None:
                    cells.append("--")
                elif st["verdict"] == INSUFFICIENT:
                    cells.append(f"n = {int(st['n_images'])}")
                else:
                    cells.append(size_cell(st["verdict"], st["delta"]))
            rows.append(cells)
    head = (["Backbone"] if len(d1.backbones) > 1 else []) + ["SR"] + [tex(SIZE_BIN_LABELS[s]) for s in SIZE_BINS]
    return _table("D1: Rank-1 difference with bicubic (pp) and verdict by short side of the LR image; bins with "
                  "fewer than 50 test images are not tested and show their size.", "tab:d1-size",
                  ("l" if len(d1.backbones) > 1 else "") + "lllll", head, rows, pos="H")


def tab_d2(d2: RunData) -> str:
    q = d2.quality.groupby("method")[["psnr", "ssim"]].mean()
    b = d2.backbones[0]
    rows = []
    for m in ["bicubic"] + d2.sr + ["hr"]:
        st = d2.stat(b, m)
        psnr = fmt_num(float(q.loc[m, "psnr"]), 2) if m in q.index else "--"
        ssim = fmt_num(float(q.loc[m, "ssim"]), 3) if m in q.index else "--"
        rows.append([tex(LABEL.get(m, m)), tex(d2.trained_for(m)), psnr, ssim,
                     tex(fmt_mean_sd(d2.per_seed(b, m, "rank1"))),
                     tex(fmt_delta(st["delta"])) if st is not None else "", tex(st["verdict"]) if st is not None else ""])
    return _table(f"D2: LFW (people with at least 20 images), synthetic $\\times${d2.cfg['scale']}, "
                  f"{tex(backbone_display(b))}, seeds {', '.join(map(str, d2.seeds))}. PSNR and SSIM on the Y channel "
                  "of the test images; Rank-1 in \\%.", "tab:d2", "llrrrrl",
                  ["SR", "Trained for", "PSNR (dB)", "SSIM", "Rank-1 (\\%)", "$\\Delta$ (pp)", "Verdict"], rows)


def tab_summary(runs: dict[str, RunData]) -> str:
    rows = []
    for key, r in runs.items():
        for b in r.backbones:
            best = max(r.sr, key=lambda m, b=b: r.mean(b, m, "rank1")) if r.sr else None
            st = r.stat(b, best) if best else None
            ds = Path(str(r.cfg["dataset"])).name
            rows.append([tex(DEMO_NAMES.get(key, key)), tex(ds), tex(r.cfg["mode"]), tex(r.cfg["protocol"]),
                         str(len(r.seeds)), f"{r.n_classes:,}", f"{r.n_test:,}", tex(best or "n/a"),
                         tex(fmt_delta(st["delta"])) if st is not None else "", tex(st["verdict"]) if st is not None else "",
                         f"{r.hours:.1f}"])
    return _table("The demonstrations: best SR model, its Rank-1 difference with bicubic and the verdict. The runs vary "
                  "dataset, mode and protocol together, so they illustrate the workflow across domains rather than "
                  "isolate one factor. Hours: wall-clock time of the whole run on the reference machine.",
                  "tab:summary", "lllllrrlrlr",
                  ["Demo", "Dataset", "Mode", "Protocol", "Seeds", "Classes", "Test", "Best SR", "$\\Delta$ (pp)",
                   "Verdict", "Hours"], rows, star=True)


def tab_latency(run: RunData) -> str:
    lat = run.latency
    rows = []
    for m in ["bicubic"] + run.sr:
        r = lat[(lat.component == "sr") & (lat.sr == m)]
        row = [tex(m), f"{r.median_ms.iloc[0]:.1f} / {r.p95_ms.iloc[0]:.1f}" if len(r) else "n/a"]
        for b in run.backbones:
            p = lat[(lat.component == "pipeline") & (lat.sr == m) & (lat.backbone == b)]
            row.append(f"{p.median_ms.iloc[0]:.1f} / {p.p95_ms.iloc[0]:.1f}" if len(p) else "n/a")
        rows.append(row)
    rec = "; ".join(f"{tex(backbone_display(r.backbone))} {r.median_ms:.1f} / {r.p95_ms:.1f}"
                    for r in lat[lat.component == "recognizer"].itertuples())
    c = run.cfg["latency"]
    return _table(f"CPU latency in ms (median / p95): batch 1, LR input {c['lr_size']} $\\times$ {c['lr_size']}, "
                  f"{c['threads']} threads, {tex(run.fp['environment']['cpu'])}, 10 warm-up and 100 timed runs. "
                  f"Recognizer alone: {rec}.", "tab:latency", "l" + "r" * (1 + len(run.backbones)),
                  ["SR", "SR only"] + [f"SR + {tex(backbone_display(b))}" for b in run.backbones], rows)


def tab_validation(validation: Path) -> str:
    v = yaml.safe_load(validation.read_text())
    rows = [[tex(c["component"]), tex(c["reference"]), tex(c["criterion"]), tex(c.get("result", "pending"))]
            for c in v["checks"]]
    col = ">{\\raggedright\\arraybackslash}p"
    return _table("Correctness checks of SR4Rec against reference implementations.", "tab:validation",
                  f"{col}{{2.4cm}}{col}{{3.0cm}}{col}{{2.6cm}}{col}{{4.6cm}}",
                  ["Component", "Reference", "Criterion", "Result"], rows, pos="H")


# ----------------------------------------------------------------------------- figures


def fig_d1_delta_by_size(d1: RunData, out: Path) -> None:
    style.apply()
    n = len(d1.backbones)
    fig, axes = plt.subplots(1, n, figsize=(max(4.2, 3.4 * n), 2.9), squeeze=False)
    width = 0.8 / max(1, len(d1.sr))
    for ax, b in zip(axes[0], d1.backbones, strict=True):
        ax.axhline(0, color="black", linewidth=0.8)
        for i, m in enumerate(d1.sr):
            st = style.style_for(i + 1)
            for k, sb in enumerate(SIZE_BINS):
                s = d1.stat(b, m, sb)
                if s is None:
                    continue
                x = k + (i - (len(d1.sr) - 1) / 2) * width
                if s["verdict"] == INSUFFICIENT:
                    ax.plot(x, s["delta"], marker=st["marker"], markerfacecolor="white", color=st["color"], linestyle="none")
                else:
                    ax.errorbar(x, s["delta"], yerr=[[s["delta"] - s["ci_low"]], [s["ci_high"] - s["delta"]]],
                                marker=st["marker"], color=st["color"], capsize=2, linestyle="none")
            ax.plot([], [], marker=st["marker"], color=st["color"], linestyle="none", label=m)
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_xticks(range(len(SIZE_BINS)), [SIZE_BIN_LABELS[s] for s in SIZE_BINS], fontsize=7)
        ax.set_xlabel("Short side of the LR image")
        ax.set_ylabel("Rank-1 difference with bicubic (pp)")
        if n > 1:
            ax.set_title(backbone_display(b), fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_d2_fidelity(d2: RunData, out: Path) -> None:
    style.apply()
    q = d2.quality.groupby("method")[["psnr", "ssim"]].mean()
    b = d2.backbones[0]
    markers = {"bicubic degradation": "o", "real-world degradation": "^", "n/a": "s", "unknown": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, col, xlabel in zip(axes, ("psnr", "ssim"), ("PSNR (dB, Y channel)", "SSIM (Y channel)"), strict=True):
        for m in ["bicubic"] + d2.sr:
            acc = 100 * d2.mean(b, m, "rank1")
            sd = 100 * float(np.std(d2.per_seed(b, m, "rank1"), ddof=1)) if len(d2.seeds) > 1 else 0.0
            kind = d2.trained_for(m)
            ax.errorbar(q.loc[m, col], acc, yerr=sd, marker=markers[kind], color="black", capsize=2, linestyle="none")
            ax.annotate(m, (q.loc[m, col], acc), textcoords="offset points", xytext=(5, 3), fontsize=6.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Rank-1, {backbone_display(b)} (%)")
        ax.margins(x=0.25, y=0.15)
    handles = [plt.Line2D([], [], color="black", marker=mk, linestyle="none", label=k)
               for k, mk in markers.items() if k != "unknown"]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- numbers


def macros(runs: dict[str, RunData], extra: dict[str, str]) -> str:
    out = ["% Generated by scripts/make_paper_assets.py. Do not edit."]
    for key, r in runs.items():
        k = MACRO_KEYS[key]
        out += [f"\\newcommand{{\\{k}NumTest}}{{{r.n_test:,}}}",
                f"\\newcommand{{\\{k}NumClasses}}{{{r.n_classes:,}}}",
                f"\\newcommand{{\\{k}NumSeeds}}{{{len(r.seeds)}}}",
                f"\\newcommand{{\\{k}Hours}}{{{r.hours:.1f}}}",
                f"\\newcommand{{\\{k}Device}}{{{tex(r.device)}}}"]
    for k, v in extra.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d1", type=Path, help="run folder of d1_earvn (main demonstration)")
    ap.add_argument("--d1-fixed", type=Path, help="run folder of d1_earvn_fixed (protocol comparison)")
    ap.add_argument("--d2", type=Path, help="run folder of d2_lfw (controlled, synthetic)")
    ap.add_argument("--d3", type=Path, help="run folder of d3_cub (other domain, single run)")
    ap.add_argument("--validation", type=Path, default=ROOT / "paper" / "validation.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {k: RunData(p) for k, p in (("d1", a.d1), ("d1_fixed", a.d1_fixed), ("d2", a.d2), ("d3", a.d3)) if p}
    extra: dict[str, str] = {}
    if "d1" in runs:
        d1 = runs["d1"]
        (a.out / "tab_d1_main.tex").write_text(tab_d1_main(d1))
        (a.out / "tab_d1_by_size.tex").write_text(tab_d1_by_size(d1))
        fig_d1_delta_by_size(d1, a.out / "fig_d1_delta_by_size.pdf")
        if d1.latency is not None:
            (a.out / "tab_latency.tex").write_text(tab_latency(d1))
        from make_qualitative_figure import make_figure

        try:
            c, d = make_figure(d1.path, 4, a.out / "fig_d1_qualitative.png", caption_in_image=False)
            extra.update({"QualCorrected": f"{c:,}", "QualDegraded": f"{d:,}"})
        except SystemExit as err:
            print(f"Qualitative figure skipped: {err}")
    if "d2" in runs and runs["d2"].quality is not None:
        (a.out / "tab_d2_fidelity.tex").write_text(tab_d2(runs["d2"]))
        fig_d2_fidelity(runs["d2"], a.out / "fig_d2_fidelity.pdf")
    if runs:
        (a.out / "tab_summary.tex").write_text(tab_summary(runs))
    if a.validation.is_file():
        (a.out / "tab_validation.tex").write_text(tab_validation(a.validation))
    (a.out / "macros.tex").write_text(macros(runs, extra))
    print(f"Wrote paper assets to {a.out}: " + ", ".join(sorted(p.name for p in a.out.iterdir())))


if __name__ == "__main__":
    main()
