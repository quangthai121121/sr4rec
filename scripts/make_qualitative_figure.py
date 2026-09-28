"""Paper Figure 4: stack comparison panels of a run (same drawing function as runs/<run>/comparisons/).

Takes half "corrected" and half "degraded" panels from comparisons/index.csv (fixed order, no manual
picking) and writes one PNG with a caption line giving the total number of corrected and degraded
test images for the chosen backbone and seed.

Usage: python scripts/make_qualitative_figure.py runs/reproduce_d1_earvn --rows 4 --out figure4.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402


def make_figure(run: Path, rows: int, out: Path, caption_in_image: bool = True) -> tuple[int, int]:
    """Write the stacked panels to ``out``; return (corrected, degraded) test-image counts."""
    idx = pd.read_csv(run / "comparisons" / "index.csv")
    half = rows // 2
    chosen = pd.concat([idx[idx.reason == "corrected"].head(half), idx[idx.reason == "degraded"].head(rows - half)])
    if chosen.empty:
        raise SystemExit("No corrected or degraded panels in this run (use comparisons.selection: mixed).")
    cfg = yaml.safe_load((run / "sr4rec.yaml").read_text())
    backbone = cfg["comparisons"]["backbone"] or cfg["recognizer"]["backbones"][0]
    seed = cfg["recognizer"]["seeds"][0]
    pred = pd.read_csv(run / "predictions.csv")
    sub = pred[(pred.backbone == backbone) & (pred.seed == seed)].pivot(index="path", columns="method", values="correct")
    srs = [c for c in sub.columns if c not in ("bicubic", "hr")]
    corrected = int(((sub["bicubic"] == 0) & (sub[srs] == 1).any(axis=1)).sum()) if srs else 0
    degraded = int(((sub["bicubic"] == 1) & (sub[srs] == 0).any(axis=1)).sum()) if srs else 0
    images = [Image.open(run / "comparisons" / f) for f in chosen.file]
    fig, axes = plt.subplots(len(images), 1, figsize=(10, 2.6 * len(images)), squeeze=False)
    for ax, im in zip(axes[:, 0], images, strict=False):
        ax.imshow(im)
        ax.axis("off")
    if caption_in_image:
        fig.text(0.5, 0.0, f"Test images corrected by at least one SR model: {corrected}; degraded: {degraded} "
                 f"({backbone}, seed {seed}).", ha="center", fontsize=9, family="DejaVu Sans")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return corrected, degraded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--rows", type=int, default=4, help="number of panels (half corrected, half degraded)")
    ap.add_argument("--out", type=Path, default=Path("figure4.png"))
    a = ap.parse_args()
    make_figure(a.run, a.rows, a.out)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
