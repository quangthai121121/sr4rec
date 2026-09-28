"""Correctness check of SR loading and of PSNR/SSIM (paper, Appendix C): run the validated SR weights
through SR4Rec on a standard benchmark and compare PSNR/SSIM with the values published by the
authors. Only models trained for bicubic degradation have published values.

The benchmark folder holds the HR images (for example Set5/HR). LR inputs are either the
benchmark's own bicubic LR images (--lr-dir, recommended: the published numbers were computed from
them) or SR4Rec's MATLAB-style downsampling. PSNR/SSIM follow SR4Rec: Y channel, border = scale.

Usage:
    python scripts/check_sr_fidelity.py --hr-dir Set5/HR --lr-dir Set5/LR_bicubic/X4 --benchmark Set5
    python scripts/check_sr_fidelity.py --hr-dir Set14/HR --benchmark Set14 --update paper/validation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from sr4rec.eval.metrics import psnr_y, ssim_y
from sr4rec.lr.resize import downsample
from sr4rec.sr.pth_source import SpandrelSR
from sr4rec.utils import load_rgb, mod_crop, pil_to_tensor, tensor_to_uint8

# Published x4 results (PSNR dB / SSIM, Y channel). Sources: SwinIR paper, Table 2 (DIV2K-trained
# SwinIR); SPAN (Wan et al., CVPRW 2024, Table 1, 48-channel).
PUBLISHED = {
    "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth": {"Set5": (32.72, 0.9021), "Set14": (28.94, 0.7914)},
    "spanx4_ch48.pth": {"Set5": (32.20, 0.8953), "Set14": (28.66, 0.7834)},
}
TOLERANCE_DB = 0.05


def lr_for(hr_file: Path, lr_dir: Path | None, scale: int) -> torch.Tensor:
    if lr_dir is not None:
        cands = [p for p in lr_dir.iterdir() if p.stem.startswith(hr_file.stem)]
        if not cands:
            raise SystemExit(f"No LR image for {hr_file.name} in {lr_dir}")
        return pil_to_tensor(load_rgb(cands[0]))
    return downsample(pil_to_tensor(mod_crop(load_rgb(hr_file), scale)), scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr-dir", type=Path, required=True)
    ap.add_argument("--lr-dir", type=Path)
    ap.add_argument("--benchmark", required=True, help="name used to look up the published values, e.g. Set5")
    ap.add_argument("--weights-dir", type=Path, default=Path("weights"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--update", type=Path, help="write the results into this validation.yaml")
    a = ap.parse_args()
    device = torch.device(a.device)
    lines, ok_all = [], True
    for fname, table in PUBLISHED.items():
        w = a.weights_dir / fname
        ref = table.get(a.benchmark)
        if not w.is_file() or ref is None:
            print(f"skip {fname}: {'weights not found' if not w.is_file() else 'no published value for ' + a.benchmark}")
            continue
        src = SpandrelSR(Path(fname).stem, w, 4).to(device)
        ps, ss = [], []
        for f in sorted(a.hr_dir.iterdir()):
            if f.suffix.lower() not in (".png", ".bmp", ".jpg"):
                continue
            hr = np.asarray(mod_crop(load_rgb(f), 4))
            with torch.no_grad():
                sr = src.upscale(lr_for(f, a.lr_dir, 4)[None].to(device))[0].cpu()
            out = tensor_to_uint8(sr)
            ps.append(psnr_y(out, hr, 4))
            ss.append(ssim_y(out, hr, 4))
        p, s = float(np.mean(ps)), float(np.mean(ss))
        ok = abs(p - ref[0]) <= TOLERANCE_DB
        ok_all &= ok
        lines.append(f"{Path(fname).stem} {a.benchmark} x4: {p:.2f} dB / {s:.4f} (published {ref[0]:.2f} / {ref[1]:.4f})")
        print(("OK   " if ok else "FAIL ") + lines[-1])
    if a.update and lines:
        v = yaml.safe_load(a.update.read_text())
        for c in v["checks"]:
            if c["component"].startswith("SR loading and PSNR/SSIM"):
                prev = c.get("result", "")
                keep = [x for x in prev.split("; ") if x and not x.startswith("pending") and a.benchmark not in x]
                c["result"] = "; ".join(keep + lines) + ("" if ok_all else " (outside tolerance)")
        a.update.write_text(yaml.safe_dump(v, sort_keys=False, width=110))
        print(f"Updated {a.update}")


if __name__ == "__main__":
    main()
