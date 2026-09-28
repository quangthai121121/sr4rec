"""The SR weight files validated by SR4Rec (see README, "Validated SR models")."""

from __future__ import annotations

REAL_WORLD = "real-world degradation"
BICUBIC = "bicubic degradation"

# SHA-256 -> information. File names are the official ones; download links are in the README.
VALIDATED_WEIGHTS = {
    "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1": {
        "file": "RealESRGAN_x4plus.pth", "model": "Real-ESRGAN x4plus", "trained_for": REAL_WORLD},
    "129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a": {
        "file": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth", "model": "SwinIR-M x4 (classical)", "trained_for": BICUBIC},
    "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6": {
        "file": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", "model": "SwinIR-M x4 (real-world, GAN)", "trained_for": REAL_WORLD},
}

# Validated files whose SHA-256 is not recorded yet are recognised by file name only.
VALIDATED_BY_NAME = {
    "spanx4_ch48.pth": {"model": "SPAN x4 (ch48)", "trained_for": BICUBIC},
}


def lookup(sha256: str, filename: str) -> dict | None:
    if sha256 in VALIDATED_WEIGHTS:
        return {**VALIDATED_WEIGHTS[sha256], "verified": True}
    if filename in VALIDATED_BY_NAME:
        return {**VALIDATED_BY_NAME[filename], "file": filename, "verified": False}
    return None
