# Reading the report

`runs/<run_name>/report.md` is the file to read first. It has the same 11 sections in every run.

1. **Answer** (top of the file): at most three sentences generated from the verdicts: which SR
   models improved or reduced Rank-1 accuracy significantly, on how many backbones.
2. **Summary**: per backbone, the SR model with the highest mean Rank-1, its Delta and verdict.
3. **Setup**: dataset, mode, protocol, split (declared ratios and actual counts), recognizer input,
   training recipe, seeds, statistics and device; the SR models with the degradation they were
   trained for, their file and SHA-256.
4. **Main results**: Rank-1 mean ± sd over seeds, Delta vs bicubic, 95% CI, Holm-adjusted p, same
   sign, verdict. The best value per backbone is bold; the baseline row and (synthetic) the HR upper
   bound are labelled.
5. **Rank-5, CMC, precision, recall, F1**: descriptive metrics and the CMC plot.
6. **Results by image size**: Delta and verdict per size bin; bins with fewer than 50 test images
   show `insufficient data (n)`.
7. **Image quality vs recognition** (synthetic only): PSNR/SSIM next to Rank-1 and F1, to show
   whether the best-looking SR model is also the best for recognition.
8. **CPU latency**.
9. **Visual comparisons**: two embedded panels; all panels are in `comparisons/`.
10. **Warnings and notes**: domain mismatch of the SR weights, real-world models on bicubic LR
    images, the `fixed_recognizer` caveat, small size bins, converted images, experimental backbones.
11. **How to read this report** and **Reproducibility**.

## Verdict rules

| Verdict | Condition |
|---|---|
| `▲ significant gain` | Delta > 0, the 95% CI excludes 0, p (Holm) < 0.05 |
| `▼ significant harm` | Delta < 0, the 95% CI excludes 0, p (Holm) < 0.05 |
| `● no difference` | otherwise |
| `insufficient data` | size bin with fewer than 50 test images (not tested) |

## Number formats

Accuracy: one decimal; Delta: signed (`+0.4`, `−3.2`); p: two significant digits, `<0.001` below
0.001; large counts with thousands separators.

## Limits to keep in mind

- `native-lr` has no HR reference: no PSNR/SSIM and no HR upper bound.
- Public SR weights were trained on natural images and may not match your data; the report says
  which degradation each validated model was trained for.
- With `fixed_recognizer`, SR outputs are out of the recognizer's training domain; use `matched` to
  compare systems built with each SR model.
