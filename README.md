# SR4Rec

[![CI](https://github.com/quangthai121121/sr4rec/actions/workflows/ci.yml/badge.svg)](https://github.com/quangthai121121/sr4rec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)

**Does placing a super-resolution (SR) model in front of a recognizer improve closed-set
recognition, compared with plain bicubic upscaling?** SR4Rec answers this question for your own
dataset and your own SR models with one configuration file and one command, under a fixed,
statistically sound protocol, and writes a report that can be read on its own.

## Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Project layout](#3-project-layout)
4. [Quick start](#4-quick-start)
5. [Preparing your dataset](#5-preparing-your-dataset)
6. [Adding SR models](#6-adding-sr-models)
7. [Examples](#7-examples)
8. [Configuration reference](#8-configuration-reference)
9. [Reading the report](#9-reading-the-report)
10. [Output files](#10-output-files)
11. [Reproducing the paper](#11-reproducing-the-paper)
12. [FAQ](#12-faq)
13. [Limitations, roadmap, contributing, licences](#13-limitations-roadmap-contributing-licences)
14. [Citation, license, acknowledgements](#14-citation-license-acknowledgements)

## 1. Overview

SR4Rec is for researchers and engineers who work with low-resolution images of identities (faces,
ears, ...) or object classes (species, breeds, ...) and need to decide whether an SR model is worth
adding to a recognition system.

```
images/ + labels.csv ──> split ──> LR images ──> bicubic | SR model 1 | ... | SR model k
                        (every class           (synthetic: MATLAB bicubic ↓scale;
                         in train/val/test)      native-lr: images as given)
        ──> letterbox + ImageNet normalization ──> recognizer per backbone × seed (protocol matched or fixed_recognizer)
        ──> Rank-1/Rank-5/CMC, macro P/R/F1, PSNR/SSIM ──> paired statistics vs bicubic
        ──> report.md, CSV files, plots, LaTeX table, before/after-SR panels
```

What SR4Rec enforces for you:

- the same test images and the same preprocessing for every method: letterbox to a square (aspect
  ratio kept, never stretched, ImageNet-mean padding) and ImageNet normalization;
- a closed-set split in which every class is in train, val and test and no image (not even a
  renamed copy) is in two splits; checkpoints chosen on val, never on test;
- protocol `matched` by default: the recognizer of each SR model is trained on images that went
  through that SR model;
- paired statistics over test images (bootstrap CI, sign-flip permutation test, Holm correction)
  and a verdict per SR model and backbone, also by image size;
- a fingerprint (versions, hardware, data and weight hashes) for every run.

Three commands only: `sr4rec init`, `sr4rec run`, `sr4rec reproduce`.

## 2. Installation

Requirements: Python 3.10-3.12, PyTorch 2.3 or newer. A CUDA GPU is recommended for real datasets;
Apple-silicon GPUs (MPS) are supported; the examples run on a CPU. With `runtime.device: auto`
SR4Rec uses a CUDA GPU if there is one, otherwise an Apple GPU, otherwise the CPU.

```bash
git clone https://github.com/quangthai121121/sr4rec.git
cd sr4rec
pip install -e ".[dev]"
pytest -m "not slow"                    # quick self-test (about 30 s)
```

Install the PyTorch build for your GPU first if needed (<https://pytorch.org/get-started/>).

**Offline use.** The ImageNet-pretrained backbones are downloaded by timm from Hugging Face on
first use and cached in `~/.cache/huggingface/` (change with the `HF_HOME` environment variable).
On a machine without internet access, run once on a connected machine:

```bash
python -c "import timm; [timm.create_model(m, pretrained=True) for m in ['resnet18', 'mobilenetv3_small_100', 'convnext_tiny']]"
```

and copy the Hugging Face cache folder to the offline machine (same `HF_HOME`). SR4Rec never
downloads SR weights; you provide them.

## 3. Project layout

Keep everything for one study in one project folder and run every command from it:

```
my_project/                      # project folder; run every command from here
├── sr4rec.yaml                  # written by `sr4rec init`, edited by you
├── data/                        # datasets prepared by you
│   └── <dataset_name>/
│       ├── images/              # all images (any sub-folder structure)
│       └── labels.csv           # one row per image (see "Preparing your dataset")
├── weights/                     # SR weight files downloaded or trained by you
│   ├── RealESRGAN_x4plus.pth
│   ├── 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth
│   ├── 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
│   └── spanx4_ch48.pth
├── sr_models/                   # optional: your own SR model classes (method 2)
├── sr_images/                   # optional: SR outputs produced by external tools (method 3)
│   └── <method_name>/           # same relative paths as the dataset images, .png
└── runs/                        # created by SR4Rec; never edit by hand
    ├── .cache/                  # cached LR/SR images, splits and recognizer checkpoints
    └── <run_name>/              # one folder per run (report.md, CSV files, plots, ...)
```

- Every path in `sr4rec.yaml` is relative to the folder that contains `sr4rec.yaml` (or absolute),
  so the whole project folder can be moved to another machine.
- `sr4rec init data/<dataset_name>` writes `sr4rec.yaml` into the current folder with
  `dataset: data/<dataset_name>`; it never overwrites an existing file without `--force`.
- `weights/`, `sr_models/` and `sr_images/` are recommendations: SR4Rec reads exactly the paths you
  declare and never searches for files.
- `runs/` is created by SR4Rec (`runtime.output_dir`); the cache is `runs/.cache/`
  (`runtime.cache_dir`). Deleting the cache only makes the next run slower.

## 4. Quick start

```bash
cd my_project
sr4rec init data/my_dataset/        # 1. validate the dataset, write ./sr4rec.yaml
# 2. edit sr4rec.yaml: put your SR weights in weights/ and list them under `sr:`,
#    set the split if needed, confirm every [CHECK] line (then delete the marker)
sr4rec run sr4rec.yaml --dry-run    #    check everything in under a minute
sr4rec run sr4rec.yaml              # 3. run; then open runs/<run_name>/report.md
```

`sr4rec run` refuses to start while a `[CHECK]` marker remains. At the end it prints a 4-5 line
summary and the path of `report.md`. Running again reuses every cached step (SR outputs and trained
recognizers); adding an SR model trains only the recognizers of that model.

Tip for large studies (protocol `matched` trains one recognizer per SR model, backbone and seed;
`--dry-run` prints the number of trainings and an estimated time): screen many SR models first with
`protocol: fixed_recognizer`, then run `matched` on the promising ones.

Python API (identical results to the CLI):

```python
from sr4rec import Run, load_config

result = Run(load_config("sr4rec.yaml")).execute()
print(result.report)          # path of report.md
```

## 5. Preparing your dataset

```
data/<dataset_name>/
├── images/        # all images; any sub-folder structure is allowed
└── labels.csv     # columns: path,label[,split]
```

| Column | Required | Rule |
|---|---|---|
| `path` | yes | relative to `images/`, with `/`, unique |
| `label` | yes | class to recognise (identity, species, ...), any string |
| `split` | no | `train`, `val` or `test` |

Split (block `split` of `sr4rec.yaml`): `source: auto` uses the `split` column when present and
the ratios otherwise; `column` requires the column (without `val` rows, `val_from_train` = 10% of
each class's train images become val); `ratios` always splits by `ratios` (default 0.7 / 0.1 / 0.2,
each strictly between 0 and 1, summing to 1), per class: test = max(1, ⌊n × test + 0.5⌋),
val = max(1, ⌊n × val + 0.5⌋), train = the rest (at least 1), so a class needs at least 3 images with
the defaults. The split seed is independent of the recognizer seeds. The report prints the declared
ratios and the actual counts.

```python
# Folder-per-label layout (images/<label>/*.jpg) -> labels.csv
from pathlib import Path
import pandas as pd

root = Path("data/my_dataset/images")
exts = {".jpg", ".jpeg", ".png", ".bmp"}
rows = [{"path": p.relative_to(root).as_posix(), "label": p.parent.name}
        for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
pd.DataFrame(rows).to_csv("data/my_dataset/labels.csv", index=False)
```

```python
# Identity encoded in file names (e.g. 017_s1_03.jpg -> person 017) -> labels.csv
from pathlib import Path
import pandas as pd

root = Path("data/my_dataset/images")
rows = []
for p in sorted(root.glob("*.jpg")):
    person = p.stem.split("_")[0]
    rows.append({"path": p.name, "label": person})
pd.DataFrame(rows).to_csv("data/my_dataset/labels.csv", index=False)
```

`sr4rec init` checks the format (missing columns, missing or unreadable files, duplicate paths,
classes too small for the ratios, identical images in two splits) and reports errors by line
number of `labels.csv`. Full guide, conversion scripts and common errors:
[docs/preparing_dataset.md](docs/preparing_dataset.md).

## 6. Adding SR models

| I have ... | Use | CPU latency measured? |
|---|---|---|
| a `.pth` / `.safetensors` file of an architecture spandrel supports | 1. `weights` | yes |
| the PyTorch code of the model (with or without weights) | 2. `module` | yes |
| only SR output images, or a model that is not PyTorch | 3. `images` | no |

```yaml
sr:
  - name: realesrgan                          # method 1: weights file (spandrel)
    weights: weights/RealESRGAN_x4plus.pth
  - name: my_sr                               # method 2: your PyTorch class
    module: sr_models/tiny_espcn.py:TinyESPCN
    weights: weights/tiny_espcn_x4.pth        # optional; passed to __init__(weights=...)
  - name: my_method                           # method 3: pre-computed SR images
    images: sr_images/my_method/
```

Method 2 interface: a `torch.nn.Module` with an integer class attribute `scale`,
`__init__(weights=None)`, and `forward(x)` mapping (N, 3, H, W) in [0, 1] to
(N, 3, H·scale, W·scale). Method 3: PNG files with the same relative paths as the dataset images
(`.png` extension), made from the LR images SR4Rec writes to `runs/<run_name>/lr/` (synthetic) or
from the dataset images (native-lr); `matched` needs train, val and test, `fixed_recognizer` test only.

**Always run `sr4rec run sr4rec.yaml --dry-run` first**:

```
Config              OK
Dataset             OK  (300 images, 10 classes)
Split               OK  train 180 / val 20 / test 100 images
SR models
  bicubic           OK  x4  built-in baseline
  realesrgan        OK  x4  weights  ESRGAN (spandrel)  sha256 4fa0d38...
  my_sr             OK  x4  module   sr_models/tiny_espcn.py:TinyESPCN
  my_method         OK  x4  images   300/300 images found
Plan                protocol matched: 4 image pipelines (bicubic + 3 SR) + HR, 3 backbone(s), 3 seed(s)
                    = 45 recognizer trainings (45 not cached), 900 SR images to compute
Estimated time      about 2.1 h on NVIDIA GeForce RTX 3090 (from a timed mini-batch)
Dry run passed. Start the real run with: sr4rec run sr4rec.yaml
```

**Validated SR models** (download them yourself into `weights/`):

| Model | File | Trained for | SHA-256 | Source |
|---|---|---|---|---|
| Real-ESRGAN x4plus | `RealESRGAN_x4plus.pth` | real-world degradation | `4fa0d389...9682f1` | [Real-ESRGAN v0.1.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0) |
| SwinIR-M x4 classical | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth` | bicubic degradation | `129dc773...98610a` | [SwinIR v0.0](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) |
| SwinIR-M x4 real-world (GAN) | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | real-world degradation | `b9afb61e...cbcfc6` | [SwinIR v0.0](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) |
| SPAN x4 | `spanx4_ch48.pth` | bicubic degradation | recorded at release | [SPAN](https://github.com/hongyuanyu/SPAN) |

Full hashes and step-by-step guides for the three methods: [docs/adding_sr_models.md](docs/adding_sr_models.md).

**Security:** method 2 runs the Python file you give it; only use code you trust.

Validated recognition backbones (timm): `resnet18`, `mobilenetv3_small_100`, `convnext_tiny`.
Any other timm model name runs with an "experimental" warning.

## 7. Examples

All examples use small subsets of **real** public datasets and run on a CPU. The subsets are
rebuilt from your own downloads (SR4Rec does not redistribute the images):

```bash
python scripts/make_examples.py --pets <path_to_oxford_iiit_pet>   # examples/data/pets_mini
python scripts/make_examples.py --earvn <path_to_EarVN1.0>         # examples/data/earvn_mini
python scripts/make_examples.py --lfw <path_to_lfw>                # examples/data/lfw_mini (faces)
python examples/sr_models/train_tiny_espcn.py                      # example SR weights
```

| Config | Shows | Time (laptop CPU) |
|---|---|---|
| `examples/configs/full_reference.yaml` | every key with full comments (= `sr4rec init` output) | n/a |
| `01_quickstart.yaml` | minimal workflow (= `sr4rec reproduce quickstart`) | ~5 min |
| `02_synthetic_pth.yaml` | method 1 with the 4 validated weight files | 1-2 h (minutes on a GPU) |
| `03_native_lr.yaml` | `mode: native-lr`, results by image size (ears) | ~5 min |
| `04_fixed_recognizer.yaml` | cheap screening protocol | ~3 min |
| `05_split_ratios.yaml` | your own split ratios | ~5 min |
| `06_sr_module.yaml` | method 2 (Python class) | ~5 min |
| `07_sr_images.yaml` | method 3 (pre-computed images) | ~5 min |
| `08_faces_lfw.yaml` | face identity recognition | ~5 min |

Run for example `sr4rec run examples/configs/01_quickstart.yaml` from the repository root.

**Privacy note (faces):** face images are biometric data. SR4Rec ships no face image. Follow the
terms of the original dataset and the data-protection rules that apply to you, and use face
examples for research only. See [examples/README.md](examples/README.md).

## 8. Configuration reference

`sr4rec init` writes a configuration in which every key documents its type, whether it is
required, its default and the complete list of accepted values (or the range). Unknown keys and
invalid values are rejected with a message that names the key and the accepted values.

| Block | Keys |
|---|---|
| top level | `dataset`, `mode` (synthetic, native-lr), `scale` (2, 3, 4), `protocol` (matched, fixed_recognizer) |
| `split` | `source` (auto, column, ratios), `ratios` {train, val, test}, `val_from_train`, `seed` |
| `sr[]` | `name`, one of `weights` / `module` (+ optional `weights`) / `images`, `tile` |
| `recognizer` | `backbones`, `pretrained`, `input_size`, `seeds` |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `horizontal_flip`, `num_workers` |
| `latency` | `enabled`, `lr_size`, `threads` |
| `comparisons` | `enabled`, `count`, `selection` (mixed, random, all), `backbone` |
| `runtime` | `device` (auto, cuda, mps, cpu), `output_dir`, `cache_dir`, `run_name` |

Complete table (generated from the schema): [docs/config_reference.md](docs/config_reference.md).

## 9. Reading the report

`report.md` starts with the question and a rule-generated answer, followed by: summary, setup
(including the SR weights, their degradation type and SHA-256), main results (Rank-1 mean ± sd,
Δ vs bicubic, 95% CI, Holm-adjusted p, same-sign count, verdict), Rank-5 / CMC / macro P, R, F1,
results by image size, PSNR/SSIM vs recognition (synthetic mode), CPU latency, before/after-SR
panels, warnings, a glossary and reproducibility notes.

| Verdict | Condition |
|---|---|
| `▲ significant gain` | Δ > 0, 95% CI excludes 0, p (Holm) < 0.05 |
| `▼ significant harm` | Δ < 0, 95% CI excludes 0, p (Holm) < 0.05 |
| `● no difference` | otherwise |
| `insufficient data` | size bin with fewer than 50 test images |

In `native-lr` mode there is no HR reference: no PSNR/SSIM and no HR upper bound. Public SR weights
were trained on natural images (DIV2K and similar) and may not match your data; the report warns
about it. Details: [docs/report_guide.md](docs/report_guide.md); methods and statistics:
[docs/methods.md](docs/methods.md).

## 10. Output files

```
runs/<run_name>/
├── report.md             # human-readable report, read this first
├── plots/                # accuracy_by_size.png, cmc.png, quality_vs_accuracy.png (synthetic)
├── metrics.csv           # backbone, method, seed, size_bin, n_images, rank1, rank5, precision,
│                         #   recall, f1, psnr, ssim (fractions in [0, 1]; psnr in dB)
├── cmc.csv               # backbone, method, seed, rank1 ... rank10 (n/a when k > classes)
├── stats.csv             # backbone, size_bin, method, n_images, delta, ci_low, ci_high (pp),
│                         #   p_raw, p_holm, same_sign, verdict
├── predictions.csv       # backbone, method, seed, path, label, pred, true_rank, correct
├── quality.csv           # method, path, psnr, ssim per test image (synthetic)
├── latency.csv           # component (sr | recognizer | pipeline), sr, backbone, median_ms, p95_ms
├── checkpoints.csv       # recognizer checkpoints used (key, file, best epoch, best val Rank-1)
├── table.tex             # main results table (LaTeX, booktabs)
├── split.csv             # the frozen train/val/test split (path, label, split)
├── comparisons/          # one before/after-SR PNG per selected test image + index.csv
├── lr/                   # LR images of train/, val/, test/ (synthetic mode), for external SR
├── sr4rec.yaml           # the exact configuration used (paths relative to this folder)
└── fingerprint.yaml      # environment, versions, seeds, data and weight hashes
```

`method` is `bicubic`, the SR name from the configuration, or `hr` (HR upper bound, synthetic mode).

## 11. Reproducing the paper

Demo definitions and expected results are shipped with the package (`sr4rec reproduce list`).

| Demo | Dataset | Setting | Role in the paper |
|---|---|---|---|
| `quickstart` | pets_mini | synthetic x4, CPU | quickstart |
| `d1_earvn` | EarVN1.0 | native-lr x4, matched, ResNet-18, 3 seeds | main result, results by image size |
| `d1_earvn_fixed` | EarVN1.0 | as D1 with protocol fixed_recognizer | what the protocol changes |
| `d2_lfw` | LFW (>= 20 images per person) | synthetic x4, matched, ResNet-18, 3 seeds | does PSNR/SSIM predict recognition |
| `d3_cub` | CUB-200-2011 | synthetic x4, matched, ResNet-18, 1 seed | another domain (single run) |

```bash
python scripts/prepare_earvn.py --earvn <path> --out data/earvn   # D1
python scripts/prepare_lfw.py   --lfw   <path> --out data/lfw     # D2
python scripts/prepare_cub.py   --cub   <path> --out data/cub     # D3
# put the four validated SR weight files in weights/

python scripts/fetch_checkpoints.py d1_earvn     # level 1: published recognizers, no training
sr4rec reproduce d1_earvn --eval-only
sr4rec reproduce d1_earvn                        # level 2: full re-training on a GPU
```

Options: `--data-root` (default `data/`), `--weights-root` (default `weights/`), `--checkpoints-root`,
`--output-dir`, `--device`. Exit codes: **0** within tolerance, **2** metrics outside tolerance,
**3** missing data, weights, checkpoints or expected results (1 is any other error). The tables and
figures of the paper are generated from the run folders by `scripts/make_paper_assets.py`; the SR
correctness check of Appendix C is `scripts/check_sr_fidelity.py`. Reference hardware:
[docs/reference_environment.md](docs/reference_environment.md).

> Status of this pre-release: `sr4rec reproduce <demo>` needs that demo's dataset and SR weights
> (see above) to be present locally; without them it exits with code 3 and writes the demo
> configuration to `runs/<demo>_config.yaml`, which you can then run with `sr4rec run`. The
> `quickstart` demo has no frozen expected results in this development version, so
> `sr4rec reproduce quickstart` always exits with code 3 regardless of local data (see Appendix B
> of the paper).

## 12. FAQ

- **The GPU runs out of memory.** The run does not stop. SR retries the image with automatic tiling
  (half the size each time, down to 64 px) and finally continues on the CPU; recognizer training
  restarts with gradient accumulation (same effective batch size, smaller micro-batches) and finally
  on the CPU; evaluation halves its batch. Every fallback is listed under "Warnings and notes" in
  the report. To avoid the slowdown, set `tile: 256` on large SR models or lower `batch_size`.
- **`spandrel could not detect the architecture`.** Use method 2 with the authors' model code, or
  method 3 with images produced by their script.
- **`upscales x2, but the configuration uses scale 4`.** All SR models of a run must have the
  configured scale; run separate projects for other scales.
- **My dataset is very small.** Verdicts are computed over test images; with few test images the
  CIs are wide and most verdicts will be `● no difference`. Size bins with fewer than 50 test
  images are not tested.
- **Results differ between machines.** GPU kernels and library versions change results slightly;
  compare `fingerprint.yaml` files. `reproduce` uses tolerances of max(0.2 pp, 3 × sd over seeds).
- **The first run is slow, later runs are fast.** SR outputs and recognizers are cached in
  `runs/.cache/`; delete it to start from scratch.

## 13. Limitations, roadmap, contributing, licences

Limitations of v0.1: closed-set identification only (no verification or open-set protocols, no
attribute classification); one scale per run; a fixed training recipe (chosen for fair comparison,
sensitivity checked in demo D6); PSNR/SSIM only in synthetic mode; CPU latency only for methods 1
and 2. Roadmap: verification protocols, attribute tasks, multiple scales per run, more validated SR
models and backbones. Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) (including
the criteria for adding a validated SR model or backbone) and [CHANGELOG.md](CHANGELOG.md) for the
versioning policy (semantic versioning). Licences, sources and citations of every dataset, weight
file and checkpoint: [DATA_AND_MODEL_LICENSES.md](DATA_AND_MODEL_LICENSES.md).

## 14. Citation, license, acknowledgements

If you use SR4Rec, please cite the software (see [CITATION.cff](CITATION.cff); the Zenodo DOI is
added at the first release):

```bibtex
@software{sr4rec,
  title   = {SR4Rec: a toolkit for measuring whether super-resolution helps closed-set recognition},
  author  = {Le Quang, Thai and Hoang, V. T.},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/quangthai121121/sr4rec},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

SR4Rec is released under the [MIT License](LICENSE). It builds on PyTorch, timm, spandrel, NumPy,
SciPy, statsmodels, scikit-learn, pandas, Pillow, matplotlib, pydantic and Jinja2. SR weights,
backbone weights and datasets keep their own licences (see DATA_AND_MODEL_LICENSES.md).
