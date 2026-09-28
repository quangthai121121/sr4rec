# Reference environment

The expected results of the shipped demos (`src/sr4rec/demos/expected/`) are frozen on the machine
below. `fingerprint.yaml` of each run records the same fields, so any run can be compared with it.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 |
| CPU (latency) | 13th Gen Intel(R) Core(TM) i7-13700F |
| Operating system | Ubuntu 24.04.1 LTS |
| Python / PyTorch / CUDA | Python 3.12.3, PyTorch 2.11.0, CUDA 13.0 |
| SR4Rec | 0.1.0 |

Tolerances: full re-training `max(0.2 pp, 3 x sd over seeds)` per backbone and method;
`--eval-only` 0.01 pp (four decimals of the accuracy on the same type of hardware).

`requirements.lock` (repository root) pins one known-working set of exact package versions,
generated in a clean environment. It is not required to match the table above exactly — SR4Rec's
tolerances are designed to absorb this kind of drift between hardware and library versions across
time.