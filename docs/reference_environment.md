# Reference environment

The expected results of the paper demos (`src/sr4rec/demos/expected/`) are frozen on the machine
below. `fingerprint.yaml` of each run records the same fields, so any run can be compared with it.

| Item | Value |
|---|---|
| GPU | to be filled when the paper runs are frozen |
| CPU (latency) | to be filled |
| Operating system | to be filled |
| Python / PyTorch / CUDA | to be filled |
| SR4Rec | 0.1.0 |

Tolerances: full re-training `max(0.2 pp, 3 x sd over seeds)` per backbone and method;
`--eval-only` 0.01 pp (four decimals of the accuracy on the same type of hardware).



Hạng mục	Máy bạn	                    Kết luận
Python      3.12.3                      Đúng dải 3.10–3.12
PyTorch     2.11.0+cu130, CUDA bật      Ổn
GPU         RTX 3080                    Dùng được
SR4Rec      0.1.0.dev0                  Import và CLI OK
Test        132 passed, 1 skipped       Xanh