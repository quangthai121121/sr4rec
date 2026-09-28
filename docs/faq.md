# FAQ

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
  compare `fingerprint.yaml` files. `reproduce` uses tolerances of max(0.2 pp, 3 x sd over seeds).
- **The first run is slow, later runs are fast.** SR outputs and recognizers are cached in
  `runs/.cache/`; delete it to start from scratch.
