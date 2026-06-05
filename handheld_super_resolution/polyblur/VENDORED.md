# Vendored: Polyblur

Pure-Python core of **Polyblur** (fast blind deblurring of small anisotropic
blurs), vendored from https://github.com/teboli/polyblur (MIT, (c) 2022 Thomas
Eboli) at commit `86ca0d0368e51c08a5195f3bca75d4906a7e7829`.

Only the Python modules are vendored; the optional C++/CUDA fast-path extensions
(`domain_transform/`, `separable_convolution/`) are intentionally excluded — they
are not imported by the Python path (the prefilter uses `filters.bilateral_filter`),
and the FFT deblur path is pure `torch.fft` (GPU-portable). See LICENSE.

Used by the pipeline as an optional post-merge deblur stage (`postprocessing.deblur`,
applied in `raw2rgb.postprocess`). Local edits to run on the modern stack (numpy 2 /
torch 2.12) are marked with a `# vendored-fix` comment.

## Divergence from upstream (simplification pass)

The FFT deblur path is the only one we use, so the unused spatial/separable paths
were removed and the residual scipy coupling dropped. All changes verified
bit-identical against the pre-edit output (max-abs diff 0.0) over the deblur and
aberration stages:

- `filters.py` — dropped the dead spatial-separable convolution helpers (their
  `gaussian_*_separable_conv2d_` bodies referenced names that no longer exist —
  they would `NameError` if ever called) and the dead separable branch of
  `convolve2d`; removed the `import scipy.fft` (only `next_fast_len` was used, in
  already-commented-out code); folded `p2o`'s two `torch.roll`s into one.
- `deblurring.py`, `edgetaper.py` — removed unused `scipy` / `domain_transform`
  imports (the prefilter uses `filters.bilateral_filter`, not the C-ext path).
- `domain_transform.py` — **deleted**; it only wrapped the excluded C++ extension.
- `blur_estimation.py` — **bug fix:** the per-channel `q`-quantile normalization
  squeezed `value_min`/`value_max` to 2D, so the `(b,c,1,1)` broadcast against the
  `(b,c,h,w)` image collapsed across channels; restored the trailing singleton
  dims (`# vendored-fix`).
