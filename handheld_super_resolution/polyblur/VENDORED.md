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

Second pass (every live caller passes `method="fft"` and `q=0.0`, so the entire
spatial `method="direct"` / `direct_separable` machinery is unreachable; −271 net
lines, again bit-identical):

- `deblurring.py` — deleted the `PolyblurDeblurring` `nn.Module` (we call the
  functional `polyblur_deblurring` directly; the class's `patch_decomposition`
  branch even `NameError`ed on an undefined `handling_saturation`, proving it was
  never run) and the `compute_polynomial` dispatcher + `compute_polynomial_direct`;
  `inverse_filtering_rank3` now calls `compute_polynomial_fft` directly and the
  dead `method`/`nn`/`F` plumbing is gone.
- `filters.py` — `convolve2d` is now FFT-only (dropped the `method='direct'`
  branch and its `conv2d_` helper); removed the unused `crop`, `dirac`, and the
  numpy `gaussian_filter` (superseded by the torch `create_gaussian_filter` in
  `blur_estimation.py`, which imported but never called it).
- `edgetaper.py` — `edgetaper` lost its now-redundant `method` arg and a dead
  `_kernel` local.
- `blur_estimation.py` — `gaussian_blur_estimation` lost its `return_2d_filters`
  arg and the kernel-as-`(sigma, rho, theta)`-tuple branch it gated (used only by
  `direct_separable`; that branch's `kernel[2][...] = theta` referenced an
  undefined `theta` — another never-run NameError).
- The now-vestigial `method` arg was dropped from the public
  `polyblur_deblurring` and from both call sites (`raw2rgb._deblur`,
  `aberration/model.py`); the deblur path is unconditionally FFT.
