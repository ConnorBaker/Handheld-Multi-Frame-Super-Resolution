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
