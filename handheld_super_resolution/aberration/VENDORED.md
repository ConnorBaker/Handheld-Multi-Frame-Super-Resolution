# Vendored: two-stage optical-aberration correction (Stage B)

Spatially-varying deblur + learned chromatic defringer, from
https://github.com/teboli/fast_two_stage_psf_correction (MIT, (c) 2022 Thomas
Eboli, ECCV 2022) at commit `fbe8e24097900e15d8ce2a165cc9e9958708805e`. See LICENSE.

What's vendored and why:
- `basicblock.py` — KAIR building blocks for the ResUNet defringer (torch-only).
- `model.py` — the `ResUNet` defringer (verbatim from upstream `modules.py`) +
  `AberrationCorrection`, an adaptation of upstream `OpticsCorrection` that
  (a) REUSES our vendored Polyblur for the per-tile blur estimation + polynomial
  inverse (upstream's `polyblur_functions.py` is an older duplicate of it), and
  (b) REPLACES `kornia.contrib` tiling with `torch.nn.functional.unfold`/`fold`
  (verified identity round-trip < 1e-5) — so **no kornia dependency**.
- `checkpoints/super_tiny_epoch_1000.pt` — the defringer weights (647 KB, 0.16M
  params; loads strict). Upstream also ships a larger `tiny` (2.3 MB); super_tiny
  defringed real images at least as well and is smaller, so only it is vendored.

Used by the pipeline as the optional `postprocessing.aberration` stage (Stage B),
which takes precedence over `deblur` (Stage A) / `sharpening` in
`raw2rgb.postprocess`. Local edits to run on torch 2.12 / numpy 2.4 are marked
`# vendored-fix`. Runs pure `torch` on GPU.
