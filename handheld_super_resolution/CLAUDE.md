# handheld_super_resolution/CLAUDE.md

The classical multi-frame burst super-resolution pipeline. Public entry:
`process(burst_path, config)` → `main(ref, comps, config)`.

See the repo-root `CLAUDE.md` for the verification policy and environment, and
`test/CLAUDE.md` for how changes here are guarded.

## Pipeline (the order `main` runs)

raw → grey → pyramid **alignment** (block-matching L2-FFT + L1, then ICA
Lucas–Kanade) → **robustness** mask → steerable-kernel **covariance** → **merge**
(accumulate weighted samples into num/den) → divide → (optional post-processing).

## Backends (important)

numba can't JIT here (root `CLAUDE.md` §2), so the **live** GPU path is torch +
cupy + Warp. The original numba `@cuda.jit` *compute* kernels have been **removed**
— their torch/cupy/Warp replacements live in `cupy_ops.py` / `warp_ops.py`, and the
preserved originals used as regression oracles live in the tests, not here (see
`test/CLAUDE.md`). numba remains only for (a) array plumbing
(`cuda.to_device` / `synchronize`) and (b) the unported, default-off frame-count
denoisers in `utils_image.py`. Don't reintroduce compute kernels in numba.

## Module index

- `super_resolution.py` — orchestration: `process` (I/O, EXIF, config) and `main`
  (the algorithm above).
- `alignment.py` — builds the reference Gaussian pyramid and runs per-level
  alignment (calls block matching, then ICA).
- `block_matching.py` — coarse alignment: L2 via FFT cross-correlation; L1 via
  integer search (`block_matching_l1_cupy`).
- `ICA.py` — inverse-compositional Lucas–Kanade refinement (`ica_iterate_cupy`).
- `kernels.py` — steerable kernel covariance from the structure tensor
  (`estimate_cov_cupy`).
- `robustness.py` — per-pixel robustness mask (guide image, local stats, noise
  model, flow-discontinuity penalty).
- `merge.py` — Alg. 4 / 11 accumulation; delegates to the Warp merge.
- `cupy_ops.py` — CuPy/torch ports of the numba kernels (the live compute layer);
  re-exports the merge from `warp_ops`.
- `warp_ops.py` — the merge accumulation as NVIDIA Warp kernels (a per-thread
  gather, deliberately not tiles).
- `utils_image.py` — grey conversion (FFT / decimate), downsampling, gradients,
  Hessian, GAT, and the (dead) frame-count denoisers.
- `params.py` — config validation + SNR-based tile-size / parameter resolution.
- `utils.py` — small helpers (divide / add via cupy, timing).
- `fast_monte_carlo.py` — noise-curve estimation; `raw2rgb.py` — post-processing;
  `utils_dng.py` — DNG / raw loading.

Config schema lives in `configs/default.yaml`, resolved by `params.py`.
