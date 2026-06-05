# Eval / sweep — running notes & findings

Context: after porting the pipeline to CuPy for the GB10, the output looked soft
(especially foliage) vs the Canon ground truth. This harness was built to (a)
quantify quality vs Canon with perceptual metrics, (b) automatically find where
the algorithm/config fails, and (c) sweep parameters (Optuna) to recover detail.

## Key findings (burst 0003, Samsung, ISO 100, 14 frames; scale 2)

- **The default config's multi-frame fusion is perceptually WORSE than a single
  frame vs Canon**: DISTS 0.277 (multi) vs 0.275 (single); **90/140 tiles** have
  multi worse than single. So at low ISO, the merge averages away detail faster
  than it denoises.
- The pipeline output is also **softer than a plain bilinear demosaic** of one
  Samsung frame (Laplacian var building 163 vs 242, foliage 44 vs 112).
- Worst tiles vs Canon are **high-frequency edges against sky** (bare branches,
  power lines) and **foliage** — the registration-free detector finds tiles where
  multi-frame keeps only ~9–16% of the single-frame sharpness.
- Merge kernel covariance eigenvalues are ~0.5–0.8 px (big axis) / ~0.3 px²
  (small axis); ~46% are near-isotropic. `k_detail` has a hard floor (~0.18):
  below it the scale-1 demosaic loses cross-color support in the 3×3 window and
  the output goes NaN.
- Denoise params (`k_denoise`, `D_th`) barely move scale-1 sharpness — at scale 1
  the merge is sampling-limited (3×3 window, ~1px spacing), so detail lives in
  `scale` (SR) and in alignment quality, not kernel width.

## Root-cause hypotheses (ranked)

1. **Alignment precision on fine texture.** Multi-frame helps the static building
   (Laplacian 163→206) but hurts foliage — averaging 13 frames across sub-pixel
   alignment error smears high-freq texture. Robustness isn't protecting it (the
   accumulated-robustness mask is ~12.6/13 *everywhere*, not discriminating).
   ⇒ levers: `ica.tuning.n_iter`, `block_matching.tuning.search_radii`,
   `robustness.tuning.t` (reject more aggressively).
2. **Kernel-regression merge is inherently ~1px-soft at scale 1** (by design, for
   denoising). The detail win is supposed to come from `scale=2`.
3. **Possible port-induced softening** — *cannot be ruled out*: numba can't run on
   the GB10, so I can't diff against the original implementation. "Softer than
   bilinear" + "multi worse than single" are yellow flags. Definitive check =
   run the original numba pipeline on a numba-capable GPU and diff.

## Design decisions

- Register **output → Canon frame** (compare at Canon's detail level); DISTS as
  the objective (alignment-robust); color-match before scoring.
- Cache the homography across sweep trials (geometry ~constant across configs).
- Two failure detectors: Canon-anchored (rigorous) + registration-free (fast,
  any burst, but conflates denoising).

## Caveats

- **Parallax / non-planar foreground**: one homography can't perfectly register
  Canon (different viewpoint) to the output, so per-tile metrics on foreground
  mix algorithm error with registration error. DISTS mitigates; the
  registration-free detector cross-checks.
- **Single-burst sweep** (0003) risks overfitting the params to one scene — the
  best config should be validated on the other bursts (0000–0004, 0050) before
  trusting it. The harness takes any `--burst/--canon`, so this is a follow-up.
- **DISTS at Canon res downsamples the (larger) scale-2 output**, which smooths
  it; ranking across configs is still valid but absolute DISTS is conservative.

## Environment note

- Installing `lpips`/`DISTS-pytorch` upgraded **torch 2.11 → 2.12** in the
  `handheld` env. The pipeline still works; one unit test
  (`test_gradient_cupy_matches_torch_sobel` at 4K) was relaxed to `atol=1e-3`
  because torch 2.12 tiles large convolutions with a different float32 reduction
  order (max diff ~5e-4, pure float noise).
- CFA of this dataset is **GRBG**.

## Sweep results (80 trials, burst 0003, scale 2, objective = DISTS vs Canon)

| config | DISTS↓ | LPIPS↓ | PSNR | SSIM | sharp_ratio↑ |
|---|---|---|---|---|---|
| single-frame | 0.300 | — | — | — | — |
| **default multi** | 0.291 | 0.313 | 23.4 | 0.803 | 0.062 |
| **swept best** | **0.234** | **0.292** | 23.3 | 0.787 | **0.165** |

Best config: DISTS **−19%** vs default and below single-frame (so multi-frame now
*helps*); sharpness-vs-Canon **2.7×**. PSNR/SSIM dip slightly — expected, the
result is sharper (those metrics favour blur). Visible win on the worst tiles
(power lines / branches against sky go from mush to defined) — see
`results/eval/final/worst_tile_default_vs_best_vs_canon.png`.

Best params (`results/eval/sweep/best_config.yaml`) and what they say about the
defaults being mistuned for a clean ISO-100 burst:

| param | default | best | reading |
|---|---|---|---|
| `ica.tuning.n_iter` | 3 | **8** | more LK iterations → tighter sub-pixel alignment → less smearing |
| `ica.tuning.sigma_blur` | 0 | 0.84 | mild pre-blur stabilises the gradient/alignment |
| `robustness.tuning.t` | 0.12 | **0.43** | reject misaligned frames far more aggressively (the foliage lever) |
| `robustness.tuning.Mt` | 0.8 | 0.20 | flag more flow-irregular tiles |
| `merging.tuning.k_denoise` | 3.0 | 1.38 | much less denoising |
| `merging.tuning.D_th` | 0.71 | ~0.00 | detail mode almost everywhere (default was denoise-biased) |
| `merging.tuning.k_stretch` | 4.0 | ~1.0 | near-isotropic small kernels |
| `merging.tuning.k_detail` | 0.25 | 0.18 | smaller (sharper) kernels, at the support floor |

Takeaway: the stock defaults over-denoise and under-reject for low-ISO bursts;
pushing toward detail-preservation + aggressive robustness recovers detail and
makes fusion a net win. **Caveat:** tuned on one scene — validate across
0000–0004/0050 (the harness takes any `--burst/--canon`) before adopting.

## Adversarial review (IMPORTANT — read before trusting the sweep)

A self-critique with experiments (`eval/synthetic.py`) walked back several claims:

- **The Canon-DISTS objective is gameable and the "best" config over-sharpens.**
  Synthetic ground-truth test (downsample a sharp image into a shifted burst,
  SR back, compare to the KNOWN original — perfect registration, no parallax):
  the swept "best" gives DISTS 0.073 but **PSNR 34.8 < default 36.2**, SSIM
  0.94 < 0.96, and **sharpness 109 > GT's 96** (invents high-freq). On real data
  it has **4.9× the flat-sky high-pass noise** of the default (1.30 vs 0.27).
  So the "−19% DISTS, 2.7× sharper" win was largely the metric being gamed by
  over-sharpening/noise amplification, not true fidelity. DISTS rewards
  perceptual texture; vs a sharp JPEG it favours adding high-freq.
- **The Canon comparison is not ground truth.** Different sensor/lens/viewpoint →
  a single homography can't register a 3-D scene (parallax); Canon JPEG carries
  its own demosaic/sharpening/tone curve; my output is downsampled to Canon res;
  color-match is crude. Foreground/tree tiles mix registration error with
  algorithm error. ⇒ demote Canon-DISTS to a sanity check; tune against
  **synthetic GT fidelity** instead.
- **The implementation is, however, fundamentally correct.** On synthetic data
  the default config's multi-frame x2 cleanly beats baselines vs known GT:
  PSNR 36.2 vs 30.8 (single↑2) vs 29.7 (bilinear↑2); DISTS 0.108 vs 0.148 vs
  0.167; sharpness 67.7 vs 17.8 (3.8×). A broken merge/align/port could not
  produce a +5.4 dB SR gain. So "multi < single" / "softer than bilinear" on REAL
  data is the real-world confounds (over-denoising clean ISO-100 + sensor/lens
  MTF + parallax), not a gross port bug. (Bit-vs-numba still unverified.)
- **Generalization: poor.** Sweep = one scene, one (flawed) metric. The "best"
  (no denoise, aggressive rejection) is tuned to clean ISO-100 static and would
  hurt noisy/dynamic bursts. Defaults remain the sane general-purpose choice.
- **Patch size is constrained and global.** Tile sizes are hard-limited to
  {8,16,32,64} by the torch L2 box filters (`tile=4` → "Box filter not
  implemented"); patch size is fixed per pyramid level and cannot adapt to local
  content, so a 16-px patch over a depth/motion boundary (foliage) gets one flow
  vector → smear. On static synthetic, 16 (default) is best, 32 slightly worse;
  the dynamic-content case is untested and partly blocked by that constraint. I
  did NOT sweep patch size.

Bottom line: trust `eval/synthetic.py` (fidelity vs known GT) over the
Canon-DISTS sweep. The pipeline works; the defaults over-denoise low-ISO; the
sweep "win" was metric-gaming; the real detail ceiling is mostly the phone
optics/sensor + the Bayer/merge reconstruction, not a fixable parameter.

## Follow-up (items 1-3): fidelity sweep, independence, patch/robustness

### Item 1 — fidelity-anchored, multi-scene sweep (`eval/sweep_synthetic.py`)
Objective = mean over 3 scenes of 0.5*(1-SSIM)+0.5*DISTS vs SYNTHETIC GT
(un-gameable: anchored to truth). default loss 0.104 (SSIM 0.926, DISTS 0.133,
sharp/gt 0.53) -> best 0.071 (SSIM 0.935, DISTS 0.077, **sharp/gt 0.78 — still
< 1, no over-sharpening**, unlike the Canon sweep which hit sharp/gt > 1 + 4.9x
sky noise). Best params: `ica.n_iter` 3->6, `robustness.t` 0.12->0.31,
`k_denoise` 3->0.72, **`D_th` kept high (0.84)**, `k_detail` ~default (0.26),
`k_stretch` ~1.2. ⇒ the real win is **alignment + moderate rejection**, NOT
killing denoise / over-sharpening. Far more defensible than the Canon "best".
Saved: `results/eval/sweep_synth/best_config.yaml`.

### Item 2 — independent GT + forward-model sensitivity (`eval/groundtruth.py`)
GT from the Canon **RAW (CR2 via rawpy)**, not the JPEG. Across four downsample
forward models, default-config results vs GT:

| forward model | multi PSNR/SSIM/DISTS | single | bilinear |
|---|---|---|---|
| area     | 38.3 / .947 / .116 | 35.2 / .930 / .149 | 33.4 / .926 / .164 |
| bicubic  | 38.3 / .947 / .109 | 35.3 / .930 / .141 | 33.5 / .926 / .158 |
| lanczos  | 38.3 / .947 / .109 | 35.1 / .929 / .143 | 33.4 / .925 / .160 |
| gaussian | 34.6 / .935 / .138 | 33.1 / .921 / .166 | 32.0 / .917 / .177 |

multi > single > bilinear for EVERY model and metric ⇒ the "multi-frame SR
genuinely recovers detail" conclusion is **not circular** (independent of GT
source and forward model).

### Item 3 — patch size + robustness (`eval/diagnostics.py`, `block_matching.py`)
- Generalized the L2 box filter to **any tile size** (was hard-coded {8,16,32,64}).
- Patch size on static high-freq GT: tile **16 (default) best** PSNR 38.31 vs
  37.9 (tile 8/12); smaller → marginally more sharpness, less fidelity. Patch
  size is NOT the lever for static fine detail. (tile 24/32 fail only because the
  512-px synthetic LR is too small for the coarsest pyramid level — image-size,
  not box-filter, limit.)
- **Robustness under-rejects motion** (the real foliage lever): injecting an
  independently-moving center region (per-frame 30-px-shifted content), it is
  accepted **4.6/7 frames vs 6.2/7 static** — only ~25% down-weighted, so clearly
  inconsistent content still gets ~66% weight ⇒ ghosting/smear. The local-mean
  robustness is weak when moving content shares color statistics; `robustness.t`
  is the lever (fidelity sweep chose 0.31 vs default 0.12).

### CuPy primitives the size-generalizations could use (researched vs docs + cupy 14.1.1)

Verified against the CuPy docs and the installed build:
- `cupyx.scipy.ndimage`: `uniform_filter(+1d)`, `minimum_/maximum_filter`,
  `median_filter`, `gaussian_filter`, `correlate/convolve(+1d)`, `sobel/prewitt`,
  `generic_filter`; `map_coordinates`, `affine_transform` (has an `opencv` mode),
  `spline_filter`, `zoom`, `shift`, `rotate`.
- `cupyx.scipy.signal`: `convolve/correlate`, `fftconvolve`, `oaconvolve`,
  `convolve2d/correlate2d`, `sepfir2d` (ODD-length symmetric FIR only — errors on
  even widths, so NOT usable for an even box filter), `choose_conv_method`.
- `cupy.lib.stride_tricks`: `as_strided`, `sliding_window_view`.
- scatter: `cupyx.scatter_add` / `cupy.ndarray.scatter_add` / `cp.add.at`
  (atomicAdd, float32; == `np.add.at`).
- custom: `RawKernel`, `RawModule`, `ElementwiseKernel`, `ReductionKernel`.
- NOT in CuPy core (would need cumsum, or cucim which is not installed):
  integral image / summed-area table, template matching / NCC, `block_reduce`.

Corrected recommendations (verified):
- **Box filter** → `uniform_filter(x², t, mode='constant')*t²` (VERIFIED == the
  torch valid ones-conv on the interior), or a separable pair of 1-D ones-convs
  (6x faster than the 2-D conv). `sepfir2d` does NOT work (odd-only). No built-in
  integral image. Current 2-D `torch.ones` conv is correct but the least
  idiomatic/fastest expression.
- **L1 integer-shift sampling** → padded slicing (cupy core) or
  `sliding_window_view`; `map_coordinates` is unnecessary for integer shifts
  (verified bit-identical to slicing).
- **Merge** is a bounded 3x3 GATHER (1 thread/output). It *is* expressible with
  cupy primitives (per-offset fancy-index/`map_coordinates` gather + elementwise
  weight + per-channel masked add, or `cupyx.scatter_add`) — so a custom kernel
  isn't *required* for correctness — but that vectorized form is ~9x slower
  (full-grid temporaries; see perf section). The fused per-output kernel is the
  right call here; it now lives as a Warp kernel (`warp_ops.py`), not a RawModule.
- **Reshape+sum tile reductions / `map_coordinates(order=1)` sub-pixel warp /
  closed-form 2x2** are already idiomatic cupy — keep (no `block_reduce` in core).

### Rewrite to library primitives (done) + regression harness

All the size-generalized / hand-written kernels were rewritten to stock CuPy /
torch primitives; the ORIGINAL implementations are preserved in
`test/reference_kernels.py` and `test/test_kernel_equivalence.py` asserts the
rewrites match them (38 cases across bayer/grey, iso/steerable, scale 1/2,
robustness denoiser). All pass.

| op | before | after | verified |
|---|---|---|---|
| L2 box filter | `t x t` `torch.ones` conv | `F.avg_pool2d * t^2` (any t) | rtol 1e-3 |
| L1 block matching | `map_coordinates(order=0)` | integer zero-filled gather (`cp` slicing) | bit-identical |
| Dogson warp-upscale | `RawModule` kernel | vectorized 3x3 Dogson-weighted gather | rtol 1e-3 |
| merge accumulate | `RawModule` (atomic-free gather) | **NVIDIA Warp kernel** (`warp_ops.py`) | rtol 1e-3 |
| merge accumulate_ref | `RawModule` | **NVIDIA Warp kernel** (`warp_ops.py`) | rtol 1e-3 |

**Perf (resolved — the merge is now a Warp kernel).** The intermediate
*vectorized* merge was a real ~9x regression at scale 2 (real 12->48 MP burst
0003: **3.9 s -> 37.5 s**): a 9-tap gather + cov-interp + channel routing has no
fusable cupy expression, so it materialized many full-grid (HxWx, ~190 MB)
temporaries. That's the one op where the fused per-output kernel is genuinely
justified (bounded gather with per-output anisotropic kernels + Bayer routing).

Tried in order: (1) re-promote the `RawModule` C++ string — works but is
opaque C in a Python string; (2) `cupyx.jit.rawkernel` (Python->CUDA via NVRTC)
— **fails to compile** the merge: `threadIdx` is `uint32` and the index math
mixes it with `int32` shape args, raising `TypeError: Cannot cast from 'int32'
to uint32 with casting rule same_kind`; workable only with pervasive explicit
casts that defeat the readability goal. (3) **NVIDIA Warp** (`warp-lang`, already
in the stack) — its kernel DSL handles the int/float typing cleanly, JITs via
NVRTC, and reads as plain Python. CuPy arrays wrap zero-copy via `wp.from_dlpack`
(launch device is `cuda:<dev>`, NOT `str(cp_array.device)` which gives the
unparseable `<CUDA Device 0>`). Benchmarked at scale-2 HR (6048x8064x3, 48.8 MP):
**Warp 0.0166 s/frame vs the original RawModule 0.0257 s/frame** — Warp is ~1.5x
*faster* than the kernel it replaces, and the full burst is back to ~6.5 s. The
RawModule/`map_coordinates` originals are preserved in
`test/reference_kernels.py`; `test/test_kernel_equivalence.py` (38 cases) asserts
the Warp merge matches them. The other rewrites (box filter, L1, Dogson) are
equal-or-faster pure cupy/torch — keep.

**Aside — Warp tiles.** Warp 1.14 *does* expose `wp.tile_load` /
`wp.tile_matmul` / `wp.tile_zeros` / `wp.tile_store` as kernel builtins (an
earlier note here that it didn't was wrong — they're not module-level attrs, so
`hasattr(wp, "tile_load")` is False, but they compile and run inside `@wp.kernel`;
verified a tile_load+tile_matmul kernel matches `numpy @` on the GB10). They're
not used for the merge because the merge is a *gather*, not a matmul — a plain
per-output Warp kernel fits it. Warp tiles would suit the tile-structured
align/block-matching/Hessian ops if those are ever moved off torch.

### Net correction to earlier claims
- Pipeline + port: **correct** (multi-frame SR beats baselines vs true GT across
  forward models). Earlier real-data "multi < single" was over-denoising clean
  ISO-100 + the unreliable Canon metric, not a bug.
- The Canon-DISTS sweep "best" was **over-claimed** (metric-gamed, over-sharpens).
  The fidelity sweep's config is the defensible one; the biggest real lever is
  **alignment iterations + robustness rejection**, not the merge kernel or patch
  size.
