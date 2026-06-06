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

Warp **also has device-side FFT** — `wp.tile_fft` / `tile_ifft` (cuFFTDx via
libmathdx, `enable_mathdx_fft=True`) plus `tile_matmul` (cuBLASDx) — same
compile-time-builtin caveat (`dir(wp)`/`hasattr` miss them). Verified on the GB10
by running Warp's shipped `example_tile_fft` (round-trips correctly, exit 0) and
`example_tile_filtering`. So a fused on-device FFT Wiener/RL deconvolution is
implementable in Warp (a "Warp has no FFT" note elsewhere would be WRONG). See the
deconvolution section below; `eval/deconv.py` drives the actual deblur on
`torch.fft` for now (full-frame 24 MP; the Warp path needs row/col tile decomp).

### Net correction to earlier claims
- Pipeline + port: **correct** (multi-frame SR beats baselines vs true GT across
  forward models). Earlier real-data "multi < single" was over-denoising clean
  ISO-100 + the unreliable Canon metric, not a bug.
- The Canon-DISTS sweep "best" was **over-claimed** (metric-gamed, over-sharpens).
  The fidelity sweep's config is the defensible one; the biggest real lever is
  **alignment iterations + robustness rejection**, not the merge kernel or patch
  size.

## Real vs hallucinated detail — DxO 41-frame burst (`eval/hallucination.py`)

The user shot a 41-frame handheld burst (Sony a6300, ISO 200, dusk park) and
exported a per-frame DxO PhotoLab **DeepPRIME** render of every frame
(`*_DxO-reference.dng`, scene-linear). That gives **41 independent single-frame
reconstructions of the SAME static scene** at different sub-pixel offsets — the
setup needed to separate REAL detail from HALLUCINATED detail without a ground
truth.

**Principle.** DeepPRIME is a single-image learned reconstructor. Real scene
detail is **cross-frame consistent** (static scene, fixed optics) and survives a
cross-frame consensus; noise/aliasing-seeded **invention** is per-frame and is
destroyed by the consensus. On a per-ROI registered stack the single-frame HF
power splits orthogonally into consistent (REAL) + per-frame (invented + noise);
`real_fraction = consistent / total`.

**Method validity (all executable, in `test/test_hallucination.py` + the driver):**
- Shift convention round-trips; **resampling-only control scores `real_fraction`
  0.90–0.998**, so low values are genuine, not cubic-interpolation artifacts.
- The burst PANS ~71 px with **parallax** (corner spread to 48 px at ~0° rotation),
  so registration is **per-ROI** (template-match + sub-pixel phase corr), never a
  single global homography — different depths register independently.

**Findings.**
- **≈77 % of DxO's single-frame HF energy is cross-frame REAL; ≈23 % is per-frame
  (invented + residual noise)** (HF-power-weighted over ~300 tiles). This 77 % is
  an **upper bound on real**: DeepPRIME is deterministic, so *scene-conditioned*
  hallucination repeats every frame and is miscounted as real — cross-frame
  consistency only catches the *noise/aliasing-seeded* kind.
- **Noise vs invention** (`inconsistency_structure`): in **flat sky** the
  inconsistent component is **white** (lag-1 autocorr ≈ 0.01 vs 0.00 control) =
  residual under-denoising; in **textured regions** it is **structured at ~3 px**
  (autocorr 0.42–0.48) = synthesized texture. Visible in the montages: tree/bush
  *shape* is identical across frames 0/20/40 but the fine *leaf texture differs*.
- **Cross-check vs our prior-free multi-frame merge:** HF(merge) ≈ HF(DxO
  consensus) on real detail (bush ratio 0.97) — two independent reconstructions
  converging on the same "real" detail. Our merge keeps only the consistent ~77 %
  and skips the per-frame invention.

**Honest negatives (do not overstate):**
- The scene has **no clean step edges** (0 found even at relaxed thresholds:
  natural foliage/sky/grass; man-made bits are thin bars, not steps), so the
  edge-width (ESF) check is **legitimately N/A** here. The earlier
  "merge σ1.58 < DxO σ1.97" claim **did not reproduce** and was dropped;
  `eval/psf.py` (validated ESF estimator, round-trips σ≥0.8 to ±0.02) and
  `find_step_edges` / `compare_edge_sharpness` are kept for a scene that has them.
- In **vegetation**, wind motion is also cross-frame inconsistent and is **not
  separable** from invention here — needs a static high-contrast target.

**Reproduce:** `python -m eval._halluc_run --burst <dxo_dng_dir> --out eval/halluc_out
[--merge auto|merge.npy]`. Cache + montages/heatmap land in `eval/halluc_out/`
(gitignored, multi-GB).

### Second scene — building facade (a6300, 28 frames; ARW + DxO DNG)

A burst with abundant straight edges (`Pictures/{ARW,DNG}/2025-09-06/007`,
`DSC*_DxO.dng`) — the case the park scene lacked — so the edge-width (ESF/PSF)
cross-check actually runs. Rigid man-made structure ⇒ **no wind confound** (the
park's caveat is gone). Run:
`python -m eval._halluc_run --burst <DNG dir> --pattern 'DSC*_DxO.dng' --arw <ARW dir> --merge auto --out eval/halluc_out_007`.

**Real vs hallucinated:** GLOBAL real_fraction **0.72** (28% per-frame). Resampling
floor 0.995 (genuine). The high-detail tiles are the building's horizontal louvers
**at the sampling limit**: each single DxO frame *aliases* them differently
(structured per-frame texture, lag-1 autocorr **0.75**) — a textbook single-frame
reconstruction ambiguity that our multi-frame merge resolves from the combined
sub-pixel samples. So the same story as the park, on rigid structure: edges/shape
consistent (real), finest texture substantially per-frame (invented).

**PSF / edge sigma (the new result — verified, and it corrects the park):** on the
3 gate-passing clean step edges,
- DxO single ≈ DxO consensus (1.61≈1.55, 2.19≈2.15, 1.88≈1.87) ⇒ DxO's edge WIDTH
  is cross-frame consistent = **real**, not per-frame.
- **DxO consensus edge sigma ≈ 1.87 px (median) vs our raw merge ≈ 2.14 px** ⇒ DxO
  resolves sharper REAL edges by ~0.25 px. Verified NOT an artifact: merge sigma is
  identical with/without the align-resample (2.14 both ways), and DxO's overshoot is
  low (0.05) so it is not winning by haloing.
- **Our shipped Polyblur deblur cannot close it**: inert at the shipped sigma_b=0.45
  (2.14→2.13), and even maxed out (sigma_b=0) only reaches 2.07. Polyblur is a
  conservative Gaussian-inverse tuned for ~1px softness; with c=0.352 and the
  facade's high max-gradient it under-deconvolves. DxO's lens-measured deconvolution
  is genuinely stronger on edges.

**Net, honest reconciliation:** the park-scene impression that "our merge is sharper
than DxO" was **never substantiated** (no clean edges there; the old σ1.58<1.97 claim
was dropped). On a scene with *measurable* edges, **DxO wins real edge resolution**
(its deconvolution), while **we win texture honesty** (the merge keeps only the
cross-frame-consistent ~72%; DxO's extra fine texture is ~28% per-frame invention) —
and noise. Closing the edge gap needs a PSF-informed deconvolution (we now have the
measurement: merge edge sigma ≈ 2.1 px), not the current blind sigma_b.

**Tooling:** the driver gained `--arw` (ARW and DNG in separate dirs), N-frame-aware
montage/edge frame selection (28≠41 frames), a scene_map.npz cache, and a dead-flat
ROI guard in `inconsistency_structure`.

### Deconvolution — matching DxO's edge sharpness from the merge (`eval/deconv.py`)

Follow-up to the facade-scene PSF result (merge edge ~sigma 2.10 px vs DxO 1.87).
Since we MEASURED the merge PSF (~Gaussian sigma 2.1), a non-blind deconvolution can
target it directly -- and it invents nothing: it is a deterministic operator on the
merge, which already holds only cross-frame-consistent (real) detail. The risk is
noise/ringing (bounded here, measured via overshoot + flat noise), not per-frame
fabrication.

**Conv/deconv backend survey (verified on the GB10; two meanings of "deconvolution"):**
- *Transposed conv* (the deep-learning "deconv" upsampling layer): PyTorch
  `nn.ConvTranspose2d` is built in; NOT an inverse filter.
- *Inverse-filter deblur* (Wiener / Richardson-Lucy): none ships a turnkey 2-D
  `deblur()`. PyTorch -> roll on `torch.fft`; CuPy `signal.deconvolve` is 1-D only
  (`wiener` is a denoiser) -> roll on `cupy.fft`; **Warp HAS `wp.tile_fft`/`tile_ifft`
  (cuFFTDx) + `tile_matmul`** -> a fused on-device Wiener/RL is implementable
  (verified the FFT examples run on sm_121). Primary path here is `torch.fft` (full
  24 MP frame); a separable-Gaussian Warp conv kernel is also provided + benchmarked
  (agrees with scipy to ~1e-4).

**Why Polyblur could not do this (it IS deconvolution, just bounded):** its
`compute_polynomial_fft` is a rank-3 polynomial inverse `p(K)=a3 K^3+..+beta`; the
high-frequency gain is capped at `p(0)=beta` (3x at default), but inverting a 2.1 px
Gaussian needs >=10x at the frequencies it kills. Plus its blind sigma-estimate is
fooled by high-contrast edges (large gradient -> small sigma^). So it caps at 2.07.

**Result (RL beats Wiener -- less halo + far less noise gain):**
| method | edge sigma | flat noise | deslanted edge overshoot |
|---|---|---|---|
| DxO consensus (target) | 1.87 | 0.0005 | 1.8% |
| our merge (raw) | 2.10 | 0.0015 | 2.0% |
| Wiener full(s2.1) nsr=0.03 | 1.39 | 0.0021 | (overshoots target + amplifies noise 2-4x) |
| **merge + RL deconv (sigma_r~1.1, 3-10 it)** | **1.88-1.90** | 0.0020-0.0026 | 2.6% |

A measured-PSF Richardson-Lucy deconvolution takes the merge to DxO's edge resolution
(1.88 == 1.87) using only real detail (deterministic on the consensus merge -> adds
no invention), at a deslanted edge overshoot ~2.6% (vs DxO 1.8%, raw merge 2.0% --
comparable, NOT ringing; undershoot ~0.1%, RL being non-negative). Cost: a modest
noise increase (0.0015 -> 0.0020 at the gentle iters=3 setting; fewer iters = less
noise), still well below DxO's grain -- in a level-matched 1:1 render OUR column
surface is SMOOTHER than DxO's (DxO adds the ~28% per-frame texture). So DxO's sharper
edges are reproducible from our merge WITHOUT its invention and WITHOUT blockiness.

Caveats learned the hard way (both were measurement/viz errors, not the method):
- An earlier "overshoot 0.008, 6x less than DxO" claim was WRONG -- the
  `_deconv_sweep` `overshoot()` helper averages a 1deg-slanted edge band and
  dilutes/aliases the bump. The DESLANTED per-edge numbers above are the honest ones
  (~2.6%).
- A quick comparison viz used per-panel percentile-stretch + nearest-neighbor zoom,
  which mapped +-0.001 flat-region noise across full black->white and FABRICATED
  "blockiness" the merge does not have: at 1:1 flat patches are smooth and their FFT
  is featureless (no ~16px tile-grid period; the merge's nearest flow-upscaling does
  NOT show here). Render level-matched with a shared stretch + cubic zoom.

`deblur(img, deconv_sigma, method='rl')` wraps it; `test/test_deconv.py` guards it
(CPU, torch.fft). NOT yet wired into the shipped pipeline (the deconv sigma needs
per-camera/scale PSF calibration like `sigma_b`); `eval/_deconv_sweep.py` reproduces
the sweep.

### Adversarial review — verified corrections (deconvolution)

A 3-agent adversarial pass + re-execution corrected several deconvolution claims
(the `real_fraction`~0.72 hallucination result survived; the *global* deconv result
did not). All re-verified on the cached merge/stack:

- **The merge PSF is strongly FIELD-VARYING: edge sigma 0.91-2.41 px across the 8
  found edges (median 1.88, std 0.53).** "merge ~2.10" was the SOFT right-side
  cluster the reliability gate happens to select; center/left edges are much sharper.
  => a SINGLE global deconv sigma is WRONG -- it over-sharpens the already-sharp
  field. The correct design is FIELD-VARYING (per-tile), which the pipeline's
  aberration (Stage B) per-tile Polyblur ALREADY is; the lever is to feed IT a
  measured/calibrated PSF, not bolt on a global stage.
- **Flat-noise is region-dependent (earlier "smoother than DxO" was misstated):**
  flat SKY -- DxO 0.0005 << raw merge 0.0017 < merge+RL 0.0025 (DeepPRIME denoises
  flats hard; the deconv makes flat noise WORSE). Textured COLUMN -- merge 0.0079 <
  DxO 0.0115 (DxO adds invented grain there). Both true; always state the region.
- **Overshoot is method-fragile and was reported too low.** Canonical deslanted ESF:
  DxO ~15%, raw merge ~7%, merge+RL ~8% -- DxO HALOS MORE than us (we reach similar
  sharpness with less halo), but the absolute "1.8%/2.6%" do NOT reproduce; trust
  the direction, not the number.
- **"Adds no invention" was too strong.** RL is nonlinear and shapes the merge's
  (already nearest-flow-structured) flat noise (lag-1 autocorr 0.58->0.66); it stays
  AT/BELOW the DxO cross-frame-real ceiling only at gentle settings (HF/consensus
  0.98 @ it5, but 1.12 @ it10). Use few iters + a gate.
- **Held up:** `real_fraction`~0.72 robust to tiling/registration/resampling/channel
  (only scale-dependent: 0.68@hp1.0 -> 0.89@hp3.5; mildly pessimistic from an
  untrimmed HF mean, ~0.77 robustified). The deconv DOES reach target on the soft
  cluster, with less halo than DxO.

### Avenues (verified)

- **Noise-controlled deconv:** measured-PSF RL (few iters) + edge-preserving
  bilateral -> edge sigma 1.84 at flat-sky noise 0.0016 (BELOW raw merge 0.0017)
  while preserving column texture (0.0080 ~ plain RL 0.0082). The bilateral strips
  only the noise the deconv amplified. TV-regularized RL is a smaller win.
- **Edge gate = the STRUCTURE TENSOR, not the robustness mask.** Robustness is
  frame-AGREEMENT (~constant on a static scene) -- wrong signal. The structure tensor
  (l1 eigenvalue = across-edge energy = edge strength) is ALREADY computed per pixel
  for the steerable kernels (`kernels.py` estimate_cov / `cupy_ops.py`). Verified
  edge/flat contrast 6.8x; gate excludes ~85% (flat) of the frame. Gate by sqrt(l1).
- **Params:** merge PSF sigma is DERIVABLE per burst (auto esf over found edges =
  2.03; or per-camera calibration like sigma_b). sigma_r = ~1.08 *
  partial_gaussian_sigma(merge_sigma, target) (the ~8% fudge is the non-Gaussian
  edge; Gaussian model 1.02 -> 1.93, fudge 1.1 -> 1.88). HYPERparameters: iters/nsr,
  method, denoise/gate strength; target sharpness is an external anchor. The
  radial-frequency-falloff PSF estimate FAILS (confounded by the scene's own
  f^-beta spectrum) -- ESF on edges is the estimator.
- **Optuna:** extend `eval/sweep_synthetic.py` (synthetic-GT or DxO-consensus
  fidelity anchor, NOT the gameable Canon-DISTS sweep `eval/sweep.py`). DERIVE
  sigma_r (don't sweep). Sweep the few regularizers against a ONE-SIDED objective
  w1*DISTS_vs_GT + w2*|edge_sigma-target| + w3*max(0,flat_noise-merge_noise) +
  w4*max(0,overshoot-merge_overshoot) so it cannot win by over-sharpening/noise (the
  failure that broke the old Canon-DISTS sweep).

### Field-varying edge-gated deconvolution (eval/deconv.py)

Added `edge_gate` (structure-tensor sqrt(l1) -- the steerable-kernel signal, NOT
the robustness mask), `deblur_gated` (gate + edge-preserving bilateral), and
`deblur_field_varying` (per-tile local-PSF estimate -> per-tile deconv -> feather
blend). VERIFIED field-wide with fit-reliability flags (the discipline the global
result lacked):
- On the 4 edges where BOTH DxO and the merge erf-fit cleanly (rms<0.04):
  **DxO 1.71, raw merge 2.10, global-RL 1.84, field-varying 1.88 px.** Measured-PSF
  deconv CLOSES ~half the merge->DxO gap but does NOT reach DxO -- DxO's per-region
  deconvolution is stronger. (An "all-8 median 1.50 < DxO 1.57" looked like a win
  but is contaminated by 4 thin-bar edges with bad fits ~0.5 px; gate on fit
  reliability before quoting a distribution.)
- Tradeoff: field-varying controls flat NOISE far better (flat-sky 0.96x raw, BELOW
  the merge, vs global-RL 1.63x) via the gate+bilateral; global-RL is marginally
  sharper by sharpening noise too.
- The residual gap to DxO is NOT closeable by more deconvolution. Beating DxO needs
  real sub-pixel detail: SUPER-RESOLUTION (scale=2, untested here) + field-varying
  deconv, vs DxO's single-frame guess.

### "Beat DxO at every resolution?" -- honest MATCHED result (eval/_beat_dxo_matched.py)

Ran scale=1 AND scale=2 SR merges on the 007 burst (eval/_run_scale2.py),
auto-calibrated gated measured-PSF deconv (eval/calibrate_deconv.py auto merge
sigma = 1.96, high-conf), and compared PHYSICAL edge resolution to DxO -- MATCHED
(same physical edges, reliability-gated; NOT the convenient-sample trap):

  physical edge sigma (sensor px; median over 4 reliably-fit MATCHED edges):
    DxO 1.71 | s1 2.10 | s1+deconv 1.81 | s2 2.15 | s2+deconv 1.81

- We MATCH DxO closely (1.81 vs 1.71) but do NOT beat it on edge sharpness, at
  scale=1 OR scale=2. DxO's single-frame deconvolution is strong, and scale=2 SR
  does NOT help these edges (s2 2.15 ~ s1 2.10): the building columns are COARSE,
  already resolved at 1x, so SR (whose advantage is sub-Nyquist detail) has nothing
  to add there.
- LESSON AGAIN: a NON-matched distribution gave s2+deconv 1.43 < DxO 1.71 ("beat!"),
  but it compared different edge SETS; the MATCHED comparison shows ~1.81 = no beat.
  The cross-scale s2 sigma measurement (upsample 1x template -> align in 2x -> /2)
  is itself fragile; what's robust is "neither clearly beats DxO on these edges."
- Where we genuinely win: REAL-DETAIL HONESTY (DxO hallucinates ~24% of fine
  texture, per-frame inconsistent; ours is multi-frame-real) and ABOVE-native output
  (DxO has no SR -> bicubic; we have real fusion; at 2x our surface is cleaner --
  beat_dxo_2x.png, with a faint block artifact in the scale=2 deconv to investigate).
- "Beat at EVERY resolution" on edge sharpness is NOT supported by the data here. A
  definitive SR-beats-DxO demonstration needs a SUB-NYQUIST / resolution-chart
  target (coarse building edges can't isolate the SR gain) -- and that regime is
  exactly where DxO hallucinates, so the honest framing is real-detail vs
  invented-detail, not sharper-vs-softer.

### Block artifact = fixed-tile (nearest) flow upscaling -- FIXED (default -> bilinear)

The faint block artifact (visible at scale=2, amplified by the deconv; noticed
across multiple outputs) is the merge's flow_upscale_mode=nearest: the per-tile
optical flow was upsampled PIECEWISE-CONSTANT, so tile-boundary flow discontinuities
cut a tile grid into the merge. Confirmed by A/B (eval/_flow_ab.py): the
nearest-vs-bicubic difference is RMS 0.013 (maxabs 0.10) and STRONGLY periodic
(peak/mean 215) -- the 20x difference image is literally the tile grid. This is the
fixed-tile discretization; the fix is a CONTINUOUS flow field.

Default flow_upscale_mode nearest -> bilinear: block residual halved (bilinear-vs-
bicubic 0.0062 vs nearest-vs-bicubic 0.0129), edge sharpness preserved/improved
(phys sigma 2.14 vs 2.17; edge-fit rms 0.009 vs 0.017 -- the blocks were corrupting
the fit), full suite 230 passed. bicubic is marginally smoother (C1) but risks flow
overshoot at motion/depth boundaries, so bilinear is the safe default. A concrete
step toward the continuous formulation (continuous flow vs piecewise-constant tiles).

### Out-resolve DxO at <=1x? PARITY, not a beat (eval/_resolution_sweep.py)

Downscaled our scale=2 (bilinear-flow, de-aliased) merge + gated deconv and DxO's
single-frame output to 1.0/0.75x; matched physical edge sigma (sensor px):
  1.00x: DxO 1.81 vs ours+deconv 1.88   |   0.75x: DxO 1.98 vs ours 1.90
=> PARITY within +-0.07px (measurement noise); neither out-resolves the other on
resolvable edges. (Fairness matters: comparing our RAW merge to DxO's DECONVOLVED
render wrongly showed DxO leading; with OUR deconv it's a tie.) At 0.5x edge sigma is
filter-limited (both ~1px output edges, esf unreliable); the coarse louvers resolve
in both (not aliased -> no anti-aliasing difference on this content).

Why no out-resolve win here: multi-frame's genuine resolution advantage is
ANTI-ALIASING of NEAR-NYQUIST content (detail a single frame aliases) -- this scene's
columns/louvers are RESOLVABLE (not aliased), so they don't engage it, and DxO's
deconvolution handles resolvable edges fine. A definitive out-resolve demo needs a
fine resolution chart near the sensor Nyquist. Durable wins remain: real-detail
HONESTY (DxO hallucinates ~24% fine texture; ours doesn't), noise, and the
block-artifact fix (continuous flow).
