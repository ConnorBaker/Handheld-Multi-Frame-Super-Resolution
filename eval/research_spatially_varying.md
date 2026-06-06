# Research: making the pipeline spatially-varying

Grounded in `feat/warp-port`. Goal: remove fixed/global tile sizes and use
spatially-adaptive processing throughout. Driving finding (verified, `eval/NOTES.md`
"Adversarial review — verified corrections (deconvolution)"): the merge's effective
PSF is **field-varying — edge sigma 0.91–2.41 px across one scene** (median 1.88,
std 0.53), and the DxO reference corrects per-region. But the pipeline's
discretization is almost entirely **uniform**, derived from a single global SNR.

All claims below were checked against the source on this machine (granularity
numbers re-derived numerically; the Polyblur clamp and the `tile_size` fan-out
read straight from the code).

---

## 0. The existence proof: one stage is already per-pixel

`kernels.estimate_kernels` → `cupy_ops.estimate_cov_cupy` computes a **structure
tensor per Bayer quad** and emits a covariance field of shape `(H//2, W//2, 2, 2)`
(`cupy_ops.py:387`) — one anisotropic kernel **per output pixel**. The merge
(`warp_ops.py:_accumulate_warp`) bilinearly samples that field per HR pixel
(`warp_ops.py:68–81`). Measured granularity (3024×4032 raw, tile=16):

| field | grid | cells |
|---|---|---|
| merge covariance / structure tensor | `H//2 × W//2` | 1,512 × 2,016 |
| finest block-match patch grid (tile 16) | `H//16 × W//16` | 189 × 252 |

→ the steerable merge already runs at **64× the spatial resolution** of the
alignment grid (256× at tile=32). Per-pixel adaptation is demonstrably feasible
and cheap here. **Everything else is the anomaly, not the merge.**

Critically, `eval/deconv.py:edge_gate` (lines 143–168) **already recomputes that
exact `l1` eigenvalue field** (Sobel → box-filtered 2×2 tensor → closed-form max
eigenvalue) on CPU to gate the experimental deconvolution. So the field-varying
signal a spatially-varying deconv needs **already exists inside the pipeline** —
it is just thrown away after the merge instead of being exposed.

---

## 1. Where the fixed / uniform discretizations are

### (a) Block-matching tile size — global, SNR-chosen, quantized to {16,32,64}
`params.update_snr_config` (`params.py:56–73`) maps **one global burst SNR**
(`super_resolution.py:257–270`, `mean(ref_raw)/std`) to a single `Ts ∈ {64,32,16}`,
then `tile_sizes = [Ts*f for f in tile_size_factors]`. That scalar fans out to:
- alignment padding (`alignment.py:21`),
- the per-level patch grid for block-matching L1/L2, ICA, and the Hessian
  (`block_matching.py:13,60`, `ICA.py:28`, `utils_image.py:389`),
- the **merge's** tile→flow lookup (`merge.py:99` → `warp_ops.py:34`, `px=lr_x/tile`),
- robustness S-map upsampling and the flow lookup (`robustness.py:96,109,116`).

Limit: tile sizes are hard-bounded to {8,16,32,64} by the L2 box filter
(`avg_pool2d`, `block_matching.py:42`; `tile=4` → "Box filter not implemented",
`eval/NOTES.md`). One flow vector per tile ⇒ a 16-px patch straddling a
depth/motion boundary (foliage, occlusion edges) gets a single flow → smear
(the verified foliage failure, `eval/NOTES.md` items 1 & 3).
**Spatially-varying version:** a per-tile (or quadtree) tile size — fine over
high-gradient / low-local-SNR content, coarse over flat/low-SNR regions — so each
flow vector covers content with one consistent motion. (§2.)

### (b) Aberration (Stage B) patch grid — fixed 400 px, uniform stride
`aberration/model.py:90` `patch_size=400`, `overlap=0.25`; `forward` lays a
**uniform** F.unfold/F.fold grid (lines 122–164) and Polyblur-deblurs each tile.
This IS spatially-varying in that **each tile estimates its own blur kernel**
(`gaussian_blur_estimation`, one Gaussian per patch) — but on a **fixed uniform
partition** with a **bounded** estimator. (§2 on the real limitation.)

### (c) Merge window / kernel-regression support — fixed 3×3 gather
`warp_ops.py:93–94` (`di,dj ∈ {-1,0,1}`). The *shape* (anisotropy/width) of the
kernel is per-pixel via `covs`, but the **support is always 3×3**. At scale 1
this caps the merge at sampling-limited ~1 px softness (`eval/NOTES.md`: "merge is
inherently ~1px-soft at scale 1"). The `accumulated_robustness_denoiser.merge`
path can widen the ref-merge radius (`warp_ops.py:177,199` `rad_max`) but only as
a global, robustness-gated single-frame fallback — not content-adaptive.
**Spatially-varying version:** larger support in low-SNR/flat regions, tight 3×3
on detail. Lower priority — the cov field already shapes the kernel; widening
support is mostly a denoise lever, and global SNR is a fine prior for it (§4).

### (d) Noise / SNR — single global scalar
SNR is one number for the whole burst (`super_resolution.py:262`); it sets the
tile size **and** `k_detail/k_denoise/D_th/D_tr` (`params.py:75–90`) **and** the
deblur prefiltering bool (`params.py:99–101`). The noise *model* itself (the
`std_curve`/`diff_curve` indexed by local brightness, `cupy_ops.py:230`) is
brightness-adaptive per pixel inside robustness — but the SNR that drives the
**discretization choices** is global. A bright sky tile and a dark shadow tile in
the same frame get the same tile size and the same kernel-tuning constants.
**Spatially-varying version:** a per-region SNR map (the noise model already gives
per-pixel `sigma_t`; integrate it over candidate tiles) to drive (a) and the
kernel constants. (§2.)

**Summary — where uniformity hurts vs. where it's fine** (full table in §4):
field-varying **PSF/deconv** is where uniformity demonstrably hurts (measured
0.9–2.4 px spread, single global sigma over-sharpens the sharp field). The
**alignment tile size** hurts at motion/depth boundaries. **Global SNR** as a
*prior* for kernel-tuning constants and merge support is largely fine.

---

## 2. Spatially-adaptive tile sizes / patches

### Block-matching pyramid — recommended: per-region SNR, NOT a quadtree first
A quadtree / variable-size tiling is the textbook answer but it is a **poor fit**
for this pyramid: the whole align path assumes a **regular grid** that
`F.unfold(0,ts,ts).unfold(1,ts,ts)` (`alignment.py:54`) builds, that
`upscale_lvl` rescales between levels by an integer `repeat_factor`
(`alignment.py:154`), and that reshape-sum reductions exploit
(`cupy_ops.py:109,168`, `utils_image.py:394`). Irregular tiles break the batched
FFT cross-correlation and every reshape. High cost, high regression risk.

Three options, least- to most-disruptive:

1. **Per-region SNR → choose tile size per coarse block (recommended first step).**
   Keep the regular grid, but instead of one global `Ts`, compute a coarse SNR map
   (the noise model already yields per-pixel `sigma_t` from local brightness,
   `cupy_ops.py:230`) and pick `Ts` per macro-region. Cheapest concrete form:
   run alignment at **two tile sizes** (e.g. 32 and 16) and **select per output
   tile** by local SNR/gradient when sampling the flow in the merge. This is small
   and leaves the batched primitives intact. It directly targets the verified
   failure (low-SNR wants 64 for stability, high-freq foliage wants 16 for
   boundary fidelity — currently the *whole scene* is forced to one).

2. **Overlapping partition-of-unity with locally-chosen bandwidth.** Already
   the merge's model (cov field + 3×3 gather). Extending it to alignment means
   overlapping search windows blended by a feather window — but the flow field is
   piecewise-constant-per-tile by construction; the cleaner lever is to **densify
   the flow** (ICA already refines sub-pixel; raising `ica.n_iter` was the single
   biggest verified win, `eval/NOTES.md` 3→6/8) rather than vary the partition.

3. **Quadtree / gradient-or-SNR split (most disruptive).** Real content-adaptive
   tiling. Defer: it requires re-plumbing unfold/upscale/reshape to ragged grids
   and a scatter merge lookup. Only worth it if (1) proves insufficient at
   boundaries. The honest assessment (`eval/NOTES.md` item 3): **patch size was
   NOT the lever for static fine detail** (tile 16 already best); its value is
   specifically at **motion/depth boundaries**, which is the dynamic case that
   remains untested — so quadtree is a bet on a benefit not yet measured.

### Aberration stage — the real limitation is the *bounded Polyblur*, not the grid
The Stage-B per-patch path **already estimates blur per patch** (a kernel per
F.unfold tile). The verified limits (`eval/NOTES.md`, "Why Polyblur could not do
this"; confirmed in source here):
- **Bounded estimate:** `compute_gaussian_parameters` clamps
  `sigma² ∈ [0.09, 16]` (`blur_estimation.py:169`) ⇒ estimated **sigma ∈ [0.3, 4.0]
  px**, and its blind estimate is **fooled by high-contrast edges** (large gradient
  → tiny sigma).
- **Bounded inverse:** `compute_polynomial_fft` is a rank-3 polynomial inverse
  whose HF gain is capped at `beta` (`deblurring.py:118–123`, ~3× at default);
  inverting a 2.1 px Gaussian needs ≥10× → it caps at ~2.07 (measured).
- **Uniform grid** is the *least* of the three: 400 px is far coarser than the
  measured PSF variation scale, but the per-tile estimate would already capture
  most of the field variation **if the estimator and inverse weren't bounded.**

→ Verdict: **the uniform grid is secondary; the bound is primary.** The aberration
stage is the right *place* for field-varying deblur (it's already per-tile,
already feathered/folded) but the right *fix* is to replace its bounded blind
Polyblur with a **measured-PSF deconvolution per tile** (§3), and shrink/adapt the
tile to the PSF variation scale.

---

## 3. Field-varying PSF / deconvolution (the highest-value change)

This is where uniformity **demonstrably hurts** (measured 0.9–2.4 px spread). The
prototype already exists in `eval/deconv.py:deblur_field_varying` (lines 258–313):
overlapping tiles → per-tile ESF measurement of the local merge PSF
(`_local_sigma`, requires a clean step edge in the tile) → per-tile RL deconv to a
common target → raised-cosine feather. Verified result (`eval/NOTES.md`
"Field-varying edge-gated deconvolution"): closes ~half the merge→DxO edge gap
(raw 2.10 → field-varying 1.88, DxO 1.71) while controlling flat noise far better
than a global RL (0.96× raw vs 1.63×).

Three ways to make it production, best path first:

### Path A (recommended) — fold a measured-PSF deconv into Stage B, keyed on the cov field
Replace the aberration stage's per-tile **blind, bounded** Polyblur with a per-tile
**measured-PSF** deconv (RL, few iters), and **drive the per-tile PSF and the
edge-gate from the structure tensor the merge already computes** rather than
re-measuring ESF per tile. Concretely:
- **Expose the cov / `l1` field.** `estimate_kernels` already has the structure
  tensor; today it's consumed only by the merge and discarded. Return the `l1`
  eigenvalue field (across-edge energy) alongside `covs`. `eval/deconv.py:edge_gate`
  is literally recomputing this — wire the pipeline's own field through instead.
- **Edge gate = sqrt(l1)** (verified: edge/flat contrast ~6.8×, gates out ~85% flat,
  `eval/NOTES.md` "Edge gate = the STRUCTURE TENSOR"). This is the *right* signal;
  the robustness mask is frame-agreement (~constant on static scenes) and wrong.
- **Per-tile PSF:** the merge PSF is per-burst-derivable by auto-ESF over found
  edges (~2.03, `eval/NOTES.md`) or per-camera calibration like `sigma_b`. The
  field *variation* can be carried by the `l1` field as a cheap proxy (strong-edge
  tiles read sharper) instead of the brittle per-tile ESF fit in
  `_local_sigma` (which needs a clean step edge to exist in every tile and
  inherits the field median otherwise — fragile on foliage scenes).

Why this is the least-disruptive high-value path: Stage B is **already** per-tile,
overlapped, feathered, folded, and cached (`raw2rgb.py:_aberration`). Swapping the
per-tile operator + feeding it the existing field is contained, and it kills the
two bounds (estimate clamp + polynomial-inverse cap) that the notes prove block
the result. Cost: per-tile RL (3–5 FFT iters/tile) is heavier than the polynomial
inverse, but the deconv backend study (`eval/deconv.py` header) confirms
`torch.fft` handles full 24 MP and Warp has device-side `tile_fft` for a fused
on-device path if needed.

### Path B — per-pixel-kernel-coupled deconv (could the cov field DRIVE the deconv?)
The structure tensor gives, per pixel, the **across-edge direction and strength**
(`l1`, eigenvector `e1`). In principle a **steerable, per-pixel anisotropic
deconvolution** could deconvolve *more along the across-edge direction* and *less
along the edge* — exactly the geometry the merge kernel already encodes. This is
the most "spatially-varying-throughout" design and reuses the merge's covariance
field directly. **But:** (i) deconvolution is non-local (an inverse filter has
support far beyond 3×3), so a true per-pixel kernel can't be a bounded gather like
the merge — it needs per-pixel FFTs or an iterative scheme; (ii) the merge cov is
a *regression* kernel (denoise/SR sample weighting), not the *optical* PSF, so
coupling them conflates two different operators. Verdict: elegant, high
complexity/risk, and the measured payoff (§3 result) is already ~reached by the
much simpler tile-wise Path A. **Defer unless Path A's tile blockiness shows.**

### Path C — keep `eval/deconv.py:deblur_field_varying` as a postprocess stage
Lowest effort: wire the existing function into `raw2rgb.postprocess` as a new
stage. Honest downside: it re-measures the PSF by per-tile ESF (needs clean edges,
fragile on foliage) instead of using the pipeline's own field, and it's a bolt-on
rather than fixing the bounded Stage B that's already there. Use only as a
stopgap / A-B reference for Path A.

---

## 4. Per-region everything — coherent design, and where uniformity is fine

A single per-tile (or per-macro-region) **content descriptor** computed once and
reused by every stage, instead of one global SNR:

```
descriptor[region] = { local_SNR  (noise model integrated over the region),
                       l1, e1      (structure tensor — already computed per quad),
                       flow_disagreement (already in robustness, compute_s) }
```

- **Alignment tile size** ← local_SNR + l1: small/fine on high-l1 (edges) and
  high-SNR; large on flat/low-SNR. (§2 option 1 — per-region SNR on the regular
  grid; quadtree only if boundaries demand it.)
- **Robustness** ← already the most per-pixel stage (per-pixel noise model + local
  min + flow-discontinuity penalty). The lever is the **threshold `t`**, not the
  discretization: it under-rejects moving content (verified 4.6/7 vs 6.2/7,
  `eval/NOTES.md` item 3). Could be made content-adaptive (reject harder where l1
  is high = where smear is visible), but the bigger win is just tuning `t` up.
- **Merge kernel** ← already per-pixel via the cov field. Make the **3×3 support**
  content-adaptive (wider in flat/low-SNR) only if denoising demands it — low
  priority.
- **Deconvolution** ← l1 (gate) + per-region PSF (§3 Path A). **This is the
  keystone**: it's the one stage where the field-varying need is *measured*, the
  signal (l1) is *already computed*, and the place to put it (Stage B) is *already
  per-tile*.

### Where uniformity is actually FINE (do NOT spend complexity here)
- **Global SNR as a prior** for the kernel-tuning constants
  (`k_detail/k_denoise/D_th/D_tr`) and the deblur prefiltering bool — these are
  soft knobs; the verified fidelity sweep found the real levers were alignment
  iterations + robustness rejection, not these constants (`eval/NOTES.md` Item 1).
  A global SNR prior is reasonable; per-region refinement is diminishing returns.
- **Merge 3×3 support** at scale 1 — sampling-limited regardless; detail comes
  from `scale=2`, not support width (`eval/NOTES.md`).
- **The per-pixel noise model** is already brightness-adaptive — no change needed;
  just *surface* it as the per-region SNR source for the discretization choices.

### Where uniformity HURTS (spend complexity here, in order)
1. **Field-varying PSF / deconvolution** — measured 0.9–2.4 px; a single global
   deconv sigma over-sharpens the sharp field (verified). **Highest value.**
2. **Alignment tile size at motion/depth boundaries** — one flow per 16-px tile
   smears foliage (verified). Medium value; the per-region-SNR variant is cheap,
   the quadtree is not.

---

## Prioritized recommendation

1. **Expose the structure-tensor `l1` field from `estimate_kernels`** (it's already
   computed and discarded; `eval/deconv.py:edge_gate` proves the pipeline
   recomputes it). Near-zero cost, unblocks everything below. *(Small.)*
2. **Path A: field-varying, measured-PSF deconvolution inside Stage B**, gated by
   sqrt(l1), per-tile PSF from per-burst ESF or per-camera calibration. Replaces
   the doubly-bounded blind Polyblur. This is the highest measured-value change and
   the place for it (per-tile, feathered, folded) already exists. *(Medium —
   contained to `aberration/model.py` + `raw2rgb._aberration`; main cost is per-tile
   RL vs polynomial inverse, and PSF calibration.)*
3. **Per-region SNR → adaptive alignment tile size on the regular grid** (two-size
   select by local SNR/gradient), keeping batched FFT/reshape intact. Targets the
   foliage-boundary smear without the quadtree rewrite. *(Medium.)*
4. **Defer:** quadtree alignment (ragged-grid rewrite, benefit unmeasured on
   dynamic content) and per-pixel-coupled anisotropic deconvolution (Path B —
   elegant but high risk, payoff already ~reached by Path A). *(Large; do only if
   3 / Path A prove insufficient.)*

Do **not** make global SNR or the merge 3×3 support per-region — verified low
value. The whole program is essentially: *the merge already runs a per-pixel
adaptive field; route that field (l1) into a measured-PSF deconvolution, then
extend per-region adaptation outward to the alignment tile size.*
