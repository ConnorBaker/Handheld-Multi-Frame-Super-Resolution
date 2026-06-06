# Continuous & alternative-domain formulations of multi-frame SR + optical correction

Deep-research note. The thesis under investigation (the user's): **this pipeline is a
*discretized* instance of a problem that is naturally *continuous*.** Fixed pixel grid,
fixed tile sizes, a discrete weighted-accumulation merge into `num`/`den`, and on-grid
Polyblur/RL deconvolution are all discretizations. "Fractional tiles" and
"spatially-varying processing" are steps *back toward* the continuous formulation. This
note researches that continuous formulation (and frequency/wavelet alternatives), maps
each to *this* code, and rates feasibility on the GB10.

Every external claim is cited (see **Sources**). Two things were **executed on this
machine**, not trusted from docs (repo `CLAUDE.md` §1):

- **The merge in this repo *is* steering kernel regression evaluated on a grid.** Read
  directly from `handheld_super_resolution/warp_ops.py` + `merge.py` + `kernels.py`: the
  Warp kernel computes, per output pixel, `w = exp(-0.5 * dᵀ Ω⁻¹ d)` over a 3×3 gather,
  routes by Bayer channel, and accumulates `w·r·c → num`, `w·r → den`; the final divide
  is `num/den`. That is verbatim Wronski Eq. 1–2 (below), which is verbatim Takeda
  zeroth-order (Nadaraya–Watson) steering kernel regression (below).
- **Warp's device-side FFT (cuFFTDx via libmathdx) compiles and runs on this GB10
  (sm_121).** A real `@wp.kernel` doing `tile_load → tile_fft → tile_ifft → tile_store`
  round-trips to **max-abs-err 2.38e-07** (`/tmp/warp_fft_probe.py`). Caveats found by
  *running* it, not reading dir(wp): (a) `hasattr(wp,'tile_fft')` is **False** — these
  are compile-time-only builtins (so the `dir(wp)` skepticism in `eval/deconv.py` is
  right); (b) cuFFTDx needs **≥2 elements/thread**, so `block_dim ≤ fft_size/2`. The
  example in `eval/deconv.py`'s docstring uses `block_dim=64` for `fft_size=64`, which
  **fails** with `ept=1`; it must be `block_dim≤32`. `wp.config.enable_mathdx_fft`
  exists and must be set before `wp.init()`.

---

## 0. What the current pipeline actually computes (the discrete object)

For each output pixel `(x,y)` and color channel, with frames `n`, local 3×3 samples `i`:

```
C(x,y) = ( Σ_n Σ_i  c_{n,i} · w_{n,i} · R̂_n ) / ( Σ_n Σ_i  w_{n,i} · R̂_n )     [Wronski Eq.1]
w_{n,i} = exp( -½ · d_iᵀ Ω⁻¹ d_i ),   d_i = [x_i - x0, y_i - y0]ᵀ              [Wronski Eq.2]
Ω̂ = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]  (structure tensor)                        [Wronski Eq.3]
Ω = [e1 e2] diag(k1,k2) [e1 e2]ᵀ  (eigh of Ω̂; k1,k2 from λ1, λ1/λ2,
       k_detail/k_denoise/k_stretch/k_shrink)                                  [Wronski Eq.4]
```

This is **`warp_ops.py` exactly** (`num[i,j,c] += w*r*c`, `den[i,j,c] += w*r`; `kernels.py`
builds `Ω` from the structure tensor; the merge inverts `Ω` per pixel). Wronski himself
frames it as a *deliberate discretization of a continuous reconstruction*: "*similarly to
the non-parametric kernel regression framework of Takeda et al. … Unlike Takeda et al., we
**don't determine kernel basis function parameters at sparse sample positions. Instead, we
evaluate them at the final resampling grid positions** … a 'gather' approach [so] every
output pixel is independently processed only once per frame*" — chosen "*for efficient
parallel evaluation on a GPU*" ([Wronski 2019, §5.1](https://3dvar.com/Wronski2019Handheld.pdf)).
So the discretization is **explicit and admitted in the source paper**: the grid, the 3×3
neighbourhood, and "same kernel for all nine samples" are the GPU-friendly approximations
of a continuous estimator.

---

## 1. The continuous formulation of multi-frame SR

**Continuous objective.** Model the latent scene as a continuous radiance field `z(·): ℝ² → ℝ^C`.
Each frame contributes irregular, sub-pixel samples `{(x_{n,i}, y_{n,i})}` (the LR pixel
centers warped by the alignment field `V_n`), with measured values `c_{n,i}` and a noise
model. Takeda's data model is exactly this:

```
y_i = z(x_i) + ε_i,   x_i scattered (NOT on a grid)                          [Takeda Eq.1]
```

Estimate `z` and its derivatives at a query point `x` by **local weighted least squares**
on a Taylor expansion of `z` about `x` ([Takeda 2007, Eq.4](https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf)):
`min_β Σ_i [ y_i − β0 − β1ᵀ(x_i−x) − … ]² K_H(x_i − x)`. The **zeroth order** (N=0) solution
is the Nadaraya–Watson estimator — a *normalized weighted average of the scattered samples*:

```
ẑ(x) = Σ_i K_H(x_i − x) y_i  /  Σ_i K_H(x_i − x)                             [Takeda Eq.6]
```

The **steering** kernel makes `K` data-adaptive: `K_steer = exp(−½ (x_i−x)ᵀ C_i (x_i−x))`
with steering matrix `C_i = γ_i U_i Λ_i U_iᵀ` obtained from the **SVD of local gradients /
the gradient covariance** `Σ ∇z ∇zᵀ` over a window — elongate along edges, shrink across
([Takeda 2007, Eq.43,48,49](https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf)).

**The current discrete merge is precisely this estimator, made approximate in three ways:**

1. **Query on a grid, not at samples.** Takeda/N-W evaluate `ẑ(x)` at *any* continuous `x`;
   the repo evaluates only at HR grid centers `(j+0.5)/scale`. Re-querying at a different
   scale or at arbitrary points needs no new math — only a different output coordinate set.
2. **Fixed 3×3 support + one Ω per output pixel.** True KR sums over *all* samples within
   the kernel's effective support with *per-sample* kernels `K(x_i−x)`; the repo truncates
   to the 3×3 nearest LR pixels and reuses one `Ω` (bilinearly upsampled from the
   quarter-res covariance map) for all nine. For narrow kernels (`k_detail` small, the SR
   regime) 3×3 is adequate; for wide denoising kernels it under-counts the tail.
3. **Zeroth order only.** The repo is N=0 (weighted average). N=1/2 (local linear/quadratic
   KR) additionally fit a local *gradient/curvature*, i.e. recover sub-pixel slope — sharper,
   at the cost of solving a tiny normal-equation per pixel instead of one divide.

Other continuous reconstructions of the same scattered-data problem:
**normalized convolution** (Knutsson–Westin: each sample carries a *certainty*; reconstruct
by `(c·a) * g / (a * g)` — structurally the repo's `num/den` with robustness `R̂` playing the
certainty `a`); **RKHS / kriging** (reconstruct as `Σ α_i k(·, x_i)` minimizing an RKHS norm
— optimal-interpolation view, [IEEE 730385](https://ieeexplore.ieee.org/abstract/document/730385/));
**adaptive Wiener / surface-fitting** non-uniform interpolation
([researchgate 228936798](https://www.researchgate.net/publication/228936798_Near_optimal_non-uniform_interpolation_for_image_super-resolution_from_multiple_images)).

**For THIS pipeline / cost / worth.** The *cheap, actionable* upgrade is **decouple the
query grid from `scale`**: the merge already takes `scale` as a float and divides
`(j+0.5)/scale`; nothing stops querying a non-integer grid or an arbitrary point set. Going
to **N=1 local-linear KR** is the principled "more continuous" step (recovers slope, the
thing 3×3 N=0 blurs) — moderate cost (a 3×3→3-unknown solve per pixel in the Warp kernel,
well within budget; Warp has `tile`/`tile_matmul` for the tiny solves). **Worth it:**
decoupled query = free and unlocks fractional scale; N=1 = a real, bounded sharpness win
worth prototyping. **Not worth it:** full per-sample KR over a large support (cost blows up,
marginal benefit over 3×3 in the SR regime). *Elegant-but-impractical:* a global RKHS solve
(dense `k(x_i,x_j)` over millions of samples).

---

## 2. Fractional-pixel / continuous tile sizes — does "tile" survive?

The pipeline has **two distinct "tile" notions**; they continuize differently.

**(a) Block-matching tiles** (`params.py update_snr_config`: `Ts ∈ {16,32,64}` by SNR;
`tile_size_factors` per pyramid level). These exist to make **alignment** a tractable
patch-wise translation search. "Fractional tile size" here is a category error — `Ts` is the
support of a *correlation/parametric flow estimate*, and the literature's continuous version
of patchwise flow is **dense/continuous optical flow** (per-pixel, sub-pixel) with a
*continuous bandwidth*, not a fractional patch. The repo *already* refines block matching to
sub-pixel with **ICA / inverse-compositional Lucas–Kanade** (`ICA.py`) — i.e. the tile
grid is already a coarse scaffold whose output is a continuous flow field. The continuous
end-state: drop the tile partition entirely and estimate a **continuous warp field** (exactly
what `SuperF` does — it parametrizes a per-frame affine `A^(t)` and optimizes it jointly,
[SuperF 2025](https://arxiv.org/html/2512.09115); and `BurstM` does — FNet optical flow +
coordinate warping, [BurstM 2024](https://arxiv.org/html/2409.15384)).

**(b) The merge "tile"/window** is *already* continuous and *already* dissolved. There is no
merge tile — the kernel covariance `Ω` **is** the local window, and its bandwidth varies
*per pixel* (`λ1` drives spatial support, `λ1/λ2` drives anisotropy, Wronski §5.1.2). This is
literally a **continuous locally-adaptive bandwidth field**. The right primitive in the
continuous formulation is **not a tile but a bandwidth/covariance field** `Ω(x)` — which the
pipeline computes — sampled by a **partition-of-unity** of overlapping kernels (the
normalized `Σw`/`Σw` *is* a partition of unity; "constant-overlap-add"
[Smith, CCRMA](https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_Decomposition.html)). The
theoretical backbone for "continuous bandwidth + automatic scale" is **Lindeberg
scale-space** — a continuous scale parameter `t`, with **automatic scale selection** from
extrema of γ-normalized derivatives ([Lindeberg 1998](https://people.kth.se/~tony/papers/cvap198.pdf)).
The merge's `λ1→`support map is a (hand-tuned) scale-selection rule; Lindeberg gives the
principled version.

**Where a fractional/continuous tile *does* appear concretely:** `eval/deconv.py`
`deblur_field_varying` already tiles the deconvolution (512 px, 25% overlap) and
**feather-blends with a raised-cosine partition-of-unity window** (`_raised_cosine_2d`,
sums-to-1 across the overlap). That is the discrete stand-in for a continuous PSF *field*.
The continuous limit is **PSF as a smooth field `φ(x)`** (theme 5), at which point the
deconv tile also dissolves into a per-pixel operator. Patch-wise field-varying deconv is
known to risk **blocking artifacts / discontinuous blur modeling** unless overlapped/blended
([Nature s41598-025-19568-5](https://www.nature.com/articles/s41598-025-19568-5)); the
feathering is exactly the mitigation.

**For THIS pipeline / worth.** Block-matching `Ts` is a *solver scaffold* — keep it; it is
not a discretization of the *image*, it is a discretization of the *search*, and "fractional"
is meaningless there. The merge window already has no tile. **Actionable:** treat `Ω(x)` and
(future) `φ(x)` as the real primitives and make the deconv genuinely per-pixel-field rather
than per-512-tile (smoother, no blend needed) — but that is a refinement, not a paradigm
change, and the current feathered tiles are a fine approximation. *Elegant-but-impractical
now:* deleting block-matching tiles for a from-scratch dense continuous-flow + INR solve
(that's `SuperF`/`BurstM` territory — a different pipeline, not a tweak).

---

## 3. Continuous image representations (queryable at any coordinate)

**Classical (linear, no learning).** Unser, *Sampling—50 years after Shannon*: drop the
band-limited/Shannon straitjacket and reconstruct in a **shift-invariant space** spanned by a
generator `φ` (B-splines are the sweet spot) — `f(x) = Σ_k c_k φ(x/T − k)` — which is a
genuinely continuous function queryable at any `x`, with cheaper and more realistic
interpolation than ideal sinc and a wider class of (non-ideal) anti-alias prefilters
([Unser 2000](https://bigwww.epfl.ch/publications/unser0001.pdf)). This is the *linear*
continuous representation: after the merge produces `num/den`, fit a spline coefficient field
once and resample at any scale/coordinate — exact, fast (separable filtering), deterministic.

**Learned implicit (INR).** LIIF represents an image as a feature map `M` + a **shared MLP
decoder** `f_θ` queried at a continuous coordinate `x_q`:

```
I(x_q) = f_θ(z*, x_q − v*)        (nearest latent z*, rel. coord)            [LIIF Eq.2]
I(x_q) = Σ_{t∈{00,01,10,11}} (S_t/S) · f_θ(z*_t, x_q − v*_t)   (local ensemble) [LIIF Eq.4]
```

Note Eq.4 is a **bilinear-weighted, normalized ensemble of 4 neighbouring latent codes** —
*structurally identical* to the merge's `num/den` gather, but the "samples" are learned
latents and the combiner is an MLP ([LIIF, Chen 2021](https://arxiv.org/pdf/2012.09161)). LTE
adds a **dominant-frequency (Fourier-feature) estimator** so the MLP captures fine texture
and beats spectral bias ([LTE, Lee 2022](https://arxiv.org/abs/2111.08918)).

**What a continuous representation buys *this* pipeline specifically:**
- **Arbitrary / fractional scale for free** — query the same representation at any grid; no
  re-run, no fixed `scale`.
- **Unify merge + deblur + resample** into one queryable field (theme 5): evaluate the
  field *and* apply a field-varying inverse operator at the same continuous coordinates.
- **Field-varying everything is native** — `Ω(x)`, `φ(x)`, robustness all become coordinate
  inputs.

**For THIS pipeline / cost / worth.** **Actionable & low-risk: the spline / shift-invariant
representation.** It is *linear, hallucination-free, deterministic* (critical given this
repo's `eval/NOTES.md` hallucination discipline — a spline cannot invent cross-frame
structure), and trivially GB10-able (separable IIR/FIR on torch/cupy). It directly gives
fractional-scale resampling of the merge and a cleaner substrate for the deconv. **INR/LIIF
is powerful but a different animal:** it *learns*, so it risks **hallucinating detail** —
exactly what `eval/hallucination.py` and the project's "merge holds only cross-frame-consistent
detail" invariant exist to prevent. Use INR only test-time-optimized on the actual burst
(`SuperF`-style, no external training set → low hallucination, [SuperF](https://arxiv.org/html/2512.09115))
or not at all. *Elegant-but-impractical for shipping:* a pretrained feed-forward INR SR
(fast, but unverifiable invention).

---

## 4. Operating in a different domain

### 4a. Frequency domain

**SR-as-dealiasing.** The original SR insight is frequency-domain: Tsai & Huang (1984) use
the **Fourier shift theorem** — sub-pixel translations are linear phase shifts — to write a
system relating the *aliased* spectra of the LR frames to the unaliased HR spectrum, and
**solve it as inversion of those equations**; multi-frame SR literally *resolves aliased high
frequencies* by combining differently-aliased frames ([Tsai-Huang via Vandewalle 2006](https://asp-eurasipjournals.springeropen.com/articles/10.1155/ASP/2006/71459)).
That EURASIP paper does **registration in the Fourier domain** (phase correlation on aliased
images) — relevant to block matching. Limitation (well known): the shift theorem only models
**global translation**, so pure frequency SR can't handle the local motion/occlusion this
pipeline's robustness model handles — which is *why* Wronski went spatial-domain + robustness.
**Wronski's own framing of why multi-frame helps is the aliasing/sampling view**: hand tremor
makes each frame alias *differently*, and the merge turns that into resolved detail (Fig. 1
Moiré example, [Wronski](https://3dvar.com/Wronski2019Handheld.pdf)).

**Fourier ptychography** = the *frequency-synthesis* cousin: each measurement captures a
different sub-region of the object's spectrum (shifted under the pupil by illumination
angle); stitch overlapping spectral patches → large space-bandwidth product
([SAVI, Science Advances](https://www.science.org/doi/10.1126/sciadv.1602564)). The analogy
to burst SR: each frame is a differently-sampled (aliased) view of the same spectrum; the
merge "stitches" them — but FP needs *phase retrieval* (intensity-only), which burst SR
sidesteps because frames are already linear in radiance.

**Frequency-domain deblur (Wiener).** Already in the repo: `eval/deconv.py wiener_fft`
implements `X = G·H̄ / (|H|² + nsr)` on `torch.fft.rfft2` — the textbook regularized inverse
filter `≈ 1/(H+c)` that avoids the zero-crossing blow-up
([general SR/Wiener refs](https://arxiv.org/pdf/1510.00143)). RL is also there (`richardson_lucy_fft`).
**Modern burst-SR has gone (continuous) frequency-domain:** `BurstM` predicts **continuous
Fourier coefficients** per frame (amplitude/frequency/phase estimators) and reconstructs
`m_i = A_i ⊙ [cos(π(⟨F_i,δ_i⟩+ph_i)), sin(…)]`, fusing in Fourier space → arbitrary scale
without retraining ([BurstM 2024](https://arxiv.org/html/2409.15384)). That is *exactly* the
user's thesis realized: replace the fixed-grid discrete merge with a continuous-frequency
representation.

### 4b. Wavelets

**Why wavelets suit this problem:** they give a **multi-scale, spatially-localized**
basis — detail coefficients are local in *both* space and frequency, so "spatially-varying,
multi-scale detail" (the user's words) is the native coordinate system. Wavelet-SR predicts
the **missing high-frequency subbands** (LH/HL/HH) from the LR approximation and inverts the
DWT ([Deep Wavelet Prediction, IEEE 8014882](https://ieeexplore.ieee.org/document/8014882/)).

**Wavelet-domain deblur — directly relevant to the merge→DxO gap in `eval/NOTES.md`:**
**ForWaRD** (Fourier-Wavelet Regularized Deconvolution, Neelamani–Choi–Baraniuk 2004) does a
*small* Fourier (Wiener) shrinkage to invert the operator, then **wavelet shrinkage** to kill
the colored noise the inverse amplifies. It provably beats pure-Wiener (**less ringing**) and
pure-wavelet WVD, because Fourier represents the operator/noise economically and wavelets
represent piecewise-smooth images economically
([ForWaRD, IEEE 1261329](https://ieeexplore.ieee.org/document/1261329/)). This is the
**principled fix** for the repo's current `RL + bilateral` hack in `deblur_gated`: the
bilateral is a crude denoiser bolted after RL; ForWaRD's wavelet shrinkage is the right tool
to remove deconvolution-amplified noise while keeping edges.

**Domain comparison — when each helps:**
| Domain | Best at | Weak at | Fit to this pipeline |
|---|---|---|---|
| **Spatial (current)** | local motion, robustness, per-pixel `Ω`, gathers | global periodic detail, ringing control | what's shipped; GPU-ideal |
| **Frequency (Fourier)** | global translation SR, full-frame inverse filter, dealiasing | local motion/occlusion, spatially-varying PSF | deblur is already FFT; `BurstM` shows continuous-freq merge works |
| **Wavelet** | multi-scale + *localized* detail, ringing-free regularized deblur (ForWaRD) | needs careful basis/threshold choice | best **deblur regularizer** (replace bilateral); also natural for field-varying detail |

**For THIS pipeline / cost / worth.** **Frequency deblur is already here and is the right
home for a full-frame inverse** (FFT is `O(N log N)`, and **Warp can do it on-device** —
verified §intro). **Actionable, high-value:** swap the post-RL `bilateral` for **wavelet
shrinkage (ForWaRD-style)** — bounded, classical, ringing-aware, no learning/hallucination,
runs on cupy/`pywt`/torch wavelets. **Actionable-but-larger:** a continuous-Fourier merge
(`BurstM`) — strong, but it's a learned net (hallucination surface; needs the project's
verification gauntlet). *Elegant-but-impractical here:* Fourier ptychography proper (needs
phase retrieval; burst frames don't need it). *Don't:* a pure frequency-domain merge — it
can't carry the robustness/local-motion model that makes this pipeline work.

---

## 5. Continuous / spatially-varying PSF & unified merge+deblur

**The continuous inverse problem.** The cleanest statement is `SuperF`'s forward model
([SuperF 2025](https://arxiv.org/html/2512.09115)):

```
y_LR^(t)(v) ≈ φ * y_HR( A^(t) v )
```

i.e. each LR frame `t` is the continuous scene `y_HR` warped by `A^(t)`, blurred by PSF `φ`,
sampled. **Merge and deblur are then a single inverse problem**: recover the continuous
`y_HR` such that, pushed through `φ` and every `A^(t)`, it reproduces all frames. The repo
currently **splits** this — merge (forward sampling kernels, `Ω`) then deblur (inverse of
`φ`) as two stages — because the split is GPU-cheap and deterministic. The continuous view
**fuses** them: one operator `(blur ∘ resample)` whose adjoint/inverse is applied once.

**Continuous PSF models.** The natural continuous PSF is **not a pixel grid of values** but a
**low-dimensional optical basis**: a wavefront expanded in **Zernike polynomials** (defocus
`Z4`, astigmatism `Z5,Z6`, coma `Z7,Z8`, spherical `Z11`, …), from which the PSF at any field
point is computed `|F{pupil·e^{iW}}|²` ([Zernike review, IOP](https://iopscience.iop.org/article/10.1088/2040-8986/ac9e08);
[PSF-from-wavefront, Nature s41598-024-84790-6](https://www.nature.com/articles/s41598-024-84790-6)).
A handful of Zernike coefficients (optionally varying smoothly across the field) gives a
**continuous PSF field `φ(x)`** — far fewer parameters than a per-tile measured-PSF table.

**Spatially-varying deconvolution — the realistic middle ground.** Full per-pixel SV
deconvolution is expensive, so the field is approximated:
- **Patch/section-wise + interpolation** — assume `φ` ≈ const on small patches, blend (what
  `deblur_field_varying` does); risks blocking without overlap
  ([astro-ph/0208247](https://arxiv.org/pdf/astro-ph/0208247)).
- **Eigen-PSF / filter-flow** — express the SV PSF as a few basis PSFs with spatially-varying
  weights; SV Richardson–Lucy then costs ~k convolutions
  ([svDeconRL](https://github.com/dop-oxford/svDeconRL); [Temerinac-Ott 2011](https://lmb.informatik.uni-freiburg.de/Publications/2011/BRT11/TemerinacISBI2011.pdf)).
- **Ring deconvolution** — if the optics are ~rotationally symmetric, the PSF varies only
  with *radius* (same shape on a ring), giving an `O(N log N)` SV correction far cheaper than
  general patchwise ([Ring deconvolution, arXiv 2206.08928](https://arxiv.org/abs/2206.08928)).
  **This is highly relevant**: a phone lens's aberrations are dominantly radial (field
  curvature, lateral CA), so the merge PSF's measured *field variation* (`eval/NOTES.md`:
  edge σ 0.9–2.4 px across the scene) is probably mostly a **function of image radius** — a
  ring/radial PSF model would capture it with almost no per-tile machinery.
- **Learned SV-Wiener** (MultiWienerNet, 625–1600× faster than iterative
  [Optica 9-1-96](https://opg.optica.org/optica/fulltext.cfm?uri=optica-9-1-96)) and **SV
  Richardson–Lucy + TV** — the regularized iterative path.

**Unifying merge + deblur, concretely.** Two routes:
1. **Joint continuous solve (the elegant end-state):** represent `y_HR` continuously (spline
   field §3, or INR), put `φ(x)` (Zernike or ring) into the forward model, and minimize
   `Σ_{t,i} R̂ · ‖ (φ * y_HR ∘ A^(t))(x_i) − c_{t,i} ‖²` (+ regularizer). This is `SuperF`
   *plus* an explicit PSF — `SuperF` deliberately leaves `φ` fixed (boxcar), so **adding a
   Zernike/ring `φ(x)` to a SuperF-style test-time solve is the precise way to unify the
   merge and the deblur into one continuous inverse problem.**
2. **Keep the split, continuize the PSF (the actionable end-state):** leave the fast discrete
   merge, but replace the per-512-tile measured-σ table in `deblur_field_varying` with a
   **smooth `φ(x)` field** — fit a **radial/ring PSF** (1–3 params) or a few field-varying
   Zernikes to the measured edge-σ field (`eval/psf.py esf_sigma` already measures it), then
   deconvolve with a per-pixel-varying kernel (eigen-PSF SV-RL, a few FFTs). No tiles, no
   blocking, no blend.

**For THIS pipeline / cost / worth.** **Most actionable & well-matched: a radial/ring PSF
field.** The repo *already measures* the field-varying edge-σ (`eval/psf.py`, `eval/deconv.py`)
and already does field-varying RL; the missing piece is a **continuous low-D PSF model**, and
phone-lens physics says **radial** is the right one — cheap (`O(N log N)`-ish via ring
deconvolution), deterministic, and it kills the per-tile blocking the current method
feathers around. **Zernike field** is the more general continuous-optics model (good if CA /
coma matter, ties into the `aberration/` module) — a few more parameters, still tiny.
**Joint continuous merge+deblur (SuperF+φ)** is the *theoretically correct unification* and
the most elegant answer to the user's thesis — but it is a **research re-architecture**
(iterative optimize-per-burst, ~seconds not ms; learned components → hallucination surface
that must pass `eval/hallucination.py`). **Verdict:** ship the **radial-PSF field** as the
near-term win that *is* a genuine step toward the continuous formulation; treat the **joint
SuperF+Zernike solve** as the aspirational continuous endpoint, prototyped offline in `eval/`
and gated hard against hallucination before any thought of shipping.

---

## Bottom line (elegant-but-impractical vs actionable)

**Actionable now (deterministic, GB10-cheap, no hallucination risk):**
- Decouple the merge query grid from `scale` (free; enables fractional scale) — §1.
- Fit a **B-spline / shift-invariant continuous representation** to the merge for
  arbitrary-coordinate resampling — §3.
- Replace `deblur_gated`'s post-RL **bilateral with wavelet shrinkage (ForWaRD-style)** —
  §4b.
- Replace the per-tile PSF table with a **continuous radial/ring PSF field** (and/or
  field-varying Zernikes) for SV deconvolution — §5.
- (Moderate) **N=1 local-linear kernel regression** in the merge Warp kernel — §1.

**Elegant but (currently) impractical / research-grade:**
- A **joint continuous inverse problem** unifying merge+deblur over a continuous scene with a
  continuous PSF (SuperF-style INR + Zernike/ring `φ`) — the correct endpoint, but iterative,
  learned, and hallucination-sensitive — §5/§1.
- **Continuous-Fourier learned merge** (BurstM) and **pretrained INR SR** (LIIF/LTE) — strong
  but learned; only via test-time optimization on the actual burst, behind the project's
  verification gates — §3/§4a.
- **Fourier ptychography proper** / pure frequency-domain merge — wrong tool (phase retrieval
  / no local-motion robustness) — §4a.

**Verified on this machine (per `CLAUDE.md` §1):** the merge = on-grid steering kernel
regression (read from source); Warp `tile_fft`/`tile_ifft` (cuFFTDx) compiles & round-trips
on sm_121 to 2.4e-7 with `block_dim ≤ fft_size/2` (and the `eval/deconv.py` docstring's
`block_dim=64`/`fft_size=64` example is wrong → `ept=1` failure). **Not verified / flagged:**
all *quality* claims (would a spline/ForWaRD/radial-PSF actually beat the current deconv on
the real burst?) are untested here — they must go through `eval/` measurement + the
adversarial discipline in `eval/CLAUDE.md` before being believed. The Wronski supplemental
(exact `k1,k2` formula from `k_detail`/`k_denoise`/`k_stretch`/`k_shrink`) was not separately
fetched; the main-paper Eqs.1–4 above are sufficient and match `kernels.py`.

---

## Sources

Primary papers (equations quoted above were extracted from the actual PDFs via `pdftotext`
and verified against the paper title/authors):

- Takeda, Farsiu, Milanfar, "Kernel Regression for Image Processing and Reconstruction,"
  IEEE TIP 16(2), 2007 — https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf
- Wronski et al., "Handheld Multi-Frame Super-Resolution," ACM TOG 38(4) / SIGGRAPH 2019 —
  https://3dvar.com/Wronski2019Handheld.pdf · project: https://sites.google.com/view/handheld-super-res/ · arXiv: https://arxiv.org/abs/1905.03277
- Chen, Liu, Wang, "Learning Continuous Image Representation with Local Implicit Image
  Function" (LIIF), CVPR 2021 — https://arxiv.org/pdf/2012.09161 · https://yinboc.github.io/liif/
- Lee, Jin, "Local Texture Estimator for Implicit Representation Function" (LTE), CVPR 2022 —
  https://arxiv.org/abs/2111.08918
- Unser, "Sampling—50 Years After Shannon," Proc. IEEE 88(4), 2000 —
  https://bigwww.epfl.ch/publications/unser0001.pdf
- Tsai & Huang (1984) via Vandewalle et al., "A Frequency Domain Approach to Registration of
  Aliased Images with Application to Super-resolution," EURASIP JASP 2006 —
  https://asp-eurasipjournals.springeropen.com/articles/10.1155/ASP/2006/71459
- Neelamani, Choi, Baraniuk, "ForWaRD: Fourier-Wavelet Regularized Deconvolution," IEEE TSP
  52(2), 2004 — https://ieeexplore.ieee.org/document/1261329/ · code: https://www.ece.rice.edu/dsp/software/ward.shtml
- Lindeberg, "Feature Detection with Automatic Scale Selection," IJCV 1998 —
  https://people.kth.se/~tony/papers/cvap198.pdf · scale-space: https://www.kth.se/profile/tony/page/scale-space-theory

Continuous / domain / burst-SR & deconvolution:

- BurstM: "Deep Burst Multi-scale SR using Fourier Space with Optical Flow," ECCV 2024 —
  https://arxiv.org/html/2409.15384
- SuperF: "Neural Implicit Fields for Multi-Image Super-Resolution," 2025 —
  https://arxiv.org/html/2512.09115 · https://arxiv.org/pdf/2512.09115
- Deep Wavelet Prediction for Image Super-Resolution, IEEE 2017 —
  https://ieeexplore.ieee.org/document/8014882/
- Near-optimal non-uniform interpolation for SR from multiple images —
  https://www.researchgate.net/publication/228936798_Near_optimal_non-uniform_interpolation_for_image_super-resolution_from_multiple_images
- RKHS optimal interpolation of scattered field data, IEEE TIP —
  https://ieeexplore.ieee.org/abstract/document/730385/
- Constant-overlap-add / partition of unity (Smith, CCRMA) —
  https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_Decomposition.html

Spatially-varying PSF / Zernike / deconvolution:

- Zernike polynomials and their applications (review), IOP 2022 —
  https://iopscience.iop.org/article/10.1088/2040-8986/ac9e08
- PSF estimation from computed (Zernike) wavefronts for deconvolution, Sci. Rep. 2024 —
  https://www.nature.com/articles/s41598-024-84790-6
- Ring Deconvolution Microscopy (radial/rotational symmetry), arXiv 2206.08928 —
  https://arxiv.org/abs/2206.08928
- Spatially-variant RL + TV (svDeconRL) — https://github.com/dop-oxford/svDeconRL ·
  Temerinac-Ott 2011 — https://lmb.informatik.uni-freiburg.de/Publications/2011/BRT11/TemerinacISBI2011.pdf
- Deconvolution with a spatially-variant PSF (astronomy), astro-ph/0208247 —
  https://arxiv.org/pdf/astro-ph/0208247
- MultiWienerNet: deep learning for fast spatially-varying deconvolution, Optica 2022 —
  https://opg.optica.org/optica/fulltext.cfm?uri=optica-9-1-96
- Patch-wise SV aberration correction & blocking-artifact caveat, Sci. Rep. 2025 —
  https://www.nature.com/articles/s41598-025-19568-5

Fourier-domain / ptychography:

- SAVI: synthetic apertures via Fourier ptychography, Science Advances 2017 —
  https://www.science.org/doi/10.1126/sciadv.1602564
- Fast Single Image Super-Resolution (Wiener/frequency inverse) — https://arxiv.org/pdf/1510.00143

**Executed on this GB10 (not cited from docs):** `/tmp/warp_fft_probe.py` (Warp 1.14.0,
CUDA 12.9, sm_121) — `tile_fft`/`tile_ifft` round-trip err 2.38e-07; `hasattr(wp,'tile_fft')`
== False (compile-time builtin); cuFFTDx requires `block_dim ≤ fft_size/2`. Source read:
`handheld_super_resolution/{warp_ops.py,merge.py,kernels.py}`, `eval/{deconv.py,psf.py}`,
`handheld_super_resolution/params.py`.
