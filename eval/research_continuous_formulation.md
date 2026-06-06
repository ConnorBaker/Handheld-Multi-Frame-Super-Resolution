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

## 6. Recovering high-frequency detail: continuous approaches

This section targets the **open problem** in `eval/NOTES.md`: the pipeline reaches
**parity** with DxO and **`scale=2` does not out-resolve `scale=1`** ("Out-resolve DxO
at <=1x? PARITY"; "scale=2 SR does NOT help these edges (s2 2.15 ~ s1 2.10)"). The
symptom is that the SR appears to *interpolate*, not *resolve* real high frequencies.
The sections above mapped the merge to steering kernel regression (SKR); this section
asks, theme by theme, **which continuous direction could actually recover HF**, with the
math, the concrete code change, GB10 feasibility, and an actionable-vs-research verdict.
Everything external is cited in **Sources**; equations marked "extracted" were pulled
from the PDF via `pdftotext` and read directly (not summarized from a snippet).

### 6.0 First, the theory that says *why* it doesn't resolve (the diagnosis)

Two independent results pin the cause, and they agree.

**(a) Wronski's own resolution ceiling — the sampling ratio, not the algorithm.** The
paper reports its *measured* ceiling verbatim (extracted from the ar5iv HTML, §7.1):
*"significantly improved results at 1.5× zoom, small improvement up to 2× zoom and no
additional resolution gains … thereafter,"* explained by *"the sampling ratio for the
main cameras … in the range 1.5 to 1.8 (in the luminance channel) which are lower than
the critical sampling ratio of two"* ([Wronski 2019](https://3dvar.com/Wronski2019Handheld.pdf)).
**This is the NOTES.md result, predicted.** Multi-frame SR resolves *aliased*
(sub-Nyquist) content; how much aliased content exists is set by the optical-sampling
ratio (lens MTF vs pixel pitch), which for a phone/compact is already near or below 2.
Once you've de-aliased what aliasing there is, **there is no more real HF to recover by
merging**, and `scale=2` only interpolates — exactly the symptom. Wronski also states the
two-point-resolution law is *"proportional to SNR^p, where 0<p<1"* — HF recovery degrades
gracefully with SNR, it is not a hard wall.

**(b) Baker–Kanade — the reconstruction-constraint information collapses with
magnification.** "Limits on Super-Resolution and How to Break Them" proves (extracted
from the PDF) that *"the reconstruction constraints provide far less useful information as
the magnification factor increases"*; beyond a modest factor *"any smoothness prior leads
to overly smooth results"* and *"the prior provides more information than the
super-resolution reconstruction constraints"*. Their headline practical number: with
realistic noise and registration, the **effective reconstruction-based magnification limit
is ≈ 1.6×** ([Baker–Kanade 2002](http://www.cs.cmu.edu/afs/cs/project/sjb2/papers/journal/pami/limits/accepted_version.pdf)).
Beyond it, only **learned priors ("hallucination")** add detail — and they explicitly
warn it is *invented*, class-specific detail, the thing this repo's `eval/hallucination.py`
exists to reject.

**The crucial caveat that *loosens* Baker–Kanade for THIS pipeline (flagged: this is the
most consequential finding here).** Baker–Kanade's pessimism rests on the input being
**8-bit**: *"the required high-frequency information was lost from the reconstruction
constraints when the input images were discretized to 8-bit values"* (extracted). This
pipeline merges **14-bit raw** (`eval/NOTES.md`: GRBG raw; demosaic-in-the-merge). Raw is
12–14 bit vs ISP RGB's 8 bit — *"more information can be exploited in raw pixels since
they are typically 12 or 14 bits, whereas the color pixels produced by ISP are typically
8 bits"* ([Xu et al., Exploiting Raw Images for Real-Scene SR](https://arxiv.org/abs/2102.01579)).
The quantization noise floor is therefore ~**64× lower** (2^6) than Baker–Kanade assumed,
so the reconstruction constraints retain *far* more HF than their 8-bit limit predicts.
**Implication:** the 1.6× wall is an 8-bit artifact; on 14-bit raw the binding constraints
here are (i) the optical sampling ratio (6.0a, the real ceiling) and (ii) **registration
precision** (6.2) and (iii) the **0th-order, 3×3, single-Ω gather** (6.1) — i.e. things
this codebase controls, not a fundamental quantization wall. *This is a hypothesis from
two cited facts, not an executed measurement — it predicts a test (6.2/6.1 below).*

### 6.1 Higher-order & adaptive-bandwidth kernel regression — does N≥1 recover HF?

**The math (extracted from [Takeda 2007](https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf)).**
N-th-order KR fits a local degree-N Taylor polynomial by weighted least squares; the
estimate is `ẑ(x) = e₁ᵀ (Xᵀ W X)⁻¹ Xᵀ W y` (W = steering-kernel weights, X = the Vandermonde
of sample offsets). N=0 is Nadaraya–Watson — the repo's `num/den`. The decisive result is
Takeda's **"Equivalent Kernels"** section, read in full:

> "regardless of the order, the classical kernel regression is nothing but local weighted
> averaging of data (linear filtering), where the order determines the type and complexity
> of the weighting scheme … higher order regressions are equivalents of the zero-order
> regression but with a more complex kernel function."  *(extracted, Takeda §II-D)*

Two consequences land directly on this pipeline:

1. **Higher order is still a deterministic local weighted average — it cannot
   hallucinate.** N=1/2 reweight the *same* samples with a sharper, derivative-like
   equivalent kernel; they do not invent cross-frame structure. This makes N≥1 *compatible*
   with the repo's hallucination discipline (unlike INR).
2. **For *uniformly* sampled data, N=0, 1, 2 equivalent kernels are essentially
   identical** (odd moments vanish by symmetry: "the kernels for N=0, 1 … are essentially
   identical … the lower-order regression is preferred"). **But for *irregularly* sampled
   data the equivalent kernels differ and "tend to adapt themselves to the density of
   available samples"** (extracted, Fig. 6). **This is the burst-SR regime**: after warping
   by the per-frame flow, the pooled samples are *scattered*, not on a grid. So **N≥1 can
   help here precisely to the extent the warped multi-frame samples are genuinely irregular**
   — which is the SR case, not the single-frame case.

**Adaptive bandwidth — the continuous "h→0 as density rises" theory the user asked for.**
Takeda's smoothing-matrix model (extracted, Eq. 39–41): `H_i = h · μ · ρ_i` where `h` is a
global smoothing parameter (set by leave-one-out cross-validation, Eq. 40), `μ` a scale,
and `ρ_i` a **local density factor**, with `ρ_i ∝ (sample density)^(−1/2)` (his Eq. 41, the
Abramson/Silverman rule, density-sensitivity exponent proved optimal at 0.5). Plainly:
**where samples are dense, the kernel should shrink; where sparse, it should widen.** The
classical MISE-optimal global bandwidth scales as `h* ∝ n^(−1/(d+4))` (here d=2 →
`n^(−1/6)`) ([OALB / bandwidth theory](https://link.springer.com/article/10.1007/s11760-023-02556-5)),
so adding frames (raising local density `n`) *licenses a narrower kernel* and a sharper
reconstruction — **as long as the alignment that places those samples is accurate** (6.2).
This is the principled version of Wronski's hand-tuned `λ₁ → support` rule and of
Lindeberg's scale selection (§2).

**What this changes in THIS code, concretely.**
- `warp_ops.py` `_accumulate_warp`: today, per output pixel, one bilinearly-interpolated
  `Ω`, a fixed 3×3 gather, and the N=0 accumulation `num += w·r·c ; den += w·r`. The N=1
  upgrade replaces the divide with a tiny **3-unknown weighted normal-equation solve**
  (fit local value + 2 slopes, return the value): build `XᵀWX` (3×3 symmetric) and `XᵀWy`
  from the same 9 taps, solve, take `e₁ᵀ·`. Warp has `tile`/`tile_matmul`/`tile_cholesky`
  for the per-pixel 3×3 solve, or it inlines as a closed-form 3×3 (cheap). Cost: a small
  constant factor over the current gather — *well within budget* (the merge is already the
  fast part; `eval/NOTES.md` perf: 0.0166 s/frame at 48 MP).
- **Adaptive bandwidth** is even cheaper: scale `Ω⁻¹` per pixel by a **local sample-density
  factor** derived from the accumulated `den` (or the accumulated-robustness frame count
  the ref-merge already reads, `warp_ops.py` `acc_rob`) — i.e. shrink the kernel where many
  frames contributed clean samples. This is a `ρ_i`-style multiplier on `z` in the existing
  exponent, not a new pass.

**Verdict.** **Actionable and the best deterministic HF lever in the gather**, *with a
caveat I must flag*: Takeda's own theory says N≥1 buys little **if the warped samples are
near-uniform**, and `eval/NOTES.md` already found the merge "inherently ~1px-soft at scale
1 … detail lives in `scale` and alignment quality, not kernel width" — i.e. on the tested
content the kernel was not the binding constraint. **So N=1 should be prototyped in `eval/`
against synthetic GT (the un-gameable anchor) before believing it recovers HF**; the strong
prior is that **adaptive bandwidth driven by true sample density** + **N=1 on genuinely
irregular (well-aligned, many-frame) samples** helps most exactly where the merge is
currently softest, and helps little on already-resolved edges (consistent with the
parity result). This is *research-grade-leaning-actionable*: bounded, deterministic, cheap
to try, uncertain payoff. **Do NOT** expect it to beat the 6.0a sampling-ratio ceiling — it
sharpens the use of the information that is there; it does not create sub-optical-Nyquist
information.

### 6.2 Continuous sub-pixel flow & registration precision — the real gate

**The SR-from-aliasing math (extracted from [Park, Park & Kang 2003](https://cse.buffalo.edu/courses/cse725/peter/Park_2003.pdf)).**
The canonical statement of *why* multi-frame resolves HF, verbatim:

> "If the LR images are shifted by integer units, then each image contains the same
> information, and thus there is no new information … If the LR images have different
> subpixel shifts … and if aliasing is present, however, then each image cannot be obtained
> from the others … the new information contained in each LR image can be exploited."
> *(extracted)*

The HF *is* the aliased component; the **sub-pixel phase diversity is the key that
unmixes** the aliased HF from the true LF (Tsai–Huang frequency-domain SR, below). It
follows that **the recoverable HF is only as good as the sub-pixel offsets are known.**

**The reconstruction bound vs registration error.** Baker–Kanade make registration one of
the two terms (with noise) that set the 1.6× limit; the SR literature is unanimous that
*"accurate estimation of the sub-pixel motion … significantly affects the … HR image"* and
that mis-registered samples must be *down-weighted or excluded*
([SR review search](https://www.researchgate.net/publication/224710999_Are_Registration_Errors_Always_Bad_for_Super-Resolution)).
The rigorous well-posedness statement is **Ur & Gross via the Papoulis–Brown generalized
sampling theorem: for translational motion the anti-aliasing-via-SR problem has a
*unique, well-posed* solution** — but the governing system is *"linear in the unknown
signal coefficients but **nonlinear in the offsets**"* ([generalized-sampling SR](https://link.springer.com/chapter/10.1007/0-306-47004-7_4)).
The nonlinearity in offsets is exactly why a fraction-of-a-pixel error corrupts the HF
solve disproportionately: small offset error → large HF coefficient error. **This is the
mathematical reason ~0.1 px alignment is the gate** — at that scale the offset-nonlinearity
is locally linearizable and the HF coefficients are recoverable; past it they smear.

**Why this is the most promising direction for THIS pipeline.** `eval/NOTES.md`
independently reached the same conclusion from the *other* end (measurement): the single
biggest verified quality lever was **alignment iterations + robustness rejection**
(`ica.n_iter` 3→6/8; `robustness.t` up), *not* the kernel or patch size; and the
block-artifact fix was making the **flow continuous** (nearest→bilinear flow upscale). The
theory (6.2) and the measurements (NOTES) converge: **registration precision is the binding
HF constraint here**, more than the merge kernel. Concrete next steps already pointed at by
the code:
- **Tighten and continuize the flow.** ICA is the sub-pixel stage (`ICA.py` /
  `ica_iterate_cupy`); the flow is still **piecewise-constant per tile** then
  bilinearly upsampled (`warp_ops.py` `al[py,px]`). A **genuinely continuous (per-pixel)
  flow field** — densify ICA output, or fit a smooth field to the tile flows — removes the
  last tile-grid quantization of the sample positions (NOTES already halved the block
  residual this way; going per-pixel finishes it).
- **Measure the achieved sub-pixel accuracy** (the missing executed evidence). The frequency
  view gives a free, deterministic *check*: low frequencies are **aliasing-free** and can
  cross-validate the flow (Vandewalle/Tsai–Huang, 6.3) — phase-correlate the LF band of each
  warped frame against the reference; residual LF phase = residual mis-registration. If it's
  ≫0.1 px on the failing (foliage/branch) tiles, **registration is the fix and N≥1/Fourier
  are moot until it's closed.**
- **Robustness must not discard well-aligned HF.** NOTES: the mask under-rejects moving
  content yet is ~constant on static scenes; tuning `t`/`Mt` to reject *mis-aligned* frames
  while *keeping* the sharp aligned ones is a direct HF lever.

**Verdict.** **The single most actionable direction, and the one both theory and the repo's
own measurements endorse.** It is deterministic, needs no new representation, and attacks
the constraint that 6.0/6.2 identify as binding on 14-bit raw. Highest priority.

### 6.3 Frequency-domain de-aliasing — can Fourier resolve where the gather can't?

**Classical (Tsai–Huang), extracted from Park §"Frequency Domain Approach".** Stack the
aliased LR DFTs and relate them to samples of the HR continuous Fourier transform via the
**shift property + aliasing relation + bandlimitedness**; the result is a linear system
`Y = Φ X` where `Y` are aliased LR DFT coefficients, `X` are samples of the unknown HR CFT,
and `Φ` (p × L₁L₂) encodes the sub-pixel shifts (extracted). Solving it **explicitly
de-aliases**: it inverts the fold-back that a single frame can't undo. **This is a
*different operation* from the spatial gather** — the gather (SKR) reconstructs a continuous
signal by *weighted averaging in space*, which is an excellent *denoiser/interpolator* but
combines aliased samples rather than algebraically separating the aliased HF from the true
LF. **In principle a frequency-domain solve can resolve HF the spatial gather leaves
folded** — *if* the shifts are global-translation and known precisely (its fatal limitation:
*"the Fourier shift theorem … cannot model arbitrary displacements,"* extracted — no local
motion, no occlusion, the very things this pipeline's robustness exists for).

**Modern continuous-Fourier (BurstM), extracted from [arXiv:2409.15384](https://arxiv.org/html/2409.15384).**
BurstM keeps the de-aliasing-in-Fourier idea but makes it *continuous* and *local-motion
tolerant*: FNet predicts per-frame optical-flow offsets `δ_i^L`, warps coordinates
`x̂_i^H = B_s(δ_i^L + x_i^L)`, then **predicts a continuous Fourier representation per frame**
via learned amplitude/frequency/phase estimators and reconstructs
`m_i = A_i ⊙ [cos(π(⟨F_i, δ_i⟩ + ph_i)), sin(π(⟨F_i, δ_i⟩ + ph_i))]` (extracted), fusing in
Fourier space. It explicitly *"predicts the continuous Fourier coefficient of each frame for
representing high-frequency textures"* and uses Fourier mapping to beat INR spectral bias —
i.e. it is engineered for the exact HF-recovery failure here, and gives **arbitrary/fractional
scale** for free. **But A/F/ph are learned CNNs/MLPs (EDSR encoder + estimators + 4-layer MLP
decoder)** — a hallucination surface that must pass the repo's gauntlet.

**Generalized de-aliasing without learning (Papoulis–Gerchberg / generalized sampling).**
The deterministic cousin of BurstM's idea: **Papoulis–Gerchberg** alternates projections
between the spatial domain (impose known samples) and the spectral domain (impose
bandlimit), extrapolating the band — used for SR by fusing band-limited shifted frames
([PG SR](https://www.researchgate.net/publication/220459504_Application_Of_Papoulis-Gerchberg_Method_In_Image_Super-Resolution_and_Inpainting)).
It is **deterministic and hallucination-free** but *"exhibits very slow convergence"* and
"underperforms when input images are seriously degraded by blur, noise, and sampling"
(extracted from search). **Generalized sampling (Papoulis–Brown / Ur–Gross)** is the exact,
well-posed multichannel-de-aliasing solve for known translational offsets (6.2) — the
principled deterministic target.

**What this changes in THIS code & feasibility.** A frequency-domain *merge* is **not** a
drop-in for the spatial merge (it can't carry the per-pixel robustness/local-motion model —
the reason Wronski went spatial). But two narrower, GB10-feasible uses:
- **A frequency-domain de-aliasing *refinement* on top of the spatial merge**, applied
  *only* on edge/aliased tiles (gated by the structure-tensor `l1` the pipeline already
  exposes, `estimate_kernels(return_l1=True)`): on a near-static high-`l1` tile, run a small
  generalized-sampling/PG solve over the few warped frames to unmix the residual aliasing the
  gather averaged. Warp's **device-side `tile_fft`/`tile_ifft` (cuFFTDx) is verified on this
  sm_121** (§intro, 2.4e-7 round-trip), so a fused tile-wise spectral solve is implementable
  on-device; `torch.fft` already drives the full-frame deconv.
- **Frequency-domain registration *check*** (6.2) — free, deterministic, uses the
  aliasing-free LF band.

**Verdict.** **Research-grade but the most theoretically-promising route to *resolve*
(not interpolate) residual HF** — because it does the one thing the spatial gather provably
cannot (algebraic de-aliasing). The deterministic path (tile-wise generalized-sampling / PG
on gated aliased tiles, on-device FFT) is the one to prototype in `eval/`; it is bounded and
hallucination-free but convergence/conditioning are real risks and it only pays off where
genuine aliasing remains (6.0a says that's limited on this optics). **BurstM is the strong
learned version** — keep as an offline upper-bound reference behind `eval/hallucination.py`,
not a shipping target.

### 6.4 Continuous forward-model inversion — the principled SR+deconv unification

**The math.** The unified continuous statement (SuperF's forward model, §5):
`y_LR^(t)(v) ≈ φ * y_HR(A^(t) v)` — each LR frame is the continuous scene `y_HR` warped by
`A^(t)`, blurred by the continuous PSF `φ`, sampled. **Merge + deblur + SR are then one
inverse problem**: find the continuous `y_HR` (and `A^(t)`) minimizing
`Σ_{t,i} R̂ · ‖(φ * y_HR ∘ A^(t))(x_i) − c_{t,i}‖² + regularizer`. Solving the *joint*
problem is strictly more powerful for HF than the repo's **split** (gather, then deconv): the
deconvolution sees every frame's samples and the true sampling geometry at once, instead of
deconvolving a single pre-averaged merge whose HF the averaging already attenuated. This is
the **generalized-sampling/Unser** view made operator-complete: reconstruct the continuous
function in a chosen space (B-spline shift-invariant, §3 — deterministic, no hallucination,
*"simpler and possibly more realistic interpolation … wider class of anti-alias prefilters
not necessarily ideal low-pass,"* [Unser 2000](https://bigwww.epfl.ch/publications/unser0001.pdf))
that is consistent with all scattered, blurred samples.

**What this changes in THIS code & feasibility.**
- **Deterministic, shippable slice:** keep the fast gather, but make the deconv **part of a
  single continuous resample+inverse** rather than a post-hoc on-grid RL. Fit a **B-spline
  coefficient field** to `num/den` once (separable IIR, cheap on torch/cupy), then apply the
  **field-varying inverse of `φ(x)`** (the radial/ring or Zernike PSF field, §5) *at the
  continuous query coordinates* — one operator, no 512-tile blocking/feather (§2/§5). This is
  the §3+§5 recommendations *fused*, and it is the realistic "continuous forward-model" win.
- **Research end-state:** a true joint solve (SuperF-style test-time optimization) *with an
  explicit `φ(x)`* — SuperF deliberately fixes `φ` to a boxcar, so **adding a Zernike/ring
  PSF to a SuperF-style per-burst solve is the precise way to unify merge+deblur+SR
  continuously** (§5). Iterative (~seconds/burst), and SuperF is learned (MLP field) — but
  it is **test-time-optimized on the actual burst with no external HR training set**
  ([SuperF, arXiv:2512.09115](https://arxiv.org/abs/2512.09115)), so its hallucination surface
  is far smaller than a pretrained feed-forward INR (it can only fit the frames it's given).
  Still must pass `eval/hallucination.py`.

**Verdict.** The **B-spline-field + field-varying-PSF-inverse fusion is actionable** and is
the cleanest deterministic realization of "one continuous inverse problem"; it improves the
*use* of HF (no averaging-then-deconv loss, no block artifacts) but, like 6.1, **cannot
exceed the 6.0a sampling-ceiling** — it recovers HF the split currently wastes, it does not
manufacture sub-Nyquist detail. The **joint SuperF+φ solve is the theoretically-correct
endpoint**, research-grade, gated hard.

### 6.5 INR / learned continuous reps — only with the hallucination caveat

Covered in §3 and above: LIIF/LTE/BurstM/SuperF all *can* synthesize convincing HF, and
**that is the danger** — Baker–Kanade are explicit that past the reconstruction limit the
extra detail is *invented*, and `eval/hallucination.py` measured DxO's learned single-frame
reconstruction inventing ~24–28% per-frame texture. **Prefer deterministic** (6.1–6.4).
If learned at all: **test-time optimization on the actual burst only** (SuperF, no external
training set), behind the project's verification gates, and never as the shipping default.
The repo's durable edge is *real-detail honesty* (the merge keeps only cross-frame-consistent
detail); an INR trades that away for sharpness-on-paper.

### 6.6 Ranked synthesis for HF recovery

1. **Registration precision / continuous per-pixel flow (6.2)** — *actionable, highest
   confidence.* Theory (offset-nonlinear de-aliasing, ~0.1 px gate) and the repo's own
   measurements (alignment was the #1 lever; continuous-flow fixed the block artifact) agree
   this is the binding HF constraint on 14-bit raw. Add: measure achieved sub-pixel accuracy
   via the aliasing-free LF band; densify the flow to per-pixel.
2. **Field-varying PSF inverse fused with a B-spline continuous resample (6.4 + §3/§5)** —
   *actionable.* Recovers HF the current average-then-deconv split wastes; kills block
   artifacts; deterministic.
3. **Adaptive-bandwidth + N=1 kernel regression in the merge (6.1)** — *actionable-to-prototype.*
   Deterministic, cheap, helps exactly where the merge is softest *and where warped samples
   are irregular*; uncertain on already-resolved edges — gate against synthetic GT first.
4. **Deterministic frequency-domain de-aliasing refinement on gated aliased tiles (6.3)** —
   *research-grade, highest theoretical upside* (does the algebraic de-aliasing the gather
   can't); on-device FFT verified; bounded but convergence/conditioning risk and limited
   remaining-aliasing payoff.
5. **Joint continuous inverse / SuperF+φ; BurstM (6.4/6.3/6.5)** — *research-grade, learned*;
   offline upper-bound references behind hallucination gates only.

**The honest ceiling (flag).** 6.0a (Wronski's measured sampling-ratio ceiling) and 6.0b
(Baker–Kanade) say there is a hard cap on *resolvable* (real) HF set by the optics, and the
NOTES.md parity result is consistent with the pipeline **already near that cap on the tested
(coarse, resolvable) content**. 1–4 above recover HF the pipeline currently *wastes*
(mis-registration smear, average-then-deconv loss, fixed-bandwidth 0th-order gather) and so
can plausibly move it from parity to a genuine edge — **but only a sub-Nyquist / resolution-
chart target near the sensor Nyquist will *prove* an out-resolve**, and that regime is
exactly where learned methods hallucinate, so the durable framing stays *real-detail vs
invented-detail*. **Unverified here:** every quality claim in §6 (would N=1 / per-pixel flow
/ tile-FFT de-aliasing actually raise resolved HF on the real burst?) is theory + the repo's
prior measurements, not a new executed result — they must go through `eval/` synthetic-GT
fidelity + the `eval/CLAUDE.md` adversarial discipline before being believed.

---

## Bottom line (elegant-but-impractical vs actionable)

**For the HF-resolution / parity problem specifically, see §6** (the ranked HF-recovery
synthesis is §6.6). One-line summary: the binding constraint on 14-bit raw is **registration
precision + a continuous flow** (§6.2), not the kernel; Baker–Kanade's pessimistic 1.6×
limit is an **8-bit** artifact and is ~64× looser here (§6.0); the spatial gather can only
*average* aliased samples, so the route to genuinely *resolve* residual HF is **frequency-
domain de-aliasing** (§6.3); and no continuous method beats the optics' **sampling-ratio
ceiling** (Wronski's own 1.5×-significant/2×-marginal result, §6.0a) — which the NOTES.md
parity result is consistent with the pipeline already being near.

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

§6 — Recovering high-frequency detail (the SR limit, registration, de-aliasing). PDFs
marked "extracted" were downloaded and read via `pdftotext`; the equations/quotes in §6
came from those local text dumps, not from a search snippet:

- Baker & Kanade, "Limits on Super-Resolution and How to Break Them," IEEE TPAMI 24(9),
  2002 — *(extracted)* http://www.cs.cmu.edu/afs/cs/project/sjb2/papers/journal/pami/limits/accepted_version.pdf ·
  book chapter "Super-Resolution: Limits and Beyond" — https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2001_1/baker_simon_2001_1.pdf
  (the ≈1.6× reconstruction-based limit; HF info "lost when … discretized to 8-bit"; prior
  > constraints past the limit; "hallucination" = invented class-specific detail).
- Park, Park & Kang, "Super-Resolution Image Reconstruction: A Technical Overview," IEEE
  Signal Processing Magazine 20(3), 2003 — *(extracted)* https://cse.buffalo.edu/courses/cse725/peter/Park_2003.pdf
  (integer shifts → no new information; sub-pixel + aliasing → recoverable HF; observation
  model `y = DBM x + n`; Tsai–Huang `Y = ΦX` aliased-LR-DFT ↔ HR-CFT system).
- Takeda, Farsiu, Milanfar, IEEE TIP 2007 — *(extracted, §II-D Equivalent Kernels & §II-E
  Smoothing Matrix)* https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf
  ("regardless of the order … local weighted averaging"; N=0/1/2 equivalent kernels
  identical for *uniform* samples but adapt to density for *irregular* samples; smoothing
  matrix `H_i = h·μ·ρ_i`, density factor `ρ_i ∝ density^(−1/2)`).
- Takeda thesis / ISKR & MASK (iterative + adaptive-bandwidth SKR; 3-D ISKR) —
  https://users.soe.ucsc.edu/~milanfar/publications/dissertations/TakedaPhDThesis.pdf ·
  Super-Drizzle (adaptive KR for SR) — https://amostech.com/TechnicalPapers/2006/Imaging/Takeda.pdf ·
  3-D ISKR chapter — https://users.soe.ucsc.edu/~milanfar/publications/journal/3-DSKRChapter.pdf
- Fan & Gijbels local polynomial regression (higher order p reduces asymptotic bias without
  raising the variance *order*; derivative/order selection) — Derivative Estimation with
  Local Polynomial Fitting, JMLR 2013 — https://jmlr.csail.mit.edu/papers/volume14/debrabanter13a-deleted/debrabanter13a.pdf
- Optimum adaptive local bandwidth (OALB) for non-uniform data; MISE `h* ∝ n^(−1/(d+4))` —
  https://link.springer.com/article/10.1007/s11760-023-02556-5
- Xu, Ma, et al., "Towards/Exploiting Raw Images for Real-Scene Super-Resolution," CVPR
  2019 / TPAMI — https://arxiv.org/abs/2102.01579 ·
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf
  (raw is 12–14 bit vs ISP RGB 8 bit → more radiance info; the basis for the "Baker–Kanade
  8-bit wall is ~64× looser on raw" argument).
- Vandewalle, Süsstrunk, Vetterli, "A Frequency Domain Approach to Registration of Aliased
  Images …," EURASIP JASP 2006 — https://link.springer.com/article/10.1155/ASP/2006/71459
  (LF aliasing-free → registration; HF aliased → SR information).
- Ur & Gross / Papoulis–Brown generalized (multichannel) sampling — well-posed exact
  de-aliasing for known translational offsets; "linear in the signal coefficients, nonlinear
  in the offsets" — https://link.springer.com/chapter/10.1007/0-306-47004-7_4
- Papoulis–Gerchberg bandlimited extrapolation for SR (deterministic, slow convergence) —
  https://www.researchgate.net/publication/220459504_Application_Of_Papoulis-Gerchberg_Method_In_Image_Super-Resolution_and_Inpainting
- "Are Registration Errors Always Bad for Super-Resolution?" (registration-error effect on
  reconstruction) — https://www.researchgate.net/publication/224710999_Are_Registration_Errors_Always_Bad_for_Super-Resolution
- Burst-SR landscape (context; learned fusion, HF limitations): Deep Burst SR —
  https://arxiv.org/pdf/2101.10997 · RBSR — https://arxiv.org/pdf/2306.17595 ·
  BIPNet/base-frame selection — https://postech-cvlab.github.io/Burst_FSN/

**Executed on this GB10 (not cited from docs):** `/tmp/warp_fft_probe.py` (Warp 1.14.0,
CUDA 12.9, sm_121) — `tile_fft`/`tile_ifft` round-trip err 2.38e-07; `hasattr(wp,'tile_fft')`
== False (compile-time builtin); cuFFTDx requires `block_dim ≤ fft_size/2`. Source read:
`handheld_super_resolution/{warp_ops.py,merge.py,kernels.py}`, `eval/{deconv.py,psf.py}`,
`handheld_super_resolution/params.py`. **§6 PDFs extracted via `pdftotext` and read
directly** (Baker–Kanade TPAMI, Park SPM, Takeda TIP) — the §6 equations/quotes are from
those dumps, not from search summaries.

## §6.7 Adversarial verdict (EXECUTED): can they / will they?

The §6 directions were stress-tested adversarially with online research + executed
probes (see eval/NOTES.md "Adversarial pass on the 3 continuous HF-recovery
directions"). Verdicts: continuous per-pixel flow = NO (ill-posed tail, net-negative;
use robustness rejection); adaptive-bandwidth/N=1 KR = NO (bias-variance dial, ~noise;
N=1->N=0 after regularization, Takeda); frequency-domain de-aliasing = NO on real
(ideal ~50dB but pixel-aperture null + <4% real aliasing + registration/local-motion
mismatch; keep only the freq-domain registration CHECK). Downgrade §6.1/6.2/6.3 from
"prototype it" to "bounded/abandoned -- confirmed by execution". Real levers:
deconvolution (edges) + robustness rejection (the unalignable tail).
