# Wavelet-domain & continuous-representation SR — applicable lessons for a classical, non-hallucinating burst pipeline

Deep-research note (companion to `eval/research_continuous_formulation.md`). The user's hypothesis:
recent **wavelet** and **continuous-representation** SR contains progress I have under-weighted, and some
of it may transfer to *our* classical multi-frame pipeline. I engaged it open-mindedly, fetched the two
references the user named plus the surrounding 2017–2026 literature, then applied the `eval/CLAUDE.md`
discipline (red-team favorable claims; never repeat the "looks promising" trap that killed the 3 prior
continuous directions).

**Standing diagnosis this note must respect** (verified in `eval/NOTES.md`, not re-litigated here):
parity with DxO; **edges are OPTICS-limited** (per-frame PSF ≈2.1 px — only *deconvolution* moves them,
not SR/flow/kernel/freq); natural content is **already ~anti-aliased** (<4% exploitable aliasing; sampling
ratio 1.5–1.8 < 2.0); the merge already extracts ~**77%** real (cross-frame-consistent) HF; every
sharpening lever on noisy raw amplifies noise faster than real detail; and the project **rejects
hallucination** by design.

**Honesty flag up front.** The two papers the user asked me to weigh are **both learned**, and the most
structurally-relevant one (ContinuousSR) turns out to be a learned re-derivation of the estimator we
*already ship*. The genuinely transferable ideas in this whole space are **old and classical** (Donoho
WVD; Neelamani ForWaRD wavelet-shrinkage deconvolution) and they attack the **one lever that is actually
open here — deconvolution** — not SR. None of it lifts the optics ceiling. Specifics below; every external
claim is cited in **Sources**.

---

## What was *executed / read directly* vs. what is cited prose

Per repo `CLAUDE.md` §1, separating verified-by-me from trusted-from-source:

- **Read directly (PDF, this machine):** the full WaveHiT-SR paper (arXiv:2508.19927) — its equations
  and tables, *not* the WebFetch summary (which was vague and partly wrong: it guessed "predict HR
  subbands then IDWT", which the paper does **not** do).
- **Read directly (repo):** that the merge is steering-kernel regression — established and re-verified in
  `eval/research_continuous_formulation.md` against `warp_ops.py`/`merge.py`/`kernels.py`. I rely on that
  prior reading; I did not re-open the Warp kernel for this note.
- **Cited from source (not executed):** the ContinuousSR/GaussianSR/GSASR/BurstM/DiWa/LIIF/LTE equations
  and claims, and the classical WVD/ForWaRD/Donoho theory. The LIIF local-ensemble equation is quoted from
  the prior extraction in `research_continuous_formulation.md` (the live PDF fetch returned binary).
- **Not verified (would need a probe):** every *quality* claim about whether a technique would help **our**
  burst — flagged inline as **[unverified]**. The prior continuous note's §6 made exactly these
  predictions and an adversarial pass then **killed all three**; I treat any "could help" here the same way.

---

## PART A — The two references the user named

### A1. ContinuousSR  (`github.com/peylnog/ContinuousSR`) = "Pixel to Gaussian", arXiv:2503.06617, ICLR 2026

I confirmed the repo's paper identity from its raw README BibTeX (`peng2025pixel`, arXiv:2503.06617) — it
is **not** an INR/LIIF paper as one might assume from "ContinuousSR"; it is **2D Gaussian Splatting**.

1. **Core idea + math.** Represent the HR image as a sum of anisotropic 2D Gaussians and *render*
   (rasterize) it at any output grid:
   ```
   f_c(x,y) = Σ_{i=1}^N G_i(x,y)                                                    [their Eq. 3]
   G_i(x,y) = c_rgb · (1 / 2π|Σ_i|) · exp(−½ dᵀ Σ_i⁻¹ d),   d = (x,y) − μ_i        [their Eq. 5]
   ```
   Each Gaussian carries position `μ`, covariance `Σ` (σx², σy², ρ), and RGB. A CNN encoder produces
   `F_LR = E(I_LR)`; three small heads predict the field. Their two tricks:
   - **Deep Gaussian Prior (DGP):** over 40k natural images, ~99% of `σx², σy², ρσxσy` fall in fixed
     bounded ranges → instead of regressing Σ freely, **sample N fixed kernels from the DGP and learn
     softmax weights** `W = Softmax(M_weight(F_LR))`, `G_target = Σ w_i G_i` (their Eq. 6, 9, 10).
   - **Adaptive Position Drifting:** learn sub-pixel offsets `P_off = Tanh(M_pos(F_LR))`,
     `P_final = P_init + P_off` (Eq. 11–12) — denser kernels in textured regions.
2. **Learned, or deterministic core that transfers?** The **rendering (Eq. 3+5) is deterministic** given
   the field, and is **mathematically the estimator we already ship**: a normalized/weighted sum of
   anisotropic 2D Gaussians with per-element covariance, evaluated at arbitrary coordinates =
   steering-kernel regression (`research_continuous_formulation.md` §0/§1). **The parameter source is
   learned** (CNN heads, trained on DF2K bicubic pairs, scale U(4,8), L1+frequency loss, 1000 epochs / 8×V100).
3. **Transferable lesson.** Two, both already realized differently in our code:
   - *Render-from-a-Gaussian-field is exactly our merge.* "ContinuousSR" validates the steering-KR merge
     as a *continuous representation* — the merge **is** a queryable 2D-Gaussian field; we just don't
     re-render it at arbitrary scales because `scale` is baked into the output grid. Decoupling the query
     grid from `scale` (the cheap, deterministic win already noted in the prior note §1/§3) is the
     genuinely transferable structural idea.
   - *Bounded-covariance prior (DGP).* Their empirical finding that natural-image kernel covariances live
     in a small bounded range is a *sanity bound* on our `Ω`: it says the structure-tensor Σ should rarely
     exceed ~2.4 px var — a free clamp/validation on `k_detail`/`k_stretch` outputs, not a new method.
4. **HONEST verdict.** On our optics-limited, already-AA'd, 14-bit raw content: **the deterministic part
   is something we already do**, so it cannot recover *new* real HF — it would only re-express the merge.
   The *learned* part (where their +0.90 dB comes from) draws Σ/μ/RGB from a DF2K-trained prior → it
   **hallucinates** detail not present in the (single) input, the precise thing the project rejects. Its
   headline contribution is **speed** (arbitrary-scale render in ~1 ms), which is irrelevant to a fidelity
   pipeline. **Usable here: the framing (merge = continuous Gaussian field, re-queryable) and the bounded-Σ
   sanity check. Not usable: the network, the prior, the detail.** Net: *not a new capability for us.*

### A2. arXiv:2508.19927 = **WaveHiT-SR** (Hierarchical Wavelet Network for Efficient SR)

Read the full PDF directly (the abstract/HTML fetches were thin). It is **single-image**, **learned**, and
its "wavelet" is an **internal efficiency trick, not a signal-processing recipe**.

1. **Core idea + math.** A SwinIR-family transformer where DWT is applied to **internal feature maps**:
   ```
   F_wavelet = DWT(F) = { F_HH, F_HL, F_LH, F_LL }                                  [their Eq. 1]
   ```
   Used two ways: (a) as a **÷2 downsampler** replacing pooling inside "Wave Attention" (γ=2), and (b) to
   make attention cheap — **WaveAttention is O(hw·log(hw))** vs window-self-attention **O(N(hw)²)** (their
   Table 1). Reconstruction is **spatial-domain** (`I_HR ∈ ℝ^{3×sH×sW}`); they do **not** predict HR
   subbands and IDWT.
2. **Learned, or deterministic core that transfers?** The **DWT/IDWT is a fixed deterministic transform**,
   but it is applied to *learned features* purely to cut attention cost. There is **no transferable
   classical algorithm** here — pull out the transformer and nothing image-processing-useful remains.
3. **Transferable lesson.** Essentially none for us. The only durable, *general* idea it reuses is the old
   one (HF detail is **sparse in wavelet subbands**, so a wavelet basis is a good place to *separate*
   coarse from fine) — which is the same insight that makes the *classical* wavelet-deconvolution of Part C
   work. WaveHiT-SR itself is an efficiency paper.
4. **HONEST verdict.** Learned single-image SR; modest gains (~**0.4–0.5 dB** Urban100 ×2 vs SwinIR-Light;
   trained DIV2K bicubic), headline is **fewer params / lower FLOPs / ~6× faster**. On our content it would
   **hallucinate** texture from the DIV2K prior with no multi-frame verification, and it brings no new
   real-HF mechanism. **Nothing usable here** beyond the generic "wavelets sparsify HF" motivation, which
   we already get from the classical lineage below at zero hallucination risk.

---

## PART B — Continuous-representation SR family (LIIF → LTE → CiaoSR → Gaussian-Splatting SR → BurstM)

The user asked me to weigh this family open-mindedly. Structurally it is the most interesting, because the
*combination rule* keeps reappearing as **our merge**.

### B1. The recurring structural identity (the real finding of Part B)

- **LIIF** (Chen 2021): image = feature map `M` + a **shared MLP** `f_θ` queried at continuous `x_q` with a
  **bilinear-area-weighted, normalized ensemble of 4 neighbour latents** (Eq. 4, quoted in
  `research_continuous_formulation.md`). That ensemble is structurally `num/den` — our merge — with learned
  latents and an MLP combiner instead of measured samples and a Gaussian weight.
- **LTE** (Lee 2022): adds a learned **dominant-frequency / Fourier-feature** estimator so the MLP beats
  spectral bias on texture.
- **CiaoSR** (Cao 2023, CVPR): replaces the fixed ensemble weights with **learned implicit attention** over
  neighbour features (+ scale-aware non-local attention) — i.e. *learn* the merge weights.
- **GaussianSR** (AAAI 2025) / **ContinuousSR** (A1) / **GSASR** (ICCV 2025): drop the MLP and render an
  explicit **2D-Gaussian field**. GaussianSR's renderer (its Eq. 5):
  ```
  C(p) = Σ_i c_i · (1/2π|Σ_i|) · exp(−½ (p−μ_i)ᵀ Σ_i⁻¹ (p−μ_i)),   c_i = σ(ξ)·v_i
  ```
  — **verbatim the steering-KR merge.** And the search corroborates the family relation: "normalized
  convolution … is a simplified version of adaptive kernel regression," and Gaussian splatting sits in the
  **DARBF / radial-basis kernel-regression** family. The rendering, given params, is deterministic.

  **Crucial deterministic-vs-learned distinction (quoted, GSASR):** *"The original Gaussian Splatting is an
  optimization-based method that overfits each single scene, while in arbitrary-scale SR the goal is to
  learn a single model that generalizes."* So the splatting **representation has a per-image,
  training-free optimization mode** — only the *feed-forward parameter prediction* needs a training set.
  This is the seam where a non-hallucinating version *could* live (see verdict).

### B2. Transferable lesson

The architecture reveals **where these methods think recoverable information lives**: in a *continuous,
anisotropic, locally-adaptive kernel field*, re-sampleable at any coordinate. **We already build that
field** (the structure-tensor `Ω(x)` merge). The transferable, deterministic, hallucination-free pieces:

1. **Decouple query grid from `scale`** — re-render the existing merge field at fractional/arbitrary scale
   with no retrain (the cheap win from the prior note, re-confirmed by the entire ASSR family being built
   for exactly this).
2. **The merge *is* a continuous representation** — frame it as such; it unifies "merge + resample" and is
   the substrate the deconv should act on (prior note §3/§6.4).
3. **(Speculative, deterministic) per-burst Gaussian-field fit instead of feed-forward prediction** — fit
   `{μ_i, Σ_i, c_i}` to the *actual warped multi-frame samples* by least squares (GSASR's "overfit a single
   scene" mode, but multi-frame and training-free). This is a different parameterization of the same
   scattered-data reconstruction the merge does, with free positions — i.e. it is N≥1-KR-with-adaptive-bandwidth
   wearing a splatting hat.

### B3. HONEST verdict (continuous family)

- **LIIF/LTE/CiaoSR/GaussianSR/ContinuousSR as shipped are learned single-image SR → hallucinate.** On
  already-AA'd, optics-limited content their extra "detail" is drawn from a bicubic-pairs training prior,
  not from our frames; they would fail `eval/hallucination.py` exactly as DxO's DeepPRIME does (~24–28%
  per-frame invention). **Reject as shipping methods.**
- **The deterministic core (render an anisotropic-Gaussian field at any coordinate) is something we already
  ship**, so adopting it recovers **no new real HF** — at best it gives free fractional scale and a cleaner
  resample substrate. **[unverified]** that even the training-free per-burst Gaussian-field fit (B2.3)
  beats the current merge — and the strong prior from the prior note's adversarial pass is that it would
  **not**, because it is the same family as N=1/adaptive-bandwidth KR, which was killed as a *bias-variance
  dial, not a detail source* (real HF flat across operating points; noise the only thing that moved). It
  would need the synthetic-GT gauntlet before any belief, and the expected payoff is low.
- **BurstM** (ECCV 2024) is the one that is *multi-frame* and *continuous-Fourier* — it predicts continuous
  per-frame Fourier coefficients (A/F/phase nets) + optical-flow warping, giving arbitrary scale and
  targeting HF texture. It is **the learned realization of the user's thesis**. But A/F/phase are learned
  CNNs/MLPs (hallucination surface), and the prior note already measured the deterministic frequency-domain
  de-aliasing it descends from to **die on real conditions** (box-MTF Nyquist null; <4% aliasing;
  registration mismatch PSNR 35.7→13.8 at 0.3 px; real-data HF coherence 0.48 *falling* with displacement),
  with BurstM itself reported at only **+0.17 dB** over classical in that analysis. **Offline upper-bound
  reference behind the hallucination gate; not a shipping target.**

---

## PART C — Wavelet-domain SR & deconvolution (where the *only* transferable win actually is)

Two distinct things travel under "wavelet SR." One is a hallucination machine; the other is the genuinely
useful, classical, deterministic technique that targets **our open lever (deconvolution)**.

### C1. Learned wavelet SR — DWSR / MWCNN / Wavelet-SRNet / DWA / WaveHiT-SR / wavelet-diffusion

1. **Core idea.** Predict the **missing HF subbands** (LH/HL/HH) from the LR approximation with a CNN/
   transformer, then **IDWT** (DWSR, Wavelet-SRNet); or use DWT/IDWT as the down/up-sampler in a U-Net
   (MWCNN); or run a **diffusion model in the wavelet domain** (DiWa, WaveDiff, HDW-SR).
2. **Learned or deterministic?** DWT/IDWT is a fixed deterministic transform, but **the subband prediction
   is learned** — that is where the detail is manufactured.
3. **Transferable lesson.** Only the *motivation*: **HF detail is sparse in wavelet subbands**, so the
   wavelet basis is the natural coordinate system to *separate* and *operate on* fine detail. That same fact
   is what makes the **classical** C2 methods work — and C2 needs no learning.
4. **HONEST verdict — and the smoking gun.** The DiWa authors **state outright** that their model "*effectively
   **hallucinate**[s] high-frequency information for super-resolved images on the wavelet spectrum*" — the
   field naming the mechanism the project exists to avoid. Wavelet-diffusion SR is, by the authors' own
   words, **invention**. **Reject all of C1 as shipping methods** (single-image; trained priors; invented
   HF; would fail `eval/hallucination.py`). Wavelet **max-coefficient multi-frame fusion** also exists
   classically (DWT each frame → keep the largest |coefficient| per subband → IDWT) — deterministic, **but
   a strictly worse merge** than our robust steering-KR (no noise model, no robustness, picks noisy
   outliers as "salient"). Reject as a merge replacement.

### C2. Classical wavelet **deconvolution** — WVD (Donoho) and **ForWaRD** (Neelamani–Choi–Baraniuk) — the one transferable technique

This is the payoff of the whole investigation. The project's verified open lever is **deconvolution** of
the ~2.1 px optical PSF (`eval/NOTES.md`: edges optics-limited; only deconv moves them; current path is
`measured_rl` RL + a bilateral denoiser bolted on after). The classical wavelet literature has the
**principled version of exactly that bolt-on**.

1. **Core idea + math.**
   - **Wavelet–Vaguelette Decomposition (WVD, Donoho 1995):** to invert a convolution `K`, expand the
     target in wavelets, build the matched *vaguelette* series for `Kf`, then recover by **nonlinear
     shrinkage (thresholding) of the WVD coefficients** of the noisy data. Donoho proved WVD is
     **minimax-optimal** among linear and nonlinear estimators for inverting such operators, and it
     **preserves spatially-inhomogeneous detail** (edges) far better than SVD/Tikhonov.
   - **ForWaRD (2004):** the hybrid that fixes WVD's and Wiener's individual weaknesses — a *small* Fourier
     (Wiener) shrinkage to economically invert the operator/colored noise, **then wavelet shrinkage** to
     remove the noise the inverse amplifies, with the Fourier/wavelet balance set by minimizing an
     approximate MSE. Result: **"minimal ringing, unlike pure-Wiener; applicable to all ill-conditioned
     deconvolution, unlike pure-wavelet WVD."** Underneath is Donoho–Johnstone **soft-threshold
     denoising** (VisuShrink, universal threshold `λ = σ·√(2 log N)`): edges/lines = large sparse
     coefficients survive; noise = small dense coefficients are shrunk.
2. **Learned or deterministic?** **Fully deterministic, classical, hallucination-free.** Shrinkage cannot
   invent cross-frame structure — it can only *attenuate* coefficients. It runs on `pywt`/cupy/torch
   wavelets; no training set, no prior, no network.
3. **Transferable technique (concrete).** Replace the **post-RL bilateral** in `deblur_gated`
   (`eval/deconv.py`) — a crude edge-preserving denoiser chosen ad hoc — with **wavelet-shrinkage denoising
   of the deconvolution residual (ForWaRD-style)**. ForWaRD is *purpose-built* for "an inverse filter
   amplified colored noise; remove it without eating edges," which is **precisely** the regime
   `eval/NOTES.md` documents (RL raises flat-sky noise 0.0015→0.0025 while sharpening edges). It directly
   targets the project's only real lever, is deterministic, and slots into the existing field-varying
   deconv (`deblur_field_varying`). Soft-thresholding's known bias means **prefer a SureShrink/BayesShrink
   per-subband threshold or a small soft-threshold over the universal one**, gated on the structure tensor
   (the prior note's edge gate) so flats aren't over-smoothed.
4. **HONEST verdict.** **This is the single genuinely-applicable, deterministic, real-HF, low-hallucination
   technique in the entire wavelet/continuous space for our pipeline** — and it is from 1995/2004, not 2025.
   It does **not** recover new HF beyond the optics ceiling (deconv inverts the measured MTF on
   *cross-frame-real* merge detail; it adds no invention). It is a **strictly better noise/ringing
   regularizer for the deconvolution we already do** — a cleaner edge at lower noise than RL+bilateral, with
   provably less ringing than Wiener. **[unverified on our data]:** whether ForWaRD-shrinkage actually beats
   the current RL+bilateral on the 007 facade edges (flat-sky noise, deslanted overshoot, edge σ) — this
   must be measured with the existing `eval/deconv.py` harness against the DxO-consensus edge σ ≈ 1.87 and
   the merge-noise floor, under the one-sided objective from `eval/CLAUDE.md` (can't win by over-sharpening).
   It is the **one item worth prototyping**.

---

## PART D — The honest meta-finding

The user's intuition — "promising recent progress under-weighted" — is **half right and half a trap**:

- **Right:** the continuous-representation framing *is* illuminating, because it shows our merge is a
  **continuous anisotropic-Gaussian kernel-regression field** that the newest SOTA (ContinuousSR/GaussianSR)
  literally re-derives. That reframing (queryable field; free fractional scale; merge = the representation)
  is real and worth internalizing — but it is a **re-description of what we already compute**, not a new
  capability.
- **The trap:** essentially **all** the *recent* (2021–2026) wavelet/continuous SR that produces a numeric
  win does so with a **learned prior that invents HF** — and at least one (DiWa) **says so in print**
  ("hallucinate high-frequency information"). On our already-AA'd, optics-limited, 14-bit-raw content,
  importing any of them would re-introduce exactly the per-frame invention the project measures and rejects.
  The recent papers' gains are **perceptual-prior gains**, not **real-information gains**; this pipeline's
  whole thesis is the latter.
- **The genuinely transferable technique is old and classical**: **ForWaRD / WVD wavelet-shrinkage
  deconvolution**, which improves the *one lever that is actually open* (deconvolution), deterministically,
  with zero hallucination — and even it cannot beat the optics sampling ceiling, only spend the existing
  budget more cleanly.

This is consistent with — and adds nothing that overturns — the prior conclusion: **the pipeline already
extracts ~all the real HF the samples hold; the durable axis is real-vs-invented detail, and the only
honest sharpening lever is measured-PSF deconvolution.** The wavelet literature's contribution to *that*
lever is a better regularizer (ForWaRD), not a new source of detail.

---

## Sources (URLs actually fetched/read for this note)

ContinuousSR / Gaussian-splatting SR:
- ContinuousSR repo + README BibTeX — https://github.com/peylnog/ContinuousSR
- "Pixel to Gaussian: Ultra-Fast Continuous SR with 2D Gaussian Modeling" (ContinuousSR), arXiv:2503.06617 — https://arxiv.org/html/2503.06617v1 , https://arxiv.org/abs/2503.06617
- GaussianSR (2D GS for ASSR), arXiv:2407.18046 — https://arxiv.org/html/2407.18046v1
- GSASR ("overfits each single scene" quote), arXiv:2501.06838 — https://arxiv.org/abs/2501.06838 , https://mt-cly.github.io/GSASR.github.io/

WaveHiT-SR (the user's second reference):
- WaveHiT-SR, arXiv:2508.19927 — https://arxiv.org/abs/2508.19927 ; full PDF read directly — https://arxiv.org/pdf/2508.19927

Continuous-representation (INR) family:
- LIIF, arXiv:2012.09161 — https://arxiv.org/pdf/2012.09161 (equations via prior extraction in research_continuous_formulation.md)
- LTE (Local Texture Estimator), arXiv:2111.08918 — https://arxiv.org/abs/2111.08918
- CiaoSR (CVPR 2023) — https://ar5iv.labs.arxiv.org/html/2212.04362 , https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_CiaoSR_Continuous_Implicit_Attention-in-Attention_Network_for_Arbitrary-Scale_Image_Super-Resolution_CVPR_2023_paper.pdf

Burst / multi-frame continuous & generative:
- BurstM (ECCV 2024), arXiv:2409.15384 — https://arxiv.org/abs/2409.15384 , https://github.com/Egkang-Luis/burstm
- GenMFSR ("preserve deterministic info in aligned raw"; subspace-projected score distillation), arXiv:2603.19187 — https://arxiv.org/abs/2603.19187 , https://arxiv.org/html/2603.19187v1
- MFSR-GAN (CVPRW 2025) — https://arxiv.org/html/2502.20824v1

Learned wavelet SR + wavelet-diffusion (the hallucination cases):
- DWSR "Deep Wavelet Prediction" (IEEE 8014882) — https://ieeexplore.ieee.org/document/8014882/
- Wavelet-SRNet (ICCV 2017) — https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.pdf
- DWA (Differential Wavelet Amplifier), arXiv:2307.04593 — https://arxiv.org/pdf/2307.04593
- DiWa ("hallucinate high-frequency information" — verbatim), arXiv:2304.01994 — https://arxiv.org/abs/2304.01994 , https://github.com/Brian-Moser/diwa
- HDW-SR (HF-guided wavelet diffusion), arXiv:2511.13175 — https://arxiv.org/html/2511.13175v1

Classical wavelet/Fourier deconvolution & denoising (the transferable lineage):
- ForWaRD (Neelamani–Choi–Baraniuk, IEEE T-SP 2004) — https://ieeexplore.ieee.org/document/1261329/ , https://www.ece.rice.edu/dsp/software/ward.shtml
- Wavelet–Vaguelette Decomposition (Donoho) — discussion/overview https://imjohnstone.su.domains/WEBLIST/2004/WavDeconvDiscussion.pdf ; WVD for ill-conditioned systems https://apps.dtic.mil/sti/tr/pdf/ADA495095.pdf
- Donoho "De-noising by soft-thresholding" (1995) — https://www.fceia.unr.edu.ar/~jcgomez/wavelets/Donoho_1995.pdf
- Multi-frame DWT max-coefficient fusion (overview) — https://www.academia.edu/51637931/A_new_super_resolution_image_reconstruction_algorithm_based_on_wavelet_fusion

Cross-references (already in the repo):
- eval/research_continuous_formulation.md (merge = steering-KR; the 3 prior continuous directions)
- eval/NOTES.md (parity, optics-limited edges, ~77% real HF, deconvolution as the lever)
