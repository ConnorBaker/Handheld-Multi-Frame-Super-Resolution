# Research: deterministic correction of the optical edge skirt

Deterministic-only survey (no learned hallucination). Subagent-produced; load-bearing
framing re-checked. See repo-root `CLAUDE.md` §1 — verify citations before relying.

## Framing fact: Wronski is reconstruction; it LEAVES the lens MTF in

Verified from the paper: the merge is anisotropic-Gaussian kernel regression (Takeda–
Farsiu–Milanfar, IEEE TIP 16(2), 2007), and in their own eval they note the output "is
blurred by the lens, [so] we sharpen it with an unsharp mask filter." Narrowing the merge
kernels (small `k_detail`) raises the *reconstruction's* cutoff but **cannot invert the
optical PSF** — it can't synthesize contrast the lens already attenuated. So a measured-PSF
deconvolution stage is the correct lever, and our gated-RL (measured PSF + structure-tensor
gate + optional bilateral, below DxO overshoot) is squarely the deterministic SOTA for a
compact, ~shift-invariant skirt.

## What DxO actually does (documented vs. inferred)
- **Documented (DxO's own lens-sharpness page, verbatim):** lab-measures sharpness "for all
  frequencies" → a perceptual "BxU"; calibration is **field-, focal-length-, and
  aperture-dependent**; correction applied **per-pixel via EXIF**. So DxO's "Lens Softness"
  is a measured *spatially-varying* MTF/PSF correction — exactly Lever A below.
- **Inference (NOT first-party):** DxO never says "PSF/MTF/deconvolution" for softness — the
  mechanism (true deconvolution vs. calibration-shaped unsharp boost) is undocumented. DxO's
  edge **overshoot** is consistent with an unsharp-mask signature → our below-overshoot
  gated-RL is plausibly *better-behaved* on the no-ringing axis.
- **DeepPRIME is learned** (joint denoise+demosaic CNN). So the DxO output we benchmark
  against is a blend of deterministic lens correction + learned denoise — not a purely
  deterministic reference.

## Next deterministic levers, ranked for our setup
- **A. Spatially-varying (field-dependent) PSF — highest-confidence real gain.** Center is
  sharp, corners soft/astigmatic; our single ~2px-σ kernel is likely a center/average. ESF-
  calibrate at center/mid/corner, interpolate a per-pixel PSF, deconvolve patch-wise with
  overlap-blend or efficient filter flow. Precedent: Schuler et al., "Non-stationary
  Correction of Optical Aberrations", ICCV 2011; Hirsch et al., "Efficient Filter Flow…",
  CVPR 2010; field-dependence magnitude: Lin et al., "Learning Lens Blur Fields", 2023/24
  (arXiv:2310.11535). **Cheapest to test first: MEASURE whether our ESF actually varies
  center→corner on this APS-C crop before investing.** Pure measured inversion (no extra
  hallucination risk), attacks the under-corrected corners.
- **B. Fold deconvolution INTO the SR merge (forward-model / MAP / back-projection) — most
  correct "beyond DxO", biggest effort.** The unified forward model decimation∘blur(PSF)∘
  warp is Farsiu 2004 / Irani–Peleg ICPR 1990 ("the same method can also deblur"). Deconvolve
  the *aligned burst* jointly with the noise model + an edge-aware/hyper-Laplacian prior,
  instead of deconvolving one already-merged (noise-correlated) image. Wins: burst redundancy
  pays the noise-amplification cost at the same step; avoids demosaic→deconv ordering errors.
  Risk: real rearchitecture of the one-shot merge; ringing/noise re-enter if the prior is weak
  (drive its weight by the structure tensor).
- **C. Multi-frame deconvolution of the aligned burst — lighter B.** Replace "merge→RL" with
  "RL whose data term sums over aligned raw frames." **Key honest caveat:** for a *static*
  optical PSF the multi-frame win is **NOT** extra PSF identifiability (that only holds when
  frames have *different* blurs — Zhang–Wipf–Zhang CVPR 2011) — it is **noise** (N× photons →
  push deconv harder before noise breaks). Same detail, cleaner. Misregistration now feeds the
  deconv directly (alignment error ⇒ fake blur/ringing).
- **D. Blind / better PSF estimation — contingency only.** We have a calibrated PSF; Levin et
  al. CVPR 2009 warns naive joint MAP_{x,k} converges to the *no-blur* solution ("the key is
  the estimator, not the prior"). Use only as a small correction *centered on the measured
  kernel* (focus breathing / aperture drift), never from scratch.

## Edge-aware regularized deconvolution toolbox (for any lever)
Plain RL amplifies noise/ringing with iterations. Standard fixes: **TV-regularized RL** (Dey
et al., Microsc. Res. Tech. 69(4), 2006 — "suppresses oscillations while preserving edges");
**damped RL** (White, SPIE 1994 — stop iterating within noise); **Biggs–Andrews acceleration**
(Appl. Opt. 36(8), 1997, ~40× fewer iters). MAP alternative: **hyper-Laplacian prior** |∇|^α,
α≈0.5–0.8 (Krishnan–Fergus, NIPS 2009) — sharper edges, less staircasing than TV, ~seconds/MP.
Fast linear baseline worth trying on a mild skirt: **Wiener / ForWaRD** (Neelamani–Choi–
Baraniuk, IEEE TSP 52, 2004 — "minimal ringing unlike pure Wiener"). Our gate+bilateral is in
effect hand-built spatially-adaptive damping — legitimate and well-motivated.

## Validation discipline (no fabricated detail; specializes eval/CLAUDE.md)
1. **Cross-frame consensus ceiling** — any added HF *outside* the consensus support is
   fabrication; hunt adversarially for it in low-consensus regions.
2. **Synthetic forward-model round-trip** (we already inject PSF+CA) — true GT PSNR/SSIM over
   a *distribution* of scenes/ISO/edge-orientations, not one crop.
3. **Slanted-edge MTF** (ISO 12233) before/after vs. DxO and the ideal — an MTF *exceeding*
   ideal mid-band = unsharp overshoot (DxO); staying below = clean deconv from real detail.
4. **Ringing/noise audits** — dark-gap-floor-vs-distance profile (same tool that ruled out
   veiling glare) for ringing; flat-patch σ for noise; for B/C confirm the gain *scales with
   frame count* (else the multi-frame benefit isn't real).
5. **Symmetric skepticism** — red-team our own favorable result harder than DxO's; the failure
   mode is amplified noise correlated by the kernel into plausible-looking texture.
