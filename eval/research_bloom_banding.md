# Research: the "banding / bloom" near high-contrast edges (007 facade)

Question (user): a BANDING / BLOOM artifact appears near high-contrast edges in the
multi-frame SR output; the user suspects **lens bloom (veiling glare)**. (A separate
finding already showed the apparent BLOCKING was a nearest-neighbor render-zoom
artifact, not real tiles — the 26.9 px FFT period is the louvers; corroborated, not
the focus here.)

**Verdict (executed, triangulated):** the bloom the user sees is the **compact
~3–4 px optical + sensor PSF edge skirt** (the system MTF / the *narrow core* of the
lens PSF), made visually prominent by shadow-lifted display. It is **OPTICAL — present
identically in a single demosaiced RAW frame and in the DxO render — NOT introduced or
amplified by our pipeline.** It is **NOT lens veiling glare**: veiling glare is a wide,
field-spanning, low-frequency contrast veil with a power-law (~r⁻²) tail extending tens
to hundreds of px; the measured floor excess instead vanishes within ~4–5 px of every
edge. A *trace* (~0.5 % of source) wide component sits at the noise floor and is not
what's visible. Our measured-PSF (narrow-Gaussian) deconvolution does **not** amplify
it. **Recommendation: leave it alone** — it is the per-frame optics, already as good as
DxO's, and there is no wide glare tail to correct.

All numbers below were executed on this machine (GB10) via `micromamba run -n handheld`.
Light is **linear** throughout (un-gamma the stored luma `**2.2`); veiling glare is
additive scattered flux, which is linear, not perceptual.

---

## 1. The clean measurement: dark-floor "bloom" vs distance from a bright edge

**The discriminating test.** Veiling glare = the dark floor is *elevated near a bright
edge and decays SLOWLY with distance* (a wide low-frequency scattered-light PSF tail).
Distinguish from (a) a compact PSF/MTF edge skirt (few px), (b) deconvolution ringing
(oscillatory, goes *below* the floor), (c) the louver structure, (d) merge/tile
artifacts. So: profile the dark-gap floor brightness vs distance `d` from a bright
edge, in linear light, averaged over a tall band of rows to beat noise.

**Single best edge (genuinely isolated): src ≈ 0.8 linear with an 809 px dark run.**
Found by scanning the whole frame for the brightest source with the longest adjacent
dark run (the dense facade mostly lacks isolated edges; this one — a bright roofline /
column against deep shadow at single-frame y≈1800, x≈3003 — is clean). Floor excess as
a fraction of source brightness:

| d (sensor px) | 2 | 3 | 4 | 5 | 8 | 15 | 30 | 60 | 120 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| excess / src | +0.039 | +0.017 | +0.010 | +0.007 | +0.003 | −0.001 | +0.004 | +0.008 | +0.007 | −0.000 |

The excess drops **below 1 % of source by d≈4 px** and is at/below the noise floor
beyond. There is no slow wide decay.

**Aggregate over 67 isolated bright edges** (`eval/halluc_out_007/bloom_profile.png`,
median ± IQR, single-raw vs our-merge vs DxO, all registered):

| d (px) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| single raw | +0.192 | +0.060 | +0.021 | +0.005 | −0.002 | −0.006 |
| our merge | +0.206 | +0.057 | +0.017 | +0.003 | −0.004 | −0.008 |
| DxO | +0.163 | +0.040 | +0.012 | +0.002 | −0.004 | −0.007 |

A compact spike at d=1–2 px (the edge slope itself, for a system PSF σ≈1.8–2.1 px) that
is **gone by d≈4–5 px**. Beyond that the median goes slightly *negative* (neighbouring
structure), then returns to 0 by ~50 px — a glare tail would stay *positive*. The same
holds for **horizontal louver edges** (vertical skirt; n=232/389/156): excess 0.12→0.01
from d=2→5 px, ~0 by d=8 px.

**Wide-gap flatness cross-check.** In a 56 px-wide dark gap bounded by bright columns on
both sides, the floor is **flat** in the interior (coefficient of variation 3–5 %); a
two-edge fit gives edge-skirt **τ ≈ 1.7–2.0 px** (τ / gap-half-width ≈ 0.07). If glare
were present the two columns' tails would meet and the floor would be U-shaped with τ ~
the gap half-width (~28 px). It is not — compact edge skirts + flat interior = no glare
bridge.

**Power-law aureole fit (the King / veiling-glare signature).** Fitting the deep floor
to `core·exp(−d/τ) + A_pl·d^−p` on the 809 px-run edge: the aureole amplitude is
`A_pl = 4.0e-2 ± 1.9e-2`, `p≈1.0`, but its contribution is **0.25 % of src at d=20 px,
0.10 % at d=50, 0.05 % at d=100 — all ≤ the 0.43 %-of-src noise floor.** Direct annulus
means: d=20–60 px `+0.0035 ± 0.0036` (≈0), d=80–200 px `+0.0058 ± 0.0040` (a faint ~1.5σ
hint), d=300–600 px `+0.0006 ± 0.0035` (0). **Honest reading:** there is *at most* a
trace ~0.5 %-of-source broad component (genuine faint veiling glare, or residual scene
light in the "dark" run — can't separate at this level), ≈100× weaker than the near-edge
skirt and invisible. The dominant, visible effect is unambiguously the compact PSF core.

---

## 2. Optical or introduced by us? — OPTICAL (and DxO has it too)

Same edge, registered (template-match NCC 0.95–0.998; DxO offset −12,−12 px from pan +
its 4000×6000 vs our 4024×6048 grid; our s1 merge aligns at 0,0), measured in linear
light for: **(A)** a single bilinear-demosaiced RAW frame (DSC05183.ARW), **(B)** our s1
merge, **(C)** the DxO reference (DSC05183_DxO.dng):

- The compact edge skirt is **present and essentially identical in all three** (table
  above; the three `bloom_profile.png` curves overlap within their IQR bands).
- **Our merge ≈ single raw** at every distance (d=1: 0.206 vs 0.192; d=3: 0.017 vs
  0.021) — within noise. On horizontal louver edges our merge is if anything *slightly
  tighter* than the single frame (d=4: 0.019 vs 0.030). **We do not add bloom.**
- **DxO is marginally lower** at every distance (sharper edges from its deconvolution +
  harder flat denoise), but shows the **same compact-skirt shape and no wide tail** —
  i.e. even a lens-corrected pro pipeline neither exhibits nor removes a glare veil here.
- The deep dark floor (~0.012–0.017 linear in the gap) exists in **DxO too**
  (DxO 0.0122 < single 0.0149 < merge 0.0168; merge−single = +0.0019, +12 %, a tiny
  contrast-floor lift from the merge, not a glare). This floor is scene/sensor black
  level + ambient, not a removable glare.

⇒ The effect is in the per-frame optics (lens + pixel-aperture MTF), in the RAW before
our pipeline. It is **not** a port/merge/deconv artifact.

**What the user is actually seeing** (`eval/halluc_out_007/bloom_aggressive_stretch.png`):
under a **fair shared stretch + cubic zoom** the louvers/gaps look clean and similar in
single / merge / DxO / merge+deconv. Under an **aggressive shadow-lift** (lifting the
dark gaps hard, as a tone-mapped/auto-levels view does) the few-px edge skirt *balloons*
into a wide visible "glow / banding" in the dark gaps — and it appears **just as much in
the single RAW frame and DxO as in our merge.** This is the `eval/CLAUDE.md` failure
mode (display gain fabricating apparent wide structure from a sub-noise/near-edge signal)
applied to a *real* few-px skirt: the skirt is real and optical; its *apparent width* is
a display-gain illusion.

---

## 3. Does our measured-PSF deconvolution amplify the bloom? — NO

Our deconv (`eval/deconv.py`, opt-in `postprocessing.aberration.deblur_method=
measured_rl`) inverts a **narrow** Gaussian (calibrated merge PSF σ≈1.9–2.1 sensor px).
Re-measuring the floor near the bright edge after RL deconvolution (crop, linear):

| variant | floor/src @6 px | @20 px | @80 px | deep floor |
|---|---|---|---|---|
| raw merge | +0.016 | −0.004 | +0.001 | 0.0363 |
| global RL σ_r=1.0, it10 | +0.004 | −0.002 | +0.001 | 0.0369 |
| global RL σ_r=2.0, it10 | −0.006 | −0.003 | +0.002 | 0.0391 |
| **gated RL σ_r=1.1, it5 (shipped path)** | +0.009 | −0.004 | +0.001 | 0.0369 |

The **far floor (d=20–80 px) is unchanged** by the deconvolution in every case — exactly
as expected: a narrow-Gaussian inverse has no frequency content at the scales a wide
glare tail would occupy, so it *cannot* model or amplify one. The deconv only sharpens
the immediate edge (d≤6 px), slightly *reducing* the near-edge bump; at the strong
σ_r=2.0 setting it produces a tiny ringing *undershoot* at d=6 px (−0.006), not a bloom.
So: the deconv does not amplify the bloom, and (good news) since there is **no** wide
glare tail, no glare-PSF deconvolution stage is warranted.

---

## 4. Literature: veiling glare / flare PSF and correctability

The discriminating definitions match exactly what was tested and ruled out:

- **Veiling glare is a wide, diffuse, field-spanning, low-frequency contrast veil**, not
  a localized few-px edge effect. *"a widespread and diffuse veil that reduces image
  contrast even under normal lighting conditions, in contrast to structured artifacts
  such as lens flare or ghosting"*; *"low-frequency, spatially-varying"* from
  *"stray-light scattering from non-ideal optical surfaces and coatings"* (Learning
  Latent Transmission and Glare Maps, arXiv 2511.17353). It arises from multiple internal
  scattering and reduces contrast *everywhere* in the field (Talvala et al., *Veiling
  glare in HDR imaging*, SIGGRAPH 2007).
- **The PSF is two-component: a narrow Gaussian core (a few px) + a broad aureole / wings
  following ≈ r⁻² out to many × the FWHM** — King (1971); the aureole *"dominates optical
  instrumental PSFs out to distances a few times the FWHM"* and *"the wings… have a
  shallower fall-off than a Gaussian… more similar to an inverse-square power law"*
  (wide-angle-PSF literature). Stray-light/veiling glare *is* that aureole. **My data
  show only the narrow core (gone by 4–5 px); the r⁻² aureole is at the noise floor.**
- **Correction** historically is *"deconvolution using the Glare Spread Function (GSF) or
  image decomposition, yet these methods impose restrictive assumptions, often require
  multiple captures, or fail in low texture scenes"* (arXiv 2511.17353; also Talvala 2007
  uses a structured-occlusion *extra capture* to measure the GSF). Modern flare/glare
  removal is largely *learned* (a survey: *Toward Flare-Free Images*, arXiv 2310.14354) —
  which would *hallucinate*, against this project's deterministic-only stance.

**Worth modeling here?** No. (1) There is no measurable wide glare tail to correct
(≤0.5 % of source, at the noise floor). (2) GSF deconvolution needs an extra
structured-occlusion capture or restrictive assumptions we don't have. (3) Learned glare
removal hallucinates — contrary to the pipeline's principle (eval/NOTES.md). (4) The
visible "bloom" is the per-frame optical MTF; the *only* deterministic lever for it is
edge **deconvolution of the narrow PSF**, which the pipeline already has (opt-in
`measured_rl`) and which already reaches DxO's edge sharpness — and even that leaves the
dark-gap floor flat (§3). The far better fix for the *perception* is **display/tone**
(don't shadow-lift the dark gaps so hard), not an optical model.

---

## Honest negatives / caveats

- The dense facade has few truly isolated bright/dark boundaries; the 809 px-run edge and
  the 67-edge aggregate are the cleanest available. The faint d=80–200 px excess
  (~0.5 % src, ~1.5σ) is *consistent with* either a trace of real veiling glare or
  residual scene light in the "dark" reference — **not separable at this level**, and
  ~100× below the visible near-edge skirt either way. I do not claim glare is exactly
  zero; I claim it is invisible and not what the user perceives.
- An early single-edge fit on the s2 merge read τ≈9 px; reconciled — that was the edge
  *slope* of a brighter, longer source spanning more s2 px, **not** a wide tail: in s2
  the normalized floor/src matches s1 (few-% at d≤2 sensor px, ~0 beyond ~4 px). Trust
  the normalized, multi-edge distribution, not a single-edge τ.
- Measured on one burst (007, a6300, ISO 200). The optical-skirt conclusion should
  generalize (it's per-frame optics, confirmed in single RAW + DxO), but the trace-glare
  bound is scene-specific.

## Reproduce

Scratch scripts were under `/tmp/bloom_scratch/` (removed). Key steps, all
`micromamba run -n handheld python`: register single-RAW / s1-merge / DxO at a facade
ROI (cv2.matchTemplate on gamma luma); profile the dark-gap floor vs distance in linear
light over a tall row band; aggregate over auto-found isolated bright edges; fit
core+power-law to the deep floor; A/B the RL deconv (`eval.deconv.deblur` /
`deblur_gated`) on a crop. Figures kept (gitignored):
`eval/halluc_out_007/bloom_profile.png` (the decay, 3-way) and
`eval/halluc_out_007/bloom_aggressive_stretch.png` (why shadow-lift makes it look like
bloom). Inputs: `~/Pictures/{ARW,DNG}/2025-09-06/007/DSC05183*`,
`eval/halluc_out_007/merge_s2_flow_bilinear.npy` (s2 gamma luma),
`merge_s1_linear.npy` (s1 RGB).

## Sources
- [Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal (arXiv 2511.17353)](https://arxiv.org/html/2511.17353)
- [Talvala, Adams, Horowitz, Levoy — Veiling glare in high dynamic range imaging (SIGGRAPH 2007)](https://dl.acm.org/doi/10.1145/1276377.1276424)
- [Toward Flare-Free Images: A Survey (arXiv 2310.14354)](https://arxiv.org/pdf/2310.14354)
- [Removal of image intensifier veiling glare by mathematical deconvolution (PubMed 4010632)](https://pubmed.ncbi.nlm.nih.gov/4010632/)
- [King 1971 wide-angle PSF / aureole — A Method to Characterize the Wide-angle PSF (ApJ, IOPscience)](https://iopscience.iop.org/article/10.3847/1538-4357/ac32c6)
- [The influence of diffuse scattered light: the PSF (arXiv 1406.5508)](https://arxiv.org/pdf/1406.5508)
- [Slanted-edge ESF→MTF method (PMC9823301)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823301/)
