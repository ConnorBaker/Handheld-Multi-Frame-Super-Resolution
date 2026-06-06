# Registration / alignment + the two-green opportunity — research & critique

Scope: critique our REGISTRATION/alignment + color path for multi-frame burst SR
(branch `feat/warp-port`), and answer three questions: (1) grayscale registration,
(2) per-channel registration / lateral-CA correction, (3) the RGGB two-green
opportunity. Every mechanism claim below was verified by READING the code and by
running a probe on the GB10 (`micromamba run -n handheld`); literature is cited.

---

## What the code actually does (verified, not inferred)

Data flow in `super_resolution.main` (read + traced): for each frame, the **raw
mosaic** goes two places —
- `compute_grey_images(...)` → a single-channel **grey** image used ONLY for
  `align()` (the alignment proxy), and
- the **raw mosaic itself** → `merge()` directly (the reconstruction input).
Robustness builds its own **guide image** from the raw. The reconstruction never
sees a demosaiced image.

**Grey conversion** (`utils_image.compute_grey_images`, two methods):
- `"decimating"` (`compute_grey_images_decimate_cupy`): averages the whole 2×2
  block `(R + Gr + Gb + B)/4` → one grey pixel at HALF resolution. **Probe 1
  confirmed** output = 25.0 for block {10,20,30,40}. The two greens (and R,B) are
  averaged; the Gr/Gb offset is collapsed.
- `"FFT"` (default, `grey_method: FFT`): FFT of the **full-res** raw, zero the
  outer half of each frequency axis (a low-pass that kills the Bayer
  carrier/chroma at Nyquist), inverse FFT → a full-res luminance-ish grey at the
  RAW grid. Keeps full resolution but is still a single luminance channel; the
  per-color geometry is gone by construction.

**Alignment** (`alignment.py` → `block_matching.py` + `ICA.py`): 4-level Gaussian
pyramid (factors `[1,2,4,4]`) on the grey image; per level coarse L1/L2-FFT block
matching (integer shift) then inverse-compositional Lucas–Kanade (`ica.n_iter=3`)
for sub-pixel. One flow field, computed on grey, shared by all channels.

**Robustness guide** (`cupy_ops.compute_guide_image_cupy`): R, averaged-G, B at
HALF res. **Probe 2 confirmed** green = mean(Gr,Gb) = 25.0 → the two greens are
averaged here too.

**Merge** (`warp_ops._accumulate_warp` / `_accumulate_ref_warp`): a per-output
gather over a 3×3 raw window. Per neighbour it reads the **raw** value `comp[ii,jj]`
and routes it by CFA: `channel = cfa[ii & 1, jj & 1]`. In RGGB, `cfa[0,1]=1`
(Gr) and `cfa[1,0]=1` (Gb) — **both greens map to channel 1**, accumulated into
the **same** `num/den[...,1]` but at their **own true positions** `(ii,jj)` with
their own kernel weights.

**Probe 3 (decisive)** confirmed the merge keeps Gr and Gb DISTINCT: with Gr≡100,
Gb≡110, the green output is 103.5/105/106.5 — a true *spatial* blend of the two
green sub-lattices, not an average computed upstream. **Probe 6** confirmed the
same at `scale=2` (SR): the merge reads the raw mosaic per-channel on the 2× grid,
both greens distinct, NO demosaic before the merge. This is exactly Wronski's
kernel-regression design.

This matches the paper. From Wronski et al. 2019 (fetched): align on a grey image
("we create a single pixel from a 2×2 Bayer quad by combining four different color
channels together"); in the merge "every input raw image pixel … contributes only
to a specific output color channel," and the green samples "remain separate scatter
points in their per-channel accumulation." HDR+ docs (IPOL) add the design reason
for grey + integer-2px alignment: a 2-pixel displacement "has the convenient
feature of replacing Bayer samples with overlapping colors" — i.e. keep alignment
color-agnostic on purpose; let the merge place each color sample at its true spot.

---

## Q1 — Grayscale registration: does it make sense? YES.

**What grey buys us.** One flow field on a single luminance channel is (a) cheap
(¼–½ the pixels for decimate; one FFT pass otherwise), (b) high-SNR — luminance
combines all photons, so the *registration* signal is maximal and noise-robust,
and (c) the honest thing to align: a rigid/affine scene motion is one geometric
transform; the *content* (edges, texture) used to estimate it is shared across
colors. Aligning grey does not "lose registration-useful information" for
estimating that single motion — it concentrates it. This is the field-standard
choice (Wronski/HDR+; classical burst align).

**FFT vs decimate (our two methods).**
- *decimate*: half-res grey → alignment is at half the spatial sampling, so the
  finest sub-pixel cues are gone before LK; cheaper, coarser.
- *FFT* (our default): full-res grey via low-pass; preserves fine structure for
  LK at the cost of one full-res complex FFT per frame. Keeping full res for
  alignment is the right call for sub-pixel precision (consistent with the
  diagnosis that the binding limit is alignment accuracy, not motion search).

**Cons / when grey is suboptimal.** Grey throws away the *per-color geometry*
(lateral CA — see Q2) and any case where one channel is the only one carrying the
edge (e.g. a pure red/blue edge on a green-flat background; rare in natural
scenes). For our scene content and the verified diagnosis (edges optics-limited;
alignment-accuracy-limited, not search-limited), grey is the correct, near-optimal
choice. **Verdict: keep grey registration.** The lever is LK/robustness quality,
not the color space of alignment.

---

## Q2 — Per-channel registration: pros/cons, and lateral-CA correction.

**The idea.** Estimate R, G, B flows separately. Pro: it could *absorb lateral CA*
— the lens shifts R and B radially relative to G (you measured ca_R ≈ +1.35 px on
this a6300 lens), so a per-channel flow would carry R/B to G and remove the
residual color fringing as a free by-product of alignment. The cross-channel-prior
and red/blue-to-green warp literature is exactly this idea (align R,B to G to
cancel lateral CA).

**Why it is the wrong tool *here*, on three grounds:**
1. **SNR / sparsity.** R and B are Bayer-*sparse* (¼ the sites of the full grid,
   half of green) and noisier; per-channel LK on the sparse, noisy R/B planes is
   far less reliable than LK on the all-photons grey. NOTES.md already found that
   per-pixel LK on noisy raw injects jitter and is net-negative; splitting the
   signal three ways makes the conditioning strictly worse. You'd trade a robust
   single flow for three shaky ones.
2. **Lateral CA is a smooth, static, low-order field** — a global radial
   red↔green / blue↔green warp (a handful of parameters), NOT a per-tile
   per-frame optical-flow residual. Estimating it through the burst-alignment flow
   is a noisy, high-variance way to recover a smooth deterministic function. The
   right correction is a **calibrated radial warp** (or the learned defringe we
   already ship: `postprocessing.aberration.do_defringe`), applied once, not 3×
   alignment.
3. **Architecture.** The merge is per-channel already; a per-channel *flow* would
   need three flow fields threaded through robustness + merge, multiplying
   cost/complexity for a correction better done as a static warp.

**Verdict: do NOT do per-channel registration to fight CA.** It can in principle
correct lateral CA, but at a bad SNR/complexity trade. If lateral CA is the goal,
correct it with a **calibrated per-camera radial R/B-to-G warp on the raw**
(cheap, robust, deterministic) — or lean on the existing learned defringe. Keep a
single grey flow for motion. (A defensible *hybrid*, if ever wanted: estimate the
smooth global CA warp once from the reference, apply it; still align motion on
grey.)

---

## Q3 — The RGGB two-green opportunity (THE KEY ONE)

### (a) Do we average the two greens, or use them as distinct sub-pixel samples?

**Both — in different places, and the place that matters is already correct.**
- **Merge / SR reconstruction: DISTINCT.** Verified (Probes 3 & 6). The merge
  consumes the raw mosaic and accumulates Gr and Gb as two separate green scatter
  points at their true `(1,1)`-offset positions. The built-in sub-pixel green
  offset is **already exploited** for the reconstruction, at scale 1 and scale 2.
- **Alignment grey + robustness guide: AVERAGED** (Probes 1 & 2). These are
  *proxies*, not the reconstruction; averaging the greens (and, for decimate, R/B)
  there is the **correct** design (max-SNR luminance for motion/agreement
  estimation), and matches Wronski/HDR+.

So the premise "are we losing the green offset?" — **no, not where it counts.** The
merge is the kernel-regression scatter that Wronski designed precisely to use every
raw sample, both greens included.

### (b) Could treating Gr/Gb as two offset samples add real sampling diversity?

It *does*, and we already get it. The mechanism (Probe 4): in RGGB, Gr is at
`(0,1)`, Gb at `(1,0)` (offset `(+1,−1)` px in the raw grid). The two green
sub-lattices are each pitch-2 and interleaved → a **quincunx** green lattice with
**2× the green sample density** of either plane alone, filling the diagonal
frequencies a single green plane would alias. **Probe 3** shows 4–5 real green
samples land in each output pixel's 3×3 window (vs 1–2 for R/B). For SR this is
real HF from real samples — no hallucination — and it is the honest lever the
diagnosis calls for.

**The catch (literature, important caveat): Gr ≠ Gb radiometrically.**
"Green-green imbalance" / Gr-Gb crosstalk: the two green sites have slightly
different sensitivity (microlens/optical-path/angle differences), and
edge-sensitive processing amplifies the difference into **maze/labyrinth
artifacts**. ISPs ship Green Imbalance Correction for this. Our merge's Gaussian
RBF weighting is *soft* (not edge-sensitive demosaic), so it is relatively
forgiving, but treating the greens as identical-radiometry samples is only valid up
to the imbalance. Any "use the greens harder" change must respect this (a per-burst
Gr/Gb gain match is cheap insurance).

### (c) Does the dual-green offset matter for our SAMPLING-LIMITED HF regime?

**It matters, but it is ALREADY CAPTURED — so it is not an untapped lever.** The
verified diagnosis: edges are optics-limited (deconvolution territory), HF is
sampling-limited, and "more real samples" is the one honest HF lever. The two
greens literally *are* more real green samples — and the merge already uses them.
NOTES.md's "sub-pixel sampling diversity ≈ ideal" finding is *measured on the
current pipeline, which already includes both greens*; the dual-green contribution
is baked into that result, not additional to it.

Where dual-green's contribution is largest is the **low-diversity regime**: N=1
(single-frame fallback), or heavily robustness-rejected / occluded regions, or
static foreground where inter-frame motion is tiny. There, the burst supplies
little inter-frame phase diversity, and the sensor's built-in quincunx green
sampling is the *only* sub-pixel green diversity available — and we already exploit
it. With many well-shifted frames (burst 0003, N=14), the inter-frame offsets
dominate and the *marginal* diagnostic value of the green offset shrinks toward
redundancy; its durable benefit there is the **2× green sample count → ~√2 lower
green-channel variance (SNR)**, not new frequencies.

**Bottom line for Q3: the two-green idea is a GENUINE real-HF lever — and the
pipeline already pulls it (in the merge). It is captured, not missing.** There is
no "average the greens" bug to fix in the reconstruction. The only place greens are
averaged is the alignment/robustness proxies, where averaging is correct.

---

## Which is worth implementing?

Ranked, honestly:

1. **Nothing structural on the two greens for the *merge* — it's already optimal.**
   Do NOT "split the greens into two planes" or stop averaging in the grey/guide:
   the merge already uses both greens distinctly, and averaging in the proxies is
   the right SNR choice. Implementing a "dual-green" feature would re-derive
   behavior we already have. *This is the main finding: flag dual-green as
   ALREADY-CAPTURED, not a new lever.*

2. **Low-cost, defensible: per-burst Gr/Gb green-imbalance match** before the merge
   (a single scalar gain equalizing mean Gr vs Gb, or a flat-field calibration).
   Cheap insurance against maze artifacts from treating the two greens as one
   channel; only helps if this sensor shows measurable Gr-Gb imbalance (measure
   first — see sketch). Small, bounded, no downside if imbalance ≈ 0.

3. **Keep grey registration; do NOT go per-channel for CA.** If lateral CA
   (ca_R≈1.35 px) is worth removing, add a **calibrated radial R/B→G warp on the
   raw** (or use the shipped learned defringe), not per-channel flow. This is the
   only "registration/color" change with a real payoff, and it's a static warp, not
   an alignment change.

4. (Consistent with NOTES.md, not a Q1–3 item, listed for completeness) the
   verified HF levers remain **measured-PSF deconvolution** (edges) and **tighter
   robustness rejection** (the unalignable tail) — neither is a registration-color
   change.

---

## Concrete sketch for the worthwhile one(s)

**A. Gr/Gb imbalance check + optional equalization (do this first — it's a
measurement).**
- *Measure* on this a6300 raw: over flat-ish regions, `mean(Gr)` vs `mean(Gb)`
  (per ISO). If the ratio is within ~0.1–0.3 % of 1.0, imbalance is negligible →
  do nothing. If not, it's real fixed-pattern.
- *Equalize* (if needed): before `merge`, multiply Gb sites by `mean(Gr)/mean(Gb)`
  (or a low-order flat field) so the two green sub-lattices are radiometrically one
  channel. One elementwise pass on the raw, indexable by `cfa==1 & row-parity`.
  Guard with a config flag; default off until measured.
- *Validate*: render a flat/low-texture patch; FFT for a Nyquist
  checkerboard/maze peak before vs after (the green-imbalance signature). Follow
  eval discipline: median+spread over patches, shared-stretch viz.

**B. Calibrated lateral-CA warp (only if CA removal is a goal).**
- Calibrate once per lens: from a high-contrast reference, fit a global radial
  scale (or low-order polynomial) mapping R and B to G that minimizes color
  fringing (you already measured ca_R≈1.35 px — that anchors the radial term).
- Apply as a `map_coordinates`/`affine` resample on the R and B Bayer sub-planes
  (or post-merge per-channel) *before* the merge accumulates them; leave motion
  alignment on grey untouched.
- Validate against the DxO defringe reference already in `eval/NOTES.md`
  (per-axis), not by eye.

Both are small, bounded, opt-in, and respect the verification policy (measure the
distribution, gate on reliability, red-team favorable numbers).

---

## Sources

- Wronski et al., *Handheld Multi-Frame Super-Resolution*, SIGGRAPH 2019 —
  [ar5iv/arXiv 1905.03277](https://ar5iv.labs.arxiv.org/html/1905.03277),
  [PDF](https://3dvar.com/Wronski2019Handheld.pdf),
  [ACM TOG](https://dl.acm.org/doi/10.1145/3306346.3323024). Grey-from-quad
  alignment; per-channel kernel-regression accumulation of every raw sample.
- *An Analysis and Implementation of the HDR+ Burst Pipeline*, IPOL 2021 —
  [PDF](https://www.ipol.im/pub/art/2021/336/article_lr.pdf). 2×2 Bayer→grey,
  integer-2px alignment "replacing Bayer samples with overlapping colors",
  4-level pyramid (factors 2,4,4).
- Lateral chromatic aberration correction (align R,B→G): cross-channel prior in
  shearlet domain —
  [Springer](https://link.springer.com/chapter/10.1007/978-3-030-69532-3_7);
  *Fast Two-step Blind Optical Aberration Correction* —
  [arXiv 2208.00950](https://arxiv.org/pdf/2208.00950);
  *Correcting chromatic aberrations using image warping* —
  [ResearchGate](https://www.researchgate.net/publication/3513141_Correcting_chromatic_aberrations_using_image_warping).
- Two green channels (Gr/Gb) sampling rate + SNR, distinct positions:
  *Green Channel Guiding Denoising on Bayer Image* —
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3967728/);
  *Deep Residual Network for Joint Demosaicing and Super-Resolution* —
  [arXiv 1802.06573](https://arxiv.org/pdf/1802.06573).
- Gr-Gb green imbalance / maze artifact (the dual-green caveat):
  [DxO "What is maze?"](https://support.dxo.com/hc/en-us/articles/360016036611-What-is-maze-);
  *Gr/Gb difference in a 3M CMOS sensor* —
  [imagesensors.org PDF](https://imagesensors.org/Past%20Workshops/2007%20Workshop/2007%20Papers/070%20Kim%20et%20al.pdf);
  green-imbalance compensation patent —
  [USPTO 7830428](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7830428).

Probes (run on GB10, then deleted): decimate-grey averages all 4 (incl. both
greens); guide averages greens; **merge keeps Gr/Gb distinct per-channel at scale
1 and 2 (no pre-merge demosaic)**; Gr/Gb offset `(+1,−1)` px → 2× quincunx green
density, 4–5 green samples per output 3×3 window vs 1–2 for R/B.
