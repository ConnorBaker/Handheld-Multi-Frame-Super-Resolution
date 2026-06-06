# Research: should robustness/rejection be PER-PIXEL / adaptive, not a scene-global threshold?

Grounded in `feat/warp-port`. Trigger: the verified finding (`eval/NOTES.md`
"REJECTION ... INERT") that bumping `robustness.tuning.t` (a scene-global scalar)
was bit-identical on well-aligned scenes. The user's question: *shouldn't
rejection be per-pixel?* Short answer: **it already is per-pixel and per-frame —
`t` is only a global offset on an already-spatially-varying mask. The honest
"adaptive" win is not a per-pixel `t`; it is a SEPARATE per-pixel SNR/HDR weight
(the literature's `hn`), which is largely redundant on our single-exposure
bursts.** Details, code-grounded and probe-verified, below.

All code claims were read on this machine; the cancellation and per-pixel
behaviors were re-derived with a numeric probe (`eval/_probe_robustness_perpixel.py`).

---

## 1. What is per-pixel vs global in OUR robustness (read from source)

The mask is built in `robustness.compute_robustness` → `cupy_ops.py`, then enters
the merge in `warp_ops.py`. The math is the IPOL/Wronski Algorithm 6,
**Eq. (per pixel):**

```
R(x,y) = clip( S(x,y) * exp( -d²(x,y) / σ²(x,y) ) - t , 0, 1 )      # robustness_threshold_cupy
r(x,y) = min_{5x5} R                                                 # local_min_cupy
```

Granularity of each term, from the code:

| quantity | where | per-pixel? | notes |
|---|---|---|---|
| `d²` color distance ref-vs-warped-comp local means | `compute_dist_cupy`, `apply_noise_model_cupy` | **per pixel, per frame** | 3×3 local means of the guide image; Wiener-shrunk by the noise model |
| `σ²` noise variance | `apply_noise_model_cupy` | **per pixel** | `max(local_var, σ_t²)`, where `σ_t` is **brightness-indexed** (`std_curve[round(1000*mean)]`) → SNR-adaptive *per pixel* already |
| `S` flow-discontinuity scale | `compute_s_cupy` | **per TILE**, switched per pixel via `M` | `S = s1 if local flow span M > Mt else s2` (`s1=2, s2=12`); upsampled tile→pixel in `robustness_threshold_cupy` |
| 5×5 local-min refinement | `local_min_cupy` | **per pixel** | pessimistic pooling (Alg. 9) |
| **`t`** offset | config scalar | **GLOBAL** | one number for the whole burst |
| `s1, s2, Mt` | config scalars | **GLOBAL constants** | but `S` *selection* is per-pixel via `M` |
| noise-model `std_curve/diff_curve` | per-burst (ISO) | global curve, **per-pixel lookup** | the curve is global; the index `µ_p` is per pixel |

Then the merge applies `r` **per output pixel, per frame** as a *multiplicative*
weight on every sample (`warp_ops.py::_accumulate_warp`):

```
local_r = r[i_r, j_r]               # per-output-pixel robustness of THIS frame
wv = w * local_r * c ;  wa = w * local_r
num[i,j,ch] += wv ;  den[i,j,ch] += wa
```

with `num/den` accumulated across frames and finally `divide(num,den)`.

**So: every input to `R` except `t/s1/s2/Mt` is already spatially varying, and the
mask gates the merge per pixel per frame. `t` is a uniform downward bias on an
already-per-pixel confidence field.** (This matches the canonical sources verbatim;
§4.)

This is the same conclusion `research_spatially_varying.md` §0/§4 reaches —
"Robustness ← already the most per-pixel stage (per-pixel noise model + local
flow-irregularity)". This doc adds the *threshold-specific* analysis the user asked
for.

---

## 2. Why a global `t` is inert on aligned scenes — and exactly when it bites

`t` is a constant subtracted from a quantity that is then **the merge weight**.
Per output pixel the merged value is a weight-normalized average:

```
out = Σ_n r_n · v_n / Σ_n r_n           (r_n = clip(base_n - t, 0, 1))
```

If every frame agrees at a pixel (well aligned, similar `d`, same `S`), then all
`base_n` are ~equal, so all `r_n` shift together when `t` changes — and a common
factor in both numerator and denominator **cancels exactly**. `t` only changes the
output when frames *differ* in their `base_n` (mixed aligned + misaligned), because
then `t` re-weights them relative to each other, and the clip-at-0 hard-rejects the
ones that fall below `t`.

**Probe (`eval/_probe_robustness_perpixel.py`, run on the GB10):**

```
=== Probe 1: does t cancel? ===
  aligned  t=0.12: R(all)=0.7787  merged=0.500441
  aligned  t=0.43: R(all)=0.4687  merged=0.500441     <- BIT-IDENTICAL: t cancels
  margin   t=0.12: R(aln)=0.7787 R(ghost)=0.0285 merged=0.495839  (err vs truth 0.005660)
  margin   t=0.43: R(aln)=0.4687 R(ghost)=0.0000 merged=0.490179  (err vs truth 0.000000)
```

- *aligned*: all frames equal R → output independent of `t` (0.500441 either way).
  This reproduces the `eval/NOTES.md` "INERT on both the 007 facade and the park".
- *margin*: 8 aligned + 4 partially-misaligned (ghost) frames. At `t=0.12` the
  ghosts leak in (error 0.0057 from the aligned-only truth); at `t=0.43` they are
  clipped to exactly 0 and the output is the clean truth. **This is the entire job
  of `t`: set where on the per-pixel `R` curve the hard cutoff falls for the frames
  that disagree.**

So the right mental model is **not** "global vs per-pixel rejection" — the
rejection field is per-pixel. `t` is a *global decision boundary* on that field,
and it is provably inert wherever frames don't disagree.

---

## 3. Is the mask already spatially discriminating? (yes)

**Probe 2** drives `robustness_threshold_cupy` with a half-flat/half-misaligned `d²`
field and one flow-irregular tile:

```
=== Probe 2: does the mask vary per pixel? ===
  t=0.12: R min=0.0000 max=1.0000 mean=0.4946 std(spatial)=0.4961 frac rejected=0.500
          left-half (aligned) mean=0.9892   right-half (misaligned) mean=0.0000
  t=0.43: R min=0.0000 max=1.0000 mean=0.4770 std(spatial)=0.4857 frac rejected=0.500
          left-half (aligned) mean=0.9539   right-half (misaligned) mean=0.0000
```

The mask spans the full [0,1] range within one frame (spatial std ~0.49), keeping
aligned content at ~1 and hard-rejecting misaligned content at 0 **via `d²`/`S`
alone**. Raising `t` only erodes the aligned region slightly (0.989→0.954) — it
does **not** add new discrimination; the discrimination is already in `d²`.

**Caveat from `eval/NOTES.md` (item 3, the real-data measurement):** despite this,
on real foliage the *accumulated* mask sits at ~12.6/13 everywhere and an injected
independently-moving region is still accepted **4.6/7 frames** (~66% weight). The
failure is not "the threshold is global" — it is that **`d` (a 3×3 local-mean color
distance) is a weak misalignment detector when moving content shares the reference's
color statistics**. Wind-blown foliage has the *same* local mean as the background
it sweeps across, so `d` stays small and `R` stays high regardless of `t`. That is a
*signal* problem (what `d` measures), not a *threshold-granularity* problem.

---

## 4. Literature: the model is per-pixel by design; the "adaptive" axis is a separate SNR weight

**Wronski et al., SIGGRAPH 2019 (the paper this repo implements).** The robustness
model is introduced as: *"we assign confidence to the local neighborhood of every
pixel that we consider merging"* and the model *"computes a per-pixel weight for
every frame"*. Eq. (5):

> `R = s · exp(−d²/σ²) − t`, where *"s and t are tuned scale and threshold
> parameters used to guarantee that small differences get a weight of one, while
> large differences get fully rejected."*

i.e. `s` and `t` are **explicitly tuned global scalars by design**; `d` and `σ` are
per-pixel and noise-corrected (Eq. 6, the Wiener shrinkage + `σ=max(σ_ms,σ_md)`);
`s` is switched per pixel between `s1/s2` by the local motion extent `M` vs `Mth`
(Eqs. 7–8); a 5×5 morphological min finishes (Eq. 9). **Our `cupy_ops.py` is a
line-for-line match.** Nothing in the original suggests a per-pixel `t`; the spatial
adaptivity lives in `d`, `σ`, and the `s1/s2` switch.

**Lafenetre, Facciolo, Eboli — IPOL 2023 "Implementing Handheld Burst
Super-Resolution"** (the companion to *this* codebase). Algorithm 6, line 31:

> `R(x,y) ← min(max(s(p)·exp(−d²/σ²) − t, 0), 1)`

and the text: *"`t` ... is fixed after calibration to **0.12**, whereas [`s`] is a
**patch-wise** quantity ... `s1=2` or `s2=12`"*; *"this stage yields robustness maps
`r_n` of size H×W, like the LR frames."* So the reference spec is unambiguous: the
mask is **per-pixel/per-frame, `t` is a single calibrated constant**, and `s` is the
only spatially-switched tuning term (and only between two values).

**Lafenetre et al., "Handheld Burst SR Meets Multi-Exposure Satellite Imagery"
(2023)** — the most relevant work on making the merge weights more adaptive. They
*keep* the same per-pixel robustness `r_n` (with **the same `t=0.12`, `s2=12`**) and
note *"all the quantities are pixelwise"*. Their contribution is a **third,
separate per-pixel weight**: the merge kernel is

> `k_n(p,q) = w_n(p,q) · r_n(p,q) · h_n(p,q)`   (their Eq. 7)

where `w_n` = geometric (the steerable cov, our `covs`), `r_n` = robustness (ours),
and `h_n` = an **HDR weight ... based on local estimates of the SNR**, which *"gives
more importance to the well-exposed frames and filters out ... the darker regions
with the lowest SNRs."* Crucially they distinguish it from robustness: *"it relies
only on a single frame, whereas the robustness is defined for image pairs via `d`."*
They also explicitly flag that conditioning robustness on the noise profile
*"would further improve the quality of the robustness"* — i.e. the field's answer to
"make rejection SNR-adaptive" is a **separate per-pixel SNR weight `h_n`**, not a
spatially-varying `t`.

**HDR+ (Hasinoff et al., SIGGRAPH Asia 2016)** — the lineage: per-tile Wiener-style
shrinkage merge, `A_z = |D_z|²/(|D_z|²+cσ²)`, a **per-frequency, per-tile** blend
between the reference and the aligned tile driven by the local noise variance σ.
Again: per-location, noise-driven, with a *global* tuning constant `c` ("strength").
Same architecture: spatially-varying confidence, globally-tuned aggressiveness.

**Takeaway from the literature:** every member of this family makes the *confidence*
per-pixel and the *aggressiveness/strength* a small set of **global** tuned scalars.
Nobody makes the threshold itself a per-pixel field. The one adaptive extension that
exists (satellite `h_n`) is an **additional** per-pixel SNR weight multiplied beside
robustness, not a modification of `t`.

---

## 5. Verdict: is a per-pixel / adaptive threshold a real win or redundant?

**Per-pixel `t` (or per-region/SNR-scaled `t`): mostly REDUNDANT with the existing
per-pixel mask.** The reasons, in order of how decisive they are:

1. **The threshold operates on an already-per-pixel signal.** `d²` and the `s1/s2`
   switch already make `R` reject hard on edges/motion and accept on flats *within a
   single frame* (Probe 2). Pushing `t` around per-region just nudges the cutoff on a
   field that already encodes the local content. There is no regime where a flat and
   an edge with the *same* `d`/`σ`/`S` should get different cutoffs — `t`'s units are
   "standard deviations of disagreement", which is already content-normalized by `σ`.
2. **The classic per-pixel knobs you'd want a per-pixel `t` for already exist.**
   "Reject harder on edges/high-detail" = `s1` (small s on flow-irregular tiles) +
   the `d²/σ²` ratio (σ is small on clean flats, large on noisy ones). "Gentler on
   flats" = the noise-model floor `σ=max(σ_ms,σ_md)` that lifts confidence in flat
   noisy regions (Wronski Fig. 11, the explicit purpose of `σ_md`). A per-region `t`
   would partly *duplicate* and partly *fight* these.
3. **On aligned content a threshold of any spatial granularity is inert** (§2): if
   frames agree, no `t(x,y)` changes the normalized output. A per-pixel `t` only
   matters where frames disagree — exactly where `d²` is already large and the mask
   is already doing the work.

**Where adaptivity is NOT redundant (the honest opportunities), ranked:**

- **(A) The signal `d` is the weak link, not the threshold.** The verified foliage
  failure (accept 4.6/7 of moving content) is `d` failing to *see* misalignment when
  colors match — no `t` policy fixes a detector that reads ~0. Real levers:
  multi-scale / larger-neighborhood `d`, a structure-aware distance (penalize
  gradient/structure mismatch, not just local-mean color), or feeding the
  flow-residual into `d`. This is a *robustness-signal* upgrade, the actual recovery
  of the misaligned-foliage tail. (Consistent with `eval/NOTES.md` item 3 calling
  `d` "weak when moving content shares color statistics".)
- **(B) A separate per-pixel SNR/HDR weight `h_n` (satellite paper).** Multiply an
  `h_n(x,y)` based on local SNR alongside `r_n` in the merge. This is the literature's
  actual "adaptive rejection" and is *additive* to robustness, not a per-pixel `t`.
  **But on our single-exposure bursts it is largely redundant**: with one exposure,
  `h_n` is ~constant across frames at a pixel (same sensor/exposure), so like `t` it
  cancels in num/den except where brightness/saturation differs frame-to-frame. Its
  payoff is real only for **multi-exposure** bursts (HDR) — which we don't shoot.
  Low priority unless ME is on the roadmap.
- **(C) Tie robustness aggressiveness to the per-region SNR map that
  `research_spatially_varying.md` §1(d)/§2 already proposes.** If a per-region SNR
  map gets built for tile-size/kernel-tuning anyway, scaling `s` (or `Mt`) by it is
  nearly free. But note: `σ` *already* carries local brightness/SNR into `R` per
  pixel, so this is a second-order refinement of the `s1/s2` switch, not a new
  capability. Marginal.

**Concrete sketch (if pursued — A is the only one worth real effort):**
augment `compute_dist_cupy`/`apply_noise_model_cupy` so `d` includes a
*structure* term, e.g.
`d²_eff = d²_mean + λ · ‖∇µ_ref − ∇µ_warp‖²` (gradient-of-local-mean mismatch),
computed per pixel from the guide image — gradients mismatch under sub-pixel
misalignment even when local means agree, so this gives `R` something to reject in
foliage. Keep `t/s1/s2/Mt` global. Validate on the
moving-region synthetic in `eval/diagnostics.py` (target: drop the 4.6/7 acceptance
of injected motion without lowering the 6.2/7 on static content) before touching
defaults. **This is a per-pixel SIGNAL change, not a per-pixel THRESHOLD change —
which is the precise reframing the inert-`t` finding points to.**

---

## TL;DR

- Our robustness mask is **already per-pixel and per-frame**: `d²`, `σ²` (brightness/
  SNR-adaptive), the `s1/s2` flow-discontinuity switch, and a 5×5 min are all
  spatially varying; the mask multiplies every merge sample per output pixel per
  frame. **Only `t, s1, s2, Mt` (and the noise *curve*) are global scalars.**
- `t` is a **global offset** on that per-pixel field. It provably **cancels** on
  aligned content (probe: bit-identical output) and only bites where frames *disagree*
  (probe: clips ghosts) — so "global `t` is inert" is a property of *thresholding an
  agreement signal*, not of granularity.
- A **per-pixel/per-region `t` is largely redundant**: the per-pixel `d²/σ²` and
  `s1/s2` already do "reject hard on edges/motion, gentle on flats"; `t`'s units are
  already σ-normalized.
- The **literature agrees**: Wronski/IPOL define `s,t` as tuned **global** scalars
  over a per-pixel mask; HDR+ uses a global strength over a per-tile noise blend; the
  satellite follow-up's "adaptive" extension is a **separate per-pixel SNR weight
  `h_n`**, not a spatial `t`.
- The **real** adaptive win is upgrading the **robustness signal `d`** (structure-/
  multi-scale-aware) so it *detects* the misaligned-foliage tail it currently misses
  (`d` reads ~0 when moving content shares color) — a per-pixel *signal* change. A
  per-pixel SNR weight `h_n` helps only for multi-exposure bursts (not ours).

## Sources

- Wronski et al., *Handheld Multi-Frame Super-Resolution*, ACM TOG 38(4) / SIGGRAPH
  2019 — https://3dvar.com/Wronski2019Handheld.pdf (robustness §5.2, Eqs. 5–9;
  "per-pixel weight for every frame").
- Lafenetre, Facciolo, Eboli, *Implementing Handheld Burst Super-Resolution*, IPOL
  2023 — https://www.ipol.im/pub/art/2023/460/article_lr.pdf (Algorithm 6, line 31:
  `R = clip(s·exp(−d²/σ²)−t,0,1)`; `t=0.12` calibrated global, `s1=2/s2=12` patch-wise).
- Lafenetre, Nguyen, et al., *Handheld Burst SR Meets Multi-Exposure Satellite
  Imagery*, 2023 — https://arxiv.org/pdf/2303.05879 (merge weight `k=w·r·h`, Eq. 7;
  per-pixel HDR/SNR weight `h_n`; "all the quantities are pixelwise").
- Hasinoff et al., *Burst photography for HDR and low-light imaging on mobile
  cameras*, SIGGRAPH Asia 2016 — https://research.google/pubs/pub45586/ (per-tile
  noise-driven Wiener merge with a global strength constant).
- Reference Python impl. README — https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution
  (config `robustness.tuning: t=0.12, s1=2, s2=12, Mt=0.8`).

## Reproduce

`PYTHONPATH=$(pwd) micromamba run -n handheld python eval/_probe_robustness_perpixel.py`
(cancellation on aligned content; `t` bites at the margin; mask spans [0,1] per
frame via `d²`/`S`).
