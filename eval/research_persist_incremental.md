# Persisting a continuous-representation artifact for arbitrary-scale re-render, and incrementalizing the merge

Deep-research note (companion to `eval/research_continuous_formulation.md` and
`eval/research_wavelets_continuous.md`). The user's two questions:

1. **Arbitrary-scale artifact.** Can we *persist* something from the merge so a new
   output scale needs only a cheap re-render — not a re-align / re-estimate / full
   pipeline run?
2. **Incrementalization.** The merge accumulates `num`/`den` *additively* across
   frames. So is adding a frame a `+=` (O(1) extra frame), removing a `-=`, and can
   we keep a running merge that streams frames?

**Standing context this builds on (already established, not re-litigated):** the
steering-kernel-regression merge in this repo *is*, mathematically, a queryable
continuous representation — its per-output-pixel accumulation `w·r·c → num`,
`w·r → den`, `w = exp(−½ dᵀΩ⁻¹d)`, divide `num/den`, is verbatim Wronski Eq. 1–2 =
Takeda zeroth-order steering KR = the 2D-Gaussian-splat render `C(p)=Σ G_i(p)` of
GaussianSR/GSASR (`research_continuous_formulation.md` §0–1; `research_wavelets_continuous.md`
A1/B). Both prior notes already flagged the cheap structural win: **`scale` is baked
into the output grid; decouple the query grid from `scale` and you re-render at any
scale with no retrain.** This note turns that observation into a concrete persistable
artifact and a streaming-merge design, and **verifies the two load-bearing
properties by execution on the GB10**, not from the source.

---

## What was executed on this machine vs. cited

Per repo `CLAUDE.md` §1 (trust no finding you did not run):

- **Executed (GB10, sm_121, live torch+cupy+Warp path)** — `/tmp/verify_persist_incremental.py`,
  ALL CHECKS PASSED:
  - **(A) Scale-invariance is bit-exact.** Computed the per-frame flow, the per-pixel
    covariance field `Ω`, and the robustness mask at `scale ∈ {1,2,3}` for the same
    frames. `max|diff| = 0.000e+00` for **all three fields across all scales** — they
    do not depend on `scale` at all. Shapes (512×768 raw): flow `(32,48,2)` (tile
    grid), `Ω` `(256,384,2,2)` (grey/quad res = H/2×W/2), robustness `(512,768)` (LR res).
  - **(B) The comp-frame merge is exactly additive.** Streaming frame A then B into
    one `num`/`den` pair == merging A and B into *separate* pairs and summing them:
    `max|diff| = 0.000e+00` (num and den). Removing a frame by subtracting its own
    contribution reproduces the pre-add state to **1.19e-07** (float32 add-order
    round-off, not algebra).
  - **(C) The reference-frame merge is additive too — when the accumulated-robustness
    denoiser is OFF (the default).** `(A+B)+ref == (A+B)+(ref alone)`, `max|diff| =
    0.000e+00`. With that denoiser ON the ref kernel takes an `overwrite` branch
    (gated on `robustness_denoise != 0`) that *replaces* `num`/`den` instead of adding
    — see the caveat in §2.
- **Read directly (repo source):** `super_resolution.py` `main` (the align → robustness
  → covariance → merge → divide order), `merge.py` + `warp_ops.py` (the accumulation),
  `kernels.py` + `cupy_ops.py` `estimate_cov_cupy` (the covariance field), `robustness.py`
  + the robustness ops, `alignment.py` + `ICA.py` (the flow), `params.py` (what depends
  on SNR vs scale). Findings below trace to these.
- **Cited (not executed):** the ASSR / continuous-rep / online-SR literature in §3.

---

## 0. The pipeline, split into scale-dependent vs. reusable

`main` runs, per comparison frame `n`: build grey pyramids → **align** (block match +
ICA) → **robustness** mask `r_n` → **covariance** `Ω_n` → **merge** (accumulate into
`num`/`den`). Then merge the reference, then `divide(num, den)`.

Reading the source, **`config.scale` enters in exactly two places** (confirmed by a
grep over the whole package; the only other `scale` hits are the *dead* frame-count
denoisers in `utils_image.py`, which are postprocessing, default-off):

1. `super_resolution.py`: `output_size = (round(scale*H), round(scale*W))`, and `num`,
   `den` are allocated at that size.
2. `warp_ops.py`: the two merge kernels map each output pixel `(i,j)` back to LR coords
   — comp path `lr_x=(j+0.5)/scale`, `lr_y=(i+0.5)/scale`; ref path `crp_x=j/scale`,
   `crp_y=i/scale` — then gather a 3×3 LR neighbourhood with anisotropic weights.

Everything else is computed in **LR / grey coordinates and is independent of `scale`**:

| Field | Where | Shape | Coord frame | Scale-dependent? |
|---|---|---|---|---|
| Per-frame **flow** `V_n` | `alignment.py`/`ICA.py` | `(n_tiles_y, n_tiles_x, 2)` | LR px, tile grid | **No** (verified 0.0e+00) |
| Per-pixel **covariance** `Ω_n` | `kernels.py`/`estimate_cov_cupy` | `(H/2, W/2, 2, 2)` | grey/quad grid | **No** (verified 0.0e+00) |
| **Robustness** `r_n` | `robustness.py` | `(H, W)` | LR (quad-center) | **No** (verified 0.0e+00) |
| Ref **local means/stds** | `init_robustness` | `(c, H, W)` | LR | **No** (no `scale` arg) |
| **`num` / `den`** | merge | `(sH, sW, 3)` | OUTPUT grid | **YES — the only scale-specific product** |

`estimate_cov_cupy` takes **no `scale` argument at all** — `Ω` is a pure function of the
grey-grid gradients (structure tensor → eigen-decomposition → `k_detail/k_denoise/...`).
The robustness ops likewise take no `scale`. The flow is estimated on grey pyramids
built from the LR image. So **`Ω`, `r`, and `V` are a per-frame, scale-free, "compute
once" set**; the merge is the *only* stage that knows the output scale, and it is a
bounded per-output-pixel gather.

> Caveat — **SNR-resolved params** (`params.py update_snr_config`): `tile_size`,
> `k_detail`, `k_denoise`, `D_th`, `D_tr` are resolved from the burst SNR (a property of
> the *frames*, not the scale). They are fixed across scales for a given burst, so they
> are part of the reusable set — but the artifact must record them (they shape `Ω` and
> the tiling), because re-running at another scale must use the *same* resolved values.

---

## 1. The minimal persistable artifact for arbitrary-scale re-render

### The right artifact is the *scattered samples + the kernel field*, NOT `num`/`den`

`num`/`den` are written at the output grid for one specific `scale` — they are the
*render output*, not the representation. Persisting them locks you to that scale (and is
the larger object: `s² · H·W · 3 · 2 · 4` bytes). The **representation** is what the
render is a function of:

```
ARTIFACT  =  for the reference frame:  raw bayer J_1, Ω_1, CFA
             for each comp frame n>1:  raw bayer J_n, flow V_n, Ω_n, robustness r_n
             + the resolved config (SNR-derived k_detail/k_denoise/D_th/D_tr/tile_size,
               kernel='steerable'|'iso', cfa_pattern, white_balance, noise model)
```

This is exactly **"aligned scattered samples + per-pixel `Ω` + robustness"** — the warped
LR pixel centers are the scattered samples `{x_{n,i}}`, `J_n` carries their values
`c_{n,i}`, `Ω_n` is the per-location steering bandwidth, `r_n` the per-sample certainty.
It is the 2D-Gaussian *field*; the merge gather is its rasterizer. (This is structurally
the GSASR/ContinuousSR split: a per-image representation stage + a cheap scale-aware
render stage — except our representation is *measured*, not learned, so it cannot
hallucinate; `research_wavelets_continuous.md` A1/B.)

### The cheap re-render path

Given the artifact, a new scale `s'` is:

```
num = zeros((round(s'·H), round(s'·W), 3));  den = like(num)
for n>1:  merge(J_n, V_n, Ω_n, r_n, num, den, cfa, cfg(scale=s'))   # warp_ops kernel
merge_ref(J_1, Ω_1, num, den, cfa, cfg(scale=s'))
divide(num, den)                                                     # + optional postproc
```

i.e. **only the two Warp gather kernels + the divide re-run.** No grey pyramids, no
block matching, no ICA, no structure-tensor / covariance, no robustness, no noise-curve
Monte-Carlo. The merge already accepts `scale` as a float and divides `(j+0.5)/scale`,
so **`scale` need not even be an integer** — fractional/arbitrary output grids work
today with zero new math (and an arbitrary *point set*, not just a grid, would work by
generalizing the kernel's `(i,j)→lr` map — the Takeda "query at any x" property).

### Sizes (measured) and cost

Per the verified shapes, float32, at 512×768 (scales with image area; a 12 MP raw is
~30× larger but the *ratios* hold):

| Item | Size @512×768 | Note |
|---|---|---|
| flow `V_n` | 12 KiB | trivial (tile grid) |
| `Ω_n` | 1.5 MiB | grey res H/2×W/2 ×4 floats |
| `r_n` | 1.5 MiB | LR res ×1 float |
| raw `J_n` | 1.5 MiB | needed for the gather values |
| **per comp frame** | **~4.5 MiB** | |
| **N=8 burst** | **~36 MiB** | linear in N |

For a 12 MP burst of 8 frames this is on the order of ~1 GB of float32 — fine on the
GB10's 122 GiB unified memory, and reducible: `Ω` is symmetric (store 3 not 4 floats),
the raw is the native bit depth (store uint16, not float32 → ¼), `r` could be fp16. A
practical artifact is **a few hundred MB** for a full-res burst.

**Cost of re-render-only vs. full pipeline.** Align + ICA + robustness + covariance is
the *bulk* of `main`'s per-frame work (multi-level pyramids, FFT cross-correlation, an
iterative LK solve, a Sobel structure tensor, guide-image local stats, the noise-model
Monte-Carlo once per burst). The re-render is **two `wp.launch` gathers per frame + one
elementwise divide** — a small fraction of a full run, and it is the *only* part whose
cost grows with `s²` (output pixels). So: persist once, then sweep scales (e.g. the
`eval/_run_scale2.py` / `_resolution_sweep.py` loops that today re-run the whole pipeline
per `scale`) for the cost of the gather alone. This is the concrete redundancy removed.

> **Why not persist `num`/`den`?** Because the gather distances `d` and the 3×3 support
> are resolved *at the output grid* for a given scale (the kernel recomputes `Ω⁻¹` and
> the per-tap weights per output pixel). The mapping from samples to a coarser grid is
> not a simple downsample of a finer grid's `num`/`den` (the kernel support, the per-pixel
> `Ω` bilinear interpolation, and the 3×3 truncation all change), so a different scale
> genuinely needs a different gather. Hence persist the representation, re-render the grid.

---

## 2. Incrementalization: the merge is additive ⇒ streaming

### `num`/`den` are a running sum — verified

The comp-frame merge does `num[i,j,c] += val; den[i,j,c] += acc` (per `warp_ops.py`
`_accumulate_warp`), and §(B) above **verified that this is exactly additive**: the
merged `num`/`den` of a burst equals the sum of each frame's independent contribution
(`max|diff| = 0.0e+00`). Therefore:

- **Add a frame:** compute *only that frame's* `V_n, Ω_n, r_n` (against the fixed
  reference's stats), then `merge(...)` into the running `num`/`den`. **O(1) in the
  existing frame count** — no other frame is touched. This is a true streaming/online
  update: keep `num`/`den` resident, fold each new frame in as it arrives.
- **Remove a frame:** if you persisted that frame's contribution (or recompute it from
  the artifact), subtract it: `num -= num_n; den -= den_n`. Verified exact to float32
  (`~1e-7` add-order round-off). So a sliding window over a long sequence is `+=` new,
  `-=` old.
- **Re-weight a frame** (e.g. global exposure tweak via a scalar weight `α_n`): because
  the contribution is linear in the sample value and weight, you can subtract the old and
  add the rescaled one without redoing the gather geometry.

The render at any time is just `divide(num, den)` of the current accumulators — a snapshot
of the running average, exactly the classic online-SR "each new LR frame immediately
improves the current HR estimate" (Elad–Feuer adaptive-filtering / Kalman lineage; the
Springer "Online multi-frame SR" weighted-sum update; §3).

### What a new frame must recompute (and what it must not)

Recompute **per new frame** (all against the *fixed reference*): its grey pyramid →
its flow `V_n` (align + ICA vs. the cached reference pyramid) → its robustness `r_n`
(needs the cached reference local means/stds + `V_n`) → its covariance `Ω_n` → the merge.

Reuse **across all frames** (compute once for the reference, never per added frame):
the reference grey pyramid + tiled FFT + gradients + Hessian (`init_alignment`), the
reference local means/stds (`init_robustness`), the noise curves (`run_fast_MC`), the
SNR-resolved params, and the running `num`/`den`.

### Caveats (the three things that complicate naive streaming)

1. **The accumulated-robustness denoiser breaks ref additivity.** With it ON, the ref
   merge takes an `overwrite` branch — at output pixels where the *accumulated* robustness
   (a sum over all comp frames) is below `max_frame_count`, it *replaces* `num`/`den`
   with a single-frame, widened-kernel value instead of adding. Verified: with the
   denoiser OFF (default) the ref merge is additive (0.0e+00); the overwrite is gated on
   `robustness_denoise != 0`. So *streaming requires deferring the ref merge to render
   time* (it already runs last in `main`) and, if this denoiser is enabled, recomputing
   it whenever the frame set changes (it depends on the accumulated robustness, which
   changes as frames are added). The comp-frame accumulation itself is always additive.
2. **The `accumulated_r` mask is itself a running sum** — `main` already does
   `add(accumulated_r, r_n)` per frame, so it streams naturally; it only feeds the ref
   merge / postproc denoiser, which run at render time.
3. **Fixed reference frame.** All flows and robustness are relative to one reference
   (its pyramid + local stats). Streaming is valid as long as the reference is fixed.
   Changing the reference invalidates *every* frame's `V_n` and `r_n` (they are
   ref-relative) and the cached reference stats — that is a full recompute, not an
   increment. (A sliding window that also slides the reference is therefore not O(1);
   keep the reference pinned for cheap streaming, or accept a periodic re-anchor.)
4. **The `divide` is not incremental but is trivially cheap** — it is one elementwise
   op over the output grid, redone at each render snapshot (don't fold it into the
   running state; keep `num`/`den` raw and divide on demand).

### Feasibility verdict: streaming is essentially free

Adding a frame is "compute its three ref-relative fields + two gathers, `+=`." The
running state is just `num`, `den` (and `accumulated_r`). No architectural change is
needed — the additivity is already in the kernels; today's `main` simply happens to run
the loop in one shot and divide at the end. Exposing "init(ref) → add_frame(J_n) →
render(scale)" as an API is a refactor of control flow, not of math.

---

## 3. Literature

### Arbitrary-scale render from a stored continuous representation (the §1 analog)

- **LIIF** (Chen et al., CVPR 2021) — image = a feature map + a shared MLP queried at any
  continuous coordinate; "since the coordinates are continuous, LIIF can be presented in
  an arbitrary resolution." Its local-ensemble combiner (a bilinear-area-weighted,
  normalized sum of 4 neighbour latents) is structurally our `num/den`.
  https://arxiv.org/abs/2012.09161 , project: https://yinboc.github.io/liif/
- **A-LIIF** (Adaptive LIIF, ACCV 2022) — adds local adaptivity to fix LIIF's edge
  ringing/structural distortion (all-pixels-share-one-MLP problem).
  https://arxiv.org/abs/2208.04318
- **GSASR — Generalized & Efficient 2D Gaussian Splatting for Arbitrary-scale SR**
  (2025) — predicts image-conditioned **2D Gaussians**, then a **scale-aware
  GPU/CUDA rasterization** "render[s] super-resolved images by sampling discrete RGB
  values from the predicted continuous Gaussians" — an explicit
  *representation-stage + cheap-per-scale-render-stage* split, "for any image and unseen
  scaling factors" without retraining. This is the precise architecture of the §1
  artifact (ours: measured samples + `Ω`, not learned Gaussians).
  https://arxiv.org/abs/2501.06838
- **ContinuousSR / GaussianSR** — render an explicit anisotropic-2D-Gaussian field at
  arbitrary coordinates; renderer `C(p)=Σ c_i·N(p;μ_i,Σ_i)` is verbatim steering-KR =
  our merge (see `research_wavelets_continuous.md` A1/B for the equation-level match).
- **Progressively Deformable 2D GS for video at arbitrary resolution** (2025) — persists
  canonical 2D-Gaussian primitives per GoP and renders any scale from the fixed set
  ("a fixed set of splats can be rendered at arbitrary resolutions without retraining").
  https://arxiv.org/abs/2503.05600
- **Unser, *Sampling—50 years after Shannon*** (Proc. IEEE 2000) — the *linear,
  hallucination-free* continuous representation: reconstruct in a shift-invariant
  (B-spline) space `f(x)=Σ c_k φ(x/T−k)`, fit the coefficient field once, resample at any
  scale/coordinate. The deterministic alternative to INRs for the resample substrate.
  https://bigwww.epfl.ch/publications/unser0001.pdf

### Continuous-Fourier, multi-frame, arbitrary-scale

- **BurstM: Deep Burst Multi-scale SR using Fourier Space with Optical Flow**
  (ECCV 2024) — the *multi-frame* realization of the user's thesis: optical-flow align +
  predict each frame's **continuous Fourier coefficients**, fuse in Fourier space →
  **one model, arbitrary SR scale factors**. (Learned; behind the hallucination gate per
  `research_wavelets_continuous.md` B3 — cited here for the arbitrary-scale-from-a-stored-
  field structure, not as a shipping target.)
  https://arxiv.org/abs/2409.15384

### Incremental / streaming / online multi-frame fusion (the §2 analog)

- **Elad & Feuer, "Superresolution restoration of an image sequence: adaptive filtering
  approach"** (IEEE TIP 1999) — the foundational *recursive* SR: LS estimators that
  **adapt in time** (LMS / RLS), rederived as approximations of the **Kalman filter**,
  so each new frame updates the running HR estimate. This is the classic statement that
  multi-frame SR can be made recursive/streaming.
  https://dl.acm.org/doi/abs/10.1109/83.748893
- **Online multi-frame super-resolution of image sequences** (J. Image Video Proc.,
  Springer 2018) — explicit **online (OSR)** mode: "processes only one low-resolution
  image at a time … when an LR image is received, it is immediately used to improve the
  current HR image estimate"; fusion is a **weighted sum** giving more weight to
  lower-interpolation-error pixels — exactly our `num/den` running average with robustness
  as the weight. "Very fast and memory efficient." Direct support for §2.
  https://link.springer.com/article/10.1186/s13640-018-0376-5
- **Robust SR by fusion of interpolated frames** (Frontiers in Physics 2015) — the
  weighted-sum fusion of registered frames our merge implements; the incremental form is
  a running weighted average.
  https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2015.00028/full

### The repo merge's own framing (the source of the continuous identity)

- **Wronski et al., "Handheld Multi-Frame Super-Resolution"** (SIGGRAPH 2019) — frames
  the merge as a **deliberate discretization of a continuous reconstruction** ("similarly
  to the non-parametric kernel regression framework of Takeda… we evaluate [the kernels]
  at the final resampling grid positions … a 'gather' approach [so] every output pixel is
  independently processed only once per frame"). The "evaluate at the resampling grid"
  choice is *exactly* what makes re-rendering at another grid (§1) free, and "once per
  frame" is *exactly* the per-frame additive contribution (§2).
  https://3dvar.com/Wronski2019Handheld.pdf
- **Takeda et al., "Kernel Regression for Image Processing and Reconstruction"**
  (IEEE TIP 2007) — `ẑ(x)=Σ K_H(x_i−x)y_i / Σ K_H(x_i−x)` evaluated at *any* continuous
  `x` from scattered samples; the steering kernel = the structure-tensor `Ω`. The math
  that says "query at any coordinate/scale needs no new estimation, only a new `x`."
  https://users.soe.ucsc.edu/~milanfar/KernelRegression_Final.pdf

---

## 4. Deliverables

### (a) Minimal persistable artifact for arbitrary-scale re-render

Persist **the representation, not the render**: per comp frame `{raw J_n, flow V_n,
covariance Ω_n, robustness r_n}` + the reference `{J_1, Ω_1}` + the resolved config
(SNR-derived merge params, CFA, white balance, noise model). All of `V_n, Ω_n, r_n` are
**bit-identical across scales (verified 0.0e+00)**, so they are computed once. A new scale
re-runs *only* the two Warp gather kernels + the divide — fractional/arbitrary `scale`
works today. Size ~4.5 MiB/frame at 512×768 (≈ a few hundred MB for a full-res burst,
less with symmetric-`Ω`/uint16-raw packing); the re-render is a small fraction of a full
pipeline run and is the only part that scales with `s²`. **Do NOT persist `num`/`den`:**
they are scale-specific render output, and a coarser grid is not a downsample of a finer
one (kernel support / per-pixel `Ω` interpolation / 3×3 truncation all change per scale).

### (b) Incrementalizability of the merge

The comp-frame accumulation is **exactly additive (verified 0.0e+00)**: `num`/`den` is a
running sum of per-frame contributions. **Add a frame** = compute its ref-relative
`V_n, Ω_n, r_n` and `merge` it in (O(1) in existing frames); **remove** = `-=` its
contribution (exact to float32); render anytime = `divide(num, den)`. Reuse the cached
reference pyramid / local stats / noise curves across all frames. **Caveats:** robustness
and flow are relative to a *fixed reference* (changing it = full recompute); the
**accumulated-robustness denoiser**, if enabled, makes the *ref* merge `overwrite` rather
than add and depends on the full frame set (defer it to render time / recompute on set
change — verified that with it OFF, the ref merge is additive too); the `divide` is cheap
and stays out of the running state.

### (c) Feasibility verdict + sketch

**Both are feasible with no change to the merge math** — the additivity and
scale-as-a-query-parameter are already in the kernels; only `main`'s one-shot control
flow hides them. Proposed API (a refactor, not new algorithms):

```python
class StreamingMerge:
    def __init__(self, ref_raw, config):                 # fixed reference
        self.cfg = resolve_snr_params(config, ref_raw)   # params.py, once
        self.ref_pyr   = init_alignment(grey(ref_raw), self.cfg)
        self.ref_stats = init_robustness(ref_raw, ...)   # local means/stds, once
        self.Omega_ref = estimate_kernels(ref_raw, self.cfg)
        self.J_ref = ref_raw
        self.num = self.den = None                       # lazily sized at first render
        self.acc_r = 0                                   # running accumulated robustness
        self.frames = []                                 # {J_n, V_n, Omega_n, r_n} for re-render/removal

    def add_frame(self, raw_n):                          # O(1) in #frames
        V  = align(self.ref_pyr, grey(raw_n), self.cfg)
        r  = compute_robustness(raw_n, *self.ref_stats, V, ..., self.cfg)
        Om = estimate_kernels(raw_n, self.cfg)
        self.frames.append((raw_n, V, Om, r)); self.acc_r = self.acc_r + r
        self._dirty = True                               # comp accumulation can be incremental;
        #            keep raw num/den; fold this frame's gather in at current scale, or defer

    def render(self, scale):                             # cheap: gathers + divide only
        out = alloc(scale); num, den = zeros(out), zeros(out)
        for (J, V, Om, r) in self.frames:               # or maintain running num/den if scale fixed
            merge(J, V, Om, r, num, den, cfa, cfg(scale))
        merge_ref(self.J_ref, self.Omega_ref, num, den, cfa, cfg(scale), self.acc_r)  # last
        return divide(num, den)                          # + optional postproc
```

For a *fixed* scale, keep `num`/`den` resident and have `add_frame` do the comp gather
in place (true streaming, snapshot via `divide` anytime). For *arbitrary* scale, persist
the per-frame artifact and re-render — the comp loop above. Either way the expensive
per-frame estimation runs once and is never repeated for a scale change.

**Net:** the repo already *is* a continuous steering-KR / Gaussian-splat representation
whose render is (i) a function of an output grid we can choose freely and (ii) a sum over
per-frame contributions. Arbitrary-scale re-render and frame-streaming are therefore
matters of *persisting the representation and exposing the control flow*, not new
algorithms — and the only fidelity risk (hallucination) is absent because the
representation is measured, not learned.
