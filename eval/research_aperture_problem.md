# Research: the aperture problem in our per-tile block+ICA alignment

Deterministic-only survey (no learned hallucination). Subagent-produced, then the
load-bearing code claims were re-verified against our source on this machine. See
the repo-root `CLAUDE.md` §1 — treat citations as inputs to verify, not gospel.

## The fact that reframes everything: Wronski does NOT regularize the flow

The method we implement (Wronski et al., "Handheld Multi-Frame Super-Resolution",
ACM TOG 38(4) Art.28, SIGGRAPH Asia 2019; arXiv:1905.03277) handles the aperture
problem **entirely at merge time**, not in the flow:
- §5.1.1 — the anisotropic steerable (Takeda) kernel is *stretched along edges* using
  the structure tensor's λ1 / anisotropy, down-weighting off-edge samples. That is our
  `estimate_cov_cupy` (verified: cupy_ops.py uses `A = 1 + sqrt((l1-l2)/(l1+l2))`).
- §5.2.3 — the robustness flow-discontinuity term: large *local flow variation* (even on
  strong features) ⇒ drop `s` from s2 to s1 ⇒ `R = s·exp(-d²/σ²) - t` collapses ⇒ those
  frames barely contribute. That is our `compute_s_cupy` (verified: 3×3 max−min span on
  fx,fy, threshold `Mt`) + the 5×5 `local_min_cupy` erosion.
- Alignment itself is block-matching + **3 iterations of Lucas–Kanade**, with NO median
  filter, smoothness prior, eigenvalue gating, or forward-backward check. Our `n_iter=3`
  ICA matches exactly. Cited basis for "down-weight, don't fix": Robinson & Milanfar,
  "Fundamental performance limits in image registration", IEEE TIP 13(9), 2004 — the
  aperture problem is a Cramér–Rao estimator-variance limit, content-dependent.

So the streaking / the user's "jutting" feature is the **residual that leaks through the
soft robustness threshold** — by design those tiles are meant to be rejected, not fixed.

## We are one cheap step from the standard detector (verified in our code)

`compute_hessian_cupy` (utils_image.py:397-399) already sums per-tile `Σgx², Σgy², Σgxgy`
— the LK normal matrix = structure tensor = the matrix Shi–Tomasi/Harris use.
`ica_iterate_cupy` (cupy_ops.py:91-93) already computes `det` and a `solvable=|det|≥1e-10`
guard. So eigenvalues/coherence are ~free:
`λ_min = (tr − √(tr²−4det))/2`, flag aperture-ambiguous when `λ_min < τ` (Shi–Tomasi,
"Good Features to Track", CVPR 1994) or coherence `((λ1−λ2)/(λ1+λ2))²` high with large λ1
(Bigün/Granlund/Wiklund, IEEE TPAMI 13(8), 1991; Harris–Stephens 1988). LK foundation:
Lucas–Kanade IJCAI 1981; Baker–Matthews, "LK 20 Years On", IJCV 56(3), 2004 (for a
translation warp the LK Hessian *is* our tile Hessian).

## Deterministic recipe (low→high risk), all compatible with our per-tile field

1. **Detect** aperture-ambiguous tiles from the Hessian eigenvalues we already compute
   (Shi–Tomasi λ_min / coherence). FREE. Tie τ to our noise model (std_curve/diff_curve).
2. **Damp in place**: Tikhonov/Levenberg `(H+λI)⁻¹` in `ica_iterate_cupy`. Shrinks the
   ill-conditioned along-edge update toward the block-match prior; **cannot ghost** (uses
   no neighbor info). Smallest, most defensible change. (Baker–Matthews list LM as a
   standard LK variant.)
3. **Reject gross outliers**: coarse-to-fine search-bound gating (our `block_matching_l1`
   currently *clamps* OOB silently — flag `|refined−coarse|>search_radius` invalid and
   inherit neighbor flow instead) and/or MAD on the flow neighborhood. The ~2% catastrophic
   tiles are then handled by the *statistical* robustness term, not a silent skip.
   (Robust-stat / breakdown point: Farsiu–Robinson–Elad–Milanfar, IEEE TIP 13(10), 2004 —
   median breakdown 0.5; note they make *fusion* robust, not the flow.)
4. **Regularize**: small **weighted (cross-bilateral) vector-median** over the tile flow
   grid, weighted by guide-image similarity to protect motion boundaries. This is the
   single most empirically validated fix for "noisy per-cell flow + sparse outliers":
   Sun–Roth–Black, "Secrets of Optical Flow Estimation and Their Principles", CVPR 2010
   ("median filtering after each warp is the single most important secret"; the *weighted*
   version "prevents over-smoothing across boundaries"). Vector (not per-component) median:
   Astola–Haavisto–Neuvo, "Vector median filters", Proc. IEEE 78(4), 1990.
5. **(Only if occlusion ghosts persist)** forward-backward consistency — heaviest (needs a
   reverse alignment pass), overlaps the existing color-difference robustness; likely not
   worth it.

## The central risk (Q4): regularizing the flow can BLIND the robustness mask

`compute_s` keys on *flow variation* (eq.7). If you smooth the flow you can erase the very
signal the mask uses to reject moving-object ghosts → re-introduce the ghosts it was meant
to catch. Therefore: **damp in place (step 2) before any neighbor-borrowing (step 4); fill
from neighbors only where the flow neighborhood is locally consistent; keep the statistical
(color-difference) robustness term as the final ghost backstop.** This stays inside
Wronski's design philosophy while addressing the residual their soft threshold lets through.

## Established vs. speculative
- **Established (primary sources read):** Wronski merge-time aperture handling; Shi–Tomasi /
  Harris / Bigün detection; Tikhonov/LM as an LK variant; Sun–Roth–Black weighted median;
  Astola vector median; Farsiu robustness/breakdown; Robinson–Milanfar limits;
  Baker–Matthews LK.
- **Speculative (sound composition, no single citation for our exact setup):** eigenvector-
  subspace projection that keeps the across-edge component from the tile and fills the
  along-edge component from neighbors (closest cited analogue: Knutsson–Westin normalized
  convolution, CVPR 1993). Validate empirically.
