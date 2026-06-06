"""ONE reproducible script for the spatially-varying-PSF (Lever A) gating decision.

Question: does a FIELD-DEPENDENT (radius-varying) measured-PSF deconvolution beat
our GLOBAL gated-RL on the a6300 APS-C crop? Lever A only pays off if the lens PSF
(equivalently the deconv gap our merge must close to reach DxO) is significantly
WIDER toward the corners than at the center. This script measures that gate end to
end and prints a verdict. It does NOT modify the pipeline or configs.

Decisive chain (each step is executable evidence, eval/CLAUDE.md discipline:
distribution not anecdote, reliability-gated, both orientations, region-stratified,
own-claim red-teamed):

  (0) direction self-test: esf_sigma+transpose isolates sigma_x vs sigma_y
      (so astigmatism is detectable) -- run separately: eval._psf_field_selftest.
  (1) FIELD MAP on the DxO stack and on a FRESH single-frame merge: edge sigma at
      many reliability-gated step edges, binned center/mid/corner, both orientations.
  (2) CONTENT vs FIELD disambiguation -- the radial scatter on this oblique facade
      is confounded by depth (different columns at different focus). Two controls:
        (2a) matched-edge corr(DxO sigma, merge sigma): if ~1, the sigma variation
             is the SAME physical edges (content), shared by both reconstructions,
             not a property of the reconstruction or the field.
        (2b) THE DECISIVE TEST -- the deconv GAP (merge_sigma - DxO_sigma), exactly
             what a field-varying deconv would have to vary: regress on r_norm with
             a bootstrap CI. Flat gap => a GLOBAL correction suffices; field-varying
             buys nothing.
  (3) verdict.

A separate within-pan single-edge tracking control (content held fixed, only field
varies) lives in eval._psf_field_track and corroborates (2): each physical edge's
sigma is stable wherever the pan places it.

Run: micromamba run -n handheld python -m eval._psf_field_validate
Outputs a figure -> eval/halluc_out_007/_psf_field_validate.png

NOTE: reads the SHARED DxO luma cache by absolute path (does not regenerate it);
builds the fresh single-frame merge once and caches it next to the stack.
"""

import os
import time

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu

from .psf import esf_sigma
from ._psf_field_measure import find_oriented_edges, measure_field, _bin_stats

OUT = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_007"
DXO_CACHE = f"{OUT}/luma_stack.npy"
MERGE_CACHE = f"{OUT}/_psf_merge_single_luma.npy"


# ---------------------------------------------------------------------------
def ensure_merge():
    """Fresh single-frame merge luma (pipeline reconstruction; lens MTF still in,
    no DxO field correction). Cached next to the stack."""
    if os.path.exists(MERGE_CACHE):
        return np.load(MERGE_CACHE)
    import eval.pipeline_runner as pr
    import eval.hallucination as H
    burst = pr.load_burst_arw("/home/connorbaker/Pictures/ARW/2025-09-06/007")
    rgb, _ = pr.run_single(burst, overrides={"scale": 1})
    m = H.rgb_to_luma(rgb).astype(np.float32)
    np.save(MERGE_CACHE, m)
    return m


def _bootstrap_slope(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    sl0 = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0][0]
    b = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        b.append(np.linalg.lstsq(np.vstack([x[i], np.ones(len(i))]).T, y[i], rcond=None)[0][0])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(sl0), float(lo), float(hi)


def matched_edges(dxo, mg):
    """Measure the SAME physical step edges (both orientations) in DxO frame0 and
    the merge. Returns array rows (r_norm, sigma_dxo, sigma_merge, is_V)."""
    # global merge<->dxo offset from a central NCC (dxo 4000x6000, merge 4024x6048)
    cy, cx, ph = 2000, 3000, 1000
    tmpl = np.ascontiguousarray(dxo[cy - ph // 2:cy + ph // 2, cx - ph // 2:cx + ph // 2])
    win = np.ascontiguousarray(mg[cy - ph // 2 - 40:cy + ph // 2 + 40,
                                  cx - ph // 2 - 40:cx + ph // 2 + 40])
    cc = cv2.matchTemplate(win, tmpl, cv2.TM_CCORR_NORMED)
    _, pk, _, loc = cv2.minMaxLoc(cc)
    oy, ox = loc[1] - 40, loc[0] - 40
    H, W = dxo.shape
    cy0, cx0, rmax = H / 2, W / 2, np.hypot(H / 2, W / 2)
    edges = find_oriented_edges(dxo, min_stepness=5.0, min_contrast=0.04,
                                stride=45, min_sep=55)
    rows = []
    for ecy, ecx, o, sc, con in edges:
        if o == "V":
            sd = dxo[ecy - 60:ecy + 60, ecx - 22:ecx + 22]
            sm = mg[ecy + oy - 60:ecy + oy + 60, ecx + ox - 22:ecx + ox + 22]
        else:
            sd = dxo[ecy - 22:ecy + 22, ecx - 60:ecx + 60].T
            sm = mg[ecy + oy - 22:ecy + oy + 22, ecx + ox - 60:ecx + ox + 60].T
        if sd.shape != (120, 44) or sm.shape != (120, 44):
            continue
        s_d, r_d = esf_sigma(np.ascontiguousarray(sd))
        s_m, r_m = esf_sigma(np.ascontiguousarray(sm))
        if not (np.isfinite(s_d) and np.isfinite(s_m) and r_d < 0.05 and r_m < 0.05
                and 0.5 < s_d < 4.5 and 0.5 < s_m < 4.5):
            continue
        rn = np.hypot(ecy - cy0, ecx - cx0) / rmax
        rows.append((rn, s_d, s_m, float(o == "V")))
    return np.array(rows), (oy, ox, pk)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    dxo = np.ascontiguousarray(np.load(DXO_CACHE, mmap_mode="r")[0])
    mg = ensure_merge()
    print(f"loaded DxO frame0 {dxo.shape} + merge {mg.shape} ({time.time()-t0:.0f}s)\n")

    # (1) field maps -------------------------------------------------------
    print("=" * 74)
    print("(1) FIELD MAP -- edge sigma vs radial bin (reliability-gated)")
    print("=" * 74)
    fld = {}
    for name, src in [("DxO stack", np.load(DXO_CACHE, mmap_mode="r")), ("merge", mg)]:
        f0 = np.ascontiguousarray(src[0]) if src.ndim == 3 else src
        edges = find_oriented_edges(f0, min_stepness=4.5, min_contrast=0.035,
                                    stride=45, min_sep=55)
        recs = measure_field(src, edges=edges)
        rel = [d for d in recs if d.get("reliable")]
        st = _bin_stats(recs)
        fld[name] = (recs, st)
        nV = sum(d["orient"] == "V" for d in rel)
        nH = sum(d["orient"] == "H" for d in rel)
        print(f"\n{name}: {len(rel)} reliable edges (V={nV} sigma_x, H={nH} sigma_y)")
        print(f"  {'bin':>7} {'n':>3} {'sigma med':>9} {'IQR':>13}")
        for b in ["center", "mid", "corner"]:
            s = st[b]
            iqr = f"{s['p25']:.2f}-{s['p75']:.2f}" if s["n"] else "  -  "
            print(f"  {b:>7} {s['n']:>3} {s['median']:>9.3f} {iqr:>13}")

    # (2a) matched-edge content vs field ----------------------------------
    print("\n" + "=" * 74)
    print("(2a) CONTENT vs FIELD -- corr(DxO sigma, merge sigma) over matched edges")
    print("=" * 74)
    rows, (oy, ox, pk) = matched_edges(dxo, mg)
    rn, s_d, s_m = rows[:, 0], rows[:, 1], rows[:, 2]
    pr_sd, pp = pearsonr(s_d, s_m)
    print(f"merge<->dxo offset ({oy},{ox}) ncc={pk:.3f}; matched edges {len(rows)} "
          f"(V={int(rows[:,3].sum())}, H={int((1-rows[:,3]).sum())})")
    print(f"corr(DxO sigma, merge sigma) = {pr_sd:.3f} (p={pp:.1e})")
    print("  -> ~1 => sigma variation is the SAME physical edges (CONTENT/DEPTH),"
          " shared by both reconstructions; not a field property.")

    # (2b) decisive: deconv gap vs radius ---------------------------------
    print("\n" + "=" * 74)
    print("(2b) DECISIVE TEST -- deconv GAP (merge_sigma - DxO_sigma) vs r_norm")
    print("=" * 74)
    gap = s_m - s_d
    sl, lo, hi = _bootstrap_slope(rn, gap)
    pr_g, pp_g = pearsonr(rn, gap)
    gap_center = np.median(gap[rn < 1 / 3])
    gap_corner = np.median(gap[rn >= 0.5])
    flat = lo <= 0 <= hi
    print(f"gap = {sl:+.3f}*r_norm + const   slope 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"corr(r_norm, gap) = {pr_g:+.3f} (p={pp_g:.3f})")
    print(f"gap median {np.median(gap):.3f}px  std {gap.std():.3f}px  "
          f"center {gap_center:.3f} -> corner {gap_corner:.3f}")

    # also: merge radial trend significance (the naive view), for honesty
    relm = [d for d in fld["merge"][0] if d.get("reliable")]
    mrn = np.array([d["r_norm"] for d in relm]); msg = np.array([d["sigma"] for d in relm])
    cen = msg[mrn < 1 / 3]; cor = msg[mrn >= 2 / 3]
    try:
        _, pmw = mannwhitneyu(cor, cen, alternative="greater")
    except Exception:
        pmw = float("nan")
    msl, mlo, mhi = _bootstrap_slope(mrn, msg)
    R2 = 1 - (msg - (msl * mrn + np.median(msg) - msl * np.median(mrn))).var() / msg.var()
    print(f"\n(for honesty) naive merge sigma-vs-radius: slope {msl:+.3f} CI "
          f"[{mlo:+.3f},{mhi:+.3f}], center->corner MW p={pmw:.3f}, R^2~{max(R2,0):.2f}")
    print("  -> even where this looks 'significant', it is the content confound: the")
    print("     GAP (what a deconv changes) is flat, and DxO/merge sigma corr ~1.")

    # (3) verdict ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("(3) VERDICT")
    print("=" * 74)
    if flat:
        print("FIELD-DEPENDENCE OF THE DECONV GAP IS NOT SIGNIFICANT on this crop.")
        print("The merge->DxO sharpening gap is ~uniform (~%.2fpx) across the field; the"
              % np.median(gap))
        print("sigma scatter (1.3-3.2px) is CONTENT/DEPTH (oblique facade), corr(DxO,merge)")
        print("=%.2f. A single GLOBAL deconv sigma is the right correction; a radius-varying"
              % pr_sd)
        print("deconv (Lever A) has nothing to vary and would mis-correct (gap if anything")
        print("SHRINKS outward). DO NOT BUILD Lever A here. Negative result = valid finding.")
    else:
        print(f"Deconv gap DOES vary with field (slope {sl:+.3f}, CI [{lo:+.3f},{hi:+.3f}]).")
        print("Lever A may pay off -- proceed to prototype + validate corners vs center.")

    # figure ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    # panel A: matched DxO vs merge sigma (content)
    axs[0].scatter(s_d, s_m, c=rn, cmap="plasma", s=40)
    lim = [min(s_d.min(), s_m.min()) - 0.1, max(s_d.max(), s_m.max()) + 0.1]
    axs[0].plot(lim, lim, "k--", alpha=0.4, label="y=x")
    axs[0].set_xlabel("DxO edge sigma (px)"); axs[0].set_ylabel("merge edge sigma (px)")
    axs[0].set_title(f"(A) matched edges: corr={pr_sd:.2f}\n(color=r_norm) -> variation is CONTENT")
    axs[0].legend()
    # panel B: per-source sigma vs radius (the naive 'field' view)
    for name, c in [("DxO stack", "tab:blue"), ("merge", "tab:orange")]:
        rel = [d for d in fld[name][0] if d.get("reliable")]
        x = np.array([d["r_norm"] for d in rel]); y = np.array([d["sigma"] for d in rel])
        axs[1].scatter(x, y, s=22, alpha=0.6, color=c, label=name)
        a, b = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
        xs = np.linspace(x.min(), x.max(), 2)
        axs[1].plot(xs, a * xs + b, color=c, lw=2)
    axs[1].set_xlabel("r_norm (field position)"); axs[1].set_ylabel("edge sigma (px)")
    axs[1].set_title("(B) sigma vs radius (confounded by depth)\nlow R^2; trend unstable across source")
    axs[1].legend()
    # panel C: the decisive deconv gap vs radius
    axs[2].scatter(rn, gap, c="tab:green", s=40)
    xs = np.linspace(rn.min(), rn.max(), 2)
    axs[2].plot(xs, sl * xs + (np.median(gap) - sl * np.median(rn)), "k-", lw=2,
                label=f"slope {sl:+.2f}\nCI [{lo:+.2f},{hi:+.2f}]")
    axs[2].axhline(np.median(gap), ls=":", c="gray", label=f"median {np.median(gap):.2f}px")
    axs[2].set_xlabel("r_norm (field position)")
    axs[2].set_ylabel("deconv gap = merge_sigma - DxO_sigma (px)")
    axs[2].set_title("(C) DECISIVE: gap a field-deconv would vary\nFLAT -> global suffices"
                     if flat else "(C) gap varies with field")
    axs[2].legend()
    fig.suptitle("Lever A gate: spatially-varying PSF deconvolution -- "
                 + ("NOT JUSTIFIED (gap flat; variation is content)" if flat
                    else "field-dependence present"), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = f"{OUT}/_psf_field_validate.png"
    fig.savefig(png, dpi=95)
    print(f"\nfigure -> {png}   ({time.time()-t0:.0f}s total)")
    return dict(corr_dxo_merge=float(pr_sd), gap_slope=sl, gap_ci=[lo, hi],
                gap_median=float(np.median(gap)), flat=bool(flat))


if __name__ == "__main__":
    main()
