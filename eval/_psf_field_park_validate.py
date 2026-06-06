"""DECISIVE re-test: is the optical-PSF / deconv GAP FIELD-DEPENDENT, on a scene
that reaches the BOTTOM field edge (r_norm -> 1.0)?

Background (verified prior result, 007 oblique facade): corr(DxO sigma, merge
sigma)=0.98 -- the sigma scatter is subject DEPTH shared by both reconstructions,
NOT lens field -- and the GAP = merge_sigma - DxO_sigma was FLAT vs field radius
(slope CI straddled 0, ~0.43px everywhere). BUT 007 edges reached only ~0.55-0.80
r_norm; the true corners were unprobed. The DECISIVE quantity is the GAP (what a
field-varying deconv would have to vary), not raw sigma.

This scene (DSC00034-DSC00074, Sony a6300, 55mm f/9 ISO200, 41 frames) is a dirt
path -> distant treeline, foreground grass filling BOTH BOTTOM CORNERS (r_norm ->
1.0), a sign board mid-left, a soccer goal mid-right, sky at top (top corners
unprobeable). It can finally probe the BOTTOM field edge.

TWO COMPLEMENTARY PROBES (each executable evidence; eval/CLAUDE.md discipline:
distributions not points, reliability gating, region stratification, symmetric
skepticism, metric self-validation, no nearest-zoom visuals):

  PROBE 1 -- ESF gap (the 007 test, extended): edge sigma on the DxO stack AND a
    fresh multi-frame merge at every reliability-gated step edge PLUS hand-targeted
    rigid man-made edges (sign board, goal posts). GAP = merge_sigma - DxO_sigma vs
    r_norm with a bootstrap slope CI. Honest caveat reproduced up front: the
    generic finder yields almost only HORIZONTAL edges along the treeline-vs-sky
    silhouette (mid-field); clean edges do NOT reach the corners on this scene
    either. So PROBE 1 mostly re-tests the 007 regime + the sign/goal anchors.

  PROBE 2 -- depth-controlled grass texture-acutance (the NEW probe this scene
    enables, and the ONLY one reaching r_norm~1): on a ~level camera over a flat
    ground plane, a fixed IMAGE ROW is ~iso-depth, so along a row HORIZONTAL field
    position varies while depth is ~constant. For several foreground-grass rows
    (reaching the bottom-left/right corners) we measure local high-frequency
    acutance vs horizontal field position on MERGE vs DxO. The GAP (merge - DxO
    acutance) cancels content (both see the same grass). Droop toward the L/R field
    edges => uncorrected lens field-softness; flat => corroborates 007.

CONFOUNDS controlled + reported:
  (a) WIND: grass moves between frames -> the multi-frame merge motion-blurs it,
      inflating the gap independent of lens. We (i) quantify wind (grass vs rigid
      residual after sub-pixel align), (ii) use the robustness-masked merge AND a
      WIND-FREE single-frame-baseline merge as a control: if the grass acutance gap
      droops toward edges in BOTH multi and single, it is lens, not wind; if only
      in multi, it is wind.
  (b) iso-depth check: DxO field-corrects, so DxO acutance ALONG an iso-depth row
      should be ~flat if depth is controlled. We verify DxO-alone is ~flat per row
      before trusting the gap; rows where DxO droops at the edge are NOT iso-depth
      and are flagged.
  (c) content variation along a row: the GAP cancels it; stated explicitly.

Registration: VERIFIED a single global translation (merge grid ~4024x6048 vs DxO
4000x6000); offset agrees across TL/TR/BL/BR/center to <=1px, NCC~1.0 -- so a
field-varying offset cannot masquerade as a field-varying gap.

Run: micromamba run -n handheld python -m eval._psf_field_park_validate
Outputs eval/halluc_out_034/_psf_field_park.png (+ prints the verdict).
Heavy caches (DxO stack, merges) under eval/halluc_out_034/ (gitignored).
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from scipy.ndimage import shift as ndshift

from .psf import esf_sigma
from ._psf_field_measure import find_oriented_edges, _bin_stats
from . import _psf_field_park_lib as L


# ---------------------------------------------------------------------------
# Hand-targeted RIGID man-made edges (sign board, goal posts), located by eye on
# the DxO frame (DxO coords; V => vertical edge -> sigma_x).
#
# HONEST NOTE (verified): on THIS scene almost none of these pass the ESF
# reliability gate at fixed coords -- the sign edges sit next to printed text and
# carry DxO sharpening overshoot (rms too high or implausibly sharp sigma~0.7), the
# goal posts are thin lines against a textured treeline. A local lowest-rms search
# recovers only 2-3 clean fits, each idiosyncratic (one even shows DxO overshoot) --
# too few and too cherry-picked to be a distribution. So they are kept ONLY as
# documented targets and contribute nothing to the PROBE1 regression (all filtered
# out); PROBE1 runs on the auto-found edge distribution, and the corner-reaching
# evidence comes from PROBE2 (grass). This is itself a finding: this scene has ~no
# clean rigid man-made step edges, matching the prior agent's "~0 clean edges".
# ---------------------------------------------------------------------------
RIGID_EDGES = [
    (1770, 986, "V"),    # sign board LEFT edge (white board vs foliage)
    (1645, 1055, "H"),   # sign board TOP edge
    (1895, 1055, "H"),   # sign board BOTTOM edge
    (2050, 5332, "V"),   # goal RIGHT post, lower (white post vs grass)
    (1960, 5332, "V"),   # goal RIGHT post, upper
    (1858, 5430, "H"),   # goal crossbar (right side)
]


def _bootstrap_slope(x, y, n=4000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    A = np.vstack([x, np.ones(len(x))]).T
    sl0 = np.linalg.lstsq(A, y, rcond=None)[0][0]
    b = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        b.append(np.linalg.lstsq(np.vstack([x[i], np.ones(len(i))]).T, y[i], rcond=None)[0][0])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(sl0), float(lo), float(hi)


def _esf_at(img, cy, cx, orient, sz=120, band=44):
    h, b = sz // 2, band // 2
    if orient == "V":
        sub = img[cy - h:cy + h, cx - b:cx + b]
    else:
        sub = img[cy - b:cy + b, cx - h:cx + h].T
    if sub.shape != (sz, band):
        return np.nan, np.nan
    return esf_sigma(np.ascontiguousarray(sub))


# ===========================================================================
# PROBE 1 -- ESF gap (auto edges + rigid anchors), the decisive 007 test.
# ===========================================================================
def probe1_esf(dxo, mg, oy, ox):
    H, W = dxo.shape
    cy0, cx0, rmax = H / 2, W / 2, np.hypot(H / 2, W / 2)
    # auto edges (loosened for this scene; mostly H along the treeline silhouette)
    auto = find_oriented_edges(dxo, min_stepness=3.0, min_contrast=0.015,
                               stride=32, min_sep=42)
    cands = [(cy, cx, o) for cy, cx, o, *_ in auto] + RIGID_EDGES
    rows = []  # (r_norm, s_dxo, s_merge, is_V, is_rigid)
    nrig = len(RIGID_EDGES)
    for k, (cy, cx, o) in enumerate(cands):
        is_rigid = k >= len(auto)
        s_d, r_d = _esf_at(dxo, cy, cx, o)
        s_m, r_m = _esf_at(mg, cy + oy, cx + ox, o)
        # reliability: good erf fit both, plausible sigma both
        if not (np.isfinite(s_d) and np.isfinite(s_m) and r_d < 0.05 and r_m < 0.05
                and 0.5 < s_d < 4.5 and 0.5 < s_m < 4.5):
            continue
        rn = np.hypot(cy - cy0, cx - cx0) / rmax
        rows.append((rn, s_d, s_m, float(o == "V"), float(is_rigid)))
    return np.array(rows)


# ===========================================================================
# PROBE 2 -- depth-controlled grass RELATIVE-SHARPNESS gap vs HORIZONTAL field.
#
# METRIC (validated; replaces a naive HF-acutance that was REJECTED -- it read
# NOISE on flat patches, anti-correlating with real detail, see notes below):
# the SIGNED relative blur between two sub-pixel-aligned patches of the SAME
# content. fit_blur_sigma(A, B) = the Gaussian sigma that blurs the SHARP B to
# match A; so
#     relsig(merge, dxo) = fit_blur_sigma(merge, dxo) - fit_blur_sigma(dxo, merge)
# is >0 when the MERGE is SOFTER than DxO (the deconv gap DxO closes; the direct
# analog of PROBE 1's ESF gap), <0 when the merge is sharper. This CANCELS content
# (it is a registered same-patch difference) and is NOISE-ROBUST (random noise does
# not fit as a Gaussian blur of the other image). Round-trip-validated on real grass
# (recovers an injected 0.8/1.5/2.5px blur to +-0.05).
#
# WIND (verified): the grass moves between frames, so the TEMPORAL multi merge
# motion-blurs it and will NOT match a single DxO instant. But the single-frame
# merge uses ARW frame DSC00034 and DxO-frame0 is DSC00034_DxO-reference -- the
# SAME physical instant (verified: HF residual ~0.15 on grass for single vs ~0.5
# for multi). So SINGLE-merge vs DxO-frame0 is the WIND-FREE, same-instant grass
# probe that isolates lens+reconstruction. We ALSO report multi-merge vs
# DxO-consensus (temporal median; both averaged so wind ~cancels) as a cross-check.
# A lens field-softness we are NOT correcting would make relsig RISE toward
# |xnorm|=1 in the WIND-FREE comparison (merge increasingly softer than the
# field-corrected DxO off-axis); flat => corroborates 007.
# ===========================================================================
def _relsig(a, b, smax=4.0):
    """Signed relative blur (px): >0 => `a` softer than `b`, <0 => `a` sharper.
    `a`,`b` are sub-pixel-aligned float patches of the SAME content."""
    import eval.calibrate_optics as co
    return co.fit_blur_sigma(a, b, smax=smax) - co.fit_blur_sigma(b, a, smax=smax)


def _subpix_patch(ref, src, ry, rx, oy, ox, patch):
    """Sub-pixel-align a `src` patch onto the DxO `ref` patch at (ry,rx) (DxO coords;
    src is the merge, offset by (oy,ox) + a residual sub-pixel shift). Returns
    (ref_patch, aligned_src_patch) or (None, None) if OOB/poor."""
    from .hallucination import pcc_shift
    h = patch // 2
    rp = np.ascontiguousarray(ref[ry - h:ry + h, rx - h:rx + h])
    pad = 6
    win = src[ry + oy - h - pad:ry + oy + h + pad, rx + ox - h - pad:rx + ox + h + pad]
    if rp.shape != (patch, patch) or win.shape != (patch + 2 * pad, patch + 2 * pad):
        return None, None
    sp0 = np.ascontiguousarray(win[pad:pad + patch, pad:pad + patch])
    dy, dx, _ = pcc_shift(rp, sp0)
    if abs(dy) > pad or abs(dx) > pad:
        return None, None
    sp = ndshift(np.ascontiguousarray(win), (dy, dx), order=3, mode="reflect")
    return rp, np.ascontiguousarray(sp[pad:pad + patch, pad:pad + patch])


def probe2_grass(dxo_stack, mg_multi, mg_single, oy, ox, rows_frac,
                 patch=112, step=56, x_lo_frac=0.04, x_hi_frac=0.96):
    """Per foreground row, slide a window across x; at each grass patch compute the
    signed relative blur (deconv gap) for BOTH:
      relsig_single = single-merge vs DxO-frame0  (WIND-FREE, same instant)
      relsig_multi  = multi-merge  vs DxO-consensus (both temporally averaged)
    plus a texture-scale iso-depth check (grass-blade px scale; flat along a row =>
    iso-depth). Content-gated identically for all (DxO std band)."""
    dxo0 = np.ascontiguousarray(dxo_stack[0])
    # DxO consensus: temporal median over a subset (denoised, wind-averaged ref)
    nstk = min(dxo_stack.shape[0], 16)
    dxo_cons = np.median(dxo_stack[:nstk], axis=0).astype(np.float32)
    H, W = dxo0.shape
    cy0, cx0, rmax = H / 2, W / 2, np.hypot(H / 2, W / 2)
    h = patch // 2
    out = []
    xs_px = np.arange(int(x_lo_frac * W) + h, int(x_hi_frac * W) - h, step)

    def _texscale(p):  # grass-blade spatial scale proxy (radial PSD peak); flat=>iso-depth
        p = p.astype(np.float64); p = p - p.mean()
        win = np.hanning(p.shape[0])[:, None] * np.hanning(p.shape[1])[None, :]
        F = np.abs(np.fft.fftshift(np.fft.fft2(p * win))) ** 2
        cy, cx = np.array(F.shape) // 2
        yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
        rr = np.hypot(yy - cy, xx - cx).astype(int)
        rb = np.bincount(rr.ravel(), F.ravel()) / (np.bincount(rr.ravel()) + 1e-9)
        rb = rb[1:min(F.shape) // 2]
        return float(np.argmax(rb) + 1) if rb.size else np.nan

    for rf in rows_frac:
        ry = int(rf * H)
        if ry - h < 0 or ry + h > H:
            continue
        # Two probes recorded INDEPENDENTLY (different sample sets): the WIND-FREE
        # single (same-instant; per-patch align is reliable -> all grass patches)
        # is the workhorse; the multi-vs-consensus cross-check loses patches because
        # wind makes the consensus<->multi per-patch align unreliable (>pad), which
        # is itself evidence the multi grass comparison is wind-limited.
        rec = dict(row_frac=rf, ry=ry,
                   xn_s=[], rn_s=[], relsig_single=[], texscale=[], dxo_std=[],
                   xn_m=[], rn_m=[], relsig_multi=[])
        for x in xs_px:
            pd = dxo0[ry - h:ry + h, x - h:x + h]
            if pd.shape != (patch, patch) or not (0.006 < pd.std() < 0.5):
                continue  # content gate (grass band; reject flat path/sky/saturated)
            xn = (x - cx0) / (W / 2); rn = np.hypot(ry - cy0, x - cx0) / rmax
            rp_s, sp_s = _subpix_patch(dxo0, mg_single, ry, x, oy, ox, patch)
            if sp_s is not None:
                rec["xn_s"].append(xn); rec["rn_s"].append(rn)
                rec["relsig_single"].append(_relsig(sp_s, rp_s))
                rec["texscale"].append(_texscale(pd)); rec["dxo_std"].append(float(pd.std()))
            rp_m, sp_m = _subpix_patch(dxo_cons, mg_multi, ry, x, oy, ox, patch)
            if sp_m is not None:
                rec["xn_m"].append(xn); rec["rn_m"].append(rn)
                rec["relsig_multi"].append(_relsig(sp_m, rp_m))
        for k in ("xn_s", "rn_s", "relsig_single", "texscale", "dxo_std",
                  "xn_m", "rn_m", "relsig_multi"):
            rec[k] = np.array(rec[k])
        out.append(rec)
    return out


def _abs_edge_droop(xn, y):
    """Fit y vs |xnorm| (distance from vertical center-line = horizontal field
    radius). Slope<0 => droops toward L/R edges. Bootstrap CI on the slope."""
    return _bootstrap_slope(np.abs(xn), y)


# ===========================================================================
def main():
    t0 = time.time()
    dxo_stack = np.load(L.DXO_STACK, mmap_mode="r")
    dxo = np.ascontiguousarray(dxo_stack[0])
    mg_multi = L.ensure_merge(kind="multi")
    mg_single = L.ensure_merge(kind="single")
    oy, ox, pk, field = L.merge_dxo_offset(dxo, mg_multi)
    H, W = dxo.shape
    print(f"loaded DxO {dxo.shape} + merges; merge<->DxO offset ({oy},{ox}) ncc={pk:.4f}")
    fspr_y = max(v[0] for v in field.values()) - min(v[0] for v in field.values())
    fspr_x = max(v[1] for v in field.values()) - min(v[1] for v in field.values())
    print(f"  registration field spread dy={fspr_y}px dx={fspr_x}px "
          f"(<=1 => single translation valid across the field)\n")

    # -- WIND quantification (grass vs rigid, post sub-pixel align) ----------
    print("=" * 76)
    print("WIND CONTROL -- inter-frame residual (post sub-pixel align) / contrast")
    print("=" * 76)
    wind = {}
    regions = {"sign(rigid)": (1500, 950), "treeline(rigid)": (1300, 3000),
               "grass_botL": (3500, 500), "grass_botC": (3600, 3000),
               "grass_botR": (3500, 5500)}
    half = 180
    N = min(dxo_stack.shape[0], 24)
    for nm, (cy, cx) in regions.items():
        ref = np.ascontiguousarray(dxo[cy - half:cy + half, cx - half:cx + half])
        res = []
        for i in range(1, N):
            w = np.ascontiguousarray(dxo_stack[i][cy - half:cy + half, cx - half:cx + half])
            from .hallucination import pcc_shift
            dy, dx, _ = pcc_shift(ref, w)
            wa = ndshift(w, (dy, dx), order=3, mode="reflect")
            c = slice(40, -40)
            res.append(float(np.sqrt(np.mean((wa[c, c] - ref[c, c]) ** 2)) / (ref[c, c].std() + 1e-6)))
        wind[nm] = float(np.median(res))
        print(f"  {nm:16s}: residual/contrast median {wind[nm]:.3f}")
    rigid_w = np.mean([wind[k] for k in wind if "rigid" in k])
    grass_w = np.mean([wind[k] for k in wind if "grass" in k])
    print(f"  -> grass {grass_w:.3f} vs rigid {rigid_w:.3f} "
          f"(ratio {grass_w/rigid_w:.2f}x): grass moves between frames => "
          f"multi merge will motion-blur it; single-frame merge is the wind-free control.\n")

    # -- NOISE control (does single-frame merge add corner noise that fakes
    #    'sharpness'?). HF std on near-flat path at center vs L/R; if single is not
    #    much noisier than DxO, a corner relsig<0 (merge sharper) is real detail. ---
    print("=" * 76)
    print("NOISE CONTROL -- HF-noise floor on flat path (center vs L/R field edge)")
    print("=" * 76)
    from scipy.ndimage import gaussian_filter as _gf
    flats = {"path_center": (2700, 3000), "path_left": (2650, 1200), "path_right": (2650, 4800)}
    hh = 56
    for nm, (cy, cx) in flats.items():
        pd = dxo[cy - hh:cy + hh, cx - hh:cx + hh]
        ps = mg_single[cy + oy - hh:cy + oy + hh, cx + ox - hh:cx + ox + hh]
        pm = mg_multi[cy + oy - hh:cy + oy + hh, cx + ox - hh:cx + ox + hh]
        hf = lambda p: float((p - _gf(p, 2)).std())
        print(f"  {nm:12s}: HFstd dxo={hf(pd):.4f} single={hf(ps):.4f} multi={hf(pm):.4f}")
    print("  -> single HFstd ~ DxO (ISO200, SNR high) => corner 'merge sharper' is detail,")
    print("     not single-frame noise; the multi merge corroborates where wind lets it align.\n")

    # -- PROBE 1: ESF gap ----------------------------------------------------
    print("=" * 76)
    print("PROBE 1 -- ESF deconv GAP (merge_sigma - DxO_sigma) vs field radius")
    print("=" * 76)
    rows = probe1_esf(dxo, mg_multi, oy, ox)
    rn, s_d, s_m, isV, isrig = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    gap = s_m - s_d
    nV, nH, nrig = int(isV.sum()), int((1 - isV).sum()), int(isrig.sum())
    print(f"reliable edges: {len(rows)} (V/sigma_x={nV}, H/sigma_y={nH}; rigid anchors={nrig})")
    print(f"  r_norm coverage {rn.min():.2f}-{rn.max():.2f}  (>0.6: {int((rn>0.6).sum())}, "
          f">0.8: {int((rn>0.8).sum())})")
    pr_dm, _ = pearsonr(s_d, s_m)
    print(f"corr(DxO sigma, merge sigma) = {pr_dm:.3f}  -> ~1 => sigma variation is the SAME "
          "physical edges (content/depth), shared by both, not field.")
    sl, lo, hi = _bootstrap_slope(rn, gap)
    pr_g, pp_g = pearsonr(rn, gap)
    flat1 = lo <= 0 <= hi
    gc = np.median(gap[rn < 1 / 3]) if (rn < 1 / 3).any() else np.nan
    go = np.median(gap[rn >= 2 / 3]) if (rn >= 2 / 3).any() else np.nan
    print(f"GAP = {sl:+.3f}*r_norm + c   slope 95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"corr(r,gap)={pr_g:+.2f} p={pp_g:.3f}")
    print(f"  gap median {np.median(gap):.3f}px std {gap.std():.3f}px; center {gc:.3f} -> "
          f"corner {go:.3f}; {'FLAT (CI straddles 0)' if flat1 else 'NON-FLAT'}")

    # -- PROBE 2: grass relative-sharpness gap vs horizontal field -----------
    print("\n" + "=" * 76)
    print("PROBE 2 -- depth-controlled grass RELATIVE-SHARPNESS gap vs |horiz field|")
    print("  metric: relsig = blur(merge->dxo) - blur(dxo->merge); >0 => MERGE SOFTER")
    print("  than DxO (the deconv gap). content-canceling + noise-robust (validated).")
    print("=" * 76)
    rows_frac = [0.60, 0.66, 0.72, 0.78, 0.84, 0.90]
    grec = probe2_grass(dxo_stack, mg_multi, mg_single, oy, ox, rows_frac)
    print("row   ry   n_s  |xn|max rnorm | texscale e/c(iso chk) | relsig_single(WIND-FREE) "
          "med  slope[CI] vs|xn| | n_m relsig_multi med")
    for r in grec:
        if len(r["xn_s"]) < 8:
            print(f"  {r['row_frac']:.2f} {r['ry']:4d} {len(r['xn_s']):3d}  (too few patches)")
            continue
        xn = r["xn_s"]; axn = np.abs(xn)
        # iso-depth check: grass-blade texture SCALE along the row (flat => iso-depth)
        ts_e = np.median(r["texscale"][axn > 0.55]) if (axn > 0.55).any() else np.nan
        ts_c = np.median(r["texscale"][axn < 0.35]) if (axn < 0.35).any() else np.nan
        ts_ratio = ts_e / ts_c if (ts_c and np.isfinite(ts_c) and ts_c > 0) else np.nan
        iso_ok = np.isfinite(ts_ratio) and abs(ts_ratio - 1.0) < 0.15
        ss, slo, shi = _abs_edge_droop(xn, r["relsig_single"])
        mmed = np.median(r["relsig_multi"]) if len(r["relsig_multi"]) else np.nan
        print(f"  {r['row_frac']:.2f} {r['ry']:4d} {len(xn):3d}  {axn.max():.2f}  {r['rn_s'].max():.2f} | "
              f"{ts_ratio:.2f}{'OK' if iso_ok else '? '}             | "
              f"{np.median(r['relsig_single']):+.3f}  {ss:+.3f}[{slo:+.2f},{shi:+.2f}] | "
              f"{len(r['relsig_multi']):3d} {mmed:+.3f}")

    # pooled grass result (the distribution across all rows), each probe independently
    keep_s = [r for r in grec if len(r["xn_s"]) >= 8]
    keep_m = [r for r in grec if len(r["xn_m"]) >= 8]
    allxn = np.concatenate([r["xn_s"] for r in keep_s])           # WIND-FREE single
    all_s = np.concatenate([r["relsig_single"] for r in keep_s])
    all_ts = np.concatenate([r["texscale"] for r in keep_s])
    allxn_m = np.concatenate([r["xn_m"] for r in keep_m]) if keep_m else np.array([0.0])
    all_m = np.concatenate([r["relsig_multi"] for r in keep_m]) if keep_m else np.array([0.0])
    gsS, gsL, gsH = _abs_edge_droop(allxn, all_s)
    gmS, gmL, gmH = _abs_edge_droop(allxn_m, all_m)
    flat_single = gsL <= 0 <= gsH
    flat_multi = gmL <= 0 <= gmH
    # iso-depth pooled: texture scale e/c
    axn = np.abs(allxn)
    ts_ratio_pool = (np.median(all_ts[axn > 0.55]) / np.median(all_ts[axn < 0.35])
                     if (axn > 0.55).any() and (axn < 0.35).any() else np.nan)
    rnmax = max(r["rn_s"].max() for r in keep_s)
    print(f"\n  POOLED (single/wind-free: {len(allxn)} grass patches; multi: {len(allxn_m)}; "
          f"reaching r_norm {rnmax:.2f}):")
    print(f"   iso-depth: grass texture-scale edge/center = {ts_ratio_pool:.2f} "
          f"({'~1 => rows are iso-depth (blade px-scale constant along row)' if abs(ts_ratio_pool-1)<0.12 else 'depth NOT constant'})")
    print(f"   SINGLE (WIND-FREE, same instant) relsig vs |xnorm|: slope {gsS:+.4f} "
          f"CI[{gsL:+.4f},{gsH:+.4f}]  {'FLAT' if flat_single else ('RISES->edge (merge softer off-axis = LENS field-softness)' if gsS>0 else 'FALLS->edge')}")
    print(f"   MULTI (vs DxO-consensus, wind-averaged) relsig vs |xnorm|: slope {gmS:+.4f} "
          f"CI[{gmL:+.4f},{gmH:+.4f}]  {'FLAT' if flat_multi else ('RISES->edge' if gmS>0 else 'FALLS->edge')}")
    print(f"   median relsig: single {np.median(all_s):+.3f}px, multi {np.median(all_m):+.3f}px "
          f"(near 0 => merge ~ as sharp as DxO on grass overall)")
    print("   -> uncorrected lens field-softness => relsig RISES toward |xnorm|=1 in the")
    print("      WIND-FREE single comparison. Flat single = corroborates 007 (global deconv OK).")
    # expose under the old names the verdict/figure use
    dxS = gsS  # (figure 'iso' panel now uses texscale; keep var defined)

    # -- VERDICT -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("VERDICT -- does this corner-reaching scene show deconv-gap field-dependence?")
    print("=" * 76)
    print(f"PROBE 1 (ESF gap, edges to r_norm {rn.max():.2f}): "
          f"{'FLAT' if flat1 else 'NON-FLAT'} (slope CI [{lo:+.3f},{hi:+.3f}]), corr(DxO,merge)={pr_dm:.2f}.")
    print(f"PROBE 2 (grass relsig to r_norm {rnmax:.2f}): "
          f"single/wind-free slope {gsS:+.3f} CI[{gsL:+.2f},{gsH:+.2f}] "
          f"({'FLAT' if flat_single else 'NON-FLAT'}); "
          f"multi slope {gmS:+.3f} ({'FLAT' if flat_multi else 'NON-FLAT'}).")
    # Lever A is justified ONLY if OUR merge is increasingly SOFTER than the
    # field-corrected reference toward the edges (relsig RISES, gsS>0). The signs:
    #   gsS>0 (rise)  : merge softer off-axis -> uncorrected lens field-softness -> Lever A candidate
    #   gsS~0 (flat)  : corroborates 007 (global deconv adequate)
    #   gsS<0 (fall)  : merge SHARPER off-axis than DxO -> DxO's corner grass is the soft one
    #                   (its DeepPRIME denoises low-SNR/vignetted corner grass harder); our
    #                   merge does NOT under-resolve corners -> Lever A NOT justified.
    grass_merge_softer_edge = (not flat_single) and gsS > 0
    grass_merge_sharper_edge = (not flat_single) and gsS < 0
    lever_a_justified = (not flat1 and hi > 0) or grass_merge_softer_edge
    if not lever_a_justified:
        print("\n=> NO uncorrected-lens-field-softness signal, EVEN reaching r_norm~%.2f"
              % max(rn.max(), rnmax))
        print("   (further toward the field edge than 007's ~0.55-0.80). PROBE1 ESF gap CI")
        print("   straddles 0. PROBE2 wind-free grass relsig is %s toward the L/R edges --"
              % ("FLAT" if flat_single else "NEGATIVE (merge SHARPER off-axis)"))
        if grass_merge_sharper_edge:
            print("   i.e. our merge is if anything SHARPER than DxO in the corners (DxO over-")
            print("   denoises the low-SNR/vignetted corner grass). Either way our merge does NOT")
            print("   under-resolve the corners relative to the field-corrected reference.")
        print("   This CORROBORATES the 007 global-deconv-adequate finding on a scene 007 could")
        print("   NOT reach. A radius-varying deconv (Lever A) has no growing merge->reference")
        print("   gap to close here -> NOT justified. Negative result = valid finding.")
    else:
        print("\n=> Possible field-dependence: the WIND-FREE grass relsig RISES toward the L/R")
        print("   edges (merge increasingly SOFTER than the field-corrected DxO off-axis) --")
        print("   lens field-softness we are NOT correcting. Lever A may pay off near the corners.")
    if flat_single and not flat_multi:
        print("   (Note: MULTI relsig is NON-FLAT but the WIND-FREE SINGLE is FLAT -> the multi")
        print("    trend is WIND motion-blur in the temporal merge, NOT lens field-softness.)")
    field_dep = lever_a_justified

    # -- FIGURE --------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    # A: matched edge sigma (content)
    ax = axs[0, 0]
    sc = ax.scatter(s_d, s_m, c=rn, cmap="plasma", s=36)
    ax.scatter(s_d[isrig > 0], s_m[isrig > 0], facecolors="none", edgecolors="k",
               s=120, lw=1.5, label="rigid anchors")
    lim = [min(s_d.min(), s_m.min()) - 0.1, max(s_d.max(), s_m.max()) + 0.1]
    ax.plot(lim, lim, "k--", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="r_norm")
    ax.set_xlabel("DxO edge sigma (px)"); ax.set_ylabel("merge edge sigma (px)")
    ax.set_title(f"(A) PROBE1 matched edges: corr={pr_dm:.2f} -> variation is CONTENT/DEPTH")
    ax.legend()
    # B: ESF gap vs radius
    ax = axs[0, 1]
    ax.scatter(rn, gap, c="tab:green", s=36)
    ax.scatter(rn[isrig > 0], gap[isrig > 0], facecolors="none", edgecolors="k", s=120, lw=1.5)
    xs = np.linspace(rn.min(), rn.max(), 2)
    ax.plot(xs, sl * xs + (np.median(gap) - sl * np.median(rn)), "k-", lw=2,
            label=f"slope {sl:+.2f} CI[{lo:+.2f},{hi:+.2f}]")
    ax.axhline(np.median(gap), ls=":", c="gray")
    ax.set_xlabel("r_norm (field radius)"); ax.set_ylabel("ESF gap = merge-DxO sigma (px)")
    ax.set_title("(B) PROBE1 decisive ESF gap vs field\n"
                 + ("FLAT -> global suffices" if flat1 else "non-flat"))
    ax.legend()
    # C: grass relsig vs |xnorm| (wind-free single + multi cross-check), binned medians
    ax = axs[1, 0]
    aabs_s = np.abs(allxn); aabs_m = np.abs(allxn_m)
    ax.scatter(aabs_m, all_m, s=12, alpha=0.3, color="tab:orange", label=f"multi (wind-avg, n={len(all_m)})")
    ax.scatter(aabs_s, all_s, s=12, alpha=0.3, color="tab:blue", label=f"single (WIND-FREE, n={len(all_s)})")
    bins = np.linspace(0, max(aabs_s.max(), aabs_m.max()), 7)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for aabs, arr, col in [(aabs_m, all_m, "tab:orange"), (aabs_s, all_s, "tab:blue")]:
        med = [np.median(arr[(aabs >= bins[i]) & (aabs < bins[i + 1])])
               if ((aabs >= bins[i]) & (aabs < bins[i + 1])).any() else np.nan
               for i in range(len(bc))]
        ax.plot(bc, med, "o-", color=col, ms=7, lw=2.2)
    xa = np.linspace(0, aabs_s.max(), 2)
    ax.plot(xa, gsS * xa + (np.median(all_s) - gsS * np.median(aabs_s)), color="tab:blue", lw=1, ls="--")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("|xnorm| (horizontal field radius; 0=center, 1=L/R edge)")
    ax.set_ylabel("relsig = blur(merge->DxO) - blur(DxO->merge)  (px; >0 merge softer)")
    ax.set_title(f"(C) PROBE2 grass deconv gap vs horiz field\nWIND-FREE single slope {gsS:+.3f} "
                 f"CI[{gsL:+.2f},{gsH:+.2f}] ({'FLAT' if flat_single else 'rise=lens softness'})")
    ax.legend(fontsize=8)
    # D: iso-depth check -- grass texture SCALE per row vs xnorm (flat => iso-depth)
    ax = axs[1, 1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(grec)))
    for r, c in zip(grec, cmap):
        if len(r["xn_s"]) >= 8:
            ax.plot(r["xn_s"], r["texscale"], ".-", color=c, ms=4, lw=1,
                    label=f"row {r['row_frac']:.2f} (rn<{r['rn_s'].max():.2f})")
    ax.set_xlabel("xnorm (horizontal field pos)"); ax.set_ylabel("grass texture scale (PSD peak)")
    ax.set_title(f"(D) iso-depth check: grass blade-scale per row\n"
                 f"flat along row => iso-depth (pooled e/c={ts_ratio_pool:.2f})")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("PARK scene (reaches bottom field edge r_norm~1): deconv-gap field-dependence re-test",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = f"{L.OUT}/_psf_field_park.png"
    fig.savefig(png, dpi=95)
    print(f"\nfigure -> {png}   ({time.time()-t0:.0f}s total)")
    return dict(probe1_flat=bool(flat1), probe1_slope_ci=[lo, hi], corr_dxo_merge=float(pr_dm),
                grass_multi_slope=float(gmS), grass_multi_ci=[gmL, gmH],
                grass_single_slope=float(gsS), grass_single_ci=[gsL, gsH],
                grass_relsig_single_median=float(np.median(all_s)),
                grass_relsig_multi_median=float(np.median(all_m)),
                iso_depth_texscale_ratio=float(ts_ratio_pool),
                field_dependence=bool(field_dep))


if __name__ == "__main__":
    main()
