"""STEP 1 (decisive gate): measure the optical edge sigma across the FIELD.

Question this answers: is the lens PSF significantly wider toward the CORNERS than
at the CENTER on our a6300 APS-C crop? If not, a spatially-varying (field-dependent)
deconvolution (Lever A) cannot pay off and we should stop.

Discipline (eval/CLAUDE.md):
  * DISTRIBUTION not anecdote -- many edges, median AND spread per field bin.
  * RELIABILITY-GATE first -- drop bad erf fits (high rms), implausible sigma,
    low contrast, before quoting anything.
  * BOTH ORIENTATIONS -- vertical edges give sigma_x (tangential at left/right,
    sagittal at top/bottom); horizontal edges give sigma_y. Astigmatism shows up
    as sigma_x != sigma_y growing toward the corners.
  * STRATIFY by region -- center / mid / corner radial bins, plus a 2D field map.

Edge source per location: a few individual DxO frames spread across the burst,
median sigma + across-frame spread (the spread is a second reliability signal; a
location whose frames disagree is dropped). This avoids a full 28-frame per-edge
alignment while still being a distribution, not one number. (A consensus-image
cross-check is run separately at a handful of edges in _psf_field_validate.)

Run: micromamba run -n handheld python -m eval._psf_field_measure
"""

import sys
import time

import numpy as np

from .psf import esf_sigma

CACHE = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_007/luma_stack.npy"


# ---------------------------------------------------------------------------
# Dense oriented step-edge finder (denser than hallucination.find_step_edges,
# and returns BOTH orientations with the same stepness metric).
# ---------------------------------------------------------------------------

def _stepness(win, axis):
    """Across-`axis` step contrast / within-side std for a window. axis=1 => the
    step runs vertically (intensity changes along x): a NEAR-VERTICAL edge.
    axis=0 => near-horizontal edge."""
    if axis == 1:
        w = win.shape[1]
        left, right = win[:, : w // 2 - 4], win[:, w // 2 + 4:]
    else:
        h = win.shape[0]
        left, right = win[: h // 2 - 4, :], win[h // 2 + 4:, :]
    contrast = abs(float(right.mean()) - float(left.mean()))
    within = 0.5 * (float(left.std()) + float(right.std()))
    return contrast, within


def find_oriented_edges(img, sz=120, bw=44, stride=60, margin=70,
                        min_stepness=6.0, min_contrast=0.05, min_sep=80):
    """Greedy non-overlapping near-vertical AND near-horizontal step edges over a
    DENSE grid covering the WHOLE frame (incl. corners). Returns a list of
    (cy, cx, orient, stepness, contrast); orient in {'V','H'} (V = vertical edge ->
    sigma_x; H = horizontal edge -> sigma_y)."""
    H, W = img.shape
    cands = []
    for cy in range(margin, H - margin, stride):
        for cx in range(margin, W - margin, stride):
            # vertical edge: tall-thin window (step across x)
            wv = img[cy - sz // 2:cy + sz // 2, cx - bw // 2:cx + bw // 2]
            if wv.shape == (sz, bw):
                c, wi = _stepness(wv, axis=1)
                if c > min_contrast and c / (wi + 1e-6) > min_stepness:
                    cands.append((c / (wi + 1e-6), c, cy, cx, "V"))
            # horizontal edge: wide-short window (step across y)
            wh = img[cy - bw // 2:cy + bw // 2, cx - sz // 2:cx + sz // 2]
            if wh.shape == (bw, sz):
                c, wi = _stepness(wh, axis=0)
                if c > min_contrast and c / (wi + 1e-6) > min_stepness:
                    cands.append((c / (wi + 1e-6), c, cy, cx, "H"))
    cands.sort(reverse=True)
    picks = []
    for sc, con, cy, cx, orient in cands:
        if all(orient != p[2] or abs(cy - p[0]) > min_sep or abs(cx - p[1]) > min_sep
               for p in picks):
            picks.append((cy, cx, orient, sc, con))
    return picks


# ---------------------------------------------------------------------------
# Per-edge sigma across a few frames, with reliability gating.
# ---------------------------------------------------------------------------

def edge_sigma_multiframe(stack, cy, cx, orient, frames, sz=120, band=44,
                          rms_max=0.05, smin=0.5, smax=4.5, frame_std_max=0.30):
    """Median ESF sigma over `frames` at one edge (no inter-frame alignment: each
    frame is measured in place -- the burst's sub-pixel shift only re-phases the
    ESF, it does not change the optical width). Returns (sigma, info) with a
    reliability flag. A V edge is measured raw (-> sigma_x); an H edge is measured
    transposed (-> sigma_y)."""
    h, b = sz // 2, band // 2
    sigs = []
    for f in frames:
        if orient == "V":
            sub = stack[f][cy - h:cy + h, cx - b:cx + b]
        else:
            sub = stack[f][cy - b:cy + b, cx - h:cx + h].T   # transpose -> across-edge = rows->cols
        if sub.shape[0] < 12 or sub.shape[1] < 8:
            continue
        s, r = esf_sigma(np.ascontiguousarray(sub))
        if np.isfinite(s) and r < rms_max and smin < s < smax:
            sigs.append(s)
    sigs = np.array(sigs)
    if sigs.size < max(3, len(frames) // 2):
        return np.nan, dict(n=int(sigs.size), reliable=False, reason="too few good frames")
    med, std = float(np.median(sigs)), float(sigs.std())
    reliable = std < frame_std_max
    return med, dict(n=int(sigs.size), median=med, std=std,
                     reliable=reliable, reason="" if reliable else "frame disagreement")


# ---------------------------------------------------------------------------
# Field map
# ---------------------------------------------------------------------------

def measure_field(stack, frames=None, edges=None, **edge_kw):
    """Measure sigma at every reliable edge; return per-edge records with field
    position. `stack` may be an (N,H,W) burst or a single (H,W) image (wrapped to
    N=1, in which case frame-disagreement gating is disabled)."""
    if stack.ndim == 2:
        stack = stack[None]
    N, H, W = stack.shape
    if frames is None:
        frames = list(range(0, N, max(1, N // 6)))[:6] if N > 1 else [0]
    f0 = np.ascontiguousarray(stack[0])
    if edges is None:
        edges = find_oriented_edges(f0)
    cy0, cx0 = H / 2.0, W / 2.0
    rmax = float(np.hypot(cy0, cx0))
    recs = []
    for cy, cx, orient, sc, con in edges:
        if N == 1:
            sub = (f0[cy - 60:cy + 60, cx - 22:cx + 22] if orient == "V"
                   else f0[cy - 22:cy + 22, cx - 60:cx + 60].T)
            s, r = esf_sigma(np.ascontiguousarray(sub))
            info = dict(n=1, reliable=bool(np.isfinite(s) and r < 0.05 and 0.5 < s < 4.5),
                        std=0.0, median=float(s) if np.isfinite(s) else np.nan)
        else:
            s, info = edge_sigma_multiframe(stack, cy, cx, orient, frames, **edge_kw)
        r = float(np.hypot(cy - cy0, cx - cx0))
        recs.append(dict(cy=cy, cx=cx, orient=orient, r=r, r_norm=r / rmax,
                         sigma=s, stepness=float(sc), contrast=float(con), **info))
    return recs


def _bin_stats(recs, key="sigma", reliable_only=True):
    """median/spread/n of `key` in center/mid/corner radial bins (r_norm thirds)."""
    bins = {"center": [], "mid": [], "corner": []}
    for d in recs:
        if reliable_only and not d.get("reliable", False):
            continue
        if not np.isfinite(d[key]):
            continue
        rn = d["r_norm"]
        b = "center" if rn < 1 / 3 else ("mid" if rn < 2 / 3 else "corner")
        bins[b].append(d[key])
    out = {}
    for b, v in bins.items():
        v = np.array(v)
        out[b] = dict(n=len(v), median=float(np.median(v)) if len(v) else np.nan,
                      mean=float(v.mean()) if len(v) else np.nan,
                      std=float(v.std()) if len(v) else np.nan,
                      p25=float(np.percentile(v, 25)) if len(v) else np.nan,
                      p75=float(np.percentile(v, 75)) if len(v) else np.nan)
    return out


def report(recs, title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    nrel = sum(d.get("reliable", False) for d in recs)
    print(f"edges scanned: {len(recs)}   reliable (gated): {nrel}")
    V = [d for d in recs if d["orient"] == "V" and d.get("reliable")]
    Hh = [d for d in recs if d["orient"] == "H" and d.get("reliable")]
    print(f"  reliable V (sigma_x): {len(V)}   reliable H (sigma_y): {len(Hh)}")

    print("\n-- radial bins (reliable only), sigma in px --")
    print(f"{'bin':>8} | {'n':>3} | {'median':>7} | {'mean':>6} | {'std':>5} | {'IQR':>13}")
    st = _bin_stats(recs)
    for b in ["center", "mid", "corner"]:
        s = st[b]
        iqr = f"{s['p25']:.2f}-{s['p75']:.2f}" if s["n"] else "  -  "
        print(f"{b:>8} | {s['n']:>3} | {s['median']:>7.3f} | {s['mean']:>6.3f} | "
              f"{s['std']:>5.3f} | {iqr:>13}")

    # field-dependence magnitude + significance
    cen = [d["sigma"] for d in recs if d.get("reliable") and d["r_norm"] < 1 / 3]
    cor = [d["sigma"] for d in recs if d.get("reliable") and d["r_norm"] >= 2 / 3]
    if len(cen) >= 4 and len(cor) >= 4:
        cen, cor = np.array(cen), np.array(cor)
        from scipy import stats as ss
        try:
            U, p = ss.mannwhitneyu(cor, cen, alternative="greater")
        except Exception:
            p = float("nan")
        d_cohen = (cor.mean() - cen.mean()) / (np.sqrt(0.5 * (cor.var() + cen.var())) + 1e-9)
        print(f"\n-- center vs corner significance --")
        print(f"  center median {np.median(cen):.3f}px  corner median {np.median(cor):.3f}px  "
              f"delta {np.median(cor)-np.median(cen):+.3f}px")
        print(f"  Mann-Whitney U (corner>center) p={p:.4f}   Cohen's d={d_cohen:+.2f}")
    else:
        print(f"\n-- center vs corner: too few reliable edges in a bin "
              f"(center={len(cen)}, corner={len(cor)}) --")

    # astigmatism: sigma_x vs sigma_y per radial bin
    print("\n-- astigmatism (sigma_x from V edges vs sigma_y from H edges) --")
    print(f"{'bin':>8} | {'sigma_x med (n)':>16} | {'sigma_y med (n)':>16} | {'|dx-dy|':>7}")
    for b, (lo, hi) in [("center", (0, 1 / 3)), ("mid", (1 / 3, 2 / 3)), ("corner", (2 / 3, 9))]:
        sx = np.array([d["sigma"] for d in V if lo <= d["r_norm"] < hi])
        sy = np.array([d["sigma"] for d in Hh if lo <= d["r_norm"] < hi])
        mx = np.median(sx) if sx.size else np.nan
        my = np.median(sy) if sy.size else np.nan
        diff = abs(mx - my) if (sx.size and sy.size) else np.nan
        print(f"{b:>8} | {mx:>7.3f} ({sx.size:>3})    | {my:>7.3f} ({sy.size:>3})    | {diff:>7.3f}")


def main():
    t0 = time.time()
    stack = np.load(CACHE, mmap_mode="r")
    print(f"DxO stack {stack.shape} loaded ({time.time()-t0:.1f}s)")
    recs = measure_field(stack)
    report(recs, "DxO single-frame stack -- optical edge sigma vs field position")
    # persist for the validate script / plotting
    out = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_007/_psf_field_dxo.npy"
    np.save(out, np.array(recs, dtype=object), allow_pickle=True)
    print(f"\nsaved per-edge records -> {out}   ({time.time()-t0:.1f}s total)")
    return recs


if __name__ == "__main__":
    main()
