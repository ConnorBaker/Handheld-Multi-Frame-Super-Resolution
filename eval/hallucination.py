"""Separate REAL detail from HALLUCINATED detail in a burst of single-frame
DeepPRIME (DxO) renders, by cross-frame consensus.

We have N single-frame DxO renders of the SAME static scene at different
sub-pixel offsets (a handheld burst, fixed lens/aperture/sensor). DeepPRIME is a
single-image denoiser/demosaicer with a learned generative prior, so:

  * REAL scene detail is cross-frame CONSISTENT -- the same edges/texture appear
    in every render (modulo the sub-pixel shift), because the scene is static and
    the optics fixed. It survives a cross-frame consensus (mean/median over
    aligned frames).
  * HALLUCINATED detail is seeded by each frame's own noise/aliasing, so it is
    INCONSISTENT across renders. It is present in a single render but is DESTROYED
    by the consensus.

So on a registered stack the high-frequency (HF) power of a single render splits
orthogonally into a cross-frame-consistent part and a per-frame part:

    HF_f = highpass(frame_f);   consistent = mean_f(HF_f);   dev_f = HF_f - consistent
    real_fraction = power(consistent) / power(single)  in [0,1]

real_fraction is the share of a single DxO render's fine detail that is real.

WHAT THIS DOES AND DOES NOT CATCH (read before trusting a number):
  * Catches noise/aliasing-SEEDED invention (differs per frame).
  * Does NOT catch deterministic, scene-conditioned hallucination: DeepPRIME is a
    deterministic function, so if it invents the SAME plausible texture for the
    same blurry input every frame, that is cross-frame consistent and is counted
    as "real". => real_fraction is an UPPER bound on the real share.
  * "Inconsistent" mixes invention with residual noise. Distinguish them with
    ``inconsistency_structure`` (white spectrum => residual noise; structured /
    autocorrelated => synthesized texture).
  * In vegetation, wind motion is also inconsistent and is indistinguishable from
    invention here -- use a STATIC high-contrast target to separate them.

To catch the deterministic kind, cross-check against a prior-free reconstruction
(our multi-frame merge): it keeps only frame-consistent detail by construction,
so HF(merge) ~ HF(consensus) bounds the real-detail ceiling. ``compare_edge_sharpness``
adds the edge-width (ESF) check where the scene has clean step edges.

VALIDATION (see test/test_hallucination.py and ``resampling_floor``): the shift
convention round-trips; the resampling-only control scores real_fraction ~0.9-1.0
(so low values are genuine, not interpolation artifacts); and the decomposition
recovers a known consistent/invented power split.

Loading: DxO linear DNGs via eval.calibrate_optics.read_dng (scene-linear RGB,
NOT white-balanced). We analyse a gamma-corrected luma -- a fixed monotone
per-pixel transform, identical for every frame, so it cannot bias a cross-frame
*consistency* comparison, but it puts detail in the perceptual space DeepPRIME
works in (and where its hallucination lives).

CLI / reproduction: ``eval/_halluc_run.py`` (``python -m eval._halluc_run``)
drives the whole analysis -- drift, scene map + heatmap, resampling control,
noise-vs-invention structure, montages, and the merge/ESF cross-check.
"""

import numpy as np

from .calibrate_optics import read_dng

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_luma(path, gamma=2.2, wb=None):
    """DxO linear DNG -> gamma-corrected luma in [0,1].

    wb: optional per-channel (R,G,B) gains applied before luma (the DNG is raw
    scene-linear / green-dominant). Defaults to a gray-world balance so the luma
    is a sensible perceptual grey; the SAME gains are used for every frame.
    """
    rgb = read_dng(path)  # (H,W,3) linear [0,1]
    if wb is None:
        m = rgb.reshape(-1, 3).mean(0)
        wb = (m.mean() / np.maximum(m, 1e-6))
    rgb = np.clip(rgb * np.asarray(wb, np.float32), 0, 1)
    return np.clip(rgb @ LUMA, 0, 1) ** (1.0 / gamma)


def shared_wb(path):
    """Gray-world (R,G,B) gains from one frame, reused for the whole burst."""
    m = read_dng(path).reshape(-1, 3).mean(0)
    return (m.mean() / np.maximum(m, 1e-6)).astype(np.float32)


def load_luma_stack(paths, gamma=2.2, cache=None, wb="shared"):
    """Load N DxO DNGs to a (N,H,W) float32 gamma-luma stack (memmap-cached).

    wb="shared": compute one gray-world balance from the reference frame and use
    it for all (same scene+camera -> identical WB; keeps frames comparable).
    """
    import os
    paths = list(paths)
    if cache and os.path.exists(cache):
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[0] == len(paths):
            return arr
    gains = shared_wb(paths[0]) if wb == "shared" else None
    first = read_luma(paths[0], gamma, wb=gains)
    out = np.empty((len(paths),) + first.shape, np.float32)
    out[0] = first
    for i, p in enumerate(paths[1:], 1):
        out[i] = read_luma(p, gamma, wb=gains)
    if cache:
        np.save(cache, out)
    return out


def rgb_to_luma(rgb, gamma=2.2):
    """White-balanced linear RGB -> gamma luma (for an already-WB'd image such as
    our pipeline merge; matches the stack's luma space up to an affine level)."""
    return np.clip(np.asarray(rgb, np.float32) @ LUMA, 0, 1) ** (1.0 / gamma)


# ---------------------------------------------------------------------------
# Registration / drift
# ---------------------------------------------------------------------------

def pcc_shift(ref, mov, upsample=20):
    """Sub-pixel correction (dy,dx) that registers `mov` onto `ref`: applying
    scipy.ndimage.shift(mov, (dy,dx)) aligns mov to ref (verified by round-trip).

    Uses classic (magnitude-weighted) cross-correlation, NOT phase normalization:
    these DxO renders are denoised/smooth, and phase-whitening amplifies their
    (near-absent) high frequencies and fails to (0,0). Returns (dy, dx, err)
    where err is skimage's normalized RMS (lower = better)."""
    from skimage.registration import phase_cross_correlation
    shift, error, _ = phase_cross_correlation(
        ref.astype(np.float32), mov.astype(np.float32),
        upsample_factor=upsample, normalization=None)
    return float(shift[0]), float(shift[1]), float(error)


def _corner_windows(shape, win):
    H, W = shape
    m = win // 2
    cy, cx = [m, H - m], [m, W - m]
    return [(y, x) for y in cy for x in cx]  # TL, TR, BL, BR (y outer, x inner)


def measure_drift(stack, ref=0, win=1280, upsample=20):
    """Per-frame global translation + a rotation estimate, from 4 corner windows.

    For each frame we phase-correlate four corner windows against the reference's
    same windows. If all four shifts agree -> pure translation. A systematic
    difference across the horizontal/vertical baseline -> in-plane rotation; we
    fit the rotation angle from the differential shift over the corner baseline. A
    large corner spread with ~0 rotation indicates PARALLAX (the burst panned over
    a 3D scene), which is why downstream analysis registers PER ROI, not globally.
    Returns a list of dicts (one per frame).
    """
    H, W = stack.shape[1:]
    corners = _corner_windows((H, W), win)
    half = win // 2
    refw = [stack[ref][cy - half:cy + half, cx - half:cx + half] for cy, cx in corners]
    out = []
    for i in range(stack.shape[0]):
        shifts = []
        for (cy, cx), rw in zip(corners, refw):
            mw = stack[i][cy - half:cy + half, cx - half:cx + half]
            dy, dx, resp = pcc_shift(rw, mw, upsample)
            shifts.append((dy, dx, resp))
        sh = np.array([(s[0], s[1]) for s in shifts])  # (4,2): TL,TR,BL,BR
        resp = np.array([s[2] for s in shifts])
        baseline_y = corners[2][0] - corners[0][0]  # BL.y - TL.y
        baseline_x = corners[1][1] - corners[0][1]  # TR.x - TL.x
        ddx_dy = ((sh[2, 1] - sh[0, 1]) + (sh[3, 1] - sh[1, 1])) / 2 / baseline_y
        ddy_dx = ((sh[1, 0] - sh[0, 0]) + (sh[3, 0] - sh[2, 0])) / 2 / baseline_x
        theta = (ddx_dy - ddy_dx) / 2
        out.append(dict(
            frame=i, mean_dy=float(sh[:, 0].mean()), mean_dx=float(sh[:, 1].mean()),
            corner_dy=sh[:, 0].tolist(), corner_dx=sh[:, 1].tolist(),
            theta_deg=float(np.degrees(theta)), resp=float(resp.mean()),
            spread_px=float(np.hypot(np.ptp(sh[:, 0]), np.ptp(sh[:, 1]))),
        ))
    return out


# ---------------------------------------------------------------------------
# Per-ROI alignment (handles the burst's large pan + parallax: each ROI finds
# its OWN shift in each frame, so different depths register independently).
# ---------------------------------------------------------------------------

def _align_window(ref_roi, img, cy, cx, size, search, upsample):
    """Align `img`'s content at (cy,cx) to `ref_roi` (the template): locate it in
    a (size+2*search) window by normalized cross-correlation (robust to the large
    per-ROI displacement from pan+parallax), then refine to sub-pixel and resample.
    Returns (aligned size x size crop or None, info dict)."""
    import cv2
    from scipy.ndimage import shift as ndshift
    h = size // 2
    y0, x0 = cy - h - search, cx - h - search
    win = img[y0:y0 + size + 2 * search, x0:x0 + size + 2 * search]
    if win.shape[:2] != (size + 2 * search, size + 2 * search):
        return None, dict(peak=0.0, err=1.0, oob=True)
    cc = cv2.matchTemplate(np.ascontiguousarray(win), np.ascontiguousarray(ref_roi),
                           cv2.TM_CCORR_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(cc)  # loc = (px, py) top-left of best match
    px, py = loc
    matched = win[py:py + size, px:px + size]
    ry, rx, err = pcc_shift(ref_roi, matched, upsample)
    aligned = ndshift(matched, (ry, rx), order=3, mode="reflect")
    return aligned, dict(oy=int(py - search), ox=int(px - search),
                         ry=ry, rx=rx, peak=float(peak), err=float(err))


def align_to(ref_roi, img, cy, cx, size, search=140, upsample=20):
    """Crop `img` around (cy,cx), sub-pixel-aligned to ref_roi (or None if OOB).
    Used to bring an INDEPENDENT image (e.g. our pipeline merge) onto the DxO grid."""
    return _align_window(ref_roi, img, cy, cx, size, search, upsample)[0]


def extract_aligned_roi(stack, cy, cx, size=256, search=96, upsample=20, ref=0):
    """Sub-pixel-aligned (N,size,size) ROI stack centred at (cy,cx) in the
    reference frame. Returns (rois, info); info[i] carries the NCC match peak
    (a low peak flags a depth-discontinuity / occlusion ROI that did not register
    cleanly) and the residual sub-pixel shift."""
    h = size // 2
    ref_roi = np.ascontiguousarray(stack[ref][cy - h:cy + h, cx - h:cx + h])
    rois = np.empty((stack.shape[0], size, size), np.float32)
    info = []
    for i in range(stack.shape[0]):
        if i == ref:
            rois[i] = ref_roi
            info.append(dict(frame=i, oy=0, ox=0, ry=0.0, rx=0.0, peak=1.0, err=0.0))
            continue
        aligned, meta = _align_window(ref_roi, stack[i], cy, cx, size, search, upsample)
        rois[i] = ref_roi if aligned is None else aligned
        info.append(dict(frame=i, **meta))
    return rois, info


# ---------------------------------------------------------------------------
# Consensus decomposition: consistent (REAL) vs inconsistent (invented + noise)
# ---------------------------------------------------------------------------

def _highpass(img, sigma=1.6):
    import cv2
    return img - cv2.GaussianBlur(img.astype(np.float32), (0, 0), float(sigma))


def consensus_decomp(rois, info=None, peak_min=0.0, hp_sigma=1.6, trim=0.1):
    """Decompose an aligned ROI stack into consistent (REAL) vs inconsistent
    (invented + residual noise) high-frequency power.

    HF_f = highpass(frame_f). The cross-frame MEAN of HF is the consistent
    component (orthogonal variance split): total single-frame HF power =
    consistent + inconsistent. real_fraction = consistent / total in [0,1]. Also
    returns the trimmed-mean consensus image (robust real-detail estimate), the
    per-pixel inconsistency map (hallucination) and the consistent-detail map."""
    N = rois.shape[0]
    keep = np.ones(N, bool)
    if info is not None and peak_min > 0:
        keep = np.array([d.get("peak", 0.0) >= peak_min for d in info])
    sel = rois[keep]
    k = max(0, int(sel.shape[0] * trim / 2))                   # trimmed-mean consensus
    srt = np.sort(sel, axis=0)
    consensus = srt[k:srt.shape[0] - k].mean(0) if srt.shape[0] - 2 * k > 0 else srt.mean(0)
    hp = np.stack([_highpass(f, hp_sigma) for f in sel])      # (n,sz,sz)
    hp_mean = hp.mean(0)                                       # consistent HF
    hp_dev = hp - hp_mean                                      # per-frame inconsistency
    consistent = float((hp_mean ** 2).mean())
    inconsistent = float((hp_dev ** 2).mean())
    return dict(
        consensus=consensus, n_used=int(sel.shape[0]),
        consistent_power=consistent, inconsistent_power=inconsistent,
        real_fraction=consistent / (consistent + inconsistent + 1e-12),
        single_hf_rms=float(np.sqrt((hp ** 2).mean())),
        consensus_hf_rms=float(np.sqrt((hp_mean ** 2).mean())),
        halluc_map=hp_dev.std(0),                             # per-pixel inconsistency (HALLUCINATION)
        real_map=np.abs(_highpass(consensus, hp_sigma)),      # per-pixel consistent detail (REAL)
    )


def resampling_floor(base_roi, n=41, seed=0, hp_sigma=1.6):
    """Control: real_fraction for n copies of ONE image, each round-trip shifted
    by a random sub-pixel amount (content identical, only cubic resampling
    differs). Should be ~1.0 if resampling is harmless; this is the ceiling the
    real measurement is read against (a low real_fraction below this floor is a
    genuine cross-frame difference, not an interpolation artifact)."""
    from scipy.ndimage import shift as ndshift
    rng = np.random.default_rng(seed)
    base = np.ascontiguousarray(base_roi, np.float32)
    rois = np.empty((n,) + base.shape, np.float32)
    rois[0] = base
    for i in range(1, n):
        s = rng.uniform(-0.5, 0.5, 2)
        rois[i] = ndshift(ndshift(base, s, order=3, mode="reflect"), -s, order=3, mode="reflect")
    return consensus_decomp(rois, hp_sigma=hp_sigma)["real_fraction"]


def inconsistency_structure(rois, hp_sigma=1.6):
    """Is the inconsistent (per-frame) HF component residual NOISE or invented
    TEXTURE? Returns the lag-1 spatial autocorrelation of the per-frame deviation
    (~0 => white noise; >0 => spatially structured => synthesized texture), the
    low/high radial-spectrum band ratio, and the spectral peak scale (px)."""
    hp = np.stack([_highpass(f, hp_sigma) for f in rois])
    dev = hp - hp.mean(0)                                      # per-frame inconsistency
    ac1 = float(np.mean([np.corrcoef(dev[i][:, :-1].ravel(), dev[i][:, 1:].ravel())[0, 1]
                         for i in range(len(dev))]))
    sz = dev.shape[1]
    P = np.mean([np.abs(np.fft.fftshift(np.fft.fft2(dev[i]))) ** 2 for i in range(len(dev))], 0)
    yy, xx = np.mgrid[0:sz, 0:sz]
    rr = np.hypot(yy - sz // 2, xx - sz // 2).astype(int)
    radial = np.array([P[rr == k].mean() for k in range(1, sz // 2)])
    radial = radial / (radial.max() + 1e-12)
    pk = int(np.argmax(radial)) + 1
    third = len(radial) // 3
    return dict(
        lag1_autocorr=ac1,
        low_hi_band_ratio=float(radial[:third].mean() / (radial[third:].mean() + 1e-9)),
        peak_scale_px=float(sz / (pk * 2)) if pk > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Scene-wide map
# ---------------------------------------------------------------------------

def scene_map(stack, size=256, stride=256, search=120, peak_min=0.5, hp_sigma=1.6,
              progress=None):
    """Tile the frame; per tile run consensus_decomp. Returns grids (real_fraction,
    single-frame HF rms, consensus HF rms, match peak), tile origins, and the
    GLOBAL real_fraction (HF-power-weighted over all tiles) = sum(consistent)
    / sum(consistent+inconsistent). `progress(iy, n)` is called per tile row."""
    H, W = stack.shape[1:]
    pad = search + size // 2 + 4
    ys = list(range(pad, H - pad + 1, stride))
    xs = list(range(pad, W - pad + 1, stride))
    rf = np.full((len(ys), len(xs)), np.nan)
    single = np.zeros_like(rf)
    cons = np.zeros_like(rf)
    peak = np.zeros_like(rf)
    sum_c = sum_i = 0.0
    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            rois, info = extract_aligned_roi(stack, cy, cx, size=size, search=search)
            d = consensus_decomp(rois, info, hp_sigma=hp_sigma)
            rf[iy, ix] = d["real_fraction"]
            single[iy, ix] = d["single_hf_rms"]
            cons[iy, ix] = d["consensus_hf_rms"]
            peak[iy, ix] = float(np.median([k.get("peak", 0.0) for k in info]))
            sum_c += d["consistent_power"]
            sum_i += d["inconsistent_power"]
        if progress:
            progress(iy + 1, len(ys))
    w = (single ** 2).ravel()
    ok = np.isfinite(rf.ravel()) & (peak.ravel() > peak_min)
    return dict(
        ys=np.array(ys), xs=np.array(xs), size=size,
        grid_rf=rf, grid_single=single, grid_cons=cons, grid_peak=peak,
        global_real_fraction=float(sum_c / (sum_c + sum_i + 1e-12)),
        weighted_real_fraction=float((w[ok] * rf.ravel()[ok]).sum() / (w[ok].sum() + 1e-12)),
        n_tiles=int(ok.sum()),
    )


# ---------------------------------------------------------------------------
# Edge-width (ESF) cross-check -- only where the scene HAS clean step edges
# ---------------------------------------------------------------------------

def find_step_edges(img, sz=140, bw=60, stride=80, margin=400, min_stepness=6.0,
                    min_contrast=0.06, max_edges=8, min_sep=200):
    """Locate near-vertical STEP edges (two flat levels split by one transition --
    NOT a thin bar, which breaks the single-erf ESF model). stepness = across-edge
    contrast / within-side std. Returns greedy non-overlapping (cy,cx,stepness,
    contrast), strongest first; empty if the scene has none (e.g. natural scenes
    of foliage/sky), in which case the ESF check is legitimately N/A."""
    H, W = img.shape
    cands = []
    for cy in range(margin, H - margin, stride):
        for cx in range(margin, W - margin, stride):
            win = img[cy - sz // 2:cy + sz // 2, cx - bw // 2:cx + bw // 2]
            w = win.shape[1]
            left, right = win[:, :w // 2 - 4], win[:, w // 2 + 4:]
            contrast = abs(right.mean() - left.mean())
            within = 0.5 * (left.std() + right.std())
            if contrast > min_contrast and contrast / (within + 1e-6) > min_stepness:
                cands.append((contrast / (within + 1e-6), float(contrast), cy, cx))
    cands.sort(reverse=True)
    picks = []
    for sc, con, cy, cx in cands:
        if all(abs(cy - p[0]) > min_sep or abs(cx - p[1]) > min_sep for p in picks):
            picks.append((cy, cx, sc, con))
        if len(picks) >= max_edges:
            break
    return picks


def compare_edge_sharpness(stack, merge_luma, edges, size=160, band=60,
                           frames=(0, 8, 16, 24, 32, 40), search=140,
                           rms_max=0.04, std_max=0.15):
    """Edge sigma (PSF width, px; smaller = sharper REAL resolution) for
    DxO-single / DxO-consensus / our-merge on each step edge, gated by ESF fit
    reliability (low erf residual + low per-frame variance). Returns a dict with
    per-edge rows and a `conclusion`; if no edge passes the gate the verdict is
    'inconclusive' (do not invent a sharpness claim)."""
    from .psf import esf_sigma
    rows, reliable = [], []
    for cy, cx, *_ in edges:
        rois, info = extract_aligned_roi(stack, cy, cx, size=size, search=search)
        d = consensus_decomp(rois, info)
        sl = slice(size // 2 - band // 2, size // 2 + band // 2)
        bandf = lambda im: im[:, sl]
        ss = [esf_sigma(bandf(rois[i])) for i in frames]
        ss = np.array([s for s, r in ss if np.isfinite(s) and r < rms_max and 0.3 < s < 4])
        s_cons, r_cons = esf_sigma(bandf(d["consensus"]))
        mg = align_to(rois[0], merge_luma, cy, cx, size, search) if merge_luma is not None else None
        s_mg, r_mg = esf_sigma(bandf(mg)) if mg is not None else (np.nan, 1.0)
        ok = (ss.size >= 4 and ss.std() < std_max and r_cons < rms_max
              and np.isfinite(s_mg) and r_mg < 1.5 * rms_max)
        row = dict(cy=cy, cx=cx, single=float(ss.mean()) if ss.size else float("nan"),
                   single_std=float(ss.std()) if ss.size else float("nan"),
                   consensus=float(s_cons), merge=float(s_mg), reliable=bool(ok))
        rows.append(row)
        if ok:
            reliable.append(row)
    if reliable:
        R = np.array([(r["consensus"], r["merge"]) for r in reliable])
        conclusion = (f"{len(reliable)} reliable edges: DxO-consensus sigma median "
                      f"{np.median(R[:, 0]):.2f}px, our-merge {np.median(R[:, 1]):.2f}px; "
                      f"merge sharper on {(R[:, 1] < R[:, 0]).mean() * 100:.0f}%")
    else:
        conclusion = "inconclusive -- no clean step edge passed the reliability gate"
    return dict(rows=rows, n_reliable=len(reliable), conclusion=conclusion)
