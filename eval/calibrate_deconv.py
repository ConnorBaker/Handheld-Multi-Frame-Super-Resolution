"""Per-burst auto-calibration of the measured-PSF deconvolution.

The deconv needs to know the merge's effective PSF width (sigma). Rather than a
hand-set constant (which the adversarial review showed doesn't port across
cameras/scenes), estimate it PER BURST from the merge's own clean step edges --
the same ESF estimator used for the analysis -- with the verify-the-measurement
discipline baked in (eval/CLAUDE.md):

  * measure the DISTRIBUTION over many edges, report median AND spread (not 1 edge);
  * GATE on fit reliability (drop bad erf fits) BEFORE quoting;
  * if too few reliable edges (e.g. a foliage/sky scene with no clean steps),
    return low confidence so the caller falls back to a per-camera calibration --
    NOT a fabricated number.

This is deliberately NOT a radial/parametric PSF field (verified non-radial on the
a6300: corr(radius,sigma)=-0.58). It is a single robust scalar; spatial adaptivity
comes from the per-pixel structure-tensor GATE in the deconv, not from this number.
"""

import numpy as np

from .hallucination import find_step_edges
from .psf import esf_sigma


def auto_merge_sigma(luma, rms_max=0.04, smin=0.8, smax=4.0, min_edges=3,
                     roi=160, band=60, fallback=None):
    """Estimate the merge's effective PSF sigma (px) from its own step edges.

    luma: (H,W) float32 gamma luma of the merge. Returns (sigma, info). sigma is
    the median over reliably-fit edges; if fewer than `min_edges` pass, returns
    `fallback` with confidence='low' (caller should use a per-camera constant)."""
    luma = np.ascontiguousarray(luma, np.float32)
    edges = find_step_edges(luma)
    h, lo = roi // 2, roi // 2 - band // 2
    sl = slice(lo, lo + band)
    sigs = []
    for cy, cx, _, _ in edges:
        sub = luma[cy - h:cy + h, cx - h:cx + h]
        if sub.shape != (roi, roi):
            continue
        s, r = esf_sigma(sub[:, sl])
        if np.isfinite(s) and r < rms_max and smin < s < smax:
            sigs.append(s)
    sigs = np.array(sigs)
    if sigs.size >= min_edges:
        return float(np.median(sigs)), dict(
            confidence="high", n_reliable=int(sigs.size), n_edges=len(edges),
            median=float(np.median(sigs)), spread=float(sigs.std()),
            range=[float(sigs.min()), float(sigs.max())])
    return fallback, dict(confidence="low", n_reliable=int(sigs.size),
                          n_edges=len(edges), note="too few clean edges; use per-camera calibration")


def calibrate(merge_rgb_or_luma, target_frac=None, fallback_sigma=2.0):
    """Per-burst deconv calibration. Returns a dict ready for the measured_rl path:
    the estimated merge sigma, and -- if a target sharpening fraction is given --
    the Gaussian `deconv_sigma` to remove (sqrt(s^2 - (frac*s)^2)). With no target,
    the deconv removes the full measured blur (regularized by iters + gate)."""
    from .deconv import partial_gaussian_sigma
    luma = merge_rgb_or_luma if merge_rgb_or_luma.ndim == 2 else \
        np.clip(merge_rgb_or_luma @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1)
    sigma, info = auto_merge_sigma(luma, fallback=None)
    used = sigma if sigma is not None else fallback_sigma
    out = dict(merge_sigma=sigma, sigma_used=float(used), info=info,
               source="measured" if sigma is not None else "fallback")
    if target_frac is not None:
        out["deconv_sigma"] = float(partial_gaussian_sigma(used, target_frac * used))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-calibrate deconv sigma from a merge .npy")
    ap.add_argument("merge_npy", help="merge linear-RGB or gamma-luma .npy")
    ap.add_argument("--target-frac", type=float, default=None)
    args = ap.parse_args()
    m = np.load(args.merge_npy)
    print(calibrate(m, target_frac=args.target_frac))
