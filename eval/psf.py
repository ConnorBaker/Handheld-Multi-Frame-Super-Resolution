"""Slanted-edge ESF -> Gaussian edge-spread sigma (effective PSF width, in px).

The width of a step edge is the most direct measure of REAL resolution: a sharper
optical+reconstruction system renders a narrower edge. Unlike Laplacian-variance
"sharpness", edge sigma cannot be inflated by added texture or halos -- a halo
widens the fit residual, it does not narrow sigma. So comparing edge sigma across
reconstructions separates genuine resolution from hallucinated micro-contrast.

Method (standard slanted-edge / ISO-12233 style, hardened for noisy single
frames): smooth ALONG the edge, take the across-edge gradient, weight to the
gradient ridge only, robustly fit the (slightly slanted) edge line, project every
pixel to its signed distance from that line, bin into an oversampled edge-spread
function, and fit the ESF directly to an erf (the integral of a Gaussian). The
direct erf fit avoids differentiating to an LSF (which amplifies noise). Validated
by round-trip: recovers an injected Gaussian blur sigma >= ~0.8 px to +-0.02.
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def _erf_model(x, a0, b0, mu, s):
    return a0 + b0 * 0.5 * (1.0 + erf((x - mu) / (abs(s) * np.sqrt(2.0))))


def esf_sigma(roi, oversample=8, win=10.0, smooth_along=2.0):
    """Gaussian edge sigma (px) of a near-vertical high-contrast edge in `roi`.

    roi: 2D float, one dominant near-vertical step edge, roughly centred, with a
    slight slant (a few degrees off vertical) so rows sample sub-pixel phases.
    Returns (sigma_px, rms_residual) where rms is the erf-fit residual normalized
    by edge contrast (a halo/overshoot raises rms without lowering sigma)."""
    roi = np.asarray(roi, np.float64)
    h, w = roi.shape
    sm = gaussian_filter1d(roi, smooth_along, axis=0)          # denoise ALONG the edge
    gx = np.gradient(sm, axis=1)                               # across-edge gradient
    wts = gx ** 2
    wmask = wts * (wts > 0.25 * wts.max(axis=1, keepdims=True))  # gradient-ridge only
    xs = np.arange(w)
    pos = (wmask * xs).sum(1) / (wmask.sum(1) + 1e-12)         # per-row sub-pixel edge x
    ys = np.arange(h)
    valid = wmask.sum(1) > 0
    yy, pp = ys[valid].astype(np.float64), pos[valid]
    coef = np.array([0.0, w / 2.0])
    for _ in range(4):                                         # robust line fit pos ~ a*y + b
        A = np.vstack([yy, np.ones_like(yy)]).T
        coef, *_ = np.linalg.lstsq(A, pp, rcond=None)
        resid = pp - A @ coef
        keep = np.abs(resid) < 2.5 * (resid.std() + 1e-9)
        if keep.sum() < 8:
            break
        yy, pp = yy[keep], pp[keep]
    edge_x = coef[0] * ys[:, None] + coef[1]                  # edge x per row
    dist = (xs[None, :] - edge_x).ravel()                    # signed across-edge distance
    val = roi.ravel()
    bins = np.arange(-win, win + 1.0 / oversample, 1.0 / oversample)
    idx = np.digitize(dist, bins)
    esf = np.array([val[idx == k].mean() if np.any(idx == k) else np.nan
                    for k in range(1, len(bins))])
    centers = 0.5 * (bins[:-1] + bins[1:])
    good = np.isfinite(esf)
    centers, esf = centers[good], esf[good]
    if centers.size < 12:
        return np.nan, np.nan
    a0, b0 = esf.min(), esf.max() - esf.min()
    try:
        popt, _ = curve_fit(_erf_model, centers, esf, p0=[a0, b0, 0.0, 1.0], maxfev=20000)
    except Exception:
        return np.nan, np.nan
    rms = np.sqrt(np.mean((_erf_model(centers, *popt) - esf) ** 2)) / (abs(b0) + 1e-9)
    return float(abs(popt[3])), float(rms)


if __name__ == "__main__":
    # round-trip self-test: synthetic slanted edge, blur with known sigma, recover.
    H = W = 120
    yy, xx = np.mgrid[0:H, 0:W]
    edge = (xx > (W / 2 + 0.08 * (yy - H / 2))).astype(np.float64)  # slight slant
    from scipy.ndimage import gaussian_filter
    print("injected -> recovered (rms):")
    for s in (0.6, 0.8, 1.0, 1.4, 1.97):
        blur = gaussian_filter(edge, s)
        rng = np.random.default_rng(0)
        noisy = blur + rng.normal(0, 0.01, blur.shape)
        sig, rms = esf_sigma(noisy)
        print(f"  sigma={s:.2f} -> {sig:.3f} (rms {rms:.3f})")
