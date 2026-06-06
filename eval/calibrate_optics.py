"""Calibrate the synthetic forward model's optical degradation from a real lens.

Given two renderings of the SAME raw -- a plain demosaic (which still carries the
lens's blur + lateral CA) and an optics-corrected render (e.g. DxO PhotoLab:
lens-measured CA realignment + lens-sharpness deconvolution) -- recover the lens's

  * lateral-CA corner shift (px), per channel, and
  * an effective Gaussian PSF sigma (px),

ready to drop into ``synthetic.apply_optics`` / ``sweep_synthetic.OPTICS`` so the
benchmark's degradation matches a real lens instead of a provisional guess. See
the ``dxo-reference-eval`` note for how to export the pair (crucially: DxO with
**distortion OFF**, so both renders share the same pixel grid).

Both estimates are robust 1-parameter fits -- a radial magnification for CA, a
Gaussian blur for the PSF -- chosen over free per-tile shifts / phase correlation
because they match what ``apply_optics`` actually applies and recover a known
injected degradation exactly (see ``test/test_dxo_eval.py``).
"""

import argparse
import json

import numpy as np
import cv2


def _ncc(a, b):
    """Zero-mean normalized cross-correlation (1.0 = identical structure, scale-free)."""
    a = a.ravel().astype(np.float64); a = a - a.mean()
    b = b.ravel().astype(np.float64); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _scale_about_center(img, s):
    """Magnify ``img`` by ``s`` about its centre (s>1 zooms in); cubic + reflect."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), 0, float(s))
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _refine(score, lo, hi, *, n=49, rounds=3, maximize=True):
    """Coarse-to-fine 1-D search of ``score`` over [lo, hi]; return the best x."""
    pick = max if maximize else min
    step = (hi - lo) / (n - 1)
    best = pick(np.linspace(lo, hi, n), key=score)
    for _ in range(rounds - 1):
        best = pick(np.linspace(best - step, best + step, n), key=score)
        step = 2 * step / (n - 1)
    return float(best)


def _center_slice(shape, crop):
    h, w = shape[:2]
    return (slice(int(h * crop), int(h * (1 - crop))), slice(int(w * crop), int(w * (1 - crop))))


def fit_ca_shift(plain_ch, ref_ch, dmax=8.0, crop=0.12):
    """Corner lateral-CA shift (px) of ``plain_ch`` relative to ``ref_ch``.

    Models the CA as a radial magnification and finds the corner shift ``d`` such
    that de-magnifying plain by ``1 + d/rmax`` best matches ref (max NCC). Positive
    => plain is magnified outward vs ref (the realignment DxO would apply inward).
    """
    h, w = ref_ch.shape
    rmax = np.hypot((h - 1) / 2.0, (w - 1) / 2.0)
    sl = _center_slice(ref_ch.shape, crop)
    ref_c = ref_ch[sl]

    def score(d):
        return _ncc(_scale_about_center(plain_ch, 1.0 / (1.0 + d / rmax))[sl], ref_c)

    return _refine(score, -dmax, dmax, maximize=True)


def fit_blur_sigma(plain_gray, ref_gray, smax=5.0, crop=0.12):
    """Effective Gaussian PSF sigma (px): the blur taking the sharp ref to plain.

    Finds sigma minimizing ||GaussianBlur(ref, sigma) - plain||^2 (ref = sharp /
    corrected, plain = blurred). A field-varying blur collapses to its average.
    """
    sl = _center_slice(ref_gray.shape, crop)
    plain_c = plain_gray[sl]

    def score(s):
        blurred = ref_gray if s < 0.05 else cv2.GaussianBlur(ref_gray, (0, 0), float(s))
        return float(((blurred[sl] - plain_c) ** 2).mean())

    return _refine(score, 0.0, smax, maximize=False)


def calibrate(plain_rgb, corrected_rgb):
    """Recover (ca_px, psf_sigma) for ``apply_optics`` from a plain vs corrected pair.

    plain_rgb     : demosaic with NO optical corrections (blurred + CA), RGB [0,1].
    corrected_rgb : the SAME raw, lens-corrected (CA realigned + deblurred), same grid.

    Returns a dict including an ``OPTICS`` block ready to paste into
    ``sweep_synthetic.OPTICS``. CA is estimated per channel (R/B) as a radial
    magnification; the PSF as a single effective Gaussian sigma on luma.
    """
    plain = np.clip(plain_rgb, 0, 1).astype(np.float32)
    corr = np.clip(corrected_rgb, 0, 1).astype(np.float32)
    if plain.shape != corr.shape:
        raise ValueError(f"plain {plain.shape} and corrected {corr.shape} must be the same grid "
                         "(export DxO with distortion OFF)")
    ca_r = fit_ca_shift(plain[..., 0], corr[..., 0])
    ca_b = fit_ca_shift(plain[..., 2], corr[..., 2])
    sigma = fit_blur_sigma(plain.mean(2), corr.mean(2))
    ca_px = float((abs(ca_r) + abs(ca_b)) / 2.0)
    return dict(
        ca_R=round(ca_r, 3), ca_B=round(ca_b, 3), ca_px=round(ca_px, 3),
        psf_sigma=round(sigma, 3),
        # faithful (asymmetric) injection -- real lenses split R/B unevenly
        OPTICS=dict(psf_sigma=round(sigma, 3), psf_sigma_corner=round(sigma, 3),
                    ca_R=round(ca_r, 3), ca_B=round(ca_b, 3)),
    )


def read_dng(path):
    """Read a DEMOSAICED (linear) DNG -- e.g. DxO's linear-DNG export -- as RGB [0,1].

    DxO stores the full-res linear RGB as a JPEG-compressed uint16 TIFF IFD; rawpy
    (libraw) decompresses it to ``raw_image`` (H,W,4) with the 4th plane empty, so we
    take [R,G,B] / white_level. (tifffile needs the optional ``imagecodecs`` package
    for that JPEG, and imageio grabs the thumbnail IFD -- rawpy is the dependency-free
    path; verified on DxO ILCE-6300 linear DNGs.)
    """
    import rawpy
    with rawpy.imread(path) as raw:
        ri = raw.raw_image
        if ri.ndim != 3:
            raise ValueError(f"{path}: expected a demosaiced/linear DNG, got a {ri.ndim}D raw "
                             "(Bayer DNG). Export DxO as a linear DNG.")
        a = ri[..., :3].astype(np.float32) / float(raw.white_level)
    return np.clip(a, 0, 1)


def _load(path):
    """Load RGB [0,1]: .dng (linear, via rawpy), .npy, or imageio (8/16-bit TIFF/PNG)."""
    if path.lower().endswith(".dng"):
        return read_dng(path)
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        import imageio.v3 as iio
        a = iio.imread(path)
    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a / 255.0
    elif a.dtype == np.uint16:
        a = a / 65535.0
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return np.clip(a[..., :3], 0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plain", help="plain demosaic, NO optical corrections (.npy/.tif/.png)")
    ap.add_argument("corrected", help="same raw, lens-corrected (CA + sharpness), same grid")
    args = ap.parse_args()
    result = calibrate(_load(args.plain), _load(args.corrected))
    print(json.dumps(result, indent=2))
    print("\nPaste into eval/sweep_synthetic.py:\n  OPTICS =", result["OPTICS"])
