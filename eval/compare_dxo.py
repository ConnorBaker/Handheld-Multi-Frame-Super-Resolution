"""Compare a pipeline output against a DxO PhotoLab reference, per degradation.

DxO is a lens-MEASURED reference (lateral-CA realignment + lens-sharpness
deconvolution), but it is single-frame and bakes in its own denoise/sharpen
aesthetic -- a *reference*, not a fidelity ground truth (see ``dxo-reference-eval``).
So this does NOT report a global pixel-fidelity score as a target; it reports
per-axis numbers for {ours, DxO} side by side:

  * chromatic_residual  -- lateral-CA fringing on luma edges, global + by radius
                           (DxO sits near the floor; measure ours against it)
  * sharpness + overshoot -- detail (Laplacian variance) vs ringing, so neither
                           image "wins" by haloing
  * flat_noise          -- per-channel sigma in flat patches (ours, being
                           multi-frame, should be <= DxO's single-frame result)

plus DISTS/PSNR/SSIM as alignment-robust *correlates* (a sanity cross-check, NOT
an optimization target -- a lower number can mean better detail, not worse).

``ours`` is registered into the DxO frame (same camera + shot -> near-identity
homography, no parallax) and colour-matched first, reusing ``eval.registration``.
"""

import argparse
import json

import numpy as np
import cv2

from . import metrics as met
from . import registration as reg


def _luma(rgb):
    return np.clip(rgb, 0, 1).astype(np.float32).mean(2)


def _edge_mask(rgb, pct=90):
    """Boolean mask of the strongest ``100-pct``% luma-gradient pixels."""
    mag = np.hypot(*np.gradient(_luma(rgb).astype(np.float64)))
    return mag > np.percentile(mag, pct)


def _radius(shape):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    return r / r.max()


def chromatic_residual(rgb, n_radial=4, pct=90):
    """Lateral-CA fringing: mean (|R-G| + |B-G|) on luma edges, overall + by radius.

    Lateral CA grows toward the corners, so the per-radius profile (inner->outer
    rings) is the diagnostic; ``global`` is the scalar to compare ours vs DxO.
    """
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    fr = np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 2] - rgb[..., 1])
    em = _edge_mask(rgb, pct)
    r = _radius(rgb.shape)
    radial = []
    for i in range(n_radial):
        ring = em & (r >= i / n_radial) & (r < (i + 1) / n_radial)
        radial.append(float(fr[ring].mean()) if ring.any() else float("nan"))
    return dict(**{"global": float(fr[em].mean())}, radial=radial)


def flat_noise(rgb, tile=48, frac=0.15, hp_sigma=2.0):
    """Per-channel noise sigma: std of the high-pass residual in the flattest tiles.

    High-passing (img - Gaussian) isolates noise+fine-detail; restricting to the
    lowest-gradient tiles removes the detail, leaving ~noise. Lower = cleaner.
    """
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    hp = rgb - cv2.GaussianBlur(rgb, (0, 0), hp_sigma)
    lum = _luma(rgb)
    h, w = lum.shape
    grads, stds = [], []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            g = np.hypot(*np.gradient(lum[y:y + tile, x:x + tile].astype(np.float64))).mean()
            grads.append(g)
            stds.append([hp[y:y + tile, x:x + tile, c].std() for c in range(3)])
    if not grads:
        return [float("nan")] * 3
    order = np.argsort(grads)[:max(1, int(len(grads) * frac))]  # flattest tiles
    return np.asarray(stds)[order].mean(0).tolist()


def edge_overshoot(rgb, ksize=5, hp_sigma=1.0):
    """Halo/ringing proxy: mean excursion of edge pixels beyond a local smooth envelope.

    A sharpening halo pushes pixels past the local min/max of the underlying smooth
    signal; we measure that excursion on luma edges. Higher = more ringing.
    """
    g = _luma(rgb).astype(np.float64)
    sm = cv2.GaussianBlur(g, (0, 0), hp_sigma)
    k = np.ones((ksize, ksize), np.uint8)
    lo = cv2.erode(sm, k)
    hi = cv2.dilate(sm, k)
    excursion = np.maximum(g - hi, 0) + np.maximum(lo - g, 0)
    em = _edge_mask(rgb)
    return float(excursion[em].mean()) if em.any() else 0.0


def _axes(img):
    return dict(
        chromatic_residual=chromatic_residual(img)["global"],
        sharpness=met.sharpness(img),
        overshoot=edge_overshoot(img),
        flat_noise=flat_noise(img),
    )


def compare(ours_rgb, dxo_rgb, register_first=True):
    """Per-degradation comparison of a pipeline output vs a DxO reference.

    Registers ours into the DxO frame + colour-matches (reusing eval.registration)
    unless ``register_first=False`` (when the pair is already aligned, e.g. the
    same synthetic grid). Returns {ours, dxo, correlates, [registration]}.
    """
    if register_first:
        r = reg.register(ours_rgb, dxo_rgb)
        ours, dxo, info = r["mine"], r["canon"], r["info"]
    else:
        ours, dxo, info = np.clip(ours_rgb, 0, 1), np.clip(dxo_rgb, 0, 1), None
    out = dict(ours=_axes(ours), dxo=_axes(dxo))
    tm = met.tiled_metrics(ours, dxo, tile=256, max_tiles=64)
    out["correlates"] = dict(DISTS=tm["DISTS"], PSNR=tm["PSNR"], SSIM=tm["SSIM"])
    if info is not None:
        out["registration"] = info
    return out


def _print(result):
    o, d = result["ours"], result["dxo"]
    print(f"{'axis':<20}{'ours':>14}{'DxO (ref)':>14}   (lower CA/noise/overshoot better; sharper not always better)")
    print(f"{'chromatic_residual':<20}{o['chromatic_residual']:>14.4f}{d['chromatic_residual']:>14.4f}")
    print(f"{'sharpness':<20}{o['sharpness']:>14.2f}{d['sharpness']:>14.2f}")
    print(f"{'overshoot':<20}{o['overshoot']:>14.4f}{d['overshoot']:>14.4f}")
    for i, ch in enumerate("RGB"):
        print(f"{'flat_noise.' + ch:<20}{o['flat_noise'][i]:>14.4f}{d['flat_noise'][i]:>14.4f}")
    c = result["correlates"]
    print(f"\ncorrelates (NOT a target): DISTS={c['DISTS']:.4f} PSNR={c['PSNR']:.2f} SSIM={c['SSIM']:.4f}")
    if "registration" in result:
        print(f"registration: {result['registration']}")


if __name__ == "__main__":
    from .calibrate_optics import _load
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours", help="pipeline output (.npy/.tif/.png)")
    ap.add_argument("dxo", help="DxO PhotoLab reference of the same scene")
    ap.add_argument("--no-register", action="store_true", help="skip homography (pair already aligned)")
    args = ap.parse_args()
    res = compare(_load(args.ours), _load(args.dxo), register_first=not args.no_register)
    _print(res)
    print("\n" + json.dumps({k: v for k, v in res.items() if k != "registration"}, indent=2))
