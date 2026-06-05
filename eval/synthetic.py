"""Synthetic ground-truth SR test (no camera/parallax/registration confounds).

Take a sharp image GT, build a synthetic burst by sub-pixel shifting, 2x
downsampling (anti-aliased) and Bayer-mosaicking it, then run the pipeline at
scale 2 and compare the output back to GT. Because we KNOW the shifts and the GT,
registration is exact and the comparison directly measures detail recovery.

This is the rigorous check the Canon comparison cannot be: it isolates the
algorithm from sensor/lens MTF, color processing and parallax. If SR recovers GT
here but not from real Samsung frames, the real-world gap is sensor/parallax; if
it fails here too, the algorithm/port is the limit.
"""

import numpy as np
import cv2

from handheld_super_resolution.fast_monte_carlo import run_fast_MC
from . import pipeline_runner as pr

_CFA = {"GRBG": np.array([[1, 0], [2, 1]]), "RGGB": np.array([[0, 1], [1, 2]])}


def apply_optics(img, psf_sigma=0.0, psf_sigma_corner=None, ca_px=0.0):
    """Inject a KNOWN optical degradation into a full-res frame, BEFORE the sensor
    samples it -- a (optionally field-varying) Gaussian PSF blur + radial lateral
    chromatic aberration. This is what was missing from the forward model: it gives
    the deblur a real blur to invert and the defringer real fringing to remove, so
    a comparison to the *clean* GT can actually reward (or punish) their correction.

    Applied AFTER the scene shift, i.e. fixed in sensor/frame coordinates, exactly
    like a real lens. Defaults are no-ops (backward compatible).

    psf_sigma         : Gaussian blur std (px, at this full-res scale) at the centre.
    psf_sigma_corner  : blur std at the corner; if set != psf_sigma, the blur grows
                        radially (a crude field-varying PSF -- the case the per-tile
                        aberration deblur is meant to beat a global deblur on).
    ca_px             : lateral-CA channel shift (px) at the image corner. R is
                        magnified outward by this, B inward (opposite radial shift),
                        G left alone -- the radial R/B split real lenses produce.
    """
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    if psf_sigma and psf_sigma > 0:
        cc = psf_sigma if psf_sigma_corner is None else psf_sigma_corner
        if abs(cc - psf_sigma) < 1e-6:
            out = cv2.GaussianBlur(out, (0, 0), psf_sigma)
        else:  # blend a centre-sigma and corner-sigma blur by normalized radius
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.hypot(yy - (H - 1) / 2, xx - (W - 1) / 2)
            r = (r / r.max())[..., None]
            out = cv2.GaussianBlur(out, (0, 0), psf_sigma) * (1 - r) + \
                  cv2.GaussianBlur(out, (0, 0), cc) * r
    if ca_px and ca_px > 0:
        out = np.ascontiguousarray(out)
        rmax = np.hypot((H - 1) / 2, (W - 1) / 2)
        m = ca_px / rmax  # per-unit-radius magnification so the corner shift = ca_px
        for ch, sgn in ((0, +1.0), (2, -1.0)):  # R out, B in
            s = 1.0 + sgn * m
            Mc = np.float32([[s, 0, (1 - s) * (W - 1) / 2],
                             [0, s, (1 - s) * (H - 1) / 2]])
            out[..., ch] = cv2.warpAffine(out[..., ch], Mc, (W, H),
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1).astype(np.float32)


def make_synthetic_burst(gt, n=8, factor=2, max_shift=2.0, alpha=1e-4, beta=0.0,
                         seed=0, cfa_str="GRBG", psf_sigma=0.0, psf_sigma_corner=None, ca_px=0.0):
    """Build a (burst dict, gt) usable by pipeline_runner.run at scale=factor.

    gt: float32 RGB in [0,1], H2xW2 (multiple of 2*factor recommended).
    Frame 0 is the (unshifted) reference; the rest get known sub-pixel shifts.

    psf_sigma / psf_sigma_corner / ca_px (all default to no-op) inject a known
    optical PSF + lateral CA per frame before downsampling -- see ``apply_optics``.
    The returned ``gt`` stays the CLEAN image, so recovering it rewards deblur +
    defringe; with all three at 0 the forward model is the original (shift +
    anti-aliased downsample + mosaic + noise).
    """
    rng = np.random.default_rng(seed)
    H2, W2, _ = gt.shape
    H, W = H2 // factor, W2 // factor
    cfa = _CFA[cfa_str]
    frames, shifts = [], []
    for i in range(n):
        dy, dx = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-max_shift, max_shift, 2))
        M = np.float32([[1, 0, dx * factor], [0, 1, dy * factor]])
        sh = cv2.warpAffine(gt, M, (W2, H2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        sh = apply_optics(sh, psf_sigma, psf_sigma_corner, ca_px)  # lens PSF + lateral CA (sensor coords)
        lr = cv2.resize(sh, (W, H), interpolation=cv2.INTER_AREA)
        bayer = np.zeros((H, W), np.float32)
        for ii in range(2):
            for jj in range(2):
                bayer[ii::2, jj::2] = lr[ii::2, jj::2, cfa[ii, jj]]
        if alpha > 0:
            sigma = np.sqrt(np.maximum(alpha * bayer + beta, 0)).astype(np.float32)
            bayer = bayer + rng.normal(0, 1, bayer.shape).astype(np.float32) * sigma
        frames.append(np.clip(bayer, 0, 1).astype(np.float32))
        shifts.append((dy, dx))

    std_curve, diff_curve = run_fast_MC(max(alpha, 1e-6), beta)
    ref = frames[0]
    brightness = float(np.mean(ref))
    SNR = brightness / std_curve[round(1000 * brightness)]
    burst = dict(
        path="synthetic", ref_raw=ref, raw_comp=np.array(frames[1:]),
        ISO=100, tags=None, CFA=cfa, cfa_str=cfa_str, xyz2cam=None,
        wb=[1.0, 1.0, 1.0], ref_path=None,
        alpha=max(alpha, 1e-6), beta=beta, std_curve=std_curve, diff_curve=diff_curve, SNR=SNR,
    )
    return burst, gt, shifts


def _metrics(out, gt):
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    from . import metrics as met
    out = np.clip(out, 0, 1).astype(np.float32)
    gt = np.clip(gt, 0, 1).astype(np.float32)
    return dict(
        PSNR=float(psnr(gt, out, data_range=1.0)),
        SSIM=float(ssim(gt, out, channel_axis=2, data_range=1.0)),
        DISTS=float(met.tiled_metrics(out, gt, tile=256, max_tiles=120)["DISTS"]),
        sharp=met.sharpness(out), sharp_gt=met.sharpness(gt),
    )


def evaluate(gt, overrides=None, n=8, alpha=1e-4, seed=0):
    """Run pipeline scale=2 + single/bilinear baselines on a synthetic burst vs GT."""
    burst, gt, shifts = make_synthetic_burst(gt, n=n, alpha=alpha, seed=seed)
    ov = dict(overrides or {})
    ov["scale"] = 2
    multi, _ = pr.run(burst, ov)
    single, _ = pr.run_single(burst, overrides)          # native res
    bil = pr.run_bilinear(burst)                          # native res
    H2, W2 = gt.shape[:2]
    up = lambda im: cv2.resize(im, (W2, H2), interpolation=cv2.INTER_LINEAR)
    return dict(
        gt=gt, multi=np.clip(multi, 0, 1), single_up=up(single), bilinear_up=up(bil),
        m_multi=_metrics(multi, gt), m_single=_metrics(up(single), gt),
        m_bilinear=_metrics(up(bil), gt),
    )
