"""Stage B: two-stage optical-aberration correction (per-tile deblur + defringe).

The spatially-varying per-tile Polyblur recovers more detail than the global
Stage-A deblur (each tile gets its own estimated blur), and the learned ResUNet
defringer removes chromatic fringing -- including the fringing the deblur itself
amplifies. These tests guard both: (1) it recovers detail toward a known GT
without over-sharpening (PSNR stays above baseline), and (2) the defringer
measurably reduces a synthetic chromatic halo. Observed: synthetic sharpness/gt
0.75 -> 0.99 at PSNR 37.2 -> 38.1; Canon chromatic-fringe -58% vs deblur-only.
"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import test.test_reconstruction_accuracy as T
from eval.synthetic import make_synthetic_burst
from eval import pipeline_runner as pr
from handheld_super_resolution.raw2rgb import _aberration

_CROP = T._CROP
_DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _ab_cfg(**ov):
    """The shipped aberration config (configs/default.yaml), enabled, + overrides."""
    c = OmegaConf.load(_DEFAULT_CFG).postprocessing.aberration
    return OmegaConf.merge(c, dict(enabled=True, **ov))


def _sharp(x):
    s = slice(_CROP, -_CROP)
    g = np.gradient(x[s, s].mean(2).astype(np.float64))
    return float(np.hypot(g[0], g[1]).mean())


def _psnr(a, b):
    s = slice(_CROP, -_CROP)
    return float(10 * np.log10(1.0 / np.mean((a[s, s].clip(0, 1) - b[s, s].clip(0, 1)) ** 2)))


def _fringe(rgb):
    """Mean chromatic residual |R-G|+|B-G| on luma edges (lower = less fringing)."""
    g = rgb.mean(2)
    mag = np.hypot(*np.gradient(g.astype(np.float64)))
    em = mag > np.percentile(mag, 95)
    return float((np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 2] - rgb[..., 1]))[em].mean())


@pytest.fixture(scope="module")
def recon():
    gt = T._ground_truth()
    burst, gt, _ = make_synthetic_burst(
        gt, n=T._N_FRAMES, factor=T._FACTOR, alpha=T._ALPHA, seed=T._SEED)
    out, _ = pr.run(burst, {"scale": 2})
    return gt, out


def test_aberration_well_formed(recon):
    gt, out = recon
    ab = _aberration(out, _ab_cfg())
    assert ab.shape == out.shape
    assert np.isfinite(ab).all()
    assert 0.0 <= ab.min() and ab.max() <= 1.0


def test_aberration_recovers_detail_vs_gt(recon):
    """Per-tile deblur recovers detail toward GT without over-sharpening.

    Sharpness must rise toward the GT's AND PSNR must stay above the no-correction
    baseline (over-sharpening would push PSNR below it).
    """
    gt, out = recon
    ab = _aberration(out, _ab_cfg())
    sg = _sharp(gt)
    assert _sharp(ab) / sg > _sharp(out) / sg, "aberration did not increase sharpness toward GT"
    assert _sharp(ab) / sg > 0.90, f"sharpness/gt {_sharp(ab)/sg:.3f} below floor 0.90 (observed ~0.99)"
    assert _psnr(ab, gt) > _psnr(out, gt), "aberration dropped PSNR below baseline (over-sharpening)"


def test_defringer_reduces_chromatic_aberration():
    """The learned defringer reduces realistic lateral chromatic aberration.

    Scope note (honest): the net is trained on real CA statistics -- a *radial
    R/B channel shift* at edges, which is what lenses actually produce (and what
    it cut by ~58% on the Canon comparison). It targets that, NOT arbitrary color
    casts: an out-of-distribution symmetric color band can be made *worse* (a
    measured failure mode). We test the in-distribution case it is designed for.
    """
    H = W = 512
    yy, xx = np.mgrid[0:H, 0:W]
    disk = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2 < 120 ** 2).astype(np.float32)
    d = 0.3 + 0.5 * disk
    img = np.clip(np.stack([np.roll(d, 2, 0), d, np.roll(d, -2, 0)], -1).astype(np.float32), 0, 1)

    on = _aberration(img, _ab_cfg(do_deblur=False, do_defringe=True))
    off = _aberration(img, _ab_cfg(do_deblur=False, do_defringe=False))  # tiling identity
    assert _fringe(on) < _fringe(off), \
        f"defringer did not reduce lateral CA ({_fringe(off):.4f} -> {_fringe(on):.4f})"
