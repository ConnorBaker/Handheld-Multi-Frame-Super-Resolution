"""Polyblur deblur post-processing stage: recovers real detail vs known GT.

The merge is ~1px-soft by design (sampling-limited kernel regression) + residual
optical blur, so the reconstruction systematically under-recovers high frequency
(measured: interior sharpness ~0.75x the ground truth's). The vendored Polyblur
deblur (``postprocessing.deblur``) targets exactly that small anisotropic blur.

This test is un-gameable: it asserts on a KNOWN ground truth that the deblur
(i) increases sharpness toward GT AND (ii) increases PSNR. Both together mean it
recovered genuine detail rather than just sharpening/amplifying noise (which would
raise sharpness but LOWER PSNR -- the failure mode the remove_halo + prefiltering
mitigations prevent). Observed (deterministic): sharpness 0.75 -> 0.84,
PSNR 37.2 -> 38.5 dB.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

import test.test_reconstruction_accuracy as T
from eval.synthetic import make_synthetic_burst
from eval import pipeline_runner as pr
from handheld_super_resolution.raw2rgb import _deblur

_CROP = T._CROP


def _sharp(x):
    s = slice(_CROP, -_CROP)
    g = np.gradient(x[s, s].mean(2).astype(np.float64))
    return float(np.hypot(g[0], g[1]).mean())


def _psnr(a, b):
    s = slice(_CROP, -_CROP)
    return float(10 * np.log10(1.0 / np.mean((a[s, s].clip(0, 1) - b[s, s].clip(0, 1)) ** 2)))


@pytest.fixture(scope="module")
def recon():
    gt = T._ground_truth()
    burst, gt, _ = make_synthetic_burst(
        gt, n=T._N_FRAMES, factor=T._FACTOR, alpha=T._ALPHA, seed=T._SEED)
    out, _ = pr.run(burst, {"scale": 2})
    return gt, out


def _default_deblur_cfg():
    return OmegaConf.create(dict(enabled=True, n_iter=1, alpha=2, beta=3,
                                 remove_halo=True, prefiltering=True, edgetaping=True))


def test_deblur_runs_and_is_well_formed(recon):
    gt, out = recon
    deb = _deblur(out, _default_deblur_cfg())
    assert deb.shape == out.shape
    assert np.isfinite(deb).all()
    assert 0.0 <= deb.min() and deb.max() <= 1.0


def test_deblur_recovers_detail_vs_gt(recon):
    """Un-gameable: deblur must raise BOTH sharpness-toward-GT AND PSNR vs known GT.

    PSNR up (not just sharpness) == real detail recovered, not noise amplification.
    """
    gt, out = recon
    deb = _deblur(out, _default_deblur_cfg())
    sg = _sharp(gt)
    sharp_before, sharp_after = _sharp(out) / sg, _sharp(deb) / sg
    psnr_before, psnr_after = _psnr(out, gt), _psnr(deb, gt)

    assert sharp_after > sharp_before, \
        f"deblur did not increase sharpness toward GT ({sharp_before:.3f} -> {sharp_after:.3f})"
    assert sharp_after > 0.80, f"sharpness/gt {sharp_after:.3f} below floor 0.80 (observed ~0.835)"
    # PSNR must not drop -- if it does, the deblur is over-sharpening / amplifying
    # noise rather than recovering detail (the remove_halo+prefiltering guard).
    assert psnr_after > psnr_before, \
        f"deblur dropped PSNR ({psnr_before:.2f} -> {psnr_after:.2f}) -- over-sharpening, not recovery"
    assert psnr_after > 37.5, f"deblur PSNR {psnr_after:.2f} below floor 37.5 (observed ~38.5)"
