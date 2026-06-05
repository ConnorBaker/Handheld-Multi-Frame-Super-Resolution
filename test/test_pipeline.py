"""End-to-end pipeline test: the full GPU pipeline runs with no numba and
reconstructs a synthetic burst faithfully (correlation with the reference scene).

This exercises grey conversion, alignment (block matching L2 + L1 + ICA),
robustness, steerable/iso kernel estimation and merge -- all on CuPy/torch.
"""

from pathlib import Path

import numpy as np
import cupy as cp
import pytest
from omegaconf import OmegaConf
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

from handheld_super_resolution.params import update_snr_config, sanitize_config
from handheld_super_resolution.super_resolution import main

H = W = 512
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _scene():
    np.random.seed(0)
    base = cp.asarray(gaussian_filter(cp.asarray(np.random.rand(H, W).astype(np.float32)), 3.0))
    base = (base - base.min()) / (base.max() - base.min()) * 0.45 + 0.05
    return cp.asnumpy(base).astype(np.float32)


def _shift(img, dx, dy):
    yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij")
    warped = map_coordinates(cp.asarray(img), cp.stack([yy - dy, xx - dx], 0), order=1, mode="reflect")
    return cp.asnumpy(warped).astype(np.float32)


def _config(**overrides):
    c = OmegaConf.load(DEFAULT_CONFIG)
    c.verbose = 0
    c.noise_model.alpha = 1e-4
    c.noise_model.beta = 1e-5
    update_snr_config(c, 30.0)
    c.noise_model.std_curve = np.linspace(1e-3, 5e-2, 1001).tolist()
    c.noise_model.diff_curve = np.linspace(1e-3, 5e-2, 1001).tolist()
    c.exif = OmegaConf.create({"cfa_pattern": [[0, 1], [1, 2]], "iso": 100, "white_balance": [1.0, 1.0, 1.0]})
    c.accumulated_robustness_denoiser.enabled = False
    for key, value in overrides.items():
        OmegaConf.update(c, key, value)
    sanitize_config(c, (H, W))
    return c


def _corr(a, b):
    s = slice(20, -20)  # ignore borders
    a = a[s, s].ravel().astype(np.float64)
    b = b[s, s].ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def _abs_error(a, b):
    """Interior (mean, max) absolute error -- a per-pixel check that, unlike
    correlation, is NOT invariant to scale/offset, so a constant or rescaled
    image cannot pass it."""
    s = slice(20, -20)
    e = np.abs(a[s, s].astype(np.float64) - b[s, s].astype(np.float64))
    return float(e.mean()), float(e.max())


@pytest.mark.parametrize("kernel", ["steerable", "iso"])
@pytest.mark.parametrize("robustness", [True, False])
def test_pipeline_reconstructs_scene(kernel, robustness):
    ref = _scene()
    comps = np.stack([_shift(ref, 0.7, -0.4), _shift(ref, -1.3, 0.8), _shift(ref, 0.2, 1.1)])
    cfg = _config(**{"merging.kernel": kernel,
                     "robustness.enabled": robustness,
                     "robustness.save_mask": robustness})
    num, _ = main(ref, comps, cfg)
    out = cp.asnumpy(num)
    assert out.shape == (H, W, 3)
    assert np.isfinite(out).all()
    luma = out.mean(axis=2)
    assert _corr(luma, ref) > 0.99
    # Per-pixel reconstruction accuracy (the output luma is on the same [0.05,0.5]
    # scale as the reference scene).  Unlike the correlation above this is NOT
    # gameable by a rescaled/constant image.  Empirical (this scale-1 smooth-scene
    # burst is deterministic across runs/processes): observed MAE ~0.0014-0.0017,
    # max abs error ~0.011 (steerable) / ~0.010 (iso); thresholds carry margin.
    mae, max_ae = _abs_error(luma, ref)
    assert mae < 0.01, f"interior MAE {mae:.4f} too high (observed <= 0.0017)"
    assert max_ae < 0.05, f"interior max abs error {max_ae:.4f} too high (observed <= 0.011)"


def test_pipeline_robustness_off_with_save_mask_does_not_crash():
    """Regression: robustness disabled while save_mask stays at its True default.

    sanitize_config rejects this combo, but eval.pipeline_runner.build_config
    applies overrides AFTER sanitizing, so main() can still receive it -- and it
    used to crash dereferencing the None ref_local_means that init_robustness
    returns when robustness is off. main() now gates mask accumulation on
    robustness.enabled and runs with the all-ones accept-all mask instead.
    """
    ref = _scene()
    comps = np.stack([_shift(ref, 0.7, -0.4), _shift(ref, -1.3, 0.8)])
    cfg = _config()                      # valid + sanitized: robustness on, save_mask on
    cfg.robustness.enabled = False       # post-sanitize override, exactly as the eval harness does
    assert cfg.robustness.save_mask is True, "this test must exercise the off + save_mask combo"
    num, _ = main(ref, comps, cfg)       # must not raise
    out = cp.asnumpy(num)
    assert out.shape == (H, W, 3)
    assert np.isfinite(out).all()


def test_pipeline_super_resolution_x2():
    ref = _scene()
    comps = np.stack([_shift(ref, 0.7, -0.4), _shift(ref, -1.3, 0.8), _shift(ref, 0.2, 1.1)])
    cfg = _config(scale=2)
    num, _ = main(ref, comps, cfg)
    out = cp.asnumpy(num)
    assert out.shape == (2 * H, 2 * W, 3)
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0
