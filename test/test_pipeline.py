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
    assert _corr(out.mean(axis=2), ref) > 0.99


def test_pipeline_super_resolution_x2():
    ref = _scene()
    comps = np.stack([_shift(ref, 0.7, -0.4), _shift(ref, -1.3, 0.8), _shift(ref, 0.2, 1.1)])
    cfg = _config(scale=2)
    num, _ = main(ref, comps, cfg)
    out = cp.asnumpy(num)
    assert out.shape == (2 * H, 2 * W, 3)
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0
