"""Guards for eval.deconv (measured-PSF Wiener / Richardson-Lucy deblurring).

All CPU: the deconvolutions run on torch.fft, which works on CPU tensors, so these
need no GPU. The property that matters -- and that the whole "match DxO's edge
sharpness from the merge" result rests on -- is that deconvolving the known blur
actually narrows a known-blurred edge (without us cheating by inventing detail:
the operators are deterministic functions of the input). The Warp tile_fft path is
exercised by Warp's own shipped examples, not here.
"""

import numpy as np
import torch
import pytest
from scipy.ndimage import gaussian_filter

from eval.deconv import (
    gaussian_psf, partial_gaussian_sigma, wiener_fft, richardson_lucy_fft, deblur, _otf,
)
from eval.psf import esf_sigma


def _slanted_edge(H, W, blur):
    yy, xx = np.mgrid[0:H, 0:W]
    e = (0.15 + 0.5 * (xx > W / 2 + 0.06 * (yy - H / 2)).astype(np.float64))
    return gaussian_filter(e, blur, mode="nearest").astype(np.float32)


def test_gaussian_psf_normalized_and_odd():
    k = gaussian_psf(2.1)
    assert abs(k.sum() - 1.0) < 1e-5
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert np.argmax(k) == k.size // 2          # peak at centre


def test_partial_gaussian_sigma():
    assert abs(partial_gaussian_sigma(2.1, 1.87) - 0.955) < 0.02   # sqrt(2.1^2-1.87^2)
    assert partial_gaussian_sigma(2.0, 2.0) < 1e-2                 # equal -> ~0 (clamped)


def test_otf_forward_blur_matches_spatial():
    """The zero-phase OTF reproduces a spatial Gaussian blur (circular boundary)."""
    rng = np.random.default_rng(0)
    a = rng.random((128, 128)).astype(np.float32)
    ref = gaussian_filter(a, 2.0, mode="wrap", truncate=3.0)       # circular, 3-sigma like gaussian_psf
    t = torch.as_tensor(a)
    blur = torch.fft.irfft2(torch.fft.rfft2(t) * _otf(gaussian_psf(2.0), t.shape, t.device), s=t.shape)
    assert np.abs(blur.numpy() - ref).max() < 2e-3


@pytest.mark.parametrize("method", ["wiener", "rl"])
def test_deblur_narrows_a_known_edge(method):
    """Deconvolving the known blur narrows a known-blurred edge toward the source."""
    band = slice(70, 130)
    blurred = _slanted_edge(200, 200, 2.0)
    sb, _ = esf_sigma(blurred[:, band])
    assert 1.8 < sb < 2.3                                          # the injected blur is measured
    out = deblur(torch.as_tensor(blurred), 2.0, method=method,
                 iters=20, nsr=3e-3).clamp(0, 1).numpy()
    sd, _ = esf_sigma(out[:, band])
    assert sd < sb - 0.4, f"{method}: edge sigma should drop {sb:.2f}->{sd:.2f}"
    assert np.isfinite(out).all() and out.min() >= 0.0


def test_partial_deconv_hits_a_target_sharpness():
    """A partial-kernel deblur lands BETWEEN the blurred and a full deconvolution
    (the knob used to match DxO rather than over-sharpen)."""
    band = slice(70, 130)
    blurred = _slanted_edge(200, 200, 2.0)
    sb, _ = esf_sigma(blurred[:, band])
    mild = deblur(torch.as_tensor(blurred), 1.0, method="rl", iters=10).numpy()
    strong = deblur(torch.as_tensor(blurred), 2.0, method="rl", iters=20).numpy()
    sm, _ = esf_sigma(mild[:, band]); ss, _ = esf_sigma(strong[:, band])
    assert ss < sm < sb, f"partial {sm:.2f} should sit between full {ss:.2f} and blurred {sb:.2f}"
