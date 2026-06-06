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
    edge_gate, deblur_gated, deblur_field_varying,
)
from eval.psf import esf_sigma


def _slanted_edge(H, W, blur, x0frac=0.5, slant=0.06, lo=0.15, span=0.5):
    yy, xx = np.mgrid[0:H, 0:W]
    e = (lo + span * (xx > W * x0frac + slant * (yy - H / 2)).astype(np.float64))
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


# ---------------------------------------------------------------------------
# Field-varying, edge-gated deconvolution (the fix for the field-varying PSF)
# ---------------------------------------------------------------------------

def test_edge_gate_high_on_edge_low_on_flat():
    """The structure-tensor gate is ~1 in the edge band and ~0 on a flat patch
    (so sharpening can be confined to real edges)."""
    img = _slanted_edge(200, 200, 1.5)
    g = edge_gate(img)
    assert g.shape == img.shape and g.dtype == np.float32
    assert 0.0 <= g.min() and g.max() <= 1.0
    edge_band = g[:, 90:110].mean()         # across the transition
    flat_band = g[:, :40].mean()            # left flat side
    assert edge_band > 0.6, f"gate should fire on the edge ({edge_band:.2f})"
    assert flat_band < 0.05, f"gate should be ~0 on a flat side ({flat_band:.3f})"
    # a perfectly flat (constant) patch has no gradient -> gate identically 0
    assert edge_gate(np.full((128, 128), 0.4, np.float32)).max() < 1e-4


def test_deblur_gated_leaves_flat_unchanged_and_narrows_edge():
    """deblur_gated narrows an edge while leaving a flat region essentially
    untouched (gate ~0 on flat; bilateral does not move a flat patch)."""
    band = slice(70, 130)
    img = _slanted_edge(200, 200, 1.8)
    sb, _ = esf_sigma(img[:, band])
    out = deblur_gated(torch.as_tensor(img), 1.8, iters=10).numpy()
    sd, _ = esf_sigma(out[:, band])
    assert sd < sb - 0.3, f"edge should narrow {sb:.2f}->{sd:.2f}"
    flat_delta = np.abs(out[:, :40] - img[:, :40]).max()
    assert flat_delta < 1e-3, f"flat region moved by {flat_delta:.4f}"
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def _two_blur_field(H=512, W=1024, noise=0.006, seed=0):
    """Field with a soft (sigma 2.4) edge in the left half and a sharp (sigma 1.0)
    edge in the right half, plus flat plateaus between them and read noise."""
    left = _slanted_edge(H, W, 2.4, x0frac=0.25, slant=0.02, lo=0.12, span=0.6)
    right = _slanted_edge(H, W, 1.0, x0frac=0.75, slant=0.02, lo=0.12, span=0.6)
    xx = np.broadcast_to(np.arange(W), (H, W))
    im = np.where(xx < W // 2, left, right).astype(np.float32)
    im = im + np.random.default_rng(seed).normal(0, noise, im.shape).astype(np.float32)
    return np.clip(im, 0, 1).astype(np.float32)


def test_deblur_field_varying_reduces_both_edges_without_raising_flat_noise():
    """On a two-blur field, field-varying deconv narrows BOTH edges toward the
    target while NOT raising flat-plateau noise above the input's (the gate + the
    bilateral hold the flats; the sharp edge is not over-sharpened)."""
    import cv2
    W = 1024
    f = _two_blur_field(W=W)
    b1 = slice(W // 4 - 30, W // 4 + 30)        # soft edge band
    b2 = slice(3 * W // 4 - 30, 3 * W // 4 + 30)  # sharp edge band
    s1, _ = esf_sigma(f[:, b1]); s2, _ = esf_sigma(f[:, b2])
    out = deblur_field_varying(f, target_sigma=1.0, tile=512, iters=8)
    out = out.numpy() if hasattr(out, "numpy") else out
    o1, _ = esf_sigma(out[:, b1]); o2, _ = esf_sigma(out[:, b2])
    # both edges get sharper (soft one substantially); neither blows up
    assert o1 < s1 - 0.3, f"soft edge should narrow {s1:.2f}->{o1:.2f}"
    assert o2 <= s2 + 0.05, f"sharp edge should not soften {s2:.2f}->{o2:.2f}"
    # flat-plateau noise must not exceed the input's (measured on each plateau)
    def plat_noise(im, x0, x1):
        c = im[120:392, x0:x1]
        return float((c - cv2.GaussianBlur(c, (0, 0), 2.0)).std())
    for x0, x1 in [(330, 490), (540, 700)]:
        ni, no = plat_noise(f, x0, x1), plat_noise(out, x0, x1)
        assert no <= ni + 1e-4, f"flat noise rose {ni:.5f}->{no:.5f} on [{x0},{x1}]"
    assert np.isfinite(out).all()
