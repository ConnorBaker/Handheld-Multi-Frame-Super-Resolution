"""Guards for the SHIPPED-package deconvolution module (``handheld_super_resolution.deconv``).

This is the in-package port of the verified ``eval/deconv.py`` (the package must not
import from ``eval/``) plus ``edge_gate_from_l1`` -- the gate that consumes the
pipeline's OWN structure-tensor ``l1`` field (exposed via
``merging.expose_edge_strength``) instead of recomputing a Sobel mask.

All CPU: the deconvolutions run on ``torch.fft`` (works on CPU tensors) and the gate
is plain torch/numpy, so these need no GPU. The properties that matter: (1)
deconvolving a KNOWN blur narrows a known-blurred edge (without inventing detail --
the operators are deterministic), and (2) the gate fires on edges and is ~0 on flats,
so RL sharpening can be confined to real edges.
"""

import numpy as np
import torch
import pytest
from scipy.ndimage import gaussian_filter

from handheld_super_resolution.deconv import (
    gaussian_psf, _otf, wiener_fft, richardson_lucy_fft, deblur, edge_gate_from_l1,
)
from eval.psf import esf_sigma


def _slanted_edge(H, W, blur, x0frac=0.5, slant=0.06, lo=0.15, span=0.5):
    yy, xx = np.mgrid[0:H, 0:W]
    e = (lo + span * (xx > W * x0frac + slant * (yy - H / 2)).astype(np.float64))
    return gaussian_filter(e, blur, mode="nearest").astype(np.float32)


# ---------------------------------------------------------------------------
# PSF / OTF / deconvolution (the torch.fft port)
# ---------------------------------------------------------------------------

def test_gaussian_psf_normalized_and_odd():
    k = gaussian_psf(2.1)
    assert abs(k.sum() - 1.0) < 1e-5
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert np.argmax(k) == k.size // 2          # peak at centre


def test_otf_forward_blur_matches_spatial():
    """The zero-phase OTF reproduces a spatial Gaussian blur (circular boundary)."""
    rng = np.random.default_rng(0)
    a = rng.random((128, 128)).astype(np.float32)
    ref = gaussian_filter(a, 2.0, mode="wrap", truncate=3.0)
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
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        deblur(torch.zeros(8, 8), 1.0, method="nope")


# ---------------------------------------------------------------------------
# edge_gate_from_l1 -- consumes the pipeline's OWN structure tensor
# ---------------------------------------------------------------------------

def _structure_tensor_l1(luma, win=5):
    """Reference l1 (max eigenvalue of the box-filtered 2x2 structure tensor) the
    way the pipeline computes it -- so the test feeds edge_gate_from_l1 the SAME
    kind of field the merge exposes, not a hand-rolled gate input."""
    import cv2
    a = np.asarray(luma, np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    box = lambda m: cv2.boxFilter(m, cv2.CV_32F, (win, win), normalize=False)
    sa, sb, sd = box(gx * gx), box(gx * gy), box(gy * gy)
    tr = sa + sd
    disc = np.sqrt(np.clip((sa - sd) ** 2 + 4.0 * sb * sb, 0, None))
    return (0.5 * (tr + disc)).astype(np.float32)                  # l1


def test_edge_gate_from_l1_shape_dtype_range():
    l1 = _structure_tensor_l1(_slanted_edge(128, 128, 1.5))
    g = edge_gate_from_l1(torch.as_tensor(l1))
    assert isinstance(g, torch.Tensor) and g.dtype == torch.float32
    assert g.shape == l1.shape
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_edge_gate_high_on_edge_low_on_flat():
    """The gate is ~1 in the edge band and ~0 on a flat side (so sharpening is
    confined to real edges)."""
    l1 = _structure_tensor_l1(_slanted_edge(200, 200, 1.5))
    g = edge_gate_from_l1(torch.as_tensor(l1)).numpy()
    edge_band = g[:, 90:110].mean()         # across the transition
    flat_band = g[:, :40].mean()            # left flat side
    assert edge_band > 0.6, f"gate should fire on the edge ({edge_band:.2f})"
    assert flat_band < 0.05, f"gate should be ~0 on a flat side ({flat_band:.3f})"


def test_edge_gate_flat_field_is_all_zero():
    """A perfectly flat l1 field (all-equal) -> all-zero gate (no spurious
    sharpening when there is no edge structure anywhere)."""
    g = edge_gate_from_l1(torch.full((64, 64), 3.0))
    assert float(g.max()) < 1e-4
    # a truly constant image has zero gradient -> zero l1 -> zero gate
    g0 = edge_gate_from_l1(torch.zeros(64, 64))
    assert float(g0.max()) < 1e-4


def test_edge_gate_accepts_numpy_and_clamps_negative_l1():
    """Accepts a numpy array (not just a tensor) and tolerates tiny negative l1
    (round-off) without producing NaN."""
    l1 = _structure_tensor_l1(_slanted_edge(96, 96, 1.0)).copy()
    l1[0, 0] = -1e-6                                               # round-off negative
    g = edge_gate_from_l1(l1)                                      # numpy in
    assert torch.isfinite(g).all()
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_gate_blend_leaves_flat_unchanged_narrows_edge():
    """The intended use -- out = img + gate*(deconv - img) -- narrows the edge while
    leaving the flat side essentially untouched (gate ~0 there)."""
    img = _slanted_edge(200, 200, 1.8)
    l1 = _structure_tensor_l1(img)
    gate = edge_gate_from_l1(torch.as_tensor(l1))
    t = torch.as_tensor(img)
    dec = deblur(t, 1.8, method="rl", iters=10, clip=True)
    out = (t + gate * (dec - t)).clamp(0, 1).numpy()
    band = slice(70, 130)
    sb, _ = esf_sigma(img[:, band]); sd, _ = esf_sigma(out[:, band])
    assert sd < sb - 0.3, f"edge should narrow {sb:.2f}->{sd:.2f}"
    flat_delta = np.abs(out[:, :40] - img[:, :40]).max()
    assert flat_delta < 1e-3, f"flat region moved by {flat_delta:.4f}"
    assert np.isfinite(out).all()
