"""Measured-PSF deconvolution of the merge (Richardson-Lucy / Wiener), GPU torch.fft.

Opt-in alternative to the aberration stage's BOUNDED Polyblur deblur (see
``raw2rgb._aberration`` / ``aberration/model.py``). Polyblur's deconvolution is a
rank-3 polynomial inverse whose high-frequency gain is capped at ``beta`` and whose
blind per-tile sigma-estimate is fooled by high-contrast edges (``eval/NOTES.md``);
Richardson-Lucy is unbounded, so given a MEASURED PSF it can target the merge's
residual softness directly.

Two design rules carried over from the eval study (``eval/NOTES.md`` tail):
  * The deconvolution sigma is a CONFIG / per-camera calibration value (like the
    Polyblur ``sigma_b``), NOT a blind per-tile estimate, and the PSF is an
    isotropic Gaussian here -- no radial / parametric field is assumed.
  * Field-adaptivity comes from a per-pixel GATE, not from a parametric field. RL
    amplifies flat-region noise, so the sharpening is blended in only where there
    is real edge energy. The gate consumes the pipeline's OWN structure tensor (the
    ``l1`` eigenvalue the merge already computes for the steerable kernels) via
    ``edge_gate_from_l1`` -- it does NOT recompute a Sobel mask.

The torch.fft Wiener / RL implementations below are ported verbatim from the verified
``eval/deconv.py`` (the package must not import from ``eval/``). They run on the GB10
(sm_121); ``test/test_deconv.py`` (eval) and ``test/test_deconv_module.py`` (this
module) guard them. Dependency-light: torch + numpy only.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PSF + OTF  (ported verbatim from eval/deconv.py)
# ---------------------------------------------------------------------------

def gaussian_psf(sigma, size=None):
    """Normalized 2-D Gaussian PSF (sum=1). size defaults to ~6 sigma (odd)."""
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _otf(psf, shape, device):
    """Zero-phase OTF of `psf` at full-frame `shape` (rfft2), so convolution adds
    no spatial shift. Returns the complex half-spectrum tensor."""
    h, w = psf.shape
    H, W = shape
    big = torch.zeros((H, W), dtype=torch.float32, device=device)
    big[:h, :w] = torch.as_tensor(psf, device=device)
    big = torch.roll(big, shifts=(-(h // 2), -(w // 2)), dims=(0, 1))  # center -> origin
    return torch.fft.rfft2(big)


# ---------------------------------------------------------------------------
# Deconvolution (PyTorch / torch.fft)  (ported verbatim from eval/deconv.py)
# ---------------------------------------------------------------------------

def wiener_fft(img, psf, nsr=1e-2):
    """Wiener deconvolution X = G.conj(H)/(|H|^2 + nsr). `nsr` is the noise-to-
    signal regularization (larger => gentler, less noise/ringing). img: (H,W) float
    tensor in [0,1] on cuda. Returns the deconvolved tensor (not clamped)."""
    H = _otf(psf, img.shape, img.device)
    G = torch.fft.rfft2(img)
    X = G * torch.conj(H) / (H.real ** 2 + H.imag ** 2 + nsr)
    return torch.fft.irfft2(X, s=img.shape)


def richardson_lucy_fft(img, psf, iters=10, eps=1e-6, clip=True):
    """Richardson-Lucy deconvolution (multiplicative, non-negative). FFT convs.
    `iters` is the regularizer (more => sharper but more noise/ringing)."""
    H = _otf(psf, img.shape, img.device)
    Hc = torch.conj(H)
    u = img.clamp(min=0).clone()
    for _ in range(iters):
        conv = torch.fft.irfft2(torch.fft.rfft2(u) * H, s=img.shape)
        ratio = img / (conv + eps)
        u = u * torch.fft.irfft2(torch.fft.rfft2(ratio) * Hc, s=img.shape)
        u = u.clamp(min=0)
    return u.clamp(0, 1) if clip else u


def deblur(img, deconv_sigma, method="rl", iters=10, nsr=1e-2, clip=True):
    """Measured-PSF deblur: deconvolve a Gaussian of `deconv_sigma` from `img` (a
    (H,W) float tensor in [0,1]). method 'rl' (Richardson-Lucy -- gentler, less
    halo/noise; preferred here) or 'wiener'. `iters`/`nsr` regularize."""
    psf = gaussian_psf(deconv_sigma)
    if method == "rl":
        return richardson_lucy_fft(img, psf, iters=iters, clip=clip)
    if method == "wiener":
        out = wiener_fft(img, psf, nsr=nsr)
        return out.clamp(0, 1) if clip else out
    raise ValueError(f"unknown method {method!r} (use 'rl' or 'wiener')")


# ---------------------------------------------------------------------------
# Edge gate from the pipeline's OWN structure tensor (l1 eigenvalue field)
# ---------------------------------------------------------------------------

def edge_gate_from_l1(l1, lo_pct=50, hi_pct=90):
    """Turn the structure-tensor large-eigenvalue field `l1` into a [0,1] edge gate.

    `l1` is the across-edge gradient energy that ``cupy_ops.estimate_cov_cupy``
    already computes per pixel for the steerable merge kernels (and discards). The
    edge strength is ``sqrt(l1)``, smoothstep-normalized between the `lo_pct`/`hi_pct`
    percentiles of that strength so the gate is ~0 on flats and ~1 on strong edges
    (verified edge/flat contrast ~6.8x, ``eval/NOTES.md``). This is the SAME
    normalization as ``eval/deconv.py:edge_gate`` -- but it CONSUMES the pipeline's
    own field instead of recomputing a Sobel structure tensor.

    Accepts a torch tensor, cupy array or numpy array (anything ``torch.as_tensor``
    / ``__cuda_array_interface__`` ingests); returns a torch float32 tensor on the
    same device, shape == ``l1.shape``. A perfectly flat field (l1 all equal) gives
    an all-zero gate (no spurious sharpening)."""
    if not isinstance(l1, torch.Tensor):
        # torch.as_tensor ingests cupy / numba device arrays via __cuda_array_interface__
        # (zero-copy on GPU) and numpy arrays on CPU; covers every caller here.
        l1 = torch.as_tensor(l1)
    l1 = l1.to(torch.float32)
    strength = torch.sqrt(l1.clamp(min=0))
    lo = torch.quantile(strength.flatten(), lo_pct / 100.0)
    hi = torch.quantile(strength.flatten(), hi_pct / 100.0)
    t = ((strength - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).to(torch.float32)  # smoothstep
