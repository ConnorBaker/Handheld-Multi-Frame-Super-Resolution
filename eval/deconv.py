"""Measured-PSF deconvolution of the multi-frame merge, and a backend study of the
convolution/deconvolution each accelerator offers (PyTorch / CuPy / Warp).

Motivation (see eval/NOTES.md, building-facade scene): the merge's measured edge
PSF is ~Gaussian sigma~2.1 px; DxO reaches ~1.87 px via its own deconvolution; our
shipped Polyblur cannot (its rank-3 polynomial inverse caps high-frequency gain at
`beta`, and its blind sigma-estimate is fooled by high-contrast edges). Since we
have MEASURED the PSF, a non-blind deconvolution can target it directly. Crucially
this invents nothing: it is a deterministic operator on the merge, which already
holds only cross-frame-consistent (real) detail -- it can amplify noise/ringing
(bounded here by regularization, measured via overshoot + flat noise) but cannot
add per-frame structure.

"Deconvolution" means two different things; only the second is image deblurring:
  (a) transposed / fractionally-strided convolution (the deep-learning "deconv"
      layer for upsampling) -- NOT an inverse filter; and
  (b) inverse-filter deblurring (Wiener / Richardson-Lucy) -- what we want here.

Backends (each VERIFIED on the GB10, sm_121):
  * PyTorch  -- (a) nn.ConvTranspose2d / F.conv_transpose2d are built in. (b) no
                turnkey deblur, so Wiener + RL here are rolled on torch.fft (rfft2).
                This is the primary path: FFT is natural for a full-frame inverse.
  * CuPy     -- signal.deconvolve is built in but 1-D (inverse filtering / polynomial
                division); signal.wiener is an ADAPTIVE DENOISER, not deblur. (b) for
                2-D, roll on cupy.fft / ndimage (equivalent to the torch path).
  * Warp     -- HAS device-side FFT: wp.tile_fft / tile_ifft (cuFFTDx via libmathdx,
                enable_mathdx_fft=True here) + tile_matmul (cuBLASDx), with shipped
                FFT / FFT-convolution / FFT-filtering examples -- all run on sm_121
                (an earlier "Warp has no FFT" note here was WRONG; dir(wp) misses
                compile-time tile builtins). So a fused on-device FFT Wiener/RL is
                implementable in Warp; a separable-Gaussian spatial conv kernel is
                also provided below + benchmarked (Gaussian symmetry => the RL
                forward and adjoint blurs are the same kernel).

Validated by round-trip (``python -m eval.deconv``): blur a sharp edge by a known
sigma, deconvolve, and recover a sharper edge; and the three blur backends agree.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PSF + OTF
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
# Deconvolution (PyTorch / torch.fft)
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


def partial_gaussian_sigma(src_sigma, dst_sigma):
    """The Gaussian sigma to DECONVOLVE to take an edge from src_sigma to dst_sigma
    (variances subtract): sqrt(src^2 - dst^2). Use as the deconvolution PSF to
    *match* a target sharpness rather than fully invert (gentler, less noise).
    Empirically the real (non-Gaussian) edge needs ~10% more (e.g. 2.10->1.87
    measured at sigma_r~1.1, not the 0.96 the pure-Gaussian model predicts)."""
    return float(np.sqrt(max(src_sigma ** 2 - dst_sigma ** 2, 1e-6)))


def deblur(img, deconv_sigma, method="rl", iters=10, nsr=1e-2, clip=True):
    """Measured-PSF deblur: deconvolve a Gaussian of `deconv_sigma` from `img`
    (a (H,W) float tensor in [0,1]). method 'rl' (Richardson-Lucy -- gentler, less
    halo/noise; preferred here) or 'wiener'. On the facade merge, sigma_r~1.1 took
    the edge 2.10->1.88 px (== DxO) at a fair DESLANTED edge overshoot ~2.6% vs DxO
    ~1.8% / raw merge ~2.0% (comparable, not ringing), adding no invention (it
    operates on the already-consensus merge). Use 3-5 iters for less noise gain;
    `iters`/`nsr` regularize. (The slant-averaged overshoot() in _deconv_sweep
    under-reports the bump -- deslant per-edge for an honest number.)"""
    psf = gaussian_psf(deconv_sigma)
    if method == "rl":
        return richardson_lucy_fft(img, psf, iters=iters, clip=clip)
    if method == "wiener":
        out = wiener_fft(img, psf, nsr=nsr)
        return out.clamp(0, 1) if clip else out
    raise ValueError(f"unknown method {method!r} (use 'rl' or 'wiener')")


# ---------------------------------------------------------------------------
# NVIDIA Warp -- separable Gaussian convolution kernel (a spatial alternative to
# Warp's device-side wp.tile_fft, verified available here). Benchmarked against
# torch/cupy; RL's per-iter blur is the only conv it needs (Gaussian is symmetric
# so the forward and adjoint blurs are identical).
# ---------------------------------------------------------------------------

import warp as wp


@wp.kernel
def _conv1d_h(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float),
              w: wp.array(dtype=float), r: int):
    i, j = wp.tid()
    W = src.shape[1]
    acc = float(0.0)
    for k in range(-r, r + 1):
        jj = wp.clamp(j + k, 0, W - 1)          # clamped (replicate) border
        acc += w[k + r] * src[i, jj]
    dst[i, j] = acc


@wp.kernel
def _conv1d_v(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float),
              w: wp.array(dtype=float), r: int):
    i, j = wp.tid()
    H = src.shape[0]
    acc = float(0.0)
    for k in range(-r, r + 1):
        ii = wp.clamp(i + k, 0, H - 1)
        acc += w[k + r] * src[ii, j]
    dst[i, j] = acc


def warp_gaussian_blur(img, sigma, device="cuda:0"):
    """Separable Gaussian blur via two Warp 1-D conv kernels. `img` is a (H,W)
    cupy or torch-cuda array (wrapped zero-copy by Warp). Returns a wp.array."""
    r = int(np.ceil(3 * sigma))
    ax = np.arange(-r, r + 1)
    w = np.exp(-ax ** 2 / (2 * sigma ** 2)); w = (w / w.sum()).astype(np.float32)
    src = wp.from_dlpack(img) if not isinstance(img, wp.array) else img
    H, W = src.shape
    wgt = wp.array(w, dtype=float, device=device)
    tmp = wp.zeros((H, W), dtype=float, device=device)
    out = wp.zeros((H, W), dtype=float, device=device)
    wp.launch(_conv1d_h, dim=(H, W), inputs=[src, tmp, wgt, r], device=device)
    wp.launch(_conv1d_v, dim=(H, W), inputs=[tmp, out, wgt, r], device=device)
    return out


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    from .psf import esf_sigma

    # (1) the three blur backends agree (interior)
    rng = np.random.default_rng(0)
    a = rng.random((512, 512)).astype(np.float32)
    sig = 2.1
    ref = gaussian_filter(a, sig, mode="nearest")
    t = torch.as_tensor(a, device="cuda")
    psf = gaussian_psf(sig)
    tb = torch.fft.irfft2(torch.fft.rfft2(t) * _otf(psf, t.shape, t.device), s=t.shape).cpu().numpy()
    cb = cp.asnumpy(cnd.gaussian_filter(cp.asarray(a), sig, mode="nearest"))
    wb = warp_gaussian_blur(cp.asarray(a), sig).numpy()
    s = slice(20, -20)
    print("blur backend agreement (interior max-abs-diff):")
    print(f"  cupy.ndimage vs scipy : {np.abs(cb[s,s]-ref[s,s]).max():.2e}")
    print(f"  warp(separable) vs scipy: {np.abs(wb[s,s]-ref[s,s]).max():.2e}")
    print(f"  torch.fft(circular) vs scipy(nearest): {np.abs(tb[s,s]-ref[s,s]).max():.2e} (boundary differs, interior close)")

    # (2) deconvolution round-trip: blur a sharp slanted edge, recover sharper
    H = W = 200
    yy, xx = np.mgrid[0:H, 0:W]
    edge = (0.15 + 0.5 * (xx > 100 + 0.06 * (yy - 100)).astype(np.float64)).astype(np.float32)
    blurred = gaussian_filter(edge, 2.1, mode="nearest")
    band = slice(70, 130)
    sig0, _ = esf_sigma(edge[:, band]); sigb, _ = esf_sigma(blurred[:, band])
    tb = torch.as_tensor(blurred, device="cuda")
    for name, fn in [("Wiener nsr=3e-3", lambda: wiener_fft(tb, gaussian_psf(2.1), 3e-3)),
                     ("RL iters=20", lambda: richardson_lucy_fft(tb, gaussian_psf(2.1), 20))]:
        out = fn().clamp(0, 1).cpu().numpy()
        sd, _ = esf_sigma(out[:, band])
        print(f"  round-trip {name:18s}: edge sigma {sigb:.2f} -> {sd:.2f} (sharp src {sig0:.2f})")
