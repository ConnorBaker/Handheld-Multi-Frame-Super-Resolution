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
# Field-varying, edge-gated deconvolution
#
# A verified adversarial pass (eval/NOTES.md) showed the merge PSF is FIELD-VARYING
# (edge sigma ~0.9-2.4 px across the scene), so a single GLOBAL deconv sigma
# over-sharpens the already-sharp regions, and plain RL amplifies flat-region noise.
# Two fixes, both measured: (1) gate the sharpening to actual EDGES via the
# structure-tensor edge strength (the pipeline's steerable-kernel signal, NOT the
# robustness mask, which is cross-frame agreement -- ~constant on a static scene);
# (2) estimate the LOCAL PSF per tile and deconvolve each tile to a common target.
# ---------------------------------------------------------------------------

def _to_np(img):
    """View an (H,W) torch tensor or array as a float32 numpy array (no copy of a
    cpu tensor's storage). Returns (array, was_tensor, device)."""
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy().astype(np.float32, copy=False), True, img.device
    return np.asarray(img, dtype=np.float32), False, None


def edge_gate(luma, lo_pct=50, hi_pct=90, win=5):
    """Structure-tensor edge-strength map in [0,1] (HxW float32), used to gate
    sharpening to real edges (flats left untouched).

    Mirrors ``cupy_ops.estimate_cov_cupy``: Sobel gradients -> box-filter the 2x2
    structure tensor (a=Sum gx^2, b=Sum gx*gy, d=Sum gy^2) over a `win`x`win`
    window -> closed-form max eigenvalue l1=(tr+sqrt((a-d)^2+4b^2))/2. The edge
    strength is sqrt(l1) (across-edge gradient energy), smoothstep-normalized
    between the lo/hi percentiles of that strength so the map is ~0 on flats and
    ~1 on strong edges. cv2/numpy only (runs on CPU)."""
    import cv2
    a = np.asarray(luma, dtype=np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    box = lambda m: cv2.boxFilter(m, cv2.CV_32F, (win, win), normalize=False)
    sa = box(gx * gx)            # a
    sb = box(gx * gy)            # b (= [0,1] = [1,0])
    sd = box(gy * gy)            # d
    tr = sa + sd
    disc = np.sqrt(np.clip((sa - sd) ** 2 + 4.0 * sb * sb, 0, None))
    l1 = 0.5 * (tr + disc)       # max eigenvalue (across-edge energy)
    strength = np.sqrt(np.clip(l1, 0, None))
    lo, hi = np.percentile(strength, [lo_pct, hi_pct])
    t = np.clip((strength - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    g = t * t * (3.0 - 2.0 * t)  # smoothstep
    return g.astype(np.float32)


def _flat_noise_std(img, win=200, y=200, x=2600, hp_sigma=2.0):
    """High-pass std of a flat patch (default the facade scene's sky crop), the
    noise floor the deconv must not exceed. Falls back to a darkest-variance patch
    if the default crop is out of bounds (e.g. a synthetic test image)."""
    import cv2
    H, W = img.shape
    if y + win <= H and x + win <= W:
        c = img[y:y + win, x:x + win]
    else:                                  # pick the lowest-gradient win-tile
        w = min(win, H, W)
        grad = (np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, 3))
                + np.abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)))
        gm = cv2.boxFilter(grad, cv2.CV_32F, (w, w), normalize=True)
        valid = gm[: H - w + 1, : W - w + 1]      # top-left corners that fit a w-tile
        yy, xx = np.unravel_index(int(np.argmin(valid)), valid.shape)
        c = img[yy:yy + w, xx:xx + w]
    return float((c - cv2.GaussianBlur(c.astype(np.float32), (0, 0), hp_sigma)).std())


def deblur_gated(img, deconv_sigma, method="rl", iters=5, denoise="bilateral",
                 denoise_strength=None, gate=True, **kw):
    """Edge-gated, noise-controlled deblur. Same input/output type as ``deblur``.

    Deconvolve (via ``deblur``), then blend only at edges -- ``img + g*(out-img)``
    with ``g=edge_gate(img)`` (g=1 to disable) -- so flats are not sharpened. Then
    optionally an edge-preserving bilateral (``cv2.bilateralFilter``) strips the
    noise the deconv amplified while keeping texture; ``sigma_color`` auto-scales to
    ~3x the measured flat-region noise std when ``denoise_strength is None``. On the
    facade merge this reaches DxO's edge sigma with flat-sky noise AT/BELOW the raw
    merge (eval/NOTES.md). KEEP THIS SIGNATURE -- another component calls it."""
    import cv2
    arr, was_tensor, dev = _to_np(img)
    base = torch.as_tensor(arr)
    out = deblur(base, deconv_sigma, method=method, iters=iters, **kw).cpu().numpy()
    if gate:
        g = edge_gate(arr) if gate is True else np.asarray(gate, np.float32)
        out = arr + g * (out - arr)
    out = out.astype(np.float32)
    if denoise == "bilateral":
        sc = denoise_strength
        if sc is None:
            sc = 3.0 * _flat_noise_std(arr)
        sc = max(float(sc), 1e-4)
        out = cv2.bilateralFilter(np.ascontiguousarray(out), d=5,
                                  sigmaColor=sc, sigmaSpace=3)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return torch.as_tensor(out).to(dev) if was_tensor else out


def _raised_cosine_2d(h, w, oy, ox):
    """Separable raised-cosine (Hann) feather window, ramping over the `oy`/`ox`
    overlap margins and flat (=1) in the tile interior. Sums to ~1 across tiles
    laid on a 50%-or-more overlap grid, so feathered tiles blend seamlessly."""
    def ramp(n, o):
        w1 = np.ones(n, np.float32)
        if o > 0:
            r = 0.5 * (1.0 - np.cos(np.pi * (np.arange(o) + 0.5) / o))
            w1[:o] = r
            w1[-o:] = r[::-1]
        return w1
    return np.outer(ramp(h, oy), ramp(w, ox)).astype(np.float32)


def _local_sigma(tile, fallback_sigma, rms_max=0.04, band=60, min_contrast=0.05):
    """Local merge-PSF sigma in a tile: ESF sigma of the strongest CLEAN near-vertical
    step edge inside it (good erf fit + real contrast), else `fallback_sigma`. Reuses
    eval.hallucination.find_step_edges (single-tile search) + eval.psf.esf_sigma."""
    from .hallucination import find_step_edges
    from .psf import esf_sigma
    h, w = tile.shape
    sz = min(140, h - 2, w - 2)
    # fine stride so a search window lands centred on an isolated edge (a coarse
    # stride misses lone edges -- it only works on the busy full frame by luck).
    edges = find_step_edges(tile, sz=sz, bw=min(60, w - 2),
                            stride=max(16, sz // 8), margin=max(8, sz // 2 + 2),
                            min_contrast=min_contrast, max_edges=4, min_sep=sz // 2)
    for cy, cx, *_ in edges:                       # strongest first
        b = max(8, band // 2)
        sub = tile[cy - sz // 2:cy + sz // 2, cx - b:cx + b]
        if sub.shape[0] < 12 or sub.shape[1] < 8:
            continue
        s, rms = esf_sigma(sub)
        if np.isfinite(s) and rms < rms_max and 0.4 < s < 4.0:
            return float(s), True
    return float(fallback_sigma), False


def deblur_field_varying(img, target_sigma=1.87, tile=512, overlap=0.25, method="rl",
                         fudge=1.08, fallback_sigma=2.0, **kw):
    """Field-varying measured-PSF deconvolution: estimate the LOCAL merge PSF per
    overlapping tile and deblur each tile to a common `target_sigma`, so already-sharp
    regions are deconvolved gently and soft regions strongly (the fix the global deconv
    lacked -- a single sigma over-sharpens the sharp field). Returns a float32 (H,W).

    Two passes so the PSF FIELD is smooth (not blindly global): (1) measure the
    local sigma per tile -- ESF on the strongest clean step edge inside it
    (``_local_sigma``); (2) tiles with no clean edge (e.g. an edge that fell in the
    tile's measurement margin, or a flat tile) INHERIT the median of the tiles that
    did measure, so a soft region is not under-deconvolved by a blind fallback (only
    if NO tile measured is `fallback_sigma` used); (3) per tile
    ``deconv_sigma=fudge*partial_gaussian_sigma(local_sigma, target_sigma)`` (the
    ~8% fudge is the real edge's non-Gaussian tail), ``deblur_gated`` the tile, and
    feather-blend the overlapping tiles with a raised-cosine window. `**kw` flows to
    deblur_gated (iters/denoise/...)."""
    arr, was_tensor, dev = _to_np(img)
    H, W = arr.shape
    step = max(1, int(round(tile * (1.0 - overlap))))
    ov = (tile - step + 1) // 2                    # feather margin (>=1 if overlap>0)
    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))

    # pass 1: measure the local PSF sigma where a clean edge exists
    sig = np.full((len(ys), len(xs)), np.nan, np.float32)
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            t = arr[y0:y0 + min(tile, H - y0), x0:x0 + min(tile, W - x0)]
            s, ok = _local_sigma(t, np.nan)
            if ok:
                sig[iy, ix] = s
    measured = np.isfinite(sig)
    field_med = float(np.median(sig[measured])) if measured.any() else float(fallback_sigma)
    sig = np.where(measured, sig, field_med)       # inherit field median where unmeasured

    # pass 2: deconvolve each tile to the common target and feather-blend
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            th, tw = min(tile, H - y0), min(tile, W - x0)
            t = arr[y0:y0 + th, x0:x0 + tw]
            ds = fudge * partial_gaussian_sigma(float(sig[iy, ix]), target_sigma)
            out_t = deblur_gated(t, ds, method=method, **kw)
            if isinstance(out_t, torch.Tensor):
                out_t = out_t.cpu().numpy()
            win = _raised_cosine_2d(th, tw, min(ov, th // 2), min(ov, tw // 2))
            acc[y0:y0 + th, x0:x0 + tw] += out_t * win
            wsum[y0:y0 + th, x0:x0 + tw] += win
    out = (acc / np.maximum(wsum, 1e-8)).astype(np.float32)
    return torch.as_tensor(out).to(dev) if was_tensor else out


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
