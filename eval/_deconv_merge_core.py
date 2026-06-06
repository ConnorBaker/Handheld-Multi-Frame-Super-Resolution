"""Prototype: multi-frame Richardson-Lucy deconvolution whose DATA TERM sums over
the ALIGNED burst frames (Lever C from eval/research_edge_skirt.md), vs the current
"merge -> single-image gated-RL" postprocess.

FEASIBILITY STUDY -- NOT a production rearchitecture. Standalone numpy/torch
prototype that reuses the existing alignment (eval.hallucination per-ROI sub-pixel
registration) and the existing single-image deconv (eval.deconv), so the merge Warp
kernel is left untouched.

Forward model (Farsiu 2004; Irani-Peleg 1990 "the same method can also deblur"),
made explicit:

    g_n = D( PSF (*) warp_n(x) ) + noise_n

x      : the latent sharp HR image we want
warp_n : the (sub-pixel) geometric registration of frame n to the reference grid
PSF    : the STATIC optical+reconstruction Gaussian (one kernel for the whole burst)
D      : decimation. At the pipeline's default scale=1, D = identity, so the LR and
         HR grids coincide and warp_n is pure sub-pixel translation. The prototype
         supports s>1 (D = s-fold box-decimation, adjoint = s-fold replication) but
         the validation runs at s=1 to match the shipped merge.

Honest caveat (VERIFIED empirically by frame_count_scaling, not assumed): for a
STATIC PSF the multi-frame win is NOT extra PSF identifiability -- that needs frames
with DIFFERENT blurs (Zhang-Wipf-Zhang CVPR 2011). It is NOISE: N aligned looks ->
~sqrt(N) SNR -> the deconv can be pushed harder (more iters) before noise breaks,
recovering the SAME detail more cleanly. So the deconv data term keeps the frames
SEPARATE (average the per-frame ratios); collapsing to one mean first throws the
distinction away.

Runs on torch-cuda (GB10, sm_121). No numba compute kernels.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Forward-model operators (explicit decimation / blur / adjoints)
# ---------------------------------------------------------------------------

def gaussian_psf_t(sigma, device, size=None):
    """Normalized 2-D Gaussian PSF (sum=1) as a torch tensor on `device`."""
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    ax = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    xx = ax[None, :]
    yy = ax[:, None]
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _otf(psf, shape, device):
    """Zero-phase OTF of `psf` at `shape` (rfft2): convolution adds no spatial shift.
    Same construction as eval.deconv._otf (verified there by round-trip)."""
    h, w = psf.shape
    H, W = shape
    big = torch.zeros((H, W), dtype=torch.float32, device=device)
    big[:h, :w] = psf
    big = torch.roll(big, shifts=(-(h // 2), -(w // 2)), dims=(0, 1))
    return torch.fft.rfft2(big)


def blur_fft(img, otf):
    """PSF (*) img via FFT. The OTF is zero-phase (symmetric kernel) so this same
    op is its own adjoint -- the RL back-projection blur is identical."""
    return torch.fft.irfft2(torch.fft.rfft2(img) * otf, s=img.shape)


def decimate(hr, s):
    """s-fold decimation by box-average (ML adjoint-pair with replication below).
    s == 1 is identity."""
    if s == 1:
        return hr
    return F.avg_pool2d(hr[None, None], kernel_size=s, stride=s)[0, 0]


def upsample_adjoint(lr, s, out_hw):
    """Adjoint of box-decimation: spread each LR pixel over its s x s HR block,
    divided by s^2 (the transpose of avg_pool). s == 1 is identity."""
    if s == 1:
        return lr
    up = F.interpolate(lr[None, None], scale_factor=s, mode="nearest")[0, 0] / (s * s)
    H, W = out_hw
    return up[:H, :W]


# ---------------------------------------------------------------------------
# Single-image Richardson-Lucy (baseline; == eval.deconv.richardson_lucy_fft math)
# ---------------------------------------------------------------------------

def rl_single(g, psf, iters, eps=1e-6):
    """Plain single-image RL on observation `g` with `psf`. Same math as
    eval.deconv.richardson_lucy_fft; re-expressed so the multi-frame variant below
    is an obvious generalization (N=1 reproduces this)."""
    otf = _otf(psf, g.shape, g.device)
    u = g.clamp(min=0).clone()
    for _ in range(iters):
        conv = blur_fft(u, otf)
        ratio = g / (conv + eps)
        u = u * blur_fft(ratio, otf)
        u = u.clamp(min=0)
    return u


# ---------------------------------------------------------------------------
# MULTI-FRAME Richardson-Lucy: data term sums over the aligned frames
# ---------------------------------------------------------------------------

def rl_multiframe(aligned, psf, iters, scale=1, eps=1e-6, reg_tv=0.0, init=None):
    """Multi-frame RL whose data term sums over the ALIGNED frames `aligned`.

    aligned : (N, h, w) torch-cuda tensor. The N frames ALREADY registered to a
              common (reference) grid -- pure sub-pixel translation undone, so warp_n
              is the identity in the latent update and only PSF + (optional)
              decimation remain. (Registration is done OUTSIDE, by the same
              eval.hallucination sub-pixel aligner the harness uses everywhere;
              misregistration therefore feeds the deconv directly -- the exact Lever
              C risk, audited in the validation.)
    psf     : (k, k) torch PSF (the static optical/reconstruction Gaussian).
    scale   : decimation factor s. Latent x lives at (s*h, s*w); each iter blurs x by
              PSF, decimates to LR, compares to EVERY aligned frame, and back-projects
              the AVERAGE per-frame residual ratio. s == 1 (the shipped merge) =>
              latent and frames share the grid.
    reg_tv  : optional TV regularization weight (Dey 2006 TV-RL: suppresses RL's
              noise/ringing while preserving edges). 0 => plain RL.

    Returns the latent (s*h, s*w) clamped >= 0. KEY property vs "average then
    deconvolve once": the ratio is formed PER FRAME against PSF*x then averaged, so
    independent per-frame noise averages down inside the data term (~sqrt(N)) -- the
    whole multi-frame benefit for a static PSF.
    """
    N, h, w = aligned.shape
    H, W = h * scale, w * scale
    dev = aligned.device
    otf = _otf(psf, (H, W), dev)
    a = aligned.clamp(min=0)
    if init is None:
        mean_lr = a.mean(0)
        if scale > 1:
            u = F.interpolate(mean_lr[None, None], scale_factor=scale,
                              mode="bilinear", align_corners=False)[0, 0][:H, :W]
        else:
            u = mean_lr.clone()
        u = u.clamp(min=1e-4)
    else:
        u = init.clamp(min=1e-4).clone()
    ones_lr = torch.ones((h, w), device=dev)
    # denominator sum_n W^T 1 is constant for a static PSF -> precompute
    den_bp = blur_fft(upsample_adjoint(ones_lr, scale, (H, W)), otf) * N
    for _ in range(iters):
        lr_pred = decimate(blur_fft(u, otf), scale) + eps
        ratio_sum = (a / lr_pred[None]).sum(0)
        bp = blur_fft(upsample_adjoint(ratio_sum, scale, (H, W)), otf)
        update = bp / (den_bp + eps)
        if reg_tv > 0.0:
            update = update / (1.0 + reg_tv * _tv_term(u))
        u = (u * update).clamp(min=0)
    return u


def _tv_term(u, eps=1e-3):
    """Total-variation divergence div(grad u / |grad u|) for TV-RL."""
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    mag = torch.sqrt(gx ** 2 + gy ** 2 + eps ** 2)
    nx = gx / mag
    ny = gy / mag
    dxx = torch.zeros_like(u)
    dyy = torch.zeros_like(u)
    dxx[:, 1:] = nx[:, 1:] - nx[:, :-1]
    dyy[1:, :] = ny[1:, :] - ny[:-1, :]
    return dxx + dyy


# ---------------------------------------------------------------------------
# WARP-AWARE multi-frame RL: the per-frame shift stays INSIDE the forward model
# (warp_n is NOT pre-undone). This is the correct architecture for genuine
# super-resolution (scale>1): pre-warping frames to a common grid -- as
# rl_multiframe assumes -- collapses the sub-pixel sampling diversity SR needs, so
# rl_multiframe is a NOISE-fusion tool only. Here warp_n(x) is applied to the latent
# each iteration (forward), and the adjoint warp brings the residual back, so
# distinct sub-pixel phases are fused. This is the Lever-B-flavoured forward model
# (decimation o blur o warp); kept minimal (translation-only warps, the dominant
# burst motion) to probe whether folding HELPS at scale>1.
# ---------------------------------------------------------------------------

def _shift_subpixel(img, dy, dx):
    """Sub-pixel translation of an (H,W) tensor via grid_sample (bicubic), reflect
    border. Positive (dy,dx) shifts content down/right. Its own adjoint is the
    inverse shift (translation is orthogonal up to interpolation)."""
    H, W = img.shape
    ys = torch.linspace(-1, 1, H, device=img.device)
    xs = torch.linspace(-1, 1, W, device=img.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy = gy - 2.0 * dy / max(H - 1, 1)
    gx = gx - 2.0 * dx / max(W - 1, 1)
    grid = torch.stack([gx, gy], dim=-1)[None]
    return torch.nn.functional.grid_sample(
        img[None, None], grid, mode="bicubic", padding_mode="reflection",
        align_corners=True)[0, 0]


def rl_multiframe_warp(frames, shifts, psf, iters, scale=1, eps=1e-6, init=None):
    """Multi-frame RL with the per-frame sub-pixel warp INSIDE the forward model
    (decimation o blur o warp). frames: (N,h,w) NOT pre-aligned (native sampling).
    shifts: (N,2) the (dy,dx) of each frame's content relative to the reference, at
    LR scale (so the latent warp is dy*scale, dx*scale). Latent lives at (s*h, s*w).

    Update: u <- u * sum_n warp_n^T B^T D^T ( g_n / D B warp_n u ) / sum_n warp_n^T B^T D^T 1.
    Translation warp's adjoint is its inverse shift. This fuses distinct sub-pixel
    phases -> genuine SR; contrast with rl_multiframe (pre-aligned == noise only)."""
    N, h, w = frames.shape
    H, W = h * scale, w * scale
    dev = frames.device
    otf = _otf(psf, (H, W), dev)
    g = frames.clamp(min=0)
    if init is None:
        u = F.interpolate(g.mean(0)[None, None], scale_factor=scale, mode="bilinear",
                          align_corners=False)[0, 0][:H, :W].clamp(min=1e-4) if scale > 1 \
            else g.mean(0).clone().clamp(min=1e-4)
    else:
        u = init.clamp(min=1e-4).clone()
    ones_lr = torch.ones((h, w), device=dev)
    # denominator: sum_n warp_n^T B^T D^T 1  (depends on warps; precompute once)
    den = torch.zeros((H, W), device=dev)
    for n in range(N):
        dy, dx = float(shifts[n, 0]) * scale, float(shifts[n, 1]) * scale
        bp = blur_fft(upsample_adjoint(ones_lr, scale, (H, W)), otf)
        den = den + _shift_subpixel(bp, -dy, -dx)
    for _ in range(iters):
        num = torch.zeros((H, W), device=dev)
        for n in range(N):
            dy, dx = float(shifts[n, 0]) * scale, float(shifts[n, 1]) * scale
            wu = _shift_subpixel(u, dy, dx)              # warp_n u
            pred = decimate(blur_fft(wu, otf), scale) + eps   # D B warp_n u
            ratio = g[n] / pred
            bp = blur_fft(upsample_adjoint(ratio, scale, (H, W)), otf)  # B^T D^T r
            num = num + _shift_subpixel(bp, -dy, -dx)     # warp_n^T (.)
        u = (u * num / (den + eps)).clamp(min=0)
    return u


if __name__ == "__main__":
    # Operator self-tests (executable evidence; root CLAUDE.md). Run:
    #   micromamba run -n handheld python -m eval._deconv_merge_core
    from scipy.ndimage import gaussian_filter
    from eval.psf import esf_sigma
    dev = "cuda"
    torch.manual_seed(0)
    print("operator self-tests (GB10):")

    # (1) decimate / upsample_adjoint are a true adjoint pair
    for s in (1, 2):
        H, W = 12 * s, 16 * s
        x = torch.randn(H, W, device=dev); y = torch.randn(H // s, W // s, device=dev)
        lhs = (decimate(x, s) * y).sum().item()
        rhs = (x * upsample_adjoint(y, s, (H, W))).sum().item()
        print(f"  adjoint(decimate,s={s}): |<Dx,y> - <x,D^T y>| = {abs(lhs - rhs):.2e}")

    # (2) blur_fft self-adjoint (zero-phase OTF)
    otf = _otf(gaussian_psf_t(2.1, dev), (64, 64), dev)
    x = torch.randn(64, 64, device=dev); y = torch.randn(64, 64, device=dev)
    print(f"  blur self-adjoint: |<Bx,y> - <x,By>| = "
          f"{abs((blur_fft(x, otf) * y).sum().item() - (x * blur_fft(y, otf)).sum().item()):.2e}")

    # (3) RL round-trip recovers a sharper edge
    H = W = 200
    yy, xx = np.mgrid[0:H, 0:W]
    edge = (0.15 + 0.5 * (xx > 100 + 0.06 * (yy - 100))).astype(np.float32)
    blurred = gaussian_filter(edge, 2.1, mode="nearest").astype(np.float32)
    band = slice(70, 130)
    sb, _ = esf_sigma(blurred[:, band])
    out = rl_single(torch.as_tensor(blurred, device=dev), gaussian_psf_t(2.1, dev), 20).clamp(0, 1).cpu().numpy()
    sd, _ = esf_sigma(out[:, band])
    print(f"  RL round-trip: blurred sigma {sb:.2f} -> RL {sd:.2f} (sharpens)")

    # (4) N=1 multiframe == single RL ; (5) noise reduction with N
    g = torch.as_tensor(blurred, device=dev)
    a = rl_single(g, gaussian_psf_t(2.1, dev), 10).cpu().numpy()
    b = rl_multiframe(g[None], gaussian_psf_t(2.1, dev), 10, scale=1).cpu().numpy()
    print(f"  N=1 multiframe == single RL: max|diff| = {np.abs(a - b).max():.2e}")
    import cv2
    rng = np.random.default_rng(1)
    def fn(im):
        c = im[10:50, 10:50].astype(np.float32)
        return float((c - cv2.GaussianBlur(c, (0, 0), 2.0)).std())
    print("  multiframe noise reduction (same edge + indep noise, fixed iters):")
    for N in (1, 4, 16):
        frames = np.stack([blurred + rng.normal(0, 0.03, blurred.shape).astype(np.float32) for _ in range(N)])
        u = rl_multiframe(torch.as_tensor(frames, device=dev), gaussian_psf_t(2.1, dev), 10, scale=1).clamp(0, 1).cpu().numpy()
        sd, _ = esf_sigma(u[:, band])
        print(f"    N={N:2d}: edge sigma {sd:.3f}  flat-noise {fn(u):.4f}")
