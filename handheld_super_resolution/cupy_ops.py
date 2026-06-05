"""CuPy replacements for the numba ``@cuda.jit`` kernels.

These exist so the pipeline runs on GPUs where numba's CUDA target cannot JIT
(e.g. Blackwell / GB10, where ptxas rejects the PTX version numba emits). All
torch-based code in the pipeline already runs on such GPUs; only the numba
kernels are replaced here, with vectorized CuPy / cupyx.scipy where the math is
a standard image op, and CuPy ``RawModule`` kernels for the genuinely custom
scatter/gather (merge) and the Dogson warp-upscale.

Inputs may be torch tensors, numba device arrays or cupy arrays — all expose
``__cuda_array_interface__`` so ``cp.asarray`` ingests them zero-copy.
"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    map_coordinates,
)

from .utils import DEFAULT_NUMPY_FLOAT_TYPE

FLOAT = DEFAULT_NUMPY_FLOAT_TYPE


def to_torch(cp_array):
    """Zero-copy view of a cupy array as a torch cuda tensor (via DLPack)."""
    import torch

    return torch.from_dlpack(cp_array)


# ---------------------------------------------------------------------------
# small elementwise helpers (utils.divide / utils.add / utils_image.GAT)
# ---------------------------------------------------------------------------
def divide_cupy(num, den):
    """num /= den, in place on a cupy array."""
    num = cp.asarray(num)
    cp.divide(num, cp.asarray(den), out=num)
    return num


def add_cupy(a, b):
    """a += b, in place on a cupy array."""
    a = cp.asarray(a)
    a += cp.asarray(b)
    return a


def gat_cupy(image, alpha, beta):
    """Generalized Anscombe Transform (matches utils_image.cuda_GAT)."""
    image = cp.asarray(image, dtype=cp.float32)
    vst = alpha * image + (3.0 / 8.0) * alpha * alpha + beta
    vst = cp.maximum(vst, 0)
    return (2.0 / alpha) * cp.sqrt(vst)


# ---------------------------------------------------------------------------
# ICA (Lucas-Kanade) optical-flow iteration  (replaces ica_kernel_8/16/32/64)
# ---------------------------------------------------------------------------
def ica_iterate_cupy(ref_img, gradx, grady, hessian, moving, alignment, tile_size, n_iter):
    """Vectorized inverse-compositional Lucas-Kanade.

    Parameters mirror the numba ``ica_kernel_*`` kernels. ``alignment`` is
    (n_patch_y, n_patch_x, 2) with [...,0]=x flow, [...,1]=y flow. Returns the
    updated alignment as a cupy array (same shape).
    """
    ref = cp.asarray(ref_img, dtype=cp.float32)
    gx = cp.asarray(gradx, dtype=cp.float32)
    gy = cp.asarray(grady, dtype=cp.float32)
    A = cp.asarray(hessian, dtype=cp.float32)
    mov = cp.asarray(moving, dtype=cp.float32)
    flow = cp.asarray(alignment, dtype=cp.float32).copy()

    n_patch_y, n_patch_x, _ = flow.shape
    t = tile_size
    H, W = n_patch_y * t, n_patch_x * t

    # crop to the exact patch grid (the numba kernels only touch these pixels)
    ref = ref[:H, :W]
    gx = gx[:H, :W]
    gy = gy[:H, :W]

    A00 = A[..., 0, 0]
    A01 = A[..., 0, 1]
    A10 = A[..., 1, 0]
    A11 = A[..., 1, 1]
    det = A00 * A11 - A01 * A10
    solvable = cp.abs(det) >= 1e-10
    det_safe = cp.where(solvable, det, 1.0)

    yy, xx = cp.meshgrid(
        cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij"
    )

    for _ in range(n_iter):
        # per-pixel flow (constant within a tile)
        fx = cp.repeat(cp.repeat(flow[..., 0], t, axis=0), t, axis=1)
        fy = cp.repeat(cp.repeat(flow[..., 1], t, axis=0), t, axis=1)

        coords = cp.stack([yy + fy, xx + fx], axis=0)
        warped = map_coordinates(mov, coords, order=1, mode="constant", cval=0.0)

        gradt = warped - ref
        # B = -sum_tile(grad * gradt)
        b0 = -(gx * gradt).reshape(n_patch_y, t, n_patch_x, t).sum(axis=(1, 3))
        b1 = -(gy * gradt).reshape(n_patch_y, t, n_patch_x, t).sum(axis=(1, 3))

        # solve A x = B  ->  x = A^-1 B
        dx = (A11 * b0 - A01 * b1) / det_safe
        dy = (-A10 * b0 + A00 * b1) / det_safe

        flow[..., 0] += cp.where(solvable, dx, 0.0)
        flow[..., 1] += cp.where(solvable, dy, 0.0)

    return flow


# ---------------------------------------------------------------------------
# Block matching, L1 brute-force integer search  (replaces cuda_L1_local_search*)
# ---------------------------------------------------------------------------
def block_matching_l1_cupy(ref_lvl, moving_lvl, alignments, tile_size, search_radius):
    """Integer-shift L1 block matching (pure CuPy: integer gather + reshape-sum).

    For each tile, searches integer shifts in [-R, R]^2 around the (rounded)
    current flow and keeps the shift minimizing the per-tile sum of absolute
    differences. Out-of-bounds moving samples count as 0.

    The shifts are integer, so we sample with a plain zero-filled CuPy gather --
    equivalent to ``map_coordinates(order=0, mode='constant')`` but without the
    interpolation machinery (see test/test_kernel_equivalence.py). NB: this
    computes a correct argmin; the original numba L1 kernels had a broken
    min-search, so this is intentionally *not* a faithful port of that bug.
    """
    ref = cp.asarray(ref_lvl, dtype=cp.float32)
    mov = cp.asarray(moving_lvl, dtype=cp.float32)
    flow = cp.asarray(alignments, dtype=cp.float32).copy()

    n_patch_y, n_patch_x, _ = flow.shape
    t = tile_size
    H, W = n_patch_y * t, n_patch_x * t
    mh, mw = mov.shape
    ref = ref[:H, :W]

    base_x = cp.round(flow[..., 0])
    base_y = cp.round(flow[..., 1])
    fx = cp.repeat(cp.repeat(base_x, t, axis=0), t, axis=1).astype(cp.int32)
    fy = cp.repeat(cp.repeat(base_y, t, axis=0), t, axis=1).astype(cp.int32)

    yy, xx = cp.meshgrid(
        cp.arange(H, dtype=cp.int32), cp.arange(W, dtype=cp.int32), indexing="ij"
    )

    best_cost = cp.full((n_patch_y, n_patch_x), cp.inf, dtype=cp.float32)
    best_sx = cp.zeros((n_patch_y, n_patch_x), dtype=cp.float32)
    best_sy = cp.zeros((n_patch_y, n_patch_x), dtype=cp.float32)

    R = search_radius
    for sy in range(-R, R + 1):
        for sx in range(-R, R + 1):
            gy = yy + fy + sy
            gx = xx + fx + sx
            inb = (gy >= 0) & (gy < mh) & (gx >= 0) & (gx < mw)
            shifted = cp.where(inb, mov[cp.clip(gy, 0, mh - 1), cp.clip(gx, 0, mw - 1)], cp.float32(0))
            cost = cp.abs(ref - shifted).reshape(n_patch_y, t, n_patch_x, t).sum(axis=(1, 3))
            better = cost < best_cost
            best_cost = cp.where(better, cost, best_cost)
            best_sx = cp.where(better, sx, best_sx)
            best_sy = cp.where(better, sy, best_sy)

    flow[..., 0] = base_x + best_sx
    flow[..., 1] = base_y + best_sy
    return flow


# ---------------------------------------------------------------------------
# Robustness  (replaces the 8 numba kernels in robustness.py)
# ---------------------------------------------------------------------------
def compute_guide_image_cupy(raw_img, cfa_pattern, white_balance):
    """Algorithm 7: ComputeGuideImage. Returns (3, H//2, W//2) cupy array."""
    raw = cp.asarray(raw_img, dtype=cp.float32)
    cfa = cp.asarray(cfa_pattern).astype(cp.int32)
    wb = cp.asarray(white_balance, dtype=cp.float32)
    H, W = raw.shape
    gh, gw = H // 2, W // 2
    guide = cp.zeros((3, gh, gw), dtype=cp.float32)

    quads = {
        (0, 0): raw[0 : 2 * gh : 2, 0 : 2 * gw : 2],
        (0, 1): raw[0 : 2 * gh : 2, 1 : 2 * gw : 2],
        (1, 0): raw[1 : 2 * gh : 2, 0 : 2 * gw : 2],
        (1, 1): raw[1 : 2 * gh : 2, 1 : 2 * gw : 2],
    }
    n_green = 0
    for (i, j), sub in quads.items():
        c = int(cfa[i, j])
        if c == 1:  # green: averaged over the two green sites
            guide[1] += sub / wb[1]
            n_green += 1
        else:
            guide[c] = sub / wb[c]
    guide[1] /= n_green
    return guide


def compute_local_stats_cupy(guide_img):
    """Algorithm 8: 3x3 local mean and variance (var stored in 'stds')."""
    guide = cp.asarray(guide_img, dtype=cp.float32)
    mean = uniform_filter(guide, size=(1, 3, 3), mode="nearest")
    mean_sq = uniform_filter(guide * guide, size=(1, 3, 3), mode="nearest")
    var = mean_sq - mean * mean
    return mean, var


def compute_dist_cupy(means_1, means_2):
    return cp.abs(cp.asarray(means_1) - cp.asarray(means_2))


def apply_noise_model_cupy(d_p, ref_local_means, ref_local_stds, std_curve, diff_curve):
    d_p = cp.asarray(d_p, dtype=cp.float32)
    means = cp.asarray(ref_local_means, dtype=cp.float32)
    stds = cp.asarray(ref_local_stds, dtype=cp.float32)
    std_curve = cp.asarray(std_curve, dtype=cp.float32)
    diff_curve = cp.asarray(diff_curve, dtype=cp.float32)
    L = std_curve.shape[0]

    idx = cp.clip(cp.round(1000.0 * means).astype(cp.int32), 0, L - 1)
    d_t = diff_curve[idx]
    sigma_t = std_curve[idx]

    sigma_sq = cp.maximum(stds, sigma_t * sigma_t).sum(axis=0)
    d_p_sq = d_p * d_p
    shrink = d_p_sq / (d_p_sq + d_t * d_t + 1e-10)
    d_sq = (d_p_sq * shrink * shrink).sum(axis=0)
    return d_sq, sigma_sq


def compute_s_cupy(flows, M_th, s1, s2):
    flows = cp.asarray(flows, dtype=cp.float32)
    fx, fy = flows[..., 0], flows[..., 1]
    # 'nearest' edge-replication gives the same min/max as skipping OOB neighbours
    diff0 = maximum_filter(fx, size=3, mode="nearest") - minimum_filter(fx, size=3, mode="nearest")
    diff1 = maximum_filter(fy, size=3, mode="nearest") - minimum_filter(fy, size=3, mode="nearest")
    return cp.where(diff0 * diff0 + diff1 * diff1 > M_th * M_th, FLOAT(s1), FLOAT(s2))


def robustness_threshold_cupy(d_sq, sigma_sq, S, t, tile_size):
    d_sq = cp.asarray(d_sq, dtype=cp.float32)
    sigma_sq = cp.asarray(sigma_sq, dtype=cp.float32)
    S = cp.asarray(S, dtype=cp.float32)
    H, W = d_sq.shape
    # upsample S (per-tile) to per-pixel
    S_pix = cp.repeat(cp.repeat(S, tile_size, axis=0), tile_size, axis=1)[:H, :W]
    R = S_pix * cp.exp(-d_sq / sigma_sq) - t
    return cp.clip(R, 0.0, 1.0)


def local_min_cupy(R):
    """Algorithm 9: 5x5 local minimum."""
    return minimum_filter(cp.asarray(R, dtype=cp.float32), size=5, mode="nearest")


def _dogson_q(x):
    """Dogson quadratic kernel (vectorized)."""
    ax = cp.abs(x)
    return cp.where(ax <= 0.5, -2.0 * ax * ax + 1.0,
                    cp.where(ax <= 1.5, ax * ax - 2.5 * ax + 1.5, 0.0)).astype(cp.float32)


def upscale_warp_stats_cupy(local_stats, bayer_mode, tile_size=None, flow=None):
    """Dogson biquadratic upscale (x2 in bayer) + optional flow warp.

    Pure CuPy: a 3x3 Dogson-weighted gather (the kernel isn't one of
    map_coordinates' B-spline orders, so the taps are applied explicitly).
    Out-of-bounds samples -> inf (=> robustness 0 there), matching the original.
    """
    LR = cp.asarray(local_stats, dtype=cp.float32)
    n_channels, lny, lnx = LR.shape
    s = 2 if bayer_mode else 1
    hny, hnx = lny * s, lnx * s

    Y, X = cp.meshgrid(cp.arange(hny, dtype=cp.float32), cp.arange(hnx, dtype=cp.float32), indexing="ij")
    if flow is None:
        flow_y = flow_x = cp.float32(0)
    else:
        flow = cp.asarray(flow, dtype=cp.float32)
        np_y, np_x, _ = flow.shape
        py = cp.minimum((Y // tile_size).astype(cp.int32), np_y - 1)
        px = cp.minimum((X // tile_size).astype(cp.int32), np_x - 1)
        flow_x = flow[py, px, 0]
        flow_y = flow[py, px, 1]

    LR_y = (Y + flow_y + 0.5) / s - 0.5
    LR_x = (X + flow_x + 0.5) / s - 0.5
    oob = ~((LR_y >= 0) & (LR_y < lny) & (LR_x >= 0) & (LR_x < lnx))
    cy = cp.rint(LR_y).astype(cp.int32)
    cx = cp.rint(LR_x).astype(cp.int32)

    buf = cp.zeros((n_channels, hny, hnx), dtype=cp.float32)
    w_acc = cp.zeros((hny, hnx), dtype=cp.float32)
    for i in (-1, 0, 1):
        y_ = cp.clip(cy + i, 0, lny - 1)
        wy = _dogson_q(y_.astype(cp.float32) - LR_y)
        for j in (-1, 0, 1):
            x_ = cp.clip(cx + j, 0, lnx - 1)
            w = wy * _dogson_q(x_.astype(cp.float32) - LR_x)
            buf += LR[:, y_, x_] * w
            w_acc += w
    HR = buf / w_acc
    HR[:, oob] = cp.inf
    return HR


# ---------------------------------------------------------------------------
# Kernel (covariance) estimation  (replaces cuda_estimate_kernel in kernels.py)
# ---------------------------------------------------------------------------
SEL_HARD_THRESHOLD = 0
SEL_LINEAR = 1


def estimate_cov_cupy(full_grads, k_detail, k_denoise, D_th, D_tr,
                      k_stretch, k_shrink, selection_law):
    """Structure-tensor -> steerable covariance, per grey pixel.

    full_grads : (gh, gw, 2) gradients (gx, gy). Returns (gh, gw, 2, 2) covs.
    """
    g = cp.asarray(full_grads, dtype=cp.float32)
    gx = g[..., 0]
    gy = g[..., 1]
    Gh, Gw = gx.shape  # gradient grid; covs is one larger in each dim (grey size)

    # structure tensor at covs[idy,idx] = sum of grad outer-products over the
    # window {idy-1,idy} x {idx-1,idx} of full_grads (OOB -> 0), matching the
    # numba kernel. Output shape (Gh+1, Gw+1) == grey image shape.
    sxx = gx * gx
    sxy = gx * gy
    syy = gy * gy

    def box2(arr):
        p = cp.zeros((Gh + 2, Gw + 2), dtype=cp.float32)
        p[1:-1, 1:-1] = arr
        return (p[0:Gh + 1, 0:Gw + 1] + p[0:Gh + 1, 1:Gw + 2]
                + p[1:Gh + 2, 0:Gw + 1] + p[1:Gh + 2, 1:Gw + 2])

    a = box2(sxx)  # [0,0]
    b = box2(sxy)  # [0,1]=[1,0]
    d = box2(syy)  # [1,1]
    gh, gw = Gh + 1, Gw + 1

    # closed-form 2x2 symmetric eig (descending: l1 >= l2)
    tr = a + d
    disc = cp.sqrt(cp.clip((a - d) * (a - d) + 4.0 * b * b, 0, None))
    l1 = (tr + disc) / 2.0
    l2 = (tr - disc) / 2.0

    # eigenvector for l1 (matches linalg.get_eigen_vect_2x2 convention:
    # e1 ~ (M00+M01-l2, M10+M11-l2)); e2 orthonormal to e1)
    e1x = a + b - l2
    e1y = b + d - l2
    norm = cp.sqrt(e1x * e1x + e1y * e1y)
    isotropic = norm < 1e-20
    norm_safe = cp.where(isotropic, 1.0, norm)
    e1x = cp.where(isotropic, 1.0, e1x / norm_safe)
    e1y = cp.where(isotropic, 0.0, e1y / norm_safe)
    # e2 perpendicular, sign matching numba (e2 = (-e1y*sign(e1x), |e1x|))
    sign = cp.where(e1x >= 0, 1.0, -1.0)
    e2x = -e1y * sign
    e2y = cp.abs(e1x)

    # compute_k
    A = 1.0 + cp.sqrt(cp.clip((l1 - l2) / (l1 + l2 + 1e-10), 0, None))
    D = cp.clip(1.0 - cp.sqrt(cp.clip(l1, 0, None)) / D_tr + D_th, 0.0, 1.0)
    if selection_law == SEL_HARD_THRESHOLD:
        k1 = cp.where(A > 1.95, 1.0 / k_shrink, 1.0)
        k2 = cp.where(A > 1.95, k_stretch, 1.0)
    else:
        k1 = 1.0 + A / 2.0 * (1.0 / k_shrink - 1.0)
        k2 = 1.0 + A / 2.0 * (k_stretch - 1.0)
    k1 = k_detail * ((1.0 - D) * k1 + D * k_denoise)
    k2 = k_detail * ((1.0 - D) * k2 + D * k_denoise)

    k1_sq = k1 * k1
    k2_sq = k2 * k2
    covs = cp.empty((gh, gw, 2, 2), dtype=cp.float32)
    covs[..., 0, 0] = k1_sq * e1x * e1x + k2_sq * e2x * e2x
    covs[..., 0, 1] = k1_sq * e1x * e1y + k2_sq * e2x * e2y
    covs[..., 1, 0] = covs[..., 0, 1]
    covs[..., 1, 1] = k1_sq * e1y * e1y + k2_sq * e2y * e2y
    return covs


def _interp_cov_bilinear(covs, fy_floor, fx_floor, fy_ceil, fx_ceil, frac_y, frac_x):
    """Bilinear-interpolate the (...,2,2) cov grid and return inverse components."""
    cov_h, cov_w = covs.shape[:2]

    def at(yy, xx, a, b):
        return covs[cp.clip(yy, 0, cov_h - 1), cp.clip(xx, 0, cov_w - 1), a, b]

    def lerp(a, b):
        c00 = at(fy_floor, fx_floor, a, b)
        c01 = at(fy_floor, fx_ceil, a, b)
        c10 = at(fy_ceil, fx_floor, a, b)
        c11 = at(fy_ceil, fx_ceil, a, b)
        top = c00 + frac_x * (c01 - c00)
        bot = c10 + frac_x * (c11 - c10)
        return top + frac_y * (bot - top)

    ixx, ixy, iyy = lerp(0, 0), lerp(0, 1), lerp(1, 1)
    det = ixx * iyy - ixy * ixy
    inv = 1.0 / det
    return inv * iyy, -inv * ixy, inv * ixx


def merge_accumulate_cupy(comp_img, alignments, covs, r, bayer_mode, iso_kernel,
                          scale, tile_size, cfa_pattern, num, den):
    """Alg. 4 accumulation of a comparison frame into num/den (in place).

    Pure CuPy: one vectorized pass per 3x3 neighbour offset (a bounded gather,
    not a scatter), with elementwise cov interpolation / kernel weights and
    per-Bayer-channel masked accumulation. Matches the RawKernel
    (test/test_kernel_equivalence.py).
    """
    comp = cp.asarray(comp_img, dtype=cp.float32)
    al = cp.asarray(alignments, dtype=cp.float32)
    covs = cp.asarray(covs, dtype=cp.float32)
    r = cp.asarray(r, dtype=cp.float32)
    cfa = cp.asarray(cfa_pattern).astype(cp.int32)
    num = cp.asarray(num)
    den = cp.asarray(den)
    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = comp.shape
    np_y, np_x, _ = al.shape
    n_ch = 3 if bayer_mode else 1

    I, J = cp.meshgrid(cp.arange(hr_h, dtype=cp.float32), cp.arange(hr_w, dtype=cp.float32), indexing="ij")
    lr_x = (J + 0.5) / scale
    lr_y = (I + 0.5) / scale
    px = cp.minimum((lr_x / tile_size).astype(cp.int32), np_x - 1)
    py = cp.minimum((lr_y / tile_size).astype(cp.int32), np_y - 1)
    flowx = al[py, px, 0]
    flowy = al[py, px, 1]
    local_r = r[cp.minimum(lr_y.astype(cp.int32), lr_h - 1), cp.minimum(lr_x.astype(cp.int32), lr_w - 1)]
    lr_mov_x = lr_x + flowx
    lr_mov_y = lr_y + flowy
    mov_inb = (lr_mov_x >= 0) & (lr_mov_x < lr_w) & (lr_mov_y >= 0) & (lr_mov_y < lr_h)

    if not iso_kernel:
        if bayer_mode:
            kmap_j, kmap_i = lr_mov_x / 2.0 - 0.5, lr_mov_y / 2.0 - 0.5
        else:
            kmap_j, kmap_i = lr_mov_x - 0.5, lr_mov_y - 0.5
        ikx, iky = kmap_j.astype(cp.int32), kmap_i.astype(cp.int32)  # trunc toward zero
        frac_x, frac_y = kmap_j - ikx, kmap_i - iky
        fx_fl, fy_fl = cp.maximum(ikx, 0), cp.maximum(iky, 0)
        cov_i_xx, cov_i_xy, cov_i_yy = _interp_cov_bilinear(
            covs, fy_fl, fx_fl, fy_fl + 1, fx_fl + 1, frac_y, frac_x)

    center_j = lr_mov_x.astype(cp.int32)
    center_i = lr_mov_y.astype(cp.int32)
    lr_mov_j = lr_mov_x - 0.5
    lr_mov_i = lr_mov_y - 0.5
    val = [cp.zeros((hr_h, hr_w), cp.float32) for _ in range(n_ch)]
    acc = [cp.zeros((hr_h, hr_w), cp.float32) for _ in range(n_ch)]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            j = center_j + dj
            i = center_i + di
            nb = (j >= 0) & (j < lr_w) & (i >= 0) & (i < lr_h)
            c = comp[cp.clip(i, 0, lr_h - 1), cp.clip(j, 0, lr_w - 1)]
            dx = j.astype(cp.float32) - lr_mov_j
            dy = i.astype(cp.float32) - lr_mov_i
            if iso_kernel:
                z = 2.0 * (dx * dx + dy * dy)
            else:
                z = cov_i_xx * dx * dx + 2.0 * cov_i_xy * dx * dy + cov_i_yy * dy * dy
            w = cp.exp(-0.5 * cp.maximum(z, 0.0))
            wv = w * local_r * c * nb
            wa = w * local_r * nb
            if bayer_mode:
                chan = cfa[i & 1, j & 1]
                for ch in range(3):
                    m = chan == ch
                    val[ch] += wv * m
                    acc[ch] += wa * m
            else:
                val[0] += wv
                acc[0] += wa
    for ch in range(n_ch):
        num[..., ch] += val[ch] * mov_inb
        den[..., ch] += acc[ch] * mov_inb
    return num, den


def merge_ref_accumulate_cupy(ref_img, covs, bayer_mode, iso_kernel, scale,
                              cfa_pattern, num, den, acc_rob,
                              robustness_denoise, max_frame_count, rad_max, max_multiplier):
    """Alg. 11 accumulation of the reference frame into num/den (in place).

    Pure CuPy, mirroring the RawKernel: flow=0, the coarse position is hr/scale
    (no +0.5), the cov grid is sampled with floor (not truncation), and the
    optional robustness-aware denoiser varies the radius/blur per pixel and may
    overwrite (single-frame mode). Matches the RawKernel
    (test/test_kernel_equivalence.py).
    """
    ref = cp.asarray(ref_img, dtype=cp.float32)
    covs = cp.asarray(covs, dtype=cp.float32)
    cfa = cp.asarray(cfa_pattern).astype(cp.int32)
    num = cp.asarray(num)
    den = cp.asarray(den)
    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = ref.shape
    n_ch = 3 if bayer_mode else 1

    I, J = cp.meshgrid(cp.arange(hr_h, dtype=cp.float32), cp.arange(hr_w, dtype=cp.float32), indexing="ij")
    crp_y = I / scale
    crp_x = J / scale

    if not iso_kernel:
        if bayer_mode:
            gy, gx = (crp_y - 0.5) / 2.0, (crp_x - 0.5) / 2.0
        else:
            gy, gx = crp_y, crp_x
        floor_y = cp.maximum(cp.floor(gy).astype(cp.int32), 0)
        floor_x = cp.maximum(cp.floor(gx).astype(cp.int32), 0)
        fy = gy - cp.floor(gy)
        fx = gx - cp.floor(gx)
        cov_i_xx, cov_i_xy, cov_i_yy = _interp_cov_bilinear(
            covs, floor_y, floor_x, floor_y + 1, floor_x + 1, fy, fx)

    if robustness_denoise:
        acc_rob = cp.asarray(acc_rob if acc_rob is not None else cp.zeros((1, 1)), dtype=cp.float32)
        ah, aw = acc_rob.shape
        ir = cp.minimum(cp.rint(crp_y).astype(cp.int32), ah - 1)
        jr = cp.minimum(cp.rint(crp_x).astype(cp.int32), aw - 1)
        local_acc_r = acc_rob[cp.clip(ir, 0, ah - 1), cp.clip(jr, 0, aw - 1)]
        cond = local_acc_r <= max_frame_count
        add_denoise = cp.where(cond, cp.float32(max_multiplier), cp.float32(1.0))
        rad_pix = cp.where(cond, int(rad_max), 1)
        rad_loop = max(int(rad_max), 1)
    else:
        add_denoise = cp.float32(1.0)
        rad_pix = None
        rad_loop = 1

    center_x = cp.rint(crp_x).astype(cp.int32)
    center_y = cp.rint(crp_y).astype(cp.int32)
    val = [cp.zeros((hr_h, hr_w), cp.float32) for _ in range(n_ch)]
    acc = [cp.zeros((hr_h, hr_w), cp.float32) for _ in range(n_ch)]
    for di in range(-rad_loop, rad_loop + 1):
        for dj in range(-rad_loop, rad_loop + 1):
            pidx = center_x + dj
            pidy = center_y + di
            nb = (pidx >= 0) & (pidx < lr_w) & (pidy >= 0) & (pidy < lr_h)
            if rad_pix is not None:
                nb = nb & (abs(di) <= rad_pix) & (abs(dj) <= rad_pix)
            c = ref[cp.clip(pidy, 0, lr_h - 1), cp.clip(pidx, 0, lr_w - 1)]
            dx = pidx.astype(cp.float32) - crp_x
            dy = pidy.astype(cp.float32) - crp_y
            if iso_kernel:
                y = 2.0 * (dx * dx + dy * dy)
            else:
                y = cov_i_xx * dx * dx + 2.0 * cov_i_xy * dx * dy + cov_i_yy * dy * dy
            w = cp.exp(-0.5 * (cp.maximum(y, 0.0) / add_denoise))
            if bayer_mode:
                chan = cfa[pidy & 1, pidx & 1]
                for ch in range(3):
                    m = chan == ch
                    val[ch] += c * w * nb * m
                    acc[ch] += w * nb * m
            else:
                val[0] += c * w * nb
                acc[0] += w * nb

    if robustness_denoise:
        overwrite = local_acc_r < max_frame_count
        for ch in range(n_ch):
            num[..., ch] = cp.where(overwrite, val[ch], num[..., ch] + val[ch])
            den[..., ch] = cp.where(overwrite, acc[ch], den[..., ch] + acc[ch])
    else:
        for ch in range(n_ch):
            num[..., ch] += val[ch]
            den[..., ch] += acc[ch]
    return num, den
