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
    """Integer-shift L1 block matching.

    For each tile, searches integer shifts in [-R, R]^2 around the (rounded)
    current flow and keeps the shift minimizing the per-tile sum of absolute
    differences. Out-of-bounds moving samples count as 0 (as in the numba
    kernels). Returns updated alignments (cupy array).
    """
    ref = cp.asarray(ref_lvl, dtype=cp.float32)
    mov = cp.asarray(moving_lvl, dtype=cp.float32)
    flow = cp.asarray(alignments, dtype=cp.float32).copy()

    n_patch_y, n_patch_x, _ = flow.shape
    t = tile_size
    H, W = n_patch_y * t, n_patch_x * t
    ref = ref[:H, :W]

    base_x = cp.round(flow[..., 0])
    base_y = cp.round(flow[..., 1])
    fx = cp.repeat(cp.repeat(base_x, t, axis=0), t, axis=1)
    fy = cp.repeat(cp.repeat(base_y, t, axis=0), t, axis=1)

    yy, xx = cp.meshgrid(
        cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij"
    )

    best_cost = cp.full((n_patch_y, n_patch_x), cp.inf, dtype=cp.float32)
    best_sx = cp.zeros((n_patch_y, n_patch_x), dtype=cp.float32)
    best_sy = cp.zeros((n_patch_y, n_patch_x), dtype=cp.float32)

    R = search_radius
    for sy in range(-R, R + 1):
        for sx in range(-R, R + 1):
            coords = cp.stack([yy + fy + sy, xx + fx + sx], axis=0)
            shifted = map_coordinates(mov, coords, order=0, mode="constant", cval=0.0)
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


def upscale_warp_stats_cupy(local_stats, bayer_mode, tile_size=None, flow=None):
    """Dogson biquadratic upscale (x2 in bayer) + optional flow warp.

    Custom interpolation -> RawModule kernel.
    """
    LR = cp.ascontiguousarray(cp.asarray(local_stats, dtype=cp.float32))
    n_channels, lny, lnx = LR.shape
    upscale = 2 if bayer_mode else 1
    hny, hnx = lny * upscale, lnx * upscale
    HR = cp.empty((n_channels, hny, hnx), dtype=cp.float32)

    if flow is None:
        is_ref = 1
        flow_arr = cp.zeros((1, 1, 2), dtype=cp.float32)
        np_y = np_x = 1
        tile_size = 1
    else:
        is_ref = 0
        flow_arr = cp.ascontiguousarray(cp.asarray(flow, dtype=cp.float32))
        np_y, np_x, _ = flow_arr.shape

    kern = _get_module("dogson").get_function("upscale_dogson")
    tpb = (16, 16)
    bpg = ((hnx + 15) // 16, (hny + 15) // 16)
    kern(
        bpg, tpb,
        (LR, HR, flow_arr,
         np.int32(n_channels), np.int32(lny), np.int32(lnx),
         np.int32(hny), np.int32(hnx),
         np.int32(upscale), np.int32(is_ref), np.int32(tile_size),
         np.int32(np_y), np.int32(np_x)),
    )
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


# ---------------------------------------------------------------------------
# RawModules (compiled lazily, cached)
# ---------------------------------------------------------------------------
_MODULES = {}

_DOGSON_SRC = r'''
extern "C" {
__device__ float dogson_q(float x){
    float ax = fabsf(x);
    if (ax <= 0.5f) return -2.0f*ax*ax + 1.0f;
    else if (ax <= 1.5f) return ax*ax - 2.5f*ax + 1.5f;
    else return 0.0f;
}
__global__ void upscale_dogson(
    const float* LR, float* HR, const float* flow,
    int n_channels, int lny, int lnx, int hny, int hnx,
    int s, int is_ref, int tile_size, int np_y, int np_x)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= hnx || y >= hny) return;

    float flow_x = 0.0f, flow_y = 0.0f;
    if (!is_ref){
        int py = y / tile_size; int px = x / tile_size;
        if (py >= np_y) py = np_y - 1;
        if (px >= np_x) px = np_x - 1;
        flow_x = flow[(py*np_x + px)*2 + 0];
        flow_y = flow[(py*np_x + px)*2 + 1];
    }
    float LR_y = (y + flow_y + 0.5f)/s - 0.5f;
    float LR_x = (x + flow_x + 0.5f)/s - 0.5f;
    if (!(0.0f <= LR_y && LR_y < lny && 0.0f <= LR_x && LR_x < lnx)){
        float inf = __int_as_float(0x7f800000);
        for (int c=0;c<n_channels;c++) HR[(c*hny + y)*hnx + x] = inf;
        return;
    }
    int center_y = (int)rintf(LR_y);
    int center_x = (int)rintf(LR_x);
    float w_acc = 0.0f;
    float buf[3] = {0.0f,0.0f,0.0f};
    for (int i=-1;i<2;i++){
        int y_ = center_y + i; if (y_<0) y_=0; if (y_>lny-1) y_=lny-1;
        float wy = dogson_q((float)y_ - LR_y);
        for (int j=-1;j<2;j++){
            int x_ = center_x + j; if (x_<0) x_=0; if (x_>lnx-1) x_=lnx-1;
            float w = wy * dogson_q((float)x_ - LR_x);
            for (int c=0;c<n_channels;c++) buf[c] += LR[(c*lny + y_)*lnx + x_] * w;
            w_acc += w;
        }
    }
    for (int c=0;c<n_channels;c++) HR[(c*hny + y)*hnx + x] = buf[c]/w_acc;
}
}
'''

_MERGE_SRC = r'''
extern "C" {
__global__ void accumulate(
    const float* comp_img, const float* alignments, const float* covs, const float* r,
    int bayer_mode, int iso_kernel, float scale, int tile_size, const unsigned char* CFA,
    float* num, float* den,
    int hr_h, int hr_w, int lr_h, int lr_w, int np_y, int np_x, int cov_h, int cov_w)
{
    int hr_j = blockIdx.x*blockDim.x + threadIdx.x;
    int hr_i = blockIdx.y*blockDim.y + threadIdx.y;
    if (hr_j >= hr_w || hr_i >= hr_h) return;

    int n_channels = bayer_mode ? 3 : 1;
    float acc[3] = {0,0,0};
    float val[3] = {0,0,0};

    float lr_x = (hr_j + 0.5f)/scale;
    float lr_y = (hr_i + 0.5f)/scale;
    int px = (int)(lr_x/tile_size);
    int py = (int)(lr_y/tile_size);
    if (px >= np_x) px = np_x-1;
    if (py >= np_y) py = np_y-1;
    float flowx = alignments[(py*np_x + px)*2 + 0];
    float flowy = alignments[(py*np_x + px)*2 + 1];

    int i_r = min((int)lr_y, lr_h-1);
    int j_r = min((int)lr_x, lr_w-1);
    float local_r = r[i_r*lr_w + j_r];

    float lr_mov_x = lr_x + flowx;
    float lr_mov_y = lr_y + flowy;
    if (!(0.0f<=lr_mov_x && lr_mov_x<lr_w && 0.0f<=lr_mov_y && lr_mov_y<lr_h)) return;

    float cov_i_xx=0, cov_i_xy=0, cov_i_yy=0;
    if (!iso_kernel){
        float kmap_j, kmap_i;
        if (bayer_mode){ kmap_j = lr_mov_x/2.0f - 0.5f; kmap_i = lr_mov_y/2.0f - 0.5f; }
        else { kmap_j = lr_mov_x - 0.5f; kmap_i = lr_mov_y - 0.5f; }
        int ikx = (int)kmap_j; int iky = (int)kmap_i;
        float frac_x = kmap_j - (float)ikx;
        float frac_y = kmap_i - (float)iky;
        int floor_x = ikx>0?ikx:0; int floor_y = iky>0?iky:0;
        int ceil_x = min(floor_x+1, cov_w-1);
        int ceil_y = min(floor_y+1, cov_h-1);
        #define COV(yy,xx,a,b) covs[(((yy)*cov_w+(xx))*2 + (a))*2 + (b)]
        float tr_xx=COV(floor_y,floor_x,0,0), tr_xy=COV(floor_y,floor_x,0,1), tr_yy=COV(floor_y,floor_x,1,1);
        float tl_xx=COV(floor_y,ceil_x,0,0),  tl_xy=COV(floor_y,ceil_x,0,1),  tl_yy=COV(floor_y,ceil_x,1,1);
        float br_xx=COV(ceil_y,floor_x,0,0),  br_xy=COV(ceil_y,floor_x,0,1),  br_yy=COV(ceil_y,floor_x,1,1);
        float bl_xx=COV(ceil_y,ceil_x,0,0),   bl_xy=COV(ceil_y,ceil_x,0,1),   bl_yy=COV(ceil_y,ceil_x,1,1);
        #undef COV
        float top_xx=tr_xx+frac_x*(tl_xx-tr_xx), top_xy=tr_xy+frac_x*(tl_xy-tr_xy), top_yy=tr_yy+frac_x*(tl_yy-tr_yy);
        float bot_xx=br_xx+frac_x*(bl_xx-br_xx), bot_xy=br_xy+frac_x*(bl_xy-br_xy), bot_yy=br_yy+frac_x*(bl_yy-br_yy);
        float ixx=top_xx+frac_y*(bot_xx-top_xx), ixy=top_xy+frac_y*(bot_xy-top_xy), iyy=top_yy+frac_y*(bot_yy-top_yy);
        float det = ixx*iyy - ixy*ixy;
        float inv = 1.0f/det;
        cov_i_xx =  inv*iyy; cov_i_xy = -inv*ixy; cov_i_yy = inv*ixx;
    }

    int center_j = (int)lr_mov_x;
    int center_i = (int)lr_mov_y;
    float lr_mov_j = lr_mov_x - 0.5f;
    float lr_mov_i = lr_mov_y - 0.5f;
    for (int di=-1;di<2;di++){
        for (int dj=-1;dj<2;dj++){
            int j = center_j + dj;
            int i = center_i + di;
            if (!(0<=j && j<lr_w && 0<=i && i<lr_h)) continue;
            int channel = bayer_mode ? CFA[(i&1)*2 + (j&1)] : 0;
            float c = comp_img[i*lr_w + j];
            float dist_x = j - lr_mov_j;
            float dist_y = i - lr_mov_i;
            float z;
            if (iso_kernel) z = 2.0f*(dist_x*dist_x + dist_y*dist_y);
            else z = cov_i_xx*dist_x*dist_x + 2.0f*cov_i_xy*dist_x*dist_y + cov_i_yy*dist_y*dist_y;
            if (z < 0.0f) z = 0.0f;
            float w = expf(-0.5f*z);
            val[channel] += w*local_r*c;
            acc[channel] += w*local_r;
        }
    }
    for (int ch=0; ch<n_channels; ch++){
        num[(hr_i*hr_w + hr_j)*3 + ch] += val[ch];
        den[(hr_i*hr_w + hr_j)*3 + ch] += acc[ch];
    }
}

__global__ void accumulate_ref(
    const float* ref_img, const float* covs,
    int bayer_mode, int iso_kernel, float scale, const unsigned char* CFA,
    float* num, float* den, const float* acc_rob,
    int robustness_denoise, int max_frame_count, int rad_max, float max_multiplier,
    int hr_h, int hr_w, int lr_h, int lr_w, int cov_h, int cov_w, int accr_h, int accr_w)
{
    int hr_j = blockIdx.x*blockDim.x + threadIdx.x;
    int hr_i = blockIdx.y*blockDim.y + threadIdx.y;
    if (hr_j >= hr_w || hr_i >= hr_h) return;

    int n_channels = bayer_mode ? 3 : 1;
    float acc[3] = {0,0,0};
    float val[3] = {0,0,0};

    float crp_y = hr_i/scale;
    float crp_x = hr_j/scale;

    float cov_i_xx=0, cov_i_xy=0, cov_i_yy=0;
    if (!iso_kernel){
        float gy, gx;
        if (bayer_mode){ gy = (crp_y-0.5f)/2.0f; gx = (crp_x-0.5f)/2.0f; }
        else { gy = crp_y; gx = crp_x; }
        int floor_x = (int)floorf(gx); if (floor_x<0) floor_x=0;
        int floor_y = (int)floorf(gy); if (floor_y<0) floor_y=0;
        int ceil_x = min(floor_x+1, cov_w-1);
        int ceil_y = min(floor_y+1, cov_h-1);
        float fx = gx - floorf(gx);
        float fy = gy - floorf(gy);
        #define COV(yy,xx,a,b) covs[(((yy)*cov_w+(xx))*2 + (a))*2 + (b)]
        float c00_xx=COV(floor_y,floor_x,0,0), c00_xy=COV(floor_y,floor_x,0,1), c00_yy=COV(floor_y,floor_x,1,1);
        float c01_xx=COV(floor_y,ceil_x,0,0),  c01_xy=COV(floor_y,ceil_x,0,1),  c01_yy=COV(floor_y,ceil_x,1,1);
        float c10_xx=COV(ceil_y,floor_x,0,0),  c10_xy=COV(ceil_y,floor_x,0,1),  c10_yy=COV(ceil_y,floor_x,1,1);
        float c11_xx=COV(ceil_y,ceil_x,0,0),   c11_xy=COV(ceil_y,ceil_x,0,1),   c11_yy=COV(ceil_y,ceil_x,1,1);
        #undef COV
        float ixx = c00_xx*(1-fx)*(1-fy)+c01_xx*fx*(1-fy)+c10_xx*(1-fx)*fy+c11_xx*fx*fy;
        float ixy = c00_xy*(1-fx)*(1-fy)+c01_xy*fx*(1-fy)+c10_xy*(1-fx)*fy+c11_xy*fx*fy;
        float iyy = c00_yy*(1-fx)*(1-fy)+c01_yy*fx*(1-fy)+c10_yy*(1-fx)*fy+c11_yy*fx*fy;
        float det = ixx*iyy - ixy*ixy;
        float inv = (fabsf(det) > 1e-10f) ? 1.0f/det : 1.0f;
        cov_i_xx = inv*iyy; cov_i_xy = -inv*ixy; cov_i_yy = inv*ixx;
    }

    int rad = 1;
    float add_denoise = 1.0f;
    float local_acc_r = 0.0f;
    if (robustness_denoise){
        int ir = min((int)rintf(crp_y), accr_h-1);
        int jr = min((int)rintf(crp_x), accr_w-1);
        local_acc_r = acc_rob[ir*accr_w + jr];
        add_denoise = (local_acc_r <= max_frame_count) ? max_multiplier : 1.0f;
        rad = (local_acc_r <= max_frame_count) ? rad_max : 1;
    }

    int center_x = (int)rintf(crp_x);
    int center_y = (int)rintf(crp_y);
    for (int i=-rad;i<rad+1;i++){
        for (int j=-rad;j<rad+1;j++){
            int pidx = center_x + j;
            int pidy = center_y + i;
            if (!(0<=pidx && pidx<lr_w && 0<=pidy && pidy<lr_h)) continue;
            int channel = bayer_mode ? CFA[(pidy&1)*2 + (pidx&1)] : 0;
            float c = ref_img[pidy*lr_w + pidx];
            float dist_x = pidx - crp_x;
            float dist_y = pidy - crp_y;
            float y;
            if (iso_kernel) y = 2.0f*(dist_x*dist_x + dist_y*dist_y);
            else y = cov_i_xx*dist_x*dist_x + 2.0f*cov_i_xy*dist_x*dist_y + cov_i_yy*dist_y*dist_y;
            if (y < 0.0f) y = 0.0f;
            y /= add_denoise;
            float w = expf(-0.5f*y);
            val[channel] += c*w;
            acc[channel] += w;
        }
    }
    if (robustness_denoise && local_acc_r < max_frame_count){
        for (int ch=0; ch<n_channels; ch++){
            num[(hr_i*hr_w + hr_j)*3 + ch] = val[ch];
            den[(hr_i*hr_w + hr_j)*3 + ch] = acc[ch];
        }
    } else {
        for (int ch=0; ch<n_channels; ch++){
            num[(hr_i*hr_w + hr_j)*3 + ch] += val[ch];
            den[(hr_i*hr_w + hr_j)*3 + ch] += acc[ch];
        }
    }
}
}
'''

_MODULE_SRC = {"dogson": _DOGSON_SRC, "merge": _MERGE_SRC}


def _get_module(name):
    if name not in _MODULES:
        _MODULES[name] = cp.RawModule(code=_MODULE_SRC[name], options=("--std=c++14",))
    return _MODULES[name]


def merge_accumulate_cupy(comp_img, alignments, covs, r, bayer_mode, iso_kernel,
                          scale, tile_size, cfa_pattern, num, den):
    """Alg. 4 accumulation of a comparison frame into num/den (in place)."""
    comp = cp.ascontiguousarray(cp.asarray(comp_img, dtype=cp.float32))
    al = cp.ascontiguousarray(cp.asarray(alignments, dtype=cp.float32))
    covs = cp.ascontiguousarray(cp.asarray(covs, dtype=cp.float32))
    r = cp.ascontiguousarray(cp.asarray(r, dtype=cp.float32))
    cfa = cp.ascontiguousarray(cp.asarray(cfa_pattern).astype(cp.uint8))
    num = cp.asarray(num)
    den = cp.asarray(den)

    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = comp.shape
    np_y, np_x, _ = al.shape
    cov_h, cov_w = covs.shape[:2]

    kern = _get_module("merge").get_function("accumulate")
    tpb = (16, 16)
    bpg = ((hr_w + 15) // 16, (hr_h + 15) // 16)
    kern(bpg, tpb, (
        comp, al, covs, r,
        np.int32(bayer_mode), np.int32(iso_kernel), np.float32(scale),
        np.int32(tile_size), cfa, num, den,
        np.int32(hr_h), np.int32(hr_w), np.int32(lr_h), np.int32(lr_w),
        np.int32(np_y), np.int32(np_x), np.int32(cov_h), np.int32(cov_w),
    ))
    return num, den


def merge_ref_accumulate_cupy(ref_img, covs, bayer_mode, iso_kernel, scale,
                              cfa_pattern, num, den, acc_rob,
                              robustness_denoise, max_frame_count, rad_max, max_multiplier):
    """Alg. 11 accumulation of the reference frame into num/den (in place)."""
    ref = cp.ascontiguousarray(cp.asarray(ref_img, dtype=cp.float32))
    covs = cp.ascontiguousarray(cp.asarray(covs, dtype=cp.float32))
    cfa = cp.ascontiguousarray(cp.asarray(cfa_pattern).astype(cp.uint8))
    num = cp.asarray(num)
    den = cp.asarray(den)
    if acc_rob is None:
        acc_rob = cp.zeros((1, 1), dtype=cp.float32)
    acc_rob = cp.ascontiguousarray(cp.asarray(acc_rob, dtype=cp.float32))

    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = ref.shape
    cov_h, cov_w = covs.shape[:2]
    accr_h, accr_w = acc_rob.shape

    kern = _get_module("merge").get_function("accumulate_ref")
    tpb = (16, 16)
    bpg = ((hr_w + 15) // 16, (hr_h + 15) // 16)
    kern(bpg, tpb, (
        ref, covs,
        np.int32(bayer_mode), np.int32(iso_kernel), np.float32(scale), cfa,
        num, den, acc_rob,
        np.int32(robustness_denoise), np.int32(max_frame_count), np.int32(rad_max),
        np.float32(max_multiplier),
        np.int32(hr_h), np.int32(hr_w), np.int32(lr_h), np.int32(lr_w),
        np.int32(cov_h), np.int32(cov_w), np.int32(accr_h), np.int32(accr_w),
    ))
    return num, den
