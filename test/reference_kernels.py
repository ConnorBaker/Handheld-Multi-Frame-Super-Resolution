"""Original (custom-kernel) implementations, preserved as regression references.

These are the FIRST CuPy ports we wrote: hand-written ``RawModule`` kernels for
merge + the Dogson warp-upscale, a ``map_coordinates`` integer-shift L1 search,
and a 2-D ones-conv box filter. Production code in
``handheld_super_resolution`` has since been rewritten to use stock CuPy / torch
primitives instead; ``test/test_kernel_equivalence.py`` asserts the new
implementations match these references, so a regression in the rewrite is caught.

Do not import these from the pipeline -- they exist only for the test suite.
"""

import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from cupyx.scipy.ndimage import map_coordinates

from handheld_super_resolution.utils import DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE


# ===========================================================================
# RawModule kernels (merge + Dogson upscale)
# ===========================================================================
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
        // Clamp floor to the last valid cov cell (not just >=0): an odd LR dim at
        // x2 scale pushes the last HR pixel's index one past the end, and the
        // gather below otherwise reads covs out of bounds. The original numba
        // kernel had this same latent OOB; clamping makes the boundary defined.
        int floor_x = min(ikx>0?ikx:0, cov_w-1); int floor_y = min(iky>0?iky:0, cov_h-1);
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
        // Clamp floor to the last valid cov cell at both ends (see accumulate):
        // odd LR dim at x2 scale otherwise indexes covs one past the end.
        int floor_x = (int)floorf(gx); if (floor_x<0) floor_x=0; if (floor_x>cov_w-1) floor_x=cov_w-1;
        int floor_y = (int)floorf(gy); if (floor_y<0) floor_y=0; if (floor_y>cov_h-1) floor_y=cov_h-1;
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

_SRC = {"dogson": _DOGSON_SRC, "merge": _MERGE_SRC}


def _get_module(name):
    if name not in _MODULES:
        _MODULES[name] = cp.RawModule(code=_SRC[name], options=("--std=c++14",))
    return _MODULES[name]


def reference_dogson_upscale(local_stats, bayer_mode, tile_size=None, flow=None):
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
    bpg = ((hnx + 15) // 16, (hny + 15) // 16)
    kern(bpg, (16, 16), (LR, HR, flow_arr,
                         np.int32(n_channels), np.int32(lny), np.int32(lnx),
                         np.int32(hny), np.int32(hnx),
                         np.int32(upscale), np.int32(is_ref), np.int32(tile_size),
                         np.int32(np_y), np.int32(np_x)))
    return HR


def reference_merge_accumulate(comp_img, alignments, covs, r, bayer_mode, iso_kernel,
                               scale, tile_size, cfa_pattern, num, den):
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
    bpg = ((hr_w + 15) // 16, (hr_h + 15) // 16)
    kern(bpg, (16, 16), (comp, al, covs, r,
                         np.int32(bayer_mode), np.int32(iso_kernel), np.float32(scale),
                         np.int32(tile_size), cfa, num, den,
                         np.int32(hr_h), np.int32(hr_w), np.int32(lr_h), np.int32(lr_w),
                         np.int32(np_y), np.int32(np_x), np.int32(cov_h), np.int32(cov_w)))
    return num, den


def reference_merge_ref_accumulate(ref_img, covs, bayer_mode, iso_kernel, scale,
                                   cfa_pattern, num, den, acc_rob,
                                   robustness_denoise, max_frame_count, rad_max, max_multiplier):
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
    bpg = ((hr_w + 15) // 16, (hr_h + 15) // 16)
    kern(bpg, (16, 16), (ref, covs,
                         np.int32(bayer_mode), np.int32(iso_kernel), np.float32(scale), cfa,
                         num, den, acc_rob,
                         np.int32(robustness_denoise), np.int32(max_frame_count), np.int32(rad_max),
                         np.float32(max_multiplier),
                         np.int32(hr_h), np.int32(hr_w), np.int32(lr_h), np.int32(lr_w),
                         np.int32(cov_h), np.int32(cov_w), np.int32(accr_h), np.int32(accr_w)))
    return num, den


# ===========================================================================
# map_coordinates L1 block matching + 2-D ones-conv box filter
# ===========================================================================
def reference_block_matching_l1(ref_lvl, moving_lvl, alignments, tile_size, search_radius):
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
    yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij")
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


def reference_box_filter(search_area, tile_size):
    """2-D ones-conv windowed sum (valid) of search_area**2. search_area is 4-D
    [n, m, S, S] like in align_lvl_block_matching_L2."""
    S = search_area.shape[-1]
    box = torch.ones((1, 1, tile_size, tile_size), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    out = F.conv2d(search_area.reshape(-1, 1, S, S).square(), box, padding="valid")
    return out.view(search_area.shape[0], search_area.shape[1], out.shape[-2], out.shape[-1])
