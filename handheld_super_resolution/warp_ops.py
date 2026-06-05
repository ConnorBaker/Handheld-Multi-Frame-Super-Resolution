"""Merge accumulation as NVIDIA Warp kernels (Python -> CUDA, JIT-compiled, fused).

The merge is a bounded per-output-pixel gather (3x3 neighbourhood, anisotropic
kernel weights, Bayer channel routing) -- not a tile/matmul pattern and not
fusable by ``cupy.fuse`` (it has gathers). cupyx.jit can't easily express it
(strict uint32/int32/float typing). Warp's kernel DSL handles it cleanly and
JITs (via NVRTC) to a single fused kernel on the GB10, restoring RawKernel-class
speed while staying readable Python.

Same arithmetic as the reference RawModule in test/reference_kernels.py; verified
by test/test_kernel_equivalence.py. CuPy arrays are wrapped zero-copy via DLPack,
so writes land in the caller's num/den.
"""

import cupy as cp
import warp as wp

wp.init()


@wp.kernel
def _accumulate_warp(comp: wp.array2d(dtype=wp.float32),
                     al: wp.array3d(dtype=wp.float32),
                     covs: wp.array4d(dtype=wp.float32),
                     r: wp.array2d(dtype=wp.float32),
                     bayer: wp.int32, iso: wp.int32, scale: wp.float32, tile: wp.int32,
                     cfa: wp.array2d(dtype=wp.int32),
                     num: wp.array3d(dtype=wp.float32), den: wp.array3d(dtype=wp.float32),
                     lr_h: wp.int32, lr_w: wp.int32, np_y: wp.int32, np_x: wp.int32,
                     cov_h: wp.int32, cov_w: wp.int32):
    i, j = wp.tid()  # i = hr row, j = hr col
    lr_x = (float(j) + 0.5) / scale
    lr_y = (float(i) + 0.5) / scale
    px = int(lr_x / float(tile))
    py = int(lr_y / float(tile))
    if px >= np_x:
        px = np_x - 1
    if py >= np_y:
        py = np_y - 1
    flowx = al[py, px, 0]
    flowy = al[py, px, 1]
    i_r = wp.min(int(lr_y), lr_h - 1)
    j_r = wp.min(int(lr_x), lr_w - 1)
    local_r = r[i_r, j_r]
    lr_mov_x = lr_x + flowx
    lr_mov_y = lr_y + flowy
    if lr_mov_x < 0.0 or lr_mov_x >= float(lr_w) or lr_mov_y < 0.0 or lr_mov_y >= float(lr_h):
        return

    cov_i_xx = float(0.0)
    cov_i_xy = float(0.0)
    cov_i_yy = float(0.0)
    if iso == 0:
        if bayer != 0:
            kmap_j = lr_mov_x / 2.0 - 0.5
            kmap_i = lr_mov_y / 2.0 - 0.5
        else:
            kmap_j = lr_mov_x - 0.5
            kmap_i = lr_mov_y - 0.5
        ikx = int(kmap_j)
        iky = int(kmap_i)
        frac_x = kmap_j - float(ikx)
        frac_y = kmap_i - float(iky)
        floor_x = wp.max(ikx, 0)
        floor_y = wp.max(iky, 0)
        ceil_x = wp.min(floor_x + 1, cov_w - 1)
        ceil_y = wp.min(floor_y + 1, cov_h - 1)
        top_xx = covs[floor_y, floor_x, 0, 0] + frac_x * (covs[floor_y, ceil_x, 0, 0] - covs[floor_y, floor_x, 0, 0])
        top_xy = covs[floor_y, floor_x, 0, 1] + frac_x * (covs[floor_y, ceil_x, 0, 1] - covs[floor_y, floor_x, 0, 1])
        top_yy = covs[floor_y, floor_x, 1, 1] + frac_x * (covs[floor_y, ceil_x, 1, 1] - covs[floor_y, floor_x, 1, 1])
        bot_xx = covs[ceil_y, floor_x, 0, 0] + frac_x * (covs[ceil_y, ceil_x, 0, 0] - covs[ceil_y, floor_x, 0, 0])
        bot_xy = covs[ceil_y, floor_x, 0, 1] + frac_x * (covs[ceil_y, ceil_x, 0, 1] - covs[ceil_y, floor_x, 0, 1])
        bot_yy = covs[ceil_y, floor_x, 1, 1] + frac_x * (covs[ceil_y, ceil_x, 1, 1] - covs[ceil_y, floor_x, 1, 1])
        ixx = top_xx + frac_y * (bot_xx - top_xx)
        ixy = top_xy + frac_y * (bot_xy - top_xy)
        iyy = top_yy + frac_y * (bot_yy - top_yy)
        det = ixx * iyy - ixy * ixy
        inv = 1.0 / det
        cov_i_xx = inv * iyy
        cov_i_xy = -inv * ixy
        cov_i_yy = inv * ixx

    center_j = int(lr_mov_x)
    center_i = int(lr_mov_y)
    lr_mov_j = lr_mov_x - 0.5
    lr_mov_i = lr_mov_y - 0.5
    val0 = float(0.0)
    val1 = float(0.0)
    val2 = float(0.0)
    acc0 = float(0.0)
    acc1 = float(0.0)
    acc2 = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            jj = center_j + dj
            ii = center_i + di
            if jj >= 0 and jj < lr_w and ii >= 0 and ii < lr_h:
                channel = int(0)
                if bayer != 0:
                    channel = cfa[ii & 1, jj & 1]
                c = comp[ii, jj]
                dist_x = float(jj) - lr_mov_j
                dist_y = float(ii) - lr_mov_i
                z = float(0.0)
                if iso != 0:
                    z = 2.0 * (dist_x * dist_x + dist_y * dist_y)
                else:
                    z = cov_i_xx * dist_x * dist_x + 2.0 * cov_i_xy * dist_x * dist_y + cov_i_yy * dist_y * dist_y
                if z < 0.0:
                    z = 0.0
                w = wp.exp(-0.5 * z)
                wv = w * local_r * c
                wa = w * local_r
                if channel == 0:
                    val0 += wv
                    acc0 += wa
                elif channel == 1:
                    val1 += wv
                    acc1 += wa
                else:
                    val2 += wv
                    acc2 += wa
    num[i, j, 0] = num[i, j, 0] + val0
    den[i, j, 0] = den[i, j, 0] + acc0
    if bayer != 0:
        num[i, j, 1] = num[i, j, 1] + val1
        den[i, j, 1] = den[i, j, 1] + acc1
        num[i, j, 2] = num[i, j, 2] + val2
        den[i, j, 2] = den[i, j, 2] + acc2


@wp.kernel
def _accumulate_ref_warp(ref: wp.array2d(dtype=wp.float32),
                         covs: wp.array4d(dtype=wp.float32),
                         bayer: wp.int32, iso: wp.int32, scale: wp.float32,
                         cfa: wp.array2d(dtype=wp.int32),
                         num: wp.array3d(dtype=wp.float32), den: wp.array3d(dtype=wp.float32),
                         acc_rob: wp.array2d(dtype=wp.float32),
                         robustness_denoise: wp.int32, max_frame_count: wp.int32,
                         rad_max: wp.int32, max_multiplier: wp.float32,
                         lr_h: wp.int32, lr_w: wp.int32, cov_h: wp.int32, cov_w: wp.int32,
                         accr_h: wp.int32, accr_w: wp.int32):
    i, j = wp.tid()
    crp_y = float(i) / scale
    crp_x = float(j) / scale

    cov_i_xx = float(0.0)
    cov_i_xy = float(0.0)
    cov_i_yy = float(0.0)
    if iso == 0:
        if bayer != 0:
            gy = (crp_y - 0.5) / 2.0
            gx = (crp_x - 0.5) / 2.0
        else:
            gy = crp_y
            gx = crp_x
        floor_x = wp.max(int(wp.floor(gx)), 0)
        floor_y = wp.max(int(wp.floor(gy)), 0)
        ceil_x = wp.min(floor_x + 1, cov_w - 1)
        ceil_y = wp.min(floor_y + 1, cov_h - 1)
        fx = gx - wp.floor(gx)
        fy = gy - wp.floor(gy)
        ixx = (covs[floor_y, floor_x, 0, 0] * (1.0 - fx) * (1.0 - fy) + covs[floor_y, ceil_x, 0, 0] * fx * (1.0 - fy)
               + covs[ceil_y, floor_x, 0, 0] * (1.0 - fx) * fy + covs[ceil_y, ceil_x, 0, 0] * fx * fy)
        ixy = (covs[floor_y, floor_x, 0, 1] * (1.0 - fx) * (1.0 - fy) + covs[floor_y, ceil_x, 0, 1] * fx * (1.0 - fy)
               + covs[ceil_y, floor_x, 0, 1] * (1.0 - fx) * fy + covs[ceil_y, ceil_x, 0, 1] * fx * fy)
        iyy = (covs[floor_y, floor_x, 1, 1] * (1.0 - fx) * (1.0 - fy) + covs[floor_y, ceil_x, 1, 1] * fx * (1.0 - fy)
               + covs[ceil_y, floor_x, 1, 1] * (1.0 - fx) * fy + covs[ceil_y, ceil_x, 1, 1] * fx * fy)
        det = ixx * iyy - ixy * ixy
        inv = float(1.0)
        if wp.abs(det) > 1e-10:
            inv = 1.0 / det
        cov_i_xx = inv * iyy
        cov_i_xy = -inv * ixy
        cov_i_yy = inv * ixx

    rad = int(1)
    add_denoise = float(1.0)
    local_acc_r = float(0.0)
    if robustness_denoise != 0:
        ir = wp.min(int(wp.rint(crp_y)), accr_h - 1)
        jr = wp.min(int(wp.rint(crp_x)), accr_w - 1)
        local_acc_r = acc_rob[ir, jr]
        if local_acc_r <= float(max_frame_count):
            add_denoise = max_multiplier
            rad = rad_max
        else:
            add_denoise = 1.0
            rad = 1

    center_x = int(wp.rint(crp_x))
    center_y = int(wp.rint(crp_y))
    val0 = float(0.0)
    val1 = float(0.0)
    val2 = float(0.0)
    acc0 = float(0.0)
    acc1 = float(0.0)
    acc2 = float(0.0)
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            pidx = center_x + dj
            pidy = center_y + di
            if pidx >= 0 and pidx < lr_w and pidy >= 0 and pidy < lr_h:
                channel = int(0)
                if bayer != 0:
                    channel = cfa[pidy & 1, pidx & 1]
                c = ref[pidy, pidx]
                dist_x = float(pidx) - crp_x
                dist_y = float(pidy) - crp_y
                y = float(0.0)
                if iso != 0:
                    y = 2.0 * (dist_x * dist_x + dist_y * dist_y)
                else:
                    y = cov_i_xx * dist_x * dist_x + 2.0 * cov_i_xy * dist_x * dist_y + cov_i_yy * dist_y * dist_y
                if y < 0.0:
                    y = 0.0
                y = y / add_denoise
                w = wp.exp(-0.5 * y)
                if channel == 0:
                    val0 += c * w
                    acc0 += w
                elif channel == 1:
                    val1 += c * w
                    acc1 += w
                else:
                    val2 += c * w
                    acc2 += w

    overwrite = robustness_denoise != 0 and local_acc_r < float(max_frame_count)
    if overwrite:
        num[i, j, 0] = val0
        den[i, j, 0] = acc0
        if bayer != 0:
            num[i, j, 1] = val1
            den[i, j, 1] = acc1
            num[i, j, 2] = val2
            den[i, j, 2] = acc2
    else:
        num[i, j, 0] = num[i, j, 0] + val0
        den[i, j, 0] = den[i, j, 0] + acc0
        if bayer != 0:
            num[i, j, 1] = num[i, j, 1] + val1
            den[i, j, 1] = den[i, j, 1] + acc1
            num[i, j, 2] = num[i, j, 2] + val2
            den[i, j, 2] = den[i, j, 2] + acc2


def _wp(arr, dtype):
    """Wrap a cupy array as a Warp array (zero-copy via DLPack)."""
    return wp.from_dlpack(cp.ascontiguousarray(cp.asarray(arr, dtype=dtype)))


def _wp_device(cp_array):
    """Warp device string ('cuda:N') for a cupy array's device."""
    return "cuda:%d" % cp_array.device.id


def merge_accumulate_cupy(comp_img, alignments, covs, r, bayer_mode, iso_kernel,
                          scale, tile_size, cfa_pattern, num, den):
    """Alg. 4 accumulation of a comparison frame into num/den (in place)."""
    num = cp.ascontiguousarray(cp.asarray(num, dtype=cp.float32))
    den = cp.ascontiguousarray(cp.asarray(den, dtype=cp.float32))
    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = comp_img.shape
    np_y, np_x, _ = alignments.shape
    cov_h, cov_w = covs.shape[:2]
    wp.launch(_accumulate_warp, dim=(hr_h, hr_w), inputs=[
        _wp(comp_img, cp.float32), _wp(alignments, cp.float32), _wp(covs, cp.float32),
        _wp(r, cp.float32), int(bayer_mode), int(iso_kernel), float(scale), int(tile_size),
        _wp(cfa_pattern, cp.int32), _wp(num, cp.float32), _wp(den, cp.float32),
        int(lr_h), int(lr_w), int(np_y), int(np_x), int(cov_h), int(cov_w)],
        device=_wp_device(num))
    # The merge's per-output-pixel read-modify-write into num/den runs on Warp's
    # private CUDA stream; without this sync it races default-stream (cupy/torch)
    # work on the shared accumulators and the output is nondeterministic (~0.18/px,
    # verified). See REFACTORING_PLAN.md "align/merge nondeterminism".
    wp.synchronize_device()
    return num, den


def merge_ref_accumulate_cupy(ref_img, covs, bayer_mode, iso_kernel, scale,
                              cfa_pattern, num, den, acc_rob,
                              robustness_denoise, max_frame_count, rad_max, max_multiplier):
    """Alg. 11 accumulation of the reference frame into num/den (in place)."""
    num = cp.ascontiguousarray(cp.asarray(num, dtype=cp.float32))
    den = cp.ascontiguousarray(cp.asarray(den, dtype=cp.float32))
    if acc_rob is None:
        acc_rob = cp.zeros((1, 1), dtype=cp.float32)
    hr_h, hr_w, _ = num.shape
    lr_h, lr_w = ref_img.shape
    cov_h, cov_w = covs.shape[:2]
    accr_h, accr_w = acc_rob.shape
    wp.launch(_accumulate_ref_warp, dim=(hr_h, hr_w), inputs=[
        _wp(ref_img, cp.float32), _wp(covs, cp.float32),
        int(bayer_mode), int(iso_kernel), float(scale), _wp(cfa_pattern, cp.int32),
        _wp(num, cp.float32), _wp(den, cp.float32), _wp(acc_rob, cp.float32),
        int(robustness_denoise), int(max_frame_count), int(rad_max), float(max_multiplier),
        int(lr_h), int(lr_w), int(cov_h), int(cov_w), int(accr_h), int(accr_w)],
        device=_wp_device(num))
    # The merge's per-output-pixel read-modify-write into num/den runs on Warp's
    # private CUDA stream; without this sync it races default-stream (cupy/torch)
    # work on the shared accumulators and the output is nondeterministic (~0.18/px,
    # verified). See REFACTORING_PLAN.md "align/merge nondeterminism".
    wp.synchronize_device()
    return num, den
