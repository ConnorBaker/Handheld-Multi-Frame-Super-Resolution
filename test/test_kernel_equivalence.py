"""Regression: the rewritten pure-CuPy / library implementations must match the
original custom-kernel implementations preserved in test/reference_kernels.py.

Covers the box filter (avg_pool2d vs ones-conv), L1 block matching (integer
gather vs map_coordinates), the Dogson warp-upscale (vectorized vs RawKernel),
and merge accumulate / accumulate_ref (vectorized vs RawKernel) across the
config space (bayer/grey, iso/steerable, scale 1/2, robustness denoiser).
"""

import numpy as np
import cupy as cp
import torch
import pytest

import handheld_super_resolution.cupy_ops as ops
from . import reference_kernels as ref

CFA = np.array([[1, 0], [2, 1]])  # GRBG


def _pd_covs(ch, cw, seed=0):
    """Random per-pixel symmetric positive-definite 2x2 cov grid (invertible)."""
    rng = np.random.default_rng(seed)
    M = cp.asarray(rng.standard_normal((ch, cw, 2, 2)).astype(np.float32))
    cov = cp.matmul(M, cp.swapaxes(M, -1, -2)) + 0.5 * cp.eye(2, dtype=cp.float32)
    return cov.astype(cp.float32)


# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", [4, 8, 12, 16, 32])
def test_box_filter_avgpool_vs_conv(t):
    np.random.seed(t)
    S = 2 * 4 + t
    sa = torch.as_tensor(np.random.rand(3, 5, S, S).astype(np.float32), device="cuda")
    sq = sa.view(-1, 1, S, S).square()
    new = torch.nn.functional.avg_pool2d(sq, t, stride=1) * (t * t)
    new = new.view(3, 5, new.shape[-2], new.shape[-1])
    old = ref.reference_box_filter(sa, t)
    assert new.shape == old.shape
    assert torch.allclose(new, old, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("t,R,seed", [(8, 1, 0), (16, 4, 1), (16, 1, 2), (32, 4, 3), (4, 1, 4)])
def test_l1_gather_vs_mapcoordinates(t, R, seed):
    np.random.seed(seed)
    npy, npx = 6, 7
    H, W = npy * t, npx * t
    ref_img = cp.asarray(np.random.rand(H + 5, W + 3).astype(np.float32))
    mov = cp.asarray(np.random.rand(H + 5, W + 3).astype(np.float32))
    al = cp.asarray((np.random.rand(npy, npx, 2) * 6 - 3).astype(np.float32))
    new = ops.block_matching_l1_cupy(ref_img, mov, al, t, R)
    old = ref.reference_block_matching_l1(ref_img, mov, al, t, R)
    assert cp.allclose(new, old)  # integer gather == map_coordinates(order=0)


@pytest.mark.parametrize("bayer", [True, False])
@pytest.mark.parametrize("warp", [False, True])
def test_dogson_vectorized_vs_rawkernel(bayer, warp):
    np.random.seed(0)
    nc = 3 if bayer else 1
    lny, lnx = 40, 48
    ls = cp.asarray(np.random.rand(nc, lny, lnx).astype(np.float32))
    if warp:
        s = 2 if bayer else 1
        ts = 16
        npy, npx = (lny * s) // ts + 2, (lnx * s) // ts + 2
        flow = cp.asarray((np.random.rand(npy, npx, 2) * 4 - 2).astype(np.float32))
    else:
        ts, flow = None, None
    new = ops.upscale_warp_stats_cupy(ls, bayer, ts, flow)
    old = ref.reference_dogson_upscale(ls, bayer, ts, flow)
    assert cp.array_equal(cp.isinf(new), cp.isinf(old))  # same OOB mask
    fin = ~cp.isinf(new)
    assert cp.allclose(new[fin], old[fin], rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("bayer,iso", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_merge_accumulate_vectorized_vs_rawkernel(bayer, iso, scale, seed):
    np.random.seed(seed)
    lr_h, lr_w, tile = 32, 40, 8
    npy, npx = lr_h // tile, lr_w // tile
    comp = cp.asarray(np.random.rand(lr_h, lr_w).astype(np.float32))
    al = cp.asarray((np.random.rand(npy, npx, 2) * 4 - 2).astype(np.float32))
    # cov grid is grey-resolution: lr//2 in bayer mode, full lr in grey mode
    cov_h = lr_h // 2 if bayer else lr_h
    cov_w = lr_w // 2 if bayer else lr_w
    covs = _pd_covs(cov_h, cov_w, seed)
    r = cp.asarray(np.random.rand(lr_h, lr_w).astype(np.float32))
    hr_h, hr_w = lr_h * scale, lr_w * scale
    num0 = cp.asarray(np.random.rand(hr_h, hr_w, 3).astype(np.float32))
    den0 = cp.asarray(np.random.rand(hr_h, hr_w, 3).astype(np.float32))
    args = (comp, al, covs, r, int(bayer), int(iso), float(scale), tile, CFA)
    n_new, d_new = ops.merge_accumulate_cupy(*args, num0.copy(), den0.copy())
    n_old, d_old = ref.reference_merge_accumulate(*args, num0.copy(), den0.copy())
    assert cp.allclose(n_new, n_old, rtol=1e-3, atol=1e-4)
    assert cp.allclose(d_new, d_old, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("bayer,iso", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("rden", [False, True])
def test_merge_ref_vectorized_vs_rawkernel(bayer, iso, scale, rden):
    np.random.seed(7)
    lr_h, lr_w = 32, 40
    ref_img = cp.asarray(np.random.rand(lr_h, lr_w).astype(np.float32))
    cov_h = lr_h // 2 if bayer else lr_h
    cov_w = lr_w // 2 if bayer else lr_w
    covs = _pd_covs(cov_h, cov_w, 7)
    hr_h, hr_w = lr_h * scale, lr_w * scale
    num0 = cp.asarray(np.random.rand(hr_h, hr_w, 3).astype(np.float32))
    den0 = cp.asarray(np.random.rand(hr_h, hr_w, 3).astype(np.float32))
    if rden:
        acc_rob = cp.asarray((np.random.rand(lr_h, lr_w) * 8).astype(np.float32))
        mfc, rmax, mult = 2, 2, 8.0
    else:
        acc_rob, mfc, rmax, mult = None, 0, 0, 0.0
    args = (ref_img, covs, int(bayer), int(iso), float(scale), CFA)
    tail = (acc_rob, int(rden), mfc, rmax, float(mult))
    n_new, d_new = ops.merge_ref_accumulate_cupy(*args, num0.copy(), den0.copy(), *tail)
    n_old, d_old = ref.reference_merge_ref_accumulate(*args, num0.copy(), den0.copy(), *tail)
    assert cp.allclose(n_new, n_old, rtol=1e-3, atol=1e-4)
    assert cp.allclose(d_new, d_old, rtol=1e-3, atol=1e-4)
