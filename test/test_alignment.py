"""Alignment correctness tests for the CuPy ports (numba can't JIT here).

Verifies the CuPy block-matching (L1, integer) and ICA (subpixel) recover known
synthetic shifts on interior tiles. ``moving[y, x] = ref[y - dy, x - dx]`` so the
recovered flow should be ``(dx, dy)``.
"""

import numpy as np
import cupy as cp
import pytest
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

from handheld_super_resolution.utils_image import compute_gradient_cupy, compute_hessian_cupy
from handheld_super_resolution.cupy_ops import ica_iterate_cupy, block_matching_l1_cupy

H = 128
TILE = 16
NP = H // TILE


def _ref(seed=0):
    np.random.seed(seed)
    return cp.asarray(gaussian_filter(cp.asarray(np.random.rand(H, H).astype(np.float32)), 2.0))


def _shift(ref, dx, dy):
    yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.float32), cp.arange(H, dtype=cp.float32), indexing="ij")
    return map_coordinates(ref, cp.stack([yy - dy, xx - dx], 0), order=1, mode="reflect")


@pytest.mark.parametrize("dx,dy", [(2, -1), (3, 3), (-4, 2)])
def test_l1_recovers_integer_shift(dx, dy):
    ref = _ref()
    mov = _shift(ref, dx, dy)
    flow = block_matching_l1_cupy(ref, mov, cp.zeros((NP, NP, 2), cp.float32), TILE, 4)
    cy, cx = NP // 2, NP // 2  # interior tile
    assert float(flow[cy, cx, 0]) == dx
    assert float(flow[cy, cx, 1]) == dy


@pytest.mark.parametrize("dx,dy", [(0.5, 0.3), (-0.4, 0.6), (0.8, -0.7)])
def test_ica_recovers_subpixel_shift(dx, dy):
    ref = _ref()
    gx, gy = compute_gradient_cupy(ref)
    hess = compute_hessian_cupy(gx, gy, TILE)
    mov = _shift(ref, dx, dy)
    flow = ica_iterate_cupy(ref, gx, gy, hess, mov, cp.zeros((NP, NP, 2), cp.float32), TILE, 30)
    cy, cx = NP // 2, NP // 2
    err = abs(float(flow[cy, cx, 0]) - dx) + abs(float(flow[cy, cx, 1]) - dy)
    assert err < 0.1


@pytest.mark.parametrize("dx,dy", [(2.4, -1.3), (3.7, 2.2)])
def test_l1_then_ica_recovers_combined_shift(dx, dy):
    ref = _ref()
    gx, gy = compute_gradient_cupy(ref)
    hess = compute_hessian_cupy(gx, gy, TILE)
    mov = _shift(ref, dx, dy)
    flow = block_matching_l1_cupy(ref, mov, cp.zeros((NP, NP, 2), cp.float32), TILE, 4)
    flow = ica_iterate_cupy(ref, gx, gy, hess, mov, flow, TILE, 30)
    cy, cx = NP // 2, NP // 2
    err = abs(float(flow[cy, cx, 0]) - dx) + abs(float(flow[cy, cx, 1]) - dy)
    assert err < 0.1
