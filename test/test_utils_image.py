"""Equivalence tests for the CuPy ports in ``utils_image``.

Each CuPy implementation is checked against a reference:

* "always run" tests compare against a numba-free reference (torch / numpy).
  These are the meaningful equivalences and stay green on GPUs where numba's
  CUDA target cannot JIT.
* "requires_numba" tests compare against the original numba kernels. They are
  skipped automatically when numba cannot compile on the current GPU (e.g. a
  ptxas / PTX-version mismatch), so the suite is green there too, but they still
  verify the ports against the originals on numba-capable hardware.
"""

import math

import numpy as np
import cupy as cp
import pytest
import torch
import torch.nn.functional as F

from handheld_super_resolution.utils_image import (
    compute_grey_images,
    cuda_downsample,
    compute_grey_images_fft_cupy,
    compute_grey_images_decimate_cupy,
    cupy_downsample,
    compute_gradient_cupy,
    compute_hessian_cupy,
)


def _numba_cuda_works():
    """True if numba can actually JIT and launch a trivial CUDA kernel here."""
    from numba import cuda

    try:

        @cuda.jit
        def _probe(a):
            i = cuda.grid(1)
            if i < a.size:
                a[i] = a[i]

        d = cuda.to_device(np.zeros(4, np.float32))
        _probe[1, 4](d)
        cuda.synchronize()
        return True
    except Exception:
        return False


NUMBA_OK = _numba_cuda_works()
requires_numba = pytest.mark.skipif(
    not NUMBA_OK,
    reason="numba CUDA cannot JIT on this GPU (ptxas rejects the emitted PTX version)",
)

IMAGE_SHAPES = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2160, 3840)]
SEEDS = [0, 1, 2]
TILE_SIZES = [8, 16, 32]


# --------------------------------------------------------------------------
# numba-free references
# --------------------------------------------------------------------------
def torch_fft_grey_ref(img):
    imsize_y, imsize_x = img.shape
    t = torch.as_tensor(img, dtype=torch.float32, device="cuda")
    t = torch.fft.fftshift(torch.fft.fft2(t))
    t[: imsize_y // 4, :] = 0
    t[:, : imsize_x // 4] = 0
    t[-imsize_y // 4 :, :] = 0
    t[:, -imsize_x // 4 :] = 0
    t = torch.fft.ifft2(torch.fft.ifftshift(t))
    return t.real.cpu().numpy()


def numpy_decimate_ref(img):
    h, w = img.shape
    img = img[: 2 * (h // 2), : 2 * (w // 2)]
    return (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]) / 4


def torch_sobel_ref(img):
    sx = torch.as_tensor(np.array([[-1, 0, 1]]), dtype=torch.float32, device="cuda")[None, None]
    sy = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=torch.float32, device="cuda")[None, None]
    t = torch.as_tensor(img, dtype=torch.float32, device="cuda")[None, None]
    gradx = F.conv2d(t, sx, padding="same").squeeze().cpu().numpy()
    grady = F.conv2d(t, sy, padding="same").squeeze().cpu().numpy()
    return gradx, grady


def numpy_hessian_ref(gradx, grady, tile_size):
    """Float64 reference matching the floor-tiling numba ``compute_hessian``."""
    h, w = gradx.shape
    n_patch_y, n_patch_x = h // tile_size, w // tile_size
    gx = gradx[: n_patch_y * tile_size, : n_patch_x * tile_size].astype(np.float64)
    gy = grady[: n_patch_y * tile_size, : n_patch_x * tile_size].astype(np.float64)
    gx = gx.reshape(n_patch_y, tile_size, n_patch_x, tile_size)
    gy = gy.reshape(n_patch_y, tile_size, n_patch_x, tile_size)
    out = np.zeros((n_patch_y, n_patch_x, 2, 2))
    out[..., 0, 0] = (gx * gx).sum(axis=(1, 3))
    out[..., 1, 1] = (gy * gy).sum(axis=(1, 3))
    cross = (gx * gy).sum(axis=(1, 3))
    out[..., 0, 1] = cross
    out[..., 1, 0] = cross
    return out


def numba_hessian_ref(gradx, grady, tile_size):
    """Launch the original numba ``ICA.compute_hessian`` kernel."""
    from numba import cuda
    from handheld_super_resolution.ICA import compute_hessian

    h, w = gradx.shape
    n_patch_y, n_patch_x = h // tile_size, w // tile_size
    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), np.float32)
    tpb = (16, 16)
    bpg = (math.ceil(n_patch_x / 16), math.ceil(n_patch_y / 16))
    compute_hessian[bpg, tpb](
        cuda.to_device(gradx), cuda.to_device(grady), tile_size, hessian
    )
    return hessian.copy_to_host()


# --------------------------------------------------------------------------
# cupy vs numba-free reference (always run)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", IMAGE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_grey_fft_cupy_matches_torch(shape, seed):
    np.random.seed(seed)
    img = np.random.rand(*shape).astype(np.float32)
    got = cp.asnumpy(compute_grey_images_fft_cupy(img))
    assert got.shape == shape
    assert np.allclose(got, torch_fft_grey_ref(img), atol=1e-4)


@pytest.mark.parametrize("shape", IMAGE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_grey_decimate_cupy_matches_numpy(shape, seed):
    np.random.seed(seed)
    img = np.random.rand(*shape).astype(np.float32)
    got = cp.asnumpy(compute_grey_images_decimate_cupy(img))
    ref = numpy_decimate_ref(img)
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


@pytest.mark.parametrize("shape", [(256, 256), (512, 512), (1024, 1024), (2160, 3840)])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("factor", [1, 2, 4])
def test_downsample_cupy_matches_torch(shape, seed, factor):
    np.random.seed(seed)
    img = np.random.rand(1, 1, *shape).astype(np.float32)
    got = cp.asnumpy(cupy_downsample(cp.asarray(img), factor=factor))
    ref = cuda_downsample(torch.as_tensor(img, device="cuda"), factor=factor).cpu().numpy()
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("shape", IMAGE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_gradient_cupy_matches_torch_sobel(shape, seed):
    np.random.seed(seed)
    img = np.random.rand(*shape).astype(np.float32)
    gradx, grady = compute_gradient_cupy(img)
    ref_gradx, ref_grady = torch_sobel_ref(img)
    # atol=1e-3: cupy correlate1d vs torch conv2d use different float32 reduction
    # orders; at multi-megapixel sizes they diverge by ~5e-4 (float noise), more
    # so on torch 2.12 which tiles large convolutions differently.
    assert np.allclose(cp.asnumpy(gradx), ref_gradx, atol=1e-3)
    assert np.allclose(cp.asnumpy(grady), ref_grady, atol=1e-3)


@pytest.mark.parametrize("shape", [(128, 128), (512, 512), (1024, 1024), (2160, 3840)])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("tile_size", TILE_SIZES)
def test_hessian_cupy_matches_numpy(shape, seed, tile_size):
    np.random.seed(seed)
    gradx = np.random.rand(*shape).astype(np.float32)
    grady = np.random.rand(*shape).astype(np.float32)
    got = cp.asnumpy(compute_hessian_cupy(gradx, grady, tile_size))
    ref = numpy_hessian_ref(gradx, grady, tile_size)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-4)


# --------------------------------------------------------------------------
# cupy vs original numba kernels (skipped when numba cannot JIT here)
# --------------------------------------------------------------------------
@requires_numba
@pytest.mark.parametrize("shape", [(256, 256), (512, 512), (2160, 3840)])
@pytest.mark.parametrize("seed", SEEDS)
def test_grey_decimate_cupy_matches_numba(shape, seed):
    np.random.seed(seed)
    img = np.random.rand(*shape).astype(np.float32)
    got = cp.asnumpy(compute_grey_images_decimate_cupy(img))
    ref = compute_grey_images(img, "decimating").copy_to_host()
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


@requires_numba
@pytest.mark.parametrize("shape", [(128, 128), (512, 512), (1024, 1024)])
@pytest.mark.parametrize("tile_size", TILE_SIZES)
def test_hessian_cupy_matches_numba(shape, tile_size):
    np.random.seed(0)
    gradx = np.random.rand(*shape).astype(np.float32)
    grady = np.random.rand(*shape).astype(np.float32)
    got = cp.asnumpy(compute_hessian_cupy(gradx, grady, tile_size))
    ref = numba_hessian_ref(gradx, grady, tile_size)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-4)
