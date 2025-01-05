import cupy as cp
import numpy as np
import pytest
import torch

from handheld_super_resolution.utils_image import (
    compute_grey_images_decimate_cupy,
    compute_grey_images_decimate_numba,
    compute_grey_images_fft_cupy,
    compute_grey_images_fft_torch,
    cuda_downsample,
    cupy_downsample,
)


@pytest.mark.parametrize(
    "image_shape",
    [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_compute_grey_images_fft_equal(image_shape: tuple[int, int], seed: int):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = np.array(compute_grey_images_fft_cupy(image).get())
    result_torch = compute_grey_images_fft_torch(image).cpu().numpy()
    assert np.allclose(result_cupy, result_torch)


@pytest.mark.parametrize(
    "image_shape",
    [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_compute_grey_images_decimate_equal(image_shape: tuple[int, int], seed: int):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = np.array(compute_grey_images_decimate_cupy(image))
    result_numba = np.array(compute_grey_images_decimate_numba(image))
    assert np.allclose(result_cupy, result_numba)


@pytest.mark.parametrize(
    "image_shape",
    [
        (1, 1, 64, 64),
        (1, 1, 128, 128),
        (1, 1, 256, 256),
        (1, 1, 512, 512),
        (1, 1, 1024, 1024),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("downsample_factor", [1, 2, 4])
def test_downsample(image_shape: tuple[int, int], seed: int, downsample_factor: int):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = cupy_downsample(cp.array(image), factor=downsample_factor)
    result_cuda = cuda_downsample(
        torch.tensor(image, device="cuda"), factor=downsample_factor
    )
    assert np.allclose(result_cupy, result_cuda)
