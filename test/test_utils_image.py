import cupy as cp
import numpy as np
import pytest
import torch

from handheld_super_resolution.utils_image import (
    compute_gradient_cupy,
    compute_gradient_torch,
    compute_hessian_cupy,
    compute_hessian_numba,
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
        (2160, 3840),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_compute_grey_images_fft_equal(image_shape: tuple[int, int], seed: int):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = np.array(compute_grey_images_fft_cupy(image).get())
    result_torch = compute_grey_images_fft_torch(image).cpu().numpy()
    assert result_cupy.shape == result_torch.shape
    assert np.allclose(result_cupy, result_torch)


@pytest.mark.parametrize(
    "image_shape",
    [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2160, 3840),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_compute_grey_images_decimate_equal(image_shape: tuple[int, int], seed: int):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = np.array(compute_grey_images_decimate_cupy(image))
    result_numba = np.array(compute_grey_images_decimate_numba(image))
    assert result_cupy.shape == result_numba.shape
    assert np.allclose(result_cupy, result_numba)


@pytest.mark.parametrize(
    "image_shape",
    [
        (1, 1, 64, 64),
        (1, 1, 128, 128),
        (1, 1, 256, 256),
        (1, 1, 512, 512),
        (1, 1, 1024, 1024),
        (1, 1, 2160, 3840),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("downsample_factor", [1, 2, 4])
def test_downsample_equal(
    image_shape: tuple[int, int], seed: int, downsample_factor: int
):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = cupy_downsample(cp.array(image), factor=downsample_factor)
    result_cuda = cuda_downsample(
        torch.tensor(image, device="cuda"), factor=downsample_factor
    )
    assert result_cupy.shape == result_cuda.shape
    assert np.allclose(result_cupy, result_cuda)


# TODO: They use different methods for finding the gradient; not equal.
@pytest.mark.skip("Not equal")
@pytest.mark.parametrize(
    "image_shape", [(128, 128), (256, 256), (512, 512), (1024, 1024), (2160, 3840)]
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("sigma_blur", [0, 0.5, 1])
def test_compute_gradient_equal(
    image_shape: tuple[int, int],
    seed: int,
    sigma_blur: float,
):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    options = {"verbose": 0}
    kanade_params = {"tuning": {"sigma blur": sigma_blur}}
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy_grad_y, result_cupy_grad_x = compute_gradient_cupy(
        image, options, kanade_params
    )
    result_torch_grad_y, result_torch_grad_x = compute_gradient_torch(
        image, options, kanade_params
    )
    assert result_cupy_grad_y.shape == result_torch_grad_y.shape
    assert result_cupy_grad_x.shape == result_torch_grad_x.shape
    assert np.allclose(result_cupy_grad_y.get(), np.array(result_torch_grad_y))
    assert np.allclose(result_cupy_grad_x, result_torch_grad_x)


@pytest.mark.parametrize(
    "image_shape",
    [(128, 128), (256, 256), (512, 512), (1024, 1024), (2160, 3840)],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("tile_size", [16, 32])
def test_compute_hessian_equal(
    image_shape: tuple[int, int],
    seed: int,
    tile_size: int,
):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    options = {"verbose": 0}
    kanade_params = {
        "tuning": {
            "tileSize": tile_size,
            "tileSizes": [tile_size, tile_size, tile_size, tile_size // 2],
        }
    }
    cuda_grady = np.random.rand(*image_shape).astype(np.float32)
    cuda_gradx = np.random.rand(*image_shape).astype(np.float32)
    result_cupy_hessian = compute_hessian_cupy(
        cuda_grady, cuda_gradx, options, kanade_params
    )
    result_numba_hessian = compute_hessian_numba(
        cuda_grady, cuda_gradx, options, kanade_params
    )
    assert result_cupy_hessian.shape == result_numba_hessian.shape
    assert np.allclose(result_cupy_hessian.get(), np.array(result_numba_hessian))
