import numpy as np
import pytest

from handheld_super_resolution.block_matching import (
    init_block_matching_torch,
    init_block_matching_cupy,
)


@pytest.mark.parametrize(
    "image_shape",
    [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2160, 3840),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("tile_size", [16, 32])
def test_init_block_matching_equal(
    image_shape: tuple[int, int], seed: int, tile_size: int
):
    # Test that the two functions compute the same result
    np.random.seed(seed)
    options = {"verbose": 0}
    params = {
        "tuning": {
            "factors": [1, 2, 4, 4],
            "tileSizes": [tile_size, tile_size, tile_size, tile_size // 2],
        }
    }
    image = np.random.rand(*image_shape).astype(np.float32)
    result_cupy = init_block_matching_cupy(image, options, params)
    result_torch = init_block_matching_torch(image, options, params)

    for res_cupy, res_torch in zip(result_cupy, result_torch):
        assert np.allclose(res_cupy.get(), res_torch.get())

