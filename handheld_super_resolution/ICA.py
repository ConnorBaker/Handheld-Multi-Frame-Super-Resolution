
import numpy as np
import torch
import torch.nn.functional as F

from .utils import DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE
from .utils_image import compute_hessian_cupy
from .cupy_ops import ica_iterate_cupy, to_torch
FLOAT = DEFAULT_NUMPY_FLOAT_TYPE
SOBEL_Y = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_X = torch.as_tensor(np.array([[-1,0,1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_Y.requires_grad = False
SOBEL_X.requires_grad = False

def init_ica(image, tile_size, config):
    # Gradients via torch (runs on all GPUs); Hessian via CuPy because numba's
    # CUDA target cannot JIT on this hardware. gradx/grady stay torch tensors.
    gradx = F.conv2d(image[None, None], SOBEL_X, padding='same').squeeze()
    grady = F.conv2d(image[None, None], SOBEL_Y, padding='same').squeeze()

    hessian = compute_hessian_cupy(gradx, grady, tile_size)

    return gradx, grady, hessian


def align_lvl_ica(ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                  moving_lvl, alignment, l, config):
    tile_size = config.block_matching.tuning.tile_sizes[l]
    n_iter = config.ica.tuning.n_iter

    # CuPy replacement for the numba ica_kernel_* (numba can't JIT here).
    new_flow = ica_iterate_cupy(
        ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
        moving_lvl, alignment, tile_size, n_iter)
    # write the result back into the torch alignment tensor, in place
    alignment.copy_(to_torch(new_flow))


