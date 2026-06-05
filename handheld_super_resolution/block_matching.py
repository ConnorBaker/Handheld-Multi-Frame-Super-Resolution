import time

import torch

from .utils import DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE
from .cupy_ops import block_matching_l1_cupy, to_torch
FLOAT = DEFAULT_NUMPY_FLOAT_TYPE


def align_lvl_block_matching_L2(tyled_pyr_lvl, ref_fft_lvl, moving_lvl, alignment, l, config):
    verbose = config.verbose > 2
    currentTime = time.perf_counter()
    tileSize = config.block_matching.tuning.tile_sizes[l]
    searchRadius = config.block_matching.tuning.search_radii[l]
    distanceMetric = config.block_matching.tuning.metrics[l]

    imshape = moving_lvl.shape
    
    search_size = 2 * searchRadius + tileSize # The size of the crop in which the search is done
    corr_size = 2 * searchRadius + 1 # The size of the correlation map output

    # Extract tiles based on the optical flow
    search_area = extract_flow_patches(moving_lvl, alignment, tileSize, searchRadius)

    moving_fft = torch.fft.rfft2(search_area, dim=(-2, -1))
    corrs = torch.fft.irfft2(torch.conj(ref_fft_lvl) * moving_fft, s=(search_size, search_size))
    corrs = torch.fft.fftshift(corrs, dim=(-2, -1))

    # crop to valid region ±R around center output size search_size
    # Before cropping, corrs has an even shaped. The correlation with shift=0 is almost at the center, biased towards the bottom left. By removing 1 pixel from top and left, we center it.
    # The rest of the crop removes the phantom "circular" correlations at the borders due to the FFT
    pre_crop_size = corrs.shape[-1]
    crop = (pre_crop_size - 1 - corr_size) // 2
    corrs = corrs[..., crop+1:crop+corr_size+1, crop+1:crop+corr_size+1]


    ## Windowed L2 norm of the search patches = a box filter (windowed sum of
    ## squares). avg_pool2d (stride 1, valid) * tileSize^2 is the torch library
    ## primitive for it -- any tile size, separable, faster than a t x t ones
    ## conv. (CuPy analogue: cupyx.scipy.ndimage.uniform_filter * tileSize^2.)
    sq = search_area.view(-1, 1, search_size, search_size).square()
    L2_search = torch.nn.functional.avg_pool2d(sq, tileSize, stride=1) * (tileSize * tileSize)
    L2_search = L2_search.view(search_area.shape[0], search_area.shape[1], L2_search.shape[-2], L2_search.shape[-1])

    ## Final L2 error computation (Not the full L2, but enough to find the minimum)
    L2_error = L2_search - 2 * corrs

    L2_error_ = L2_error.flatten(-2, -1)
    max_idx = torch.argmin(L2_error_, dim=-1)
    peak_y = max_idx // corr_size
    peak_x = max_idx % corr_size
    dy = peak_y - corr_size//2
    dx = peak_x - corr_size//2

    # Test alignment here
    alignment[:, :, 0] += dx
    alignment[:, :, 1] += dy
    
def align_lvl_block_matching_L1(ref_lvl, moving_lvl, alignments, l, config):
    tile_size = config.block_matching.tuning.tile_sizes[l]
    search_radius = config.block_matching.tuning.search_radii[l]

    # CuPy replacement for the numba cuda_L1_local_search* (numba can't JIT here).
    new_flow = block_matching_l1_cupy(
        ref_lvl, moving_lvl, alignments, tile_size, search_radius)
    alignments.copy_(to_torch(new_flow))


def extract_flow_patches(frame_tgt, flow, patch_size, radius):
    ny, nx, _ = flow.shape
    p = patch_size
    r = radius
    P_search = 2 * r + p
    frame_tgt = torch.as_tensor(frame_tgt, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    flow = flow.round().long()

    dx = flow[..., 0]
    dy = flow[..., 1]

    # compute top-left corner of each patch
    top = torch.arange(ny, device=frame_tgt.device)[:, None] * p + dy
    left = torch.arange(nx, device=frame_tgt.device)[None, :] * p + dx

    offsets = torch.arange(P_search, device=frame_tgt.device) - r
    dy_offsets, dx_offsets = torch.meshgrid(offsets, offsets, indexing='ij')

    y_coords = top[:, :, None, None] + dy_offsets[None, None, :, :]
    x_coords = left[:, :, None, None] + dx_offsets[None, None, :, :]

    # clamp to image boundaries
    y_coords = y_coords.clamp(0, frame_tgt.shape[0]-1)
    x_coords = x_coords.clamp(0, frame_tgt.shape[1]-1)

    # flatten for advanced indexing
    y_flat = y_coords.reshape(-1)
    x_flat = x_coords.reshape(-1)

    aligned_patches = frame_tgt[y_flat, x_flat].view(ny, nx, P_search, P_search)
    return aligned_patches