# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

This script contains : 
    - The implementation of Alg. 4, the conventionnal accumulation
    - The implementation of Alg. 11, where the reference image is merged


@author: jamyl
"""




from .cupy_ops import merge_accumulate_cupy, merge_ref_accumulate_cupy

def merge_ref(ref_img, kernels, num, den, cfa_pattern, config, acc_rob=None):
    """
    Implementation of Alg. 11: AccumulationReference
    Accumulates the reference frame into num and den, while considering
    (if enabled) the accumulated robustness mask to enforce single frame SR if
    necessary.

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    kernels : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance Matrices Omega_1
    num : device Array[s*imshape_y, s*imshape_x, c]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x, c]
        Denominator of the accumulator
    config : OmegaConf object
        parameters.
    acc_rob : [imshape_y, imshape_x], optional
        accumulated robustness mask. The default is None.

    Returns
    -------
    None.

    """
    scale = config.scale
    iso_kernel = config.merging.kernel == 'iso'

    robustness_denoise = config.accumulated_robustness_denoiser.enabled
    if robustness_denoise:
        rad_max = config.accumulated_robustness_denoiser.merge.rad_max
        max_multiplier = config.accumulated_robustness_denoiser.merge.max_multiplier
        max_frame_count = config.accumulated_robustness_denoiser.merge.max_frame_count
    else:
        rad_max = 0
        max_multiplier = 0.0
        max_frame_count = 0

    # CuPy RawKernel replacement for the numba accumulate_ref (numba can't JIT here).
    merge_ref_accumulate_cupy(
        ref_img, kernels, 1, int(iso_kernel), float(scale),  # bayer-only pipeline
        cfa_pattern, num, den, acc_rob,
        int(robustness_denoise), int(max_frame_count), int(rad_max), float(max_multiplier))
    
    
          
    
def merge(comp_img, alignments, covs, r, num, den, cfa_pattern, config):
    """
    Implementation of Alg. 4: Accumulation
    Accumulates comp_img (J_n, n>1) into num and den, based on the alignment
    V_n, the covariance matrices Omega_n and the robustness mask estimated before.


    Parameters
    ----------
    comp_imgs : device Array [imsize_y, imsize_x]
        The non-reference image to merge (J_n)
    alignments : device Array[n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment V_n(p)
    covs : device array[imsize_y//2, imsize_x//2, 2, 2]
        covariance matrices Omega_n
    r : Device_Array[imsize_y, imsize_x]
        Robustness mask r_n
    num : device Array[s*imshape_y, s*imshape_x, c]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x, c]
        Denominator of the accumulator
        
    config : OmegaConf object
        parameters.

    Returns
    -------
    None

    """
    scale = config.scale
    iso_kernel = config.merging.kernel == 'iso'
    tile_size = config.block_matching.tuning.tile_size

    # CuPy RawKernel replacement for the numba accumulate (numba can't JIT here).
    merge_accumulate_cupy(
        comp_img, alignments, covs, r,
        1, int(iso_kernel), float(scale), int(tile_size),  # bayer-only pipeline
        cfa_pattern, num, den)


