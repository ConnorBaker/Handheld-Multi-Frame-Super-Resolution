# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

This script contains : 
    - The implementation of Algorithm 6: ComputeRobustness
    - The implementation of Algorithm 7: ComputeGuideImage
    - The implementation of Algorithm 8: ComputeLocalStatistics
    - The implementation of Algorithm 9: ComputeLocalMin


@author: jamyl
"""


import cupy as cp

from .cupy_ops import (
    compute_guide_image_cupy, compute_local_stats_cupy, upscale_warp_stats_cupy,
    compute_dist_cupy, apply_noise_model_cupy, compute_s_cupy,
    robustness_threshold_cupy, local_min_cupy,
)

def init_robustness(ref_img, cfa_pattern, white_balance, config):
    """
    Initialiazes the robustness etimation procedure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    cfa_pattern : device Array[2,2]
        Bayer pattern
    white_balance : device Array[3]
        White balance gains
    config : OmegaConf object
        parameters. 

    Returns
    -------
    local_means : device Array[imshape_y, imshape_x, channels]
        local means of the reference image.
        
    local_stds : device Array[imshape_y, imshape_x, channels]
        local standard deviations of the reference image.

    """
    bayer_mode = config.mode == 'bayer'
    if not config.robustness.enabled:
        return None, None

    # Computing guide image
    if bayer_mode:
        guide_ref_img = compute_guide_image_cupy(ref_img, cfa_pattern, white_balance)
    else:
        guide_ref_img = cp.asarray(ref_img, dtype=cp.float32)[None]

    local_means, local_stds = compute_local_stats_cupy(guide_ref_img)

    # Upscale stats to raw coarse scale (no flow for the reference)
    local_means = upscale_warp_stats_cupy(local_means, bayer_mode)
    local_stds = upscale_warp_stats_cupy(local_stds, bayer_mode)

    return local_means, local_stds
    
    
def compute_robustness(comp_img, ref_local_means, ref_local_stds, flows, cfa_pattern, white_balance, noise_model, config):
    """
    this is the implementation of Algorithm 6: ComputeRobustness
    Returns the robustnesses of the compared image J_n (n>1), based on the
    provided flow V_n(p) and the local statistics of the reference frame.

    Parameters
    ----------
    comp_img : device Array[imsize_y, imsize_x]
        Compared raw image J_n (n>1).
    ref_local_means : device Array[imsize_y, imsize_x, c]
        Local means of the reference image
    ref_local_stds : device Array[imsize_y, imsize_x, c]
        Local standard deviations of the reference image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n(p)
    cfa_pattern : device Array[2,2]
        Bayer pattern
    white_balance : device Array[3]
        White balance gains
    config : OmegaConf object
        parameters.

    Returns
    -------
    r : device Array[imsize_y, imsize_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    bayer_mode = config.mode == 'bayer'
    if not config.robustness.enabled:
        return cp.ones(comp_img.shape, dtype=cp.float32)

    tile_size = config.block_matching.tuning.tile_size
    t = config.robustness.tuning.t
    s1 = config.robustness.tuning.s1
    s2 = config.robustness.tuning.s2
    Mt = config.robustness.tuning.Mt

    std_curve, diff_curve = noise_model

    # Computing guide image
    if bayer_mode:
        guide_img = compute_guide_image_cupy(comp_img, cfa_pattern, white_balance)
    else:
        guide_img = cp.asarray(comp_img, dtype=cp.float32)[None]

    # Local stats (before flow), then upscale + warp to raw coarse scale
    comp_local_means, _ = compute_local_stats_cupy(guide_img)
    comp_local_means = upscale_warp_stats_cupy(comp_local_means, bayer_mode, tile_size, flows)

    # color distance, noise model, flow-discontinuity penalty, threshold, local min
    d_p = compute_dist_cupy(ref_local_means, comp_local_means)
    d_sq, sigma_sq = apply_noise_model_cupy(d_p, ref_local_means, ref_local_stds,
                                            std_curve, diff_curve)
    S = compute_s_cupy(flows, Mt, s1, s2)
    R = robustness_threshold_cupy(d_sq, sigma_sq, S, t, tile_size)
    r = local_min_cupy(R)
    return r


    

    
    

    
    
            

    


    
                     
    

    

    
