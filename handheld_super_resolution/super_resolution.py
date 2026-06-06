# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

This script contains : 
    - The implementation of Alg. 1, the main the body of the method
    - The implementation of Alg. 2, where the function necessary to
        compute the optical flow are called
    - All the operations necessary before its call, such as whitebalance,
        exif reading, and manipulations of the user's parameters.
    - The call to post-processing operations (if enabled)


@author: jamyl
"""

import os
import time
import warnings

from pathlib import Path
import numpy as np
import cupy as cp
from omegaconf import OmegaConf
from numba import cuda
import rawpy

from .utils_image import compute_grey_images, frame_count_denoising_gauss, frame_count_denoising_median, apply_orientation
from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, divide, add, round_iso, timer
from .alignment import align, init_alignment
from .params import sanitize_config, update_snr_config
from .robustness import init_robustness, compute_robustness
from .utils_dng import load_dng_burst
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .cupy_ops import upscale_warp_stats_cupy
from .merge import merge, merge_ref
from . import raw2rgb

NOISE_MODEL_PATH = Path(os.path.dirname(__file__)).parent / 'data' 
        

def main(ref_img, comp_imgs, config):
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N
        
    config : OmegaConf object
        parameters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains (if debugging is enabled) some debugging infos.

    """
    
    grey_method = config.grey_method
    
    ### verbose and timing related stuff
    verbose = config.verbose >= 1
    verbose_2 = config.verbose >= 2
    verbose_3 = config.verbose >= 3

    compute_grey_images_ = timer(compute_grey_images, verbose_3, end_s="- Ref grey image estimated by {}".format(grey_method))
    init_robustness_ = timer(init_robustness, verbose_2, "\nEstimating ref image local stats", 'Local stats estimated (Total)')
    compute_robustness_ = timer(compute_robustness, verbose_2, '\nEstimating robustness', 'Robustness estimated (Total)')
    estimate_kernels_ = timer(estimate_kernels, verbose_2, '\nEstimating kernels', 'Kernels estimated (Total)')
    merge_ = timer(merge, verbose_2, '\nAccumulating Image', 'Image accumulated (Total)')
    merge_ref_ = timer(merge_ref, verbose_2, '\nAccumulating ref Img', 'Ref Img accumulated (Total)')    
    divide_ = timer(divide, verbose_2, end_s='\n------------------------\nImage normalized (Total)')
    init_alignment_ = timer(init_alignment, verbose_2, '\nInitializing alignment', 'Alignment initialized (Total)')
    align_ = timer(align, verbose_2, '\nBeginning alignment', 'Image aligned (Total)')

    debug_mode = config.debug
    debug_dict = {"robustness":[],
                  "flow":[]}

    # Accumulating the robustness mask is only meaningful when robustness is
    # actually computed. With robustness disabled, init_robustness returns
    # (None, None) and compute_robustness returns an all-ones (accept-all)
    # mask, so there is nothing to accumulate -- gate on robustness.enabled to
    # avoid dereferencing the None ref_local_means below.
    accumulate_r = config.robustness.enabled and (
        config.accumulated_robustness_denoiser.enabled or config.robustness.save_mask)

    #### Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    white_balance = cuda.to_device(np.array(config.exif.white_balance))
    cfa_pattern = cuda.to_device(np.array(config.exif.cfa_pattern))
    # This running buffer is for the image being processed
    stream = cuda.stream()
    cuda_img = cuda.device_array_like(comp_imgs[0], stream=stream)
    cuda.synchronize()
    cuda_std_curve = cuda.to_device(np.array(config.noise_model.std_curve))
    cuda_diff_curve = cuda.to_device(np.array(config.noise_model.diff_curve))
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()

    #### Raw to grey (alignment proxy)
    cuda_ref_grey = compute_grey_images_(cuda_ref_img, grey_method)

    ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian = init_alignment_(cuda_ref_grey, config)

    #### Local stats estimation
    ref_local_means, ref_local_stds = init_robustness_(cuda_ref_img, cfa_pattern, white_balance, config)

    if accumulate_r:
        accumulated_r = cp.zeros(tuple(ref_local_means.shape[1:]), dtype=cp.float32)

    scale = config.scale
    native_imshape_y, native_imshape_x = cuda_ref_img.shape
    output_size = (round(scale*native_imshape_y), round(scale*native_imshape_x))
    # zeros init of num and den (CuPy: written by the merge RawKernels)
    num = cp.zeros((*output_size, 3), dtype=cp.float32)
    den = cp.zeros((*output_size, 3), dtype=cp.float32)
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')


    # comp_imgs = comp_imgs[:1]
    n_images = comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        #### Moving to GPU
        # cuda_img = cuda.to_device(comp_imgs[im_id])
        cuda.to_device(comp_imgs[im_id], to=cuda_img, stream=stream)
        
        #### Compute Grey Images (alignment proxy)
        cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
        
        cuda_final_alignment = align_(ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian,
                        cuda_im_grey, config)
        
        if debug_mode:
            # cuda_final_alignment is a torch tensor (the alignment port returns
            # torch, not a numba device array) -- copy_to_host() no longer exists.
            debug_dict["flow"].append(cuda_final_alignment.detach().cpu().numpy())
            
        #### Robustness
        cuda_robustness = compute_robustness_(cuda_img, ref_local_means, ref_local_stds, cuda_final_alignment,
                                            cfa_pattern, white_balance, (cuda_std_curve, cuda_diff_curve), config)
        if accumulate_r:
            add(accumulated_r, cuda_robustness)
        
        #### Kernel estimation
        cuda_kernels = estimate_kernels_(cuda_img, config)
        
        #### Merging
        merge_(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den, cfa_pattern, config)
        
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        if debug_mode :
            # cuda_robustness is a cupy array (the robustness port is cupy, not numba).
            debug_dict['robustness'].append(cp.asnumpy(cuda_robustness))
        stream.synchronize()
    
    #### Ref kernel estimation
    # Optionally expose the structure-tensor edge-strength field (l1) the merge
    # already computes (and otherwise discards). Default off -> identical behavior.
    expose_edge_strength = bool(config.merging.get("expose_edge_strength", False))
    if expose_edge_strength:
        cuda_kernels, ref_l1 = estimate_kernels_(cuda_ref_img, config, return_l1=True)
    else:
        cuda_kernels = estimate_kernels_(cuda_ref_img, config)

    #### Merge ref
    if accumulate_r:     
        merge_ref_(cuda_ref_img, cuda_kernels,
                   num, den, cfa_pattern,
                   config, accumulated_r)
    else:
        merge_ref_(cuda_ref_img, cuda_kernels,
                   num, den, cfa_pattern,
                   config)


        
    # num is outwritten into num/den
    divide_(num, den)
    
    if verbose :
        s = '\nTotal ellapsed time : '
        print(s, ' ' * (50 - len(s)), ': ', round((time.perf_counter() - t1), 2), 'seconds')
    
    if accumulate_r :
        debug_dict['accumulated robustness'] = cp.asnumpy(accumulated_r)

    if expose_edge_strength:
        # Bring the grey-resolution l1 field to OUTPUT resolution so a postprocess
        # gate aligns with the merged image. Reuse the pipeline's Dogson upscale
        # (grey -> native, the x2 bayer step); then resize to output size when
        # scale != 1 (the gate is a soft blend mask, so bilinear is fine).
        l1_native = upscale_warp_stats_cupy(ref_l1[None], True)[0]  # (H, W)
        # upscale_warp_stats_cupy fills out-of-bounds border samples with inf
        # (=> robustness 0 in its usual use) and its biquadratic Dogson kernel has
        # negative lobes that can undershoot a non-negative field. l1 is an
        # eigenvalue energy (>= 0); "no data" at the border means no edge -> 0, so
        # sanitize non-finite to 0 and clamp the interpolation undershoot to 0.
        l1_native = cp.nan_to_num(l1_native, nan=0.0, posinf=0.0, neginf=0.0)
        l1_native = cp.maximum(l1_native, 0.0)
        if l1_native.shape != tuple(output_size):
            import torch as _th
            import torch.nn.functional as _F
            l1_t = _th.as_tensor(l1_native)[None, None]
            l1_t = _F.interpolate(l1_t, size=tuple(output_size),
                                  mode="bilinear", align_corners=False)
            l1_native = cp.asarray(l1_t[0, 0].contiguous())
        debug_dict['edge_strength'] = cp.asnumpy(l1_native)

    return num, debug_dict


def process(burst_path, config):
    """
    Processes the burst

    Parameters
    ----------
    burst_path : str or Path
        Path of the folder where the .dng burst is located
    config : OmegaConf object
        parameters.

    Returns
    -------
    Array
        The processed image

    """
    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         config.verbose >= 1,
                                         config.verbose >= 2)
    
    # reading image stack
    ref_raw, raw_comp, ISO, tags, CFA, xyz2cam, white_balance, ref_path = load_dng_burst(burst_path)
    
    if config.noise_model.get("alpha", None) is not None:
        # User provided custom values.
        print("Using user provided alpha and beta values")
        alpha = config.noise_model.alpha
        beta = config.noise_model.beta
    ## The noise model exif are already scaled for the image ISO.
    else:
        alpha = sum([x[0] for x in tags['Image Tag 0xC761'].values[::2]])/3
        beta = sum([x[0] for x in tags['Image Tag 0xC761'].values[1::2]])/3
    config.noise_model.update({
        "alpha": alpha,
        "beta": beta
        })
    #### Packing noise model related to picture ISO
    # curve_iso = round_iso(ISO) # Rounds non standart ISO to regular ISO (100, 200, 400, ...)
    # std_noise_model_label = 'noise_model_std_ISO_{}'.format(curve_iso)
    # diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(curve_iso)
    # std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
    # diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
    
    # std_curve = np.load(std_noise_model_path)
    # diff_curve = np.load(diff_noise_model_path)
    
    # Use this to compute noise curves on the fly
    std_curve, diff_curve = run_fast_MC(alpha, beta)
    
    
    if verbose_2:   
        currentTime = getTime(currentTime, ' -- Read raw files')

    #### Estimating ref image SNR
    brightness = np.mean(ref_raw)
    
    id_noise = round(1000*brightness)
    std = std_curve[id_noise]
    
    SNR = brightness/std
    if verbose_1:
        print(" ",10*"-")
        print('|ISO : {}'.format(ISO))
        print('|Image brightness : {:.2f}'.format(brightness))
        print('|expected noise std : {:.2e}'.format(std))
        print('|Estimated SNR : {:.2f}'.format(SNR))
    
    update_snr_config(config, SNR)
    
    # checking (just in case !)
    sanitize_config(config, ref_raw.shape)
    

    config.exif = OmegaConf.create({
        "cfa_pattern": CFA.tolist(), # omegaconf doesnt like numpy...
        "iso": ISO,
        "white_balance": white_balance,
        })

    config.noise_model.update({
        "std_curve": std_curve.tolist(),
        "diff_curve": diff_curve.tolist(),
        })

    if any([x.enabled for x in [config.accumulated_robustness_denoiser.median,
                                config.accumulated_robustness_denoiser.gauss,
                                config.accumulated_robustness_denoiser.merge]]):
        config.accumulated_robustness_denoiser.enabled = True
    else:
        config.accumulated_robustness_denoiser.enabled = False
    
    
    #### Running the handheld pipeline
    handheld_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), config)
    
    
    #### Performing frame count aware denoising if enabled
    median_config = config.accumulated_robustness_denoiser.median
    gauss_config = config.accumulated_robustness_denoiser.gauss

    median = median_config.enabled
    gauss = gauss_config.enabled
    post_frame_count_denoise = (median or gauss)

    post_processing_enabled = config.postprocessing.enabled

    if post_frame_count_denoise or post_processing_enabled:
        if verbose_1:
            print('Beginning post processing')
    
    if post_frame_count_denoise : 
        if verbose_2:
            print('-- Robustness aware bluring')
        
        if median:
            handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                           median_config)
        if gauss:
            handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                          gauss_config)


    #### post processing
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        raw = rawpy.imread(ref_path)
        output_image = raw2rgb.postprocess(raw, cp.asnumpy(handheld_output),
                                           config.postprocessing.do_color_correction,
                                           config.postprocessing.do_tonemapping,
                                           config.postprocessing.do_gamma_correction,
                                           config.postprocessing.sharpening,
                                           config.postprocessing.do_devignetting,
                                           xyz2cam,
                                           deblur_config=config.postprocessing.get("deblur", None),
                                           aberration_config=config.postprocessing.get("aberration", None),
                                           edge_strength=debug_dict.get("edge_strength", None),
                                           )
    else:
        output_image = cp.asnumpy(handheld_output)

    # Applying image orientation
    if 'Image Orientation' in tags.keys():
        ori = tags['Image Orientation'].values[0]
    else:
        ori = 1
        warnings.warns('The Image Orientation EXIF tag could not be found. \
                      The image may be mirrored or misoriented.')
    output_image = apply_orientation(output_image, ori)
    if 'accumulated robustness' in debug_dict.keys():
        debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    return output_image, debug_dict

    
    
