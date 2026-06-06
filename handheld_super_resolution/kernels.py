# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

This script contains the implementation of Alg. 5: ComputeKernelCovariance, 
where the merge kernels are estimated


@author: jamyl
"""

import time

import numpy as np
from numba import cuda
import torch as th
import torch.nn.functional as F

import cupy as cp

from .utils import DEFAULT_TORCH_FLOAT_TYPE
from .utils_image import compute_grey_images_decimate_cupy
from .cupy_ops import gat_cupy, estimate_cov_cupy, to_torch


SEL_HARD_THRESHOLD = 0
SEL_LINEAR = 1


def estimate_kernels(img, config, return_l1=False):
    """
    Implementation of Alg. 5: ComputeKernelCovariance
    Returns the kernels covariance matrices for the frame J_n, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        Raw image J_n
    options : dict
        options
    config : OmegaConf object
        parameters.
    return_l1 : bool, optional
        When True, also return the structure tensor's large-eigenvalue field
        ``l1`` (imshape_y//2, imshape_x//2) -- the across-edge edge-strength signal
        the merge already computes and discards (see ``estimate_cov_cupy``). Used
        to gate a measured-PSF deconvolution. Default off so the return value is
        unchanged for the merge.

    Returns
    -------
    covs : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance matrices Omega_n, sampled at the center of each bayer quad.
    l1 : device Array[imshape_y//2, imshape_x//2]
        (only when ``return_l1``) structure-tensor large eigenvalue field.

    """
    verbose_3 = config.verbose >= 3


    k_detail = config.merging.tuning.k_detail
    k_denoise = config.merging.tuning.k_denoise
    D_th = config.merging.tuning.D_th
    D_tr = config.merging.tuning.D_tr
    k_stretch = config.merging.tuning.k_stretch
    k_shrink = config.merging.tuning.k_shrink
    
    if config.merging.selection_law == 'hard_threshold':
        selection_law = SEL_HARD_THRESHOLD
    elif config.merging.selection_law == 'linear':
        selection_law = SEL_LINEAR
    else:
        raise ValueError(f"Unknown selection law: {config.merging.selection_law}")

    alpha = config.noise_model.alpha
    beta = config.noise_model.beta

    if verbose_3:
        cuda.synchronize()
        t1 = time.perf_counter()
    
    #__ Performing Variance Stabilization Transform
    
    #__ Variance stabilization + decimate to grey (CuPy; numba can't JIT here)
    img = gat_cupy(img, alpha, beta)
    img_grey = compute_grey_images_decimate_cupy(img)

    #__ Computing grads (torch convolutions run on all GPUs)
    th_grey_img = to_torch(cp.ascontiguousarray(img_grey))[None, None]

    grad_kernel1 = np.array([[[[-0.5, 0.5]]],
                             [[[ 0.5, 0.5]]]])
    grad_kernel1 = th.as_tensor(grad_kernel1, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    grad_kernel2 = np.array([[[[0.5],
                               [0.5]]],
                             [[[-0.5],
                               [0.5]]]])
    grad_kernel2 = th.as_tensor(grad_kernel2, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

    tmp = F.conv2d(th_grey_img, grad_kernel1)
    th_full_grad = F.conv2d(tmp, grad_kernel2, groups=2)
    # shape [y, x, 2]; the convolution reduces each spatial dim by 1
    full_grads = cp.asarray(th_full_grad.squeeze().permute(1, 2, 0).contiguous())

    if return_l1:
        covs, l1 = estimate_cov_cupy(full_grads, k_detail, k_denoise, D_th, D_tr,
                                     k_stretch, k_shrink, selection_law, return_l1=True)
        return covs, l1
    covs = estimate_cov_cupy(full_grads, k_detail, k_denoise, D_th, D_tr,
                             k_stretch, k_shrink, selection_law)
    return covs


    


