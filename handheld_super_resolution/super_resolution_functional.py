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

from dataclasses import dataclass
import json
import os
from statistics import mean
import torch
from torch import Tensor
from typing import Self
import warnings

from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from numpy import typing as npt
from numba import cuda

from .utils_image import (
    compute_grey_images,
    compute_grey_images_fft,
    frame_count_denoising_gauss,
    frame_count_denoising_median,
    apply_orientation,
)
from .utils import DEFAULT_NUMPY_FLOAT_TYPE, divide, add
from .alignment import align, init_alignment
from .params import sanitize_config, update_snr_config
from .robustness import init_robustness, compute_robustness
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .merge import merge, merge_ref
from . import raw2rgb

from .io import Burst
from numba.cuda.cudadrv.devicearray import DeviceNDArray

NOISE_MODEL_PATH = Path(os.path.dirname(__file__)).parent / "data"

# TODO(@connorbaker): The problem with using objects for everything is that they hold references to other objects
# you may not need but are kept alive because their reference is held.
# If we used solely functions, we'd get the advantage of returning from a function clearing the scope and freeing
# references...

# TODO: Ensure that none of these change, or that they are wiped on each iteration


# TODO: Maybe shared should hold exactly what is necessary to perform each iteration?
@dataclass(kw_only=True, slots=True)
class Shared:
    cuda_ref_img: DeviceNDArray
    white_balance: DeviceNDArray
    cfa_pattern: DeviceNDArray
    cuda_imgs: DeviceNDArray
    cuda_std_curve: DeviceNDArray
    cuda_diff_curve: DeviceNDArray
    cuda_ref_grey: DeviceNDArray
    cuda_im_greys: Tensor  # TODO: what is this exactly?
    accumulate_r: bool
    output_template: npt.NDArray[np.float32]

    # TODO: the raw images may not be uint8 at this point

    @classmethod
    def load(
        cls,
        *,
        ref_raw: npt.NDArray[np.uint8],
        white_balance: npt.NDArray[np.float32],
        cfa: npt.NDArray[np.uint8],
        raw_comp: npt.NDArray[np.uint8],
        std_curve: npt.NDArray[np.float64],
        diff_curve: npt.NDArray[np.float64],
        accumulate_r: bool,
        output_template: npt.NDArray[np.float32],
    ) -> Self:
        #### Moving to GPU
        cuda_ref_img = cuda.to_device(ref_raw)
        white_balance = cuda.to_device(white_balance)
        cfa_pattern = cuda.to_device(cfa)
        cuda_imgs = cuda.to_device(raw_comp)
        cuda_std_curve = cuda.to_device(std_curve)
        cuda_diff_curve = cuda.to_device(diff_curve)

        #### Raw to grey (assumes bayer mode and FFT grey method)
        # NOTE: Still wants a device array, so use the compute_grey_images wrapper.
        cuda_ref_grey = compute_grey_images(cuda_ref_img, method="FFT")

        #### Compute Grey Images (assumes bayer mode and FFT grey method)
        cuda_im_greys = torch.vmap(compute_grey_images_fft)(
            torch.as_tensor(raw_comp, device="cuda")
        )

        return cls(
            cuda_ref_img=cuda_ref_img,
            white_balance=white_balance,
            cfa_pattern=cfa_pattern,
            cuda_imgs=cuda_imgs,
            cuda_std_curve=cuda_std_curve,
            cuda_diff_curve=cuda_diff_curve,
            cuda_ref_grey=cuda_ref_grey,
            cuda_im_greys=cuda_im_greys,
            accumulate_r=accumulate_r,
            output_template=output_template,
        )


# Sets up the rest of self.config and returns std_curve and diff_curve
def setup_config(
    config: OmegaConf,
    ref_raw: npt.NDArray[np.uint8],
    ISO: int,
    CFA: npt.NDArray[np.uint8],
    white_balance: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    iso_to_values = {
        entry["iso"]: {"alpha": mean(entry["a"]), "beta": mean(entry["b"])}
        for entry in json.loads((NOISE_MODEL_PATH / "sony-a6300.json").read_bytes())[
            "profiles"
        ]
    }

    alpha: float = iso_to_values[ISO]["alpha"]
    beta: float = iso_to_values[ISO]["beta"]

    config.noise_model.update({
        "alpha": alpha,
        "beta": beta,
    })

    std_curve, diff_curve = run_fast_MC(alpha, beta)

    #### Estimating ref image SNR
    brightness = ref_raw.mean()

    id_noise = round(1000 * brightness)
    std = std_curve[id_noise]

    SNR = brightness / std

    # TODO: Updates block matching and merging tuning parameters.
    # If we were optimizing over those, we'd need to be sure to run this again.
    update_snr_config(config, SNR)

    # checking (just in case !)
    sanitize_config(config, ref_raw.shape)

    # TODO: These values should not be used since we're providing them in __init__ to numba as arrays;
    # OmegaConf not handling numpy values shouldn't be an issue for us since we shouldn't include them in the config.
    config.exif = OmegaConf.create({
        "cfa_pattern": CFA.tolist(),  # omegaconf doesnt like numpy...
        "iso": ISO,
        "white_balance": white_balance.tolist(),
    })

    config.noise_model.update({
        "std_curve": std_curve.tolist(),
        "diff_curve": diff_curve.tolist(),
    })

    ard: OmegaConf = config.accumulated_robustness_denoiser
    # TODO: OmegaConf provides a merge method, which conflicts with one of our keys...
    ard.enabled = any(
        x.enabled
        for x in [
            ard.median,
            ard.gauss,
            ard.merge,
        ]
    )

    return std_curve, diff_curve


def thing(burst_path: str, glob: str, config: OmegaConf):
    burst = Burst.load(burst_path, glob)

    # TODO: Find out what isn't used and what we can free

    # TODO: setup_config mutates config in-place

    std_curve, diff_curve = setup_config(
        config, burst.ref_raw, burst.iso, burst.cfa, burst.white_balance
    )

    scale: float = config.scale
    native_imshape_y, native_imshape_x = burst.ref_raw.shape
    output_shape: tuple[int, int, int] = (
        round(scale * native_imshape_y),
        round(scale * native_imshape_x),
        3,
    )

    shared = Shared.load(
        ref_raw=burst.ref_raw,
        white_balance=burst.white_balance,
        cfa=burst.cfa,
        raw_comp=burst.raw_comp,
        std_curve=std_curve,
        diff_curve=diff_curve,
        accumulate_r=(
            config.accumulated_robustness_denoiser.enabled
            or config.robustness.save_mask
        ),
        output_template=np.zeros(output_shape, dtype=DEFAULT_NUMPY_FLOAT_TYPE),
    )


def thing1(
    config: OmegaConf,
    shared: Shared,
):
    # Ensure self.num and self.den are zeroed.
    # TODO: Not sure how to move self.num and self.den to __init__ so they can be re-used since not sure
    # how to wipe them between runs.
    # zeros init of num and den
    num = cuda.to_device(shared.output_template)
    den = cuda.to_device(shared.output_template)

    debug_dict = {"robustness": [], "flow": []}

    ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian = (
        init_alignment(shared.cuda_ref_grey, config)
    )

    #### Local stats estimation
    # TODO: This can be moved outwards, only really needs to check config to see if robustness is enabled.
    ref_local_means, ref_local_stds = init_robustness(
        shared.cuda_ref_img, shared.cfa_pattern, shared.white_balance, config
    )

    if shared.accumulate_r:
        # TODO: the shape can be derived from the shape of the reference image
        accumulated_r = cuda.to_device(np.zeros(ref_local_means.shape[1:]))

    for cuda_im_grey, cuda_img in zip(
        shared.cuda_im_greys, shared.cuda_imgs, strict=True
    ):
        thing2(
            ref_pyramid=ref_pyramid,
            tyled_pyr=tyled_pyr,
            ref_tiled_fft=ref_tiled_fft,
            ref_gradx=ref_gradx,
            ref_grady=ref_grady,
            ref_hessian=ref_hessian,
            cuda_im_grey=cuda_im_grey,
            cuda_img=cuda_img,
            config=config,
            ref_local_means=ref_local_means,
            ref_local_stds=ref_local_stds,
            cfa_pattern=shared.cfa_pattern,
            white_balance=shared.white_balance,
            cuda_std_curve=shared.cuda_std_curve,
            cuda_diff_curve=shared.cuda_diff_curve,
            accumulate_r=shared.accumulate_r,
            accumulated_r=accumulated_r,
            num=num,
            den=den,
        )


# NOTE: num and den are modified in-place.
def thing2(
    *,
    ref_pyramid,
    tyled_pyr,
    ref_tiled_fft,
    ref_gradx,
    ref_grady,
    ref_hessian,
    cuda_im_grey,
    cuda_img,
    config,
    ref_local_means,
    ref_local_stds,
    cfa_pattern,
    white_balance,
    cuda_std_curve,
    cuda_diff_curve,
    accumulate_r,
    accumulated_r,
    num,
    den,
) -> None:
    # TODO: align actually wants self.cuda_im_greys[im_id] to be a torch tensor!
    cuda_final_alignment = align(
        ref_pyramid,
        tyled_pyr,
        ref_tiled_fft,
        ref_gradx,
        ref_grady,
        ref_hessian,
        cuda_im_grey,  # im_id
        config,
    )

    #### Robustness
    cuda_robustness = compute_robustness(
        cuda_img,  # im_id
        ref_local_means,
        ref_local_stds,
        cuda_final_alignment,
        cfa_pattern,
        white_balance,
        (cuda_std_curve, cuda_diff_curve),
        config,
    )
    if accumulate_r:
        add(accumulated_r, cuda_robustness)

    #### Kernel estimation
    cuda_kernels = estimate_kernels(cuda_img, config)

    #### Merging
    merge(
        cuda_img,  # im_id
        cuda_final_alignment,
        cuda_kernels,
        cuda_robustness,
        num,
        den,
        cfa_pattern,
        config,
    )


def thing3(
    *,
    cuda_ref_img,
    num,
    den,
    cfa_pattern,
    config,
    accumulate_r,
    accumulated_r,
    debug_dict,
):
    #### Ref kernel estimation
    cuda_kernels = estimate_kernels(cuda_ref_img, config)

    #### Merge ref
    merge_ref(
        cuda_ref_img,
        cuda_kernels,
        num,
        den,
        cfa_pattern,
        config,
        accumulated_r if accumulate_r else None,
    )

    # num is outwritten into num/den
    divide(num, den)

    # NOTE: den is unused after this and can probably be deleted, assuming no need to synchronize explicitly.

    if accumulate_r:
        debug_dict["accumulated robustness"] = accumulated_r


class SRObject:
    def run(self):
        # Ensure self.num and self.den are zeroed.
        # TODO: Not sure how to move self.num and self.den to __init__ so they can be re-used since not sure
        # how to wipe them between runs.
        scale = self.config.scale
        native_imshape_y, native_imshape_x = self.cuda_ref_img.shape
        output_size = (round(scale * native_imshape_y), round(scale * native_imshape_x))
        # zeros init of num and den
        num = cuda.to_device(
            np.zeros((*output_size, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
        )
        den = cuda.to_device(
            np.zeros((*output_size, 3), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
        )

        debug_dict = {"robustness": [], "flow": []}

        ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian = (
            init_alignment(self.cuda_ref_grey, self.config)
        )

        #### Local stats estimation
        ref_local_means, ref_local_stds = init_robustness(
            self.cuda_ref_img, self.cfa_pattern, self.white_balance, self.config
        )

        if self.accumulate_r:
            accumulated_r = cuda.to_device(np.zeros(ref_local_means.shape[1:]))

        # comp_imgs = comp_imgs[:1]
        n_images = self.raw_comp.shape[0]
        for im_id in range(n_images):
            #### Moving to GPU
            # cuda.to_device(self.raw_comp[im_id], to=self.cuda_img)

            # #### Compute Grey Images (assumes bayer mode)
            # cuda_im_grey = compute_grey_images(
            #     self.raw_comp[im_id], self.config.grey_method
            # )

            # TODO: align actually wants self.cuda_im_greys[im_id] to be a torch tensor!
            cuda_final_alignment = align(
                ref_pyramid,
                tyled_pyr,
                ref_tiled_fft,
                ref_gradx,
                ref_grady,
                ref_hessian,
                # cuda_im_grey,
                self.cuda_im_greys[im_id],
                self.config,
            )

            #### Robustness
            cuda_robustness = compute_robustness(
                # self.cuda_img,
                self.cuda_imgs[im_id],
                ref_local_means,
                ref_local_stds,
                cuda_final_alignment,
                self.cfa_pattern,
                self.white_balance,
                (self.cuda_std_curve, self.cuda_diff_curve),
                self.config,
            )
            if self.accumulate_r:
                add(accumulated_r, cuda_robustness)

            #### Kernel estimation
            # cuda_kernels = estimate_kernels(self.cuda_img, self.config)
            cuda_kernels = estimate_kernels(self.cuda_imgs[im_id], self.config)

            #### Merging
            merge(
                # self.cuda_img,
                self.cuda_imgs[im_id],
                cuda_final_alignment,
                cuda_kernels,
                cuda_robustness,
                num,
                den,
                self.cfa_pattern,
                self.config,
            )

            # TODO: Is this necessary? Doesn't seem to be.
            # cuda.default_stream().synchronize()

        #### Ref kernel estimation
        cuda_kernels = estimate_kernels(self.cuda_ref_img, self.config)

        #### Merge ref
        merge_ref(
            self.cuda_ref_img,
            cuda_kernels,
            num,
            den,
            self.cfa_pattern,
            self.config,
            accumulated_r if self.accumulate_r else None,
        )

        # num is outwritten into num/den
        divide(num, den)

        # NOTE: den is unused after this and can probably be deleted, assuming no need to synchronize explicitly.

        if self.accumulate_r:
            debug_dict["accumulated robustness"] = accumulated_r

        # from run_pipeline()
        handheld_output = num

        #### Performing frame count aware denoising if enabled
        median_config = self.config.accumulated_robustness_denoiser.median
        gauss_config = self.config.accumulated_robustness_denoiser.gauss

        median = median_config.enabled
        gauss = gauss_config.enabled
        post_frame_count_denoise = median or gauss

        if post_frame_count_denoise:
            if median:
                handheld_output = frame_count_denoising_median(
                    handheld_output, debug_dict["accumulated robustness"], median_config
                )
            if gauss:
                handheld_output = frame_count_denoising_gauss(
                    handheld_output, debug_dict["accumulated robustness"], gauss_config
                )

        #### post processing
        handheld_output = torch.as_tensor(handheld_output, device="cuda")
        if self.config.postprocessing.enabled:
            pp = self.config.postprocessing
            output_image = raw2rgb.postprocess(
                self.raw,
                handheld_output,
                pp.do_color_correction,
                pp.do_tonemapping,
                pp.do_gamma_correction,
                pp.sharpening,
                pp.do_devignetting,
                cam2rgb=self.cam2rgb,
            )

        else:
            output_image = handheld_output

        # Applying image orientation
        if "Image Orientation" in self.tags.keys():
            ori = self.tags["Image Orientation"].values[0]
        else:
            ori = 1
            warnings.warns(
                "The Image Orientation EXIF tag could not be found. \
                        The image may be mirrored or misoriented."
            )
        output_image = apply_orientation(output_image, ori)
        if "accumulated robustness" in debug_dict.keys():
            debug_dict["accumulated robustness"] = apply_orientation(
                debug_dict["accumulated robustness"], ori
            )

        return output_image, debug_dict
