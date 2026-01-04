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
from typing import Self
import torch
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
from .io import Burst
from .utils import DEFAULT_NUMPY_FLOAT_TYPE, divide, add
from .alignment import align, init_alignment
from .params import sanitize_config, update_snr_config
from .robustness import init_robustness, compute_robustness
from .utils_dng import load_dng_burst
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .merge import merge, merge_ref
from . import raw2rgb
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
    cuda_im_greys: torch.Tensor  # TODO: what is this exactly?
    accumulate_r: bool
    output_template: npt.NDArray[np.float32]
    ref_local_means: DeviceNDArray
    ref_local_stds: DeviceNDArray

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
        config: OmegaConf, # Needed for init_robustness
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

        #### Local stats estimation
        ref_local_means, ref_local_stds = init_robustness(
            cuda_ref_img,
            cfa_pattern,
            white_balance,
            config,
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
            ref_local_means=ref_local_means,
            ref_local_stds=ref_local_stds,
        )


class SRObject:
    config: OmegaConf
    burst: Burst
    shared: Shared

    def __init__(self, burst_path: str, config: OmegaConf):
        self.config: OmegaConf = config
        self.burst: Burst = Burst.load(burst_path, "*.ARW")

        std_curve, diff_curve = self._setup_config(
            self.burst.ref_raw, self.burst.iso, self.burst.cfa, self.burst.white_balance
        )

        scale: float = self.config.scale
        native_imshape_y, native_imshape_x = self.burst.ref_raw.shape
        output_shape: tuple[int, int, int] = (
            round(scale * native_imshape_y),
            round(scale * native_imshape_x),
            3,
        )

        self.shared: Shared = Shared.load(
            ref_raw=self.burst.ref_raw,
            white_balance=self.burst.white_balance,
            cfa=self.burst.cfa,
            raw_comp=self.burst.raw_comp,
            std_curve=std_curve,
            diff_curve=diff_curve,
            accumulate_r=(
                self.config.accumulated_robustness_denoiser.enabled
                or self.config.robustness.save_mask
            ),
            output_template=np.zeros(output_shape, dtype=DEFAULT_NUMPY_FLOAT_TYPE),
            config=config,
        )

    # Sets up the rest of self.config and returns std_curve and diff_curve
    def _setup_config(
        self,
        ref_raw: npt.NDArray[np.uint8],
        ISO: int,
        CFA: npt.NDArray[np.uint8],
        white_balance: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        iso_to_values = {
            entry["iso"]: {"alpha": mean(entry["a"]), "beta": mean(entry["b"])}
            for entry in json.loads(
                (NOISE_MODEL_PATH / "sony-a6300.json").read_bytes()
            )["profiles"]
        }

        alpha: float = iso_to_values[ISO]["alpha"]
        beta: float = iso_to_values[ISO]["beta"]

        self.config.noise_model.update({
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
        update_snr_config(self.config, SNR)

        # checking (just in case !)
        sanitize_config(self.config, ref_raw.shape)

        # TODO: These values should not be used since we're providing them in __init__ to numba as arrays;
        # OmegaConf not handling numpy values shouldn't be an issue for us since we shouldn't include them in the config.
        self.config.exif = OmegaConf.create({
            "cfa_pattern": CFA.tolist(),  # omegaconf doesnt like numpy...
            "iso": ISO,
            "white_balance": white_balance.tolist(),
        })

        self.config.noise_model.update({
            "std_curve": std_curve.tolist(),
            "diff_curve": diff_curve.tolist(),
        })

        ard: OmegaConf = self.config.accumulated_robustness_denoiser
        ard.enabled = any(
            x.enabled
            for x in [
                ard.median,
                ard.gauss,
                ard.merge,
            ]
        )

        return std_curve, diff_curve

    def run(self):
        # Ensure self.num and self.den are zeroed.
        # TODO: Not sure how to move self.num and self.den to __init__ so they can be re-used since not sure
        # how to wipe them between runs.
        # zeros init of num and den
        num = cuda.to_device(self.shared.output_template)
        den = cuda.to_device(self.shared.output_template)

        debug_dict = {"robustness": [], "flow": []}

        # Depends on block_matching.tuning -- if we modify values there we cannot bring this into Shared.
        ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian = (
            init_alignment(self.shared.cuda_ref_grey, self.config)
        )

        if self.shared.accumulate_r:
            accumulated_r = cuda.to_device(np.zeros(self.shared.ref_local_means.shape[1:]))

        for cuda_im_grey, cuda_img in zip(self.shared.cuda_im_greys, self.shared.cuda_imgs, strict=True):
            # TODO: align actually wants self.cuda_im_greys[im_id] to be a torch tensor!
            cuda_final_alignment = align(
                ref_pyramid,
                tyled_pyr,
                ref_tiled_fft,
                ref_gradx,
                ref_grady,
                ref_hessian,
                cuda_im_grey,
                self.config,
            )

            #### Robustness
            cuda_robustness = compute_robustness(
                cuda_img,
                self.shared.ref_local_means,
                self.shared.ref_local_stds,
                cuda_final_alignment,
                self.shared.cfa_pattern,
                self.shared.white_balance,
                (self.shared.cuda_std_curve, self.shared.cuda_diff_curve),
                self.config,
            )
            if self.shared.accumulate_r:
                add(accumulated_r, cuda_robustness)

            #### Kernel estimation
            cuda_kernels = estimate_kernels(
                cuda_img,
                self.config,
            )

            #### Merging
            merge(
                cuda_img,
                cuda_final_alignment,
                cuda_kernels,
                cuda_robustness,
                num,
                den,
                self.shared.cfa_pattern,
                self.config,
            )

            # TODO: Is this necessary? Doesn't seem to be.
            # cuda.default_stream().synchronize()

        #### Ref kernel estimation
        cuda_kernels = estimate_kernels(self.shared.cuda_ref_img, self.config)

        #### Merge ref
        merge_ref(
            self.shared.cuda_ref_img,
            cuda_kernels,
            num,
            den,
            self.shared.cfa_pattern,
            self.config,
            accumulated_r if self.shared.accumulate_r else None,
        )

        # num is outwritten into num/den
        divide(num, den)

        # NOTE: den is unused after this and can probably be deleted, assuming no need to synchronize explicitly.

        if self.shared.accumulate_r:
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
                self.burst.raw,
                handheld_output,
                pp.do_color_correction,
                pp.do_tonemapping,
                pp.do_gamma_correction,
                pp.sharpening,
                pp.do_devignetting,
                cam2rgb=self.burst.cam2rgb,
            )

        else:
            output_image = handheld_output

        # Applying image orientation
        if "Image Orientation" in self.burst.tags.keys():
            ori = self.burst.tags["Image Orientation"].values[0]
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
