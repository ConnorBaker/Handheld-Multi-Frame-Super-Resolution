# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:51:42 2023

@author: jamyl
"""

import os
import glob

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Literal, TypedDict, NotRequired
import numpy as np
import torch
from skimage.util import img_as_ubyte
import cv2

from handheld_super_resolution import process
from handheld_super_resolution.utils_dng import save_as_dng


class OptionsConfig(TypedDict):
    verbose: int


class MergeTuningConfig(TypedDict):
    k_stretch: float
    k_shrink: float
    k_detail: NotRequired[float]
    k_denoise: NotRequired[float]


class MergeConfig(TypedDict):
    kernel: Literal["handheld", "iso"]
    tuning: MergeTuningConfig


class RobustnessTuningConfig(TypedDict):
    t: float
    s1: float
    s2: float
    Mt: float


class RobustnessConfig(TypedDict):
    on: bool
    tuning: RobustnessTuningConfig


class RobustnessDenoiserConfig(TypedDict):
    on: bool


class KanadeTuningConfig(TypedDict):
    kanadeIter: int


class KanadeConfig(TypedDict):
    tuning: KanadeTuningConfig


class PostSharpeningConfig(TypedDict):
    radius: float
    amount: float


PostConfig = TypedDict(
    "PostConfig",
    {
        "on": bool,
        "do sharpening": bool,
        "do tonemapping": bool,
        "do gamma": bool,
        "do devignette": bool,
        "do color correction": bool,
        "sharpening": PostSharpeningConfig,
    },
)

ParamsConfig = TypedDict(
    "ParamsConfig",
    {
        "scale": int,
        "merging": MergeConfig,
        "robustness": RobustnessConfig,
        "kanade": KanadeConfig,
        "accumulated robustness denoiser": RobustnessDenoiserConfig,
        "post processing": PostConfig,
    },
)


def mk_argument_parser() -> ArgumentParser:
    #### Argparser

    def str2bool(v):
        v = str(v)

        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise ArgumentTypeError("Boolean value expected.")

    parser = ArgumentParser()

    ## Image parameters
    image_parameters_parser = parser.add_argument_group("Image Parameters")
    image_parameters_parser.add_argument("--impath", type=str, help="input image")
    image_parameters_parser.add_argument("--outpath", type=str, help="out image")
    image_parameters_parser.add_argument(
        "--scale", type=int, default=2, help="Scaling factor"
    )
    image_parameters_parser.add_argument(
        "--verbose", type=int, default=1, help="Verbose option (0 to 4)"
    )

    ## Robustness
    robustness_parameters_parser = parser.add_argument_group("Robustness")
    robustness_parameters_parser.add_argument(
        "--t", type=float, default=0.12, help="Threshold for robustness"
    )
    robustness_parameters_parser.add_argument(
        "--s1", type=float, default=2, help="Threshold for robustness"
    )
    robustness_parameters_parser.add_argument(
        "--s2", type=float, default=12, help="Threshold for robustness"
    )
    robustness_parameters_parser.add_argument(
        "--Mt", type=float, default=0.8, help="Threshold for robustness"
    )
    robustness_parameters_parser.add_argument(
        "--R_on",
        type=str2bool,
        default=True,
        help="Whether robustness is activated or not",
    )

    robustness_parameters_parser.add_argument(
        "--R_denoising_on",
        type=str2bool,
        default=True,
        help="Whether or not the robustness based denoising should be applied",
    )

    ## Post Processing
    post_parameters_parser = parser.add_argument_group("Post Processing")
    post_parameters_parser.add_argument(
        "--post_process",
        type=str2bool,
        default=True,
        help="Whether post processing should be applied or not",
    )
    post_parameters_parser.add_argument(
        "--do_sharpening",
        type=str2bool,
        default=True,
        help="Whether sharpening should be applied during post processing",
    )
    post_parameters_parser.add_argument(
        "--radius",
        type=float,
        default=3,
        help="If sharpening is applied, radius of the unsharp mask",
    )
    post_parameters_parser.add_argument(
        "--amount",
        type=float,
        default=1.5,
        help="If sharpening is applied, amount of the unsharp mask",
    )
    post_parameters_parser.add_argument(
        "--do_tonemapping",
        type=str2bool,
        default=True,
        help="Whether tonnemaping should be applied during post processing",
    )
    post_parameters_parser.add_argument(
        "--do_gamma",
        type=str2bool,
        default=True,
        help="Whether gamma curve should be applied during post processing",
    )
    post_parameters_parser.add_argument(
        "--do_color_correction",
        type=str2bool,
        default=True,
        help="Whether color correction should be applied during post processing",
    )

    ## Merging (advanced)
    merging_parameters_parser = parser.add_argument_group("Merging (advanced)")
    merging_parameters_parser.add_argument(
        "--kernel_shape",
        type=str,
        default="handheld",
        help='"handheld" or "iso" : Whether to use steerable or isotropic kernels',
    )
    merging_parameters_parser.add_argument(
        "--k_detail", type=float, default=None, help="SNR based by default"
    )
    merging_parameters_parser.add_argument(
        "--k_denoise", type=float, default=None, help="SNR based by default"
    )
    merging_parameters_parser.add_argument("--k_stretch", type=float, default=4)
    merging_parameters_parser.add_argument("--k_shrink", type=float, default=2)

    ## Alignment (advanced)
    alignment_parameters_parser = parser.add_argument_group("Alignment (advanced)")
    alignment_parameters_parser.add_argument(
        "--ICA_iter", type=int, default=3, help="Number of ICA Iterations"
    )

    return parser


def mk_argument_config(args: Namespace) -> tuple[OptionsConfig, ParamsConfig]:
    # Bottom-up construction of the config
    options_config: OptionsConfig = {"verbose": args.verbose}

    params_config: ParamsConfig = {
        "scale": args.scale,
        "merging": {
            "kernel": args.kernel_shape,
            "tuning": {
                "k_stretch": args.k_stretch,
                "k_shrink": args.k_shrink,
            },
        },
        "robustness": {
            "on": args.R_on,
            "tuning": {
                "t": args.t,
                "s1": args.s1,
                "s2": args.s2,
                "Mt": args.Mt,
            },
        },
        "kanade": {"tuning": {"kanadeIter": args.ICA_iter}},
        "accumulated robustness denoiser": {"on": args.R_denoising_on},
        "post processing": {
            # disabling post processing for dng outputs
            "on": args.post_process and Path(args.outpath).suffix != ".dng",
            "do sharpening": args.do_sharpening,
            "do tonemapping": args.do_tonemapping,
            "do gamma": args.do_gamma,
            "do devignette": False,
            "do color correction": args.do_color_correction,
            "sharpening": {"radius": args.radius, "amount": args.amount},
        },
    }

    if args.k_detail is not None:
        params_config["merging"]["tuning"]["k_detail"] = args.k_detail

    if args.k_denoise is not None:
        params_config["merging"]["tuning"]["k_denoise"] = args.k_denoise

    return options_config, params_config


def print_argument_config(config: ParamsConfig) -> None:
    print("Parameters:")
    print("")
    print("  Upscaling factor:       %d" % config["scale"])
    print("")
    if config["scale"] == 1:
        print("    Demosaicking mode")
    else:
        print("    Super-resolution mode.")
        if config["scale"] > 2:
            print(
                "    WARNING: Since the optics and the integration on the sensor limit the aliasing, do not expect more details than that obtained at x2 (refer to our paper and the original publication)."
            )
    print("")
    if config["robustness"]["on"]:
        print("  Robustness:       enabled")
        print("  -------------------------")
        for key, value in config["robustness"]["tuning"].items():
            label = key + ":"
            print(f"  {label:<24}{value:1.2f}")
        if config["accumulated robustness denoiser"]["on"]:
            print("  Robustness denoising:   enabled")
        else:
            print("  Robustness denoising:   disabled")
        print("                            ")
    else:
        print("  Robustness:      disabled")
        print("                            ")

    print("  Alignment:")
    print("  -------------------------")
    print("  ICA Iterations:         %d" % config["kanade"]["tuning"]["kanadeIter"])
    print("")
    print("  Fusion:")
    print("  -------------------------")
    print("  Kernel shape:           %s" % config["merging"]["kernel"])
    print("  k_stretch:              %1.2f" % config["merging"]["tuning"]["k_stretch"])
    print("  k_shrink:               %1.2f" % config["merging"]["tuning"]["k_shrink"])
    if "k_detail" in config["merging"]["tuning"]:
        print(
            "  k_detail:               %1.2f" % config["merging"]["tuning"]["k_detail"]
        )
    else:
        print("  k_detail:               SNR based")
    if "k_denoise" in config["merging"]["tuning"]:
        print(
            "  k_denoise:              %1.2f" % config["merging"]["tuning"]["k_denoise"]
        )
    else:
        print("  k_denoise:              SNR based")
    print("")
    pass


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = mk_argument_parser()
    args = parser.parse_args()
    options, params = mk_argument_config(args)

    print_argument_config(params)

    #### Handheld ####
    print("Processing with handheld super-resolution")

    outpath = Path(args.outpath)

    handheld_output = process(args.impath, options, params)
    handheld_output = np.nan_to_num(handheld_output)
    handheld_output = np.clip(handheld_output, 0, 1)

    # define a faster imsave for large png images
    def imsave(fname, rgb_8bit_data):
        return cv2.imwrite(fname, cv2.cvtColor(rgb_8bit_data, cv2.COLOR_RGB2BGR))

    #### Save images ####

    if outpath.suffix == ".dng":
        if options["verbose"] >= 1:
            print("Saving output to {}".format(outpath.with_suffix(".dng").as_posix()))
        ref_img_path = glob.glob(os.path.join(args.impath, "*.dng"))[0]
        save_as_dng(handheld_output, ref_img_path, outpath)

    else:
        imsave(args.outpath, img_as_ubyte(handheld_output))
