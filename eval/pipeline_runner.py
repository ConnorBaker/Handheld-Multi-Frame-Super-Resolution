"""Reusable pipeline invocation for evaluation and parameter sweeps.

Decode a burst ONCE (``load_burst``), then run the pipeline many times with
different config overrides (``run``) without re-reading the DNGs. Also provides
single-frame and bilinear baselines used for automated failure detection.

All ``run*`` functions return float32 RGB in [0, 1].
"""

from pathlib import Path

import numpy as np
import cupy as cp
import rawpy
from omegaconf import OmegaConf
from numba import cuda
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as _bilinear

from handheld_super_resolution.utils_dng import load_dng_burst
from handheld_super_resolution.fast_monte_carlo import run_fast_MC
from handheld_super_resolution.params import update_snr_config, sanitize_config
from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.kernels import estimate_kernels
from handheld_super_resolution.merge import merge_ref
from handheld_super_resolution.utils import divide
from handheld_super_resolution import raw2rgb

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

# CFA code -> colour_demosaicing pattern string
_CFA_STR = {(0, 1, 1, 2): "RGGB", (2, 1, 1, 0): "BGGR",
            (1, 0, 2, 1): "GRBG", (1, 2, 0, 1): "GBRG"}


def load_burst(burst_path):
    """Decode a DNG burst once. Returns a dict reusable across many configs."""
    ref_raw, raw_comp, ISO, tags, CFA, xyz2cam, wb, ref_path = load_dng_burst(burst_path)
    alpha = sum(x[0] for x in tags["Image Tag 0xC761"].values[::2]) / 3
    beta = sum(x[0] for x in tags["Image Tag 0xC761"].values[1::2]) / 3
    std_curve, diff_curve = run_fast_MC(alpha, beta)
    brightness = float(np.mean(ref_raw))
    SNR = brightness / std_curve[round(1000 * brightness)]
    cfa_t = tuple(int(x) for x in np.array(CFA).ravel())
    return dict(
        path=str(burst_path),
        ref_raw=ref_raw.astype(np.float32), raw_comp=raw_comp.astype(np.float32),
        ISO=ISO, tags=tags, CFA=CFA, cfa_str=_CFA_STR.get(cfa_t, "RGGB"),
        xyz2cam=xyz2cam, wb=wb, ref_path=ref_path,
        alpha=alpha, beta=beta, std_curve=std_curve, diff_curve=diff_curve, SNR=SNR,
    )


def build_config(burst, overrides=None):
    """Build an OmegaConf config for a burst, applying scalar overrides last.

    Do NOT override block_matching.tuning.tile_size here (it is resolved from SNR
    together with tile_sizes; sweep search_radii / metrics / ica / robustness /
    merging instead).
    """
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    cfg.verbose = 0
    cfg.noise_model.alpha = burst["alpha"]
    cfg.noise_model.beta = burst["beta"]
    update_snr_config(cfg, burst["SNR"])
    sanitize_config(cfg, burst["ref_raw"].shape)
    cfg.exif = OmegaConf.create({
        "cfa_pattern": np.array(burst["CFA"]).tolist(),
        "iso": burst["ISO"], "white_balance": burst["wb"],
    })
    cfg.noise_model.std_curve = burst["std_curve"].tolist()
    cfg.noise_model.diff_curve = burst["diff_curve"].tolist()
    cfg.accumulated_robustness_denoiser.enabled = False
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)
    return cfg


def run(burst, overrides=None, n_frames=None):
    """Run the full multi-frame pipeline. Returns (linear_rgb[0,1], cfg)."""
    cfg = build_config(burst, overrides)
    comp = burst["raw_comp"] if n_frames is None else burst["raw_comp"][:n_frames]
    num, _ = main(burst["ref_raw"], comp, cfg)
    return np.clip(np.nan_to_num(cp.asnumpy(num)), 0, 1), cfg


def run_single(burst, overrides=None):
    """Single-frame baseline: reference only, same steerable-kernel merge."""
    cfg = build_config(burst, overrides)
    H, W = burst["ref_raw"].shape
    cref = cuda.to_device(burst["ref_raw"])
    cfa = cuda.to_device(np.array(cfg.exif.cfa_pattern))
    covs = estimate_kernels(cref, cfg)
    num = cp.zeros((H, W, 3), cp.float32)
    den = cp.zeros((H, W, 3), cp.float32)
    merge_ref(cref, covs, num, den, cfa, cfg)
    divide(num, den)
    return np.clip(np.nan_to_num(cp.asnumpy(num)), 0, 1), cfg


def run_bilinear(burst):
    """Naive bilinear demosaic of the reference frame (sanity baseline)."""
    img = _bilinear(burst["ref_raw"], burst["cfa_str"]).astype(np.float32)
    return np.clip(np.nan_to_num(img), 0, 1)


def postprocess(burst, linear_rgb, cfg, do_sharpening=None):
    """Apply the raw2rgb post-processing (color/gamma/sharpen) for display.

    Re-reads the reference DNG, so don't call this per sweep trial.
    """
    sh = OmegaConf.create(dict(cfg.postprocessing.sharpening))
    if do_sharpening is not None:
        sh.enabled = bool(do_sharpening)
    with rawpy.imread(burst["ref_path"]) as raw:
        out = raw2rgb.postprocess(
            raw, linear_rgb.astype(np.float32),
            cfg.postprocessing.do_color_correction, cfg.postprocessing.do_tonemapping,
            cfg.postprocessing.do_gamma_correction, sh,
            cfg.postprocessing.do_devignetting, burst["xyz2cam"],
            deblur_config=cfg.postprocessing.get("deblur", None),
            aberration_config=cfg.postprocessing.get("aberration", None))
    return np.clip(np.nan_to_num(out), 0, 1)
