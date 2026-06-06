# -*- coding: utf-8 -*-
"""Streaming / arbitrary-scale merge.

`main()` (super_resolution.py) interleaves per-frame analysis (align, robustness,
kernels) with the scale-dependent accumulation, into one num/den at config.scale.
But the analysis is *scale-invariant* -- flow and robustness are LR/grey-coord, the
covariances are per-grey-pixel, and config.scale enters ONLY (a) the num/den output
allocation and (b) the merge kernels' HR->LR coordinate map. And the merge is purely
additive (num/den += per frame). So the two phases separate cleanly:

  - BUILD (scale-free): per comp frame compute {flow, robustness, covs}; per ref
    compute {covs, local stats}. `add_frame` does one frame's analysis + stores it.
  - RENDER (scale-dependent): allocate num/den at the requested scale, replay the
    Warp gathers for every stored frame + the reference, divide.

This buys two things the monolithic `main()` cannot:
  - ARBITRARY-SCALE re-render from one stored representation (`render(scale)` for any
    scale, including fractional), without re-aligning -- the artifact is the scale-
    free per-frame fields, not a fixed-scale num/den.
  - INCREMENTAL fusion: `add_frame` is O(1) in the existing frames; because the merge
    is additive, streaming A then B is bit-identical to building [A, B] at once.

`render(scale)` reproduces `main()`'s linear output (pre-postprocessing) at that scale
-- see test/test_streaming.py. The accumulated-robustness frame-count denoiser
(default OFF) is intentionally NOT supported here: its overwrite branch breaks the
additivity this class relies on (REFACTORING_PLAN / eval/research_persist_incremental.md).
"""

import numpy as np
import cupy as cp
from numba import cuda
from omegaconf import OmegaConf

from .utils_image import compute_grey_images
from .utils import divide
from .alignment import align, init_alignment
from .robustness import init_robustness, compute_robustness
from .kernels import estimate_kernels
from .merge import merge, merge_ref


def _to_host(a):
    """numpy copy of a cupy / torch / numba-device / numpy array."""
    if a is None:
        return None
    if isinstance(a, np.ndarray):
        return a
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    if hasattr(a, "detach"):           # torch tensor (the flow)
        return a.detach().cpu().numpy()
    return np.asarray(cuda.to_device(a).copy_to_host() if hasattr(a, "copy_to_host") else a)


class StreamingMerge:
    """Incremental, arbitrary-scale steerable-kernel merge.

    smerge = StreamingMerge(ref_raw, config)
    for comp in comp_frames: smerge.add_frame(comp)
    linear_rgb_2x = smerge.render(scale=2)      # any scale, any time
    smerge.save("burst.npz"); StreamingMerge.load("burst.npz").render(1.5)
    """

    def __init__(self, ref_img, config):
        if config.accumulated_robustness_denoiser.enabled:
            raise ValueError("StreamingMerge does not support the accumulated-robustness "
                             "denoiser (its overwrite branch breaks additive streaming).")
        self.config = config
        self.grey_method = config.grey_method
        self.native_shape = tuple(ref_img.shape)
        self._cfa = np.array(config.exif.cfa_pattern)
        self._wb = np.array(config.exif.white_balance)
        self._std_curve = np.array(config.noise_model.std_curve)
        self._diff_curve = np.array(config.noise_model.diff_curve)

        self.ref_img = cuda.to_device(np.ascontiguousarray(ref_img))
        cfa_dev = cuda.to_device(self._cfa)
        wb_dev = cuda.to_device(self._wb)

        # scale-free reference state
        ref_grey = compute_grey_images(self.ref_img, self.grey_method)
        self._align_ref = init_alignment(ref_grey, config)          # pyramid etc. for aligning comps
        self._ref_means, self._ref_stds = init_robustness(self.ref_img, cfa_dev, wb_dev, config)
        self.ref_kernels = estimate_kernels(self.ref_img, config)   # Omega_1 (scale-free)
        self.frames = []                                            # list of dicts {comp, flow, r, covs}

    def add_frame(self, comp_img):
        """Analyse one comparison frame (align + robustness + kernels) and store it.
        O(1) in the number of already-added frames. No scale dependence."""
        cfa_dev = cuda.to_device(self._cfa)
        wb_dev = cuda.to_device(self._wb)
        std_dev = cuda.to_device(self._std_curve)
        diff_dev = cuda.to_device(self._diff_curve)

        comp_dev = cuda.to_device(np.ascontiguousarray(comp_img))
        grey = compute_grey_images(np.ascontiguousarray(comp_img), self.grey_method)
        flow = align(*self._align_ref, grey, self.config)
        r = compute_robustness(comp_dev, self._ref_means, self._ref_stds, flow,
                               cfa_dev, wb_dev, (std_dev, diff_dev), self.config)
        covs = estimate_kernels(comp_dev, self.config)
        self.frames.append({"comp": comp_dev, "flow": flow, "r": r, "covs": covs})
        return self

    def render(self, scale=None):
        """Accumulate every stored frame + the reference into num/den at `scale`
        (default config.scale) and divide. Returns linear output (num), pre-postproc.
        Same frame order as main() -> reproduces main()'s output at that scale."""
        scale = float(self.config.scale if scale is None else scale)
        cfg = OmegaConf.merge(self.config, OmegaConf.create({"scale": scale}))
        cfa_dev = cuda.to_device(self._cfa)
        out_h = round(scale * self.native_shape[0])
        out_w = round(scale * self.native_shape[1])
        num = cp.zeros((out_h, out_w, 3), dtype=cp.float32)
        den = cp.zeros((out_h, out_w, 3), dtype=cp.float32)
        for f in self.frames:                                       # comp frames, in insertion order
            merge(f["comp"], f["flow"], f["covs"], f["r"], num, den, cfa_dev, cfg)
        merge_ref(self.ref_img, self.ref_kernels, num, den, cfa_dev, cfg)  # then the reference
        divide(num, den)
        return num

    # ---- persistence of the scale-free representation (NOT num/den) ----
    def save(self, path):
        """Persist the scale-free artifact: ref + per-frame {raw, flow, robustness,
        covs} + the config. A new scale re-renders from this with no re-alignment."""
        blob = {"native_shape": np.array(self.native_shape), "cfa": self._cfa, "wb": self._wb,
                "std_curve": self._std_curve, "diff_curve": self._diff_curve,
                "config_yaml": OmegaConf.to_yaml(self.config),
                "ref_img": _to_host(self.ref_img), "ref_kernels": _to_host(self.ref_kernels),
                "n_frames": np.array(len(self.frames))}
        for i, f in enumerate(self.frames):
            blob[f"comp_{i}"] = _to_host(f["comp"]); blob[f"flow_{i}"] = _to_host(f["flow"])
            blob[f"r_{i}"] = _to_host(f["r"]); blob[f"covs_{i}"] = _to_host(f["covs"])
        np.savez(path, **blob)
        return path

    @classmethod
    def load(cls, path):
        """Reconstruct a render-ready StreamingMerge from a saved artifact (no
        re-analysis). render(scale) works for any scale."""
        z = np.load(path, allow_pickle=False)
        self = cls.__new__(cls)
        self.config = OmegaConf.create(str(z["config_yaml"]))
        self.grey_method = self.config.grey_method
        self.native_shape = tuple(int(x) for x in z["native_shape"])
        self._cfa = z["cfa"]; self._wb = z["wb"]
        self._std_curve = z["std_curve"]; self._diff_curve = z["diff_curve"]
        self.ref_img = cuda.to_device(np.ascontiguousarray(z["ref_img"]))
        self.ref_kernels = cp.asarray(z["ref_kernels"])
        self.frames = []
        for i in range(int(z["n_frames"])):
            self.frames.append({"comp": cuda.to_device(np.ascontiguousarray(z[f"comp_{i}"])),
                                "flow": cp.asarray(z[f"flow_{i}"]), "r": cp.asarray(z[f"r_{i}"]),
                                "covs": cp.asarray(z[f"covs_{i}"])})
        return self
