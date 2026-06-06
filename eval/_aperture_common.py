"""Shared helpers for the aperture-problem flow-regularization study.

All numbers in this study are read against the documented baseline in
eval/NOTES.md ("The user's sharp edge jutting right from the blind"):
at the scale-1 merge crop centred (cy=960, cx=1840), window y[98,130) x[50,105)
in the 160x160 DxO-aligned ROI, level-matched to the DxO cross-frame consensus,
  single-frame merge max|div| ~0.044 (rms ~0.005)
  multi-frame  merge max|div| ~0.074 (rms ~0.011)
i.e. the multi-frame fusion roughly DOUBLES the single-frame divergence from the
DxO cross-frame consensus at this real edge feature.

This module ONLY measures; it never regenerates the shared DxO luma stack (it is
read by absolute path from the shared cache). Reproduction discipline per
eval/CLAUDE.md: report distributions, gate on reliability, red-team favourable
results.

The flow-regularization recipe is opt-in via config overrides (default OFF):
  block_matching.tuning.flow_tikhonov_lambda  (float, >0 enables LM damping in ICA)
  block_matching.tuning.flow_median           (dict, enables post-ICA guided VMF)
See handheld_super_resolution/cupy_ops.py (ica_iterate_cupy) and ICA.py.
"""
import glob
import os

import numpy as np

from eval import hallucination as H

# --- shared, absolute paths (do NOT regenerate the luma stack) ---------------
SHARED_OUT = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_007"
DNG_GLOB = "/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng"
ARW_DIR = "/home/connorbaker/Pictures/ARW/2025-09-06/007"
LUMA_CACHE = os.path.join(SHARED_OUT, "luma_stack.npy")

# our own (worktree-local) merge cache so we never clobber the shared dir
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "eval", "_aperture_out")
os.makedirs(OUT, exist_ok=True)

# the documented feature (eval/NOTES.md) -- the "jut right of the blind"
FEAT_CY, FEAT_CX = 960, 1840
ROI_SIZE, ROI_SEARCH = 160, 140
WIN = (98, 130, 50, 105)  # y0, y1, x0, x1 inside the ROI


def load_stack():
    return H.load_luma_stack(sorted(glob.glob(DNG_GLOB)), cache=LUMA_CACHE)


def consensus_at(stack, cy=FEAT_CY, cx=FEAT_CX, size=ROI_SIZE, search=ROI_SEARCH):
    """DxO cross-frame consensus image + the reference ROI used to align onto."""
    rois, info = H.extract_aligned_roi(stack, cy, cx, size=size, search=search)
    cons = H.consensus_decomp(rois, info)["consensus"]
    ref = np.ascontiguousarray(stack[0][cy - size // 2:cy + size // 2,
                                        cx - size // 2:cx + size // 2])
    return cons, ref


def _level_match(a, ref):
    """Affine level-match a onto ref's mean/std (the _jut_attribute convention)."""
    return (a - a.mean()) / (a.std() + 1e-9) * ref.std() + ref.mean()


def divergence(merge_luma, stack, cons=None, ref=None, cy=FEAT_CY, cx=FEAT_CX,
               win=WIN, size=ROI_SIZE, search=ROI_SEARCH):
    """max|div| and rms of (level-matched merge - DxO consensus) in the feature
    window. This is the EXACT metric eval/_jut_attribute.py reports.

    Returns dict(maxabs, rms, xy_spread, worst_x, aligned_ok).
    """
    if cons is None or ref is None:
        cons, ref = consensus_at(stack, cy, cx, size, search)
    y0, y1, x0, x1 = win
    sl = (slice(y0, y1), slice(x0, x1))
    aligned = H.align_to(ref, merge_luma, cy, cx, size, search)
    if aligned is None:
        return dict(maxabs=float("nan"), rms=float("nan"), xy_spread=float("nan"),
                    worst_x=-1, aligned_ok=False)
    dc = cons[sl]
    d = _level_match(aligned, cons)[sl] - dc
    iy, ix = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    # x/y second-moment spread of the |div| energy (>1 => horizontally elongated)
    e = np.abs(d)
    e = np.clip(e - np.percentile(e, 60), 0, None)
    ys, xs = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    w = e.sum() + 1e-12
    mx, my = (xs * e).sum() / w, (ys * e).sum() / w
    sx = np.sqrt(((xs - mx) ** 2 * e).sum() / w)
    sy = np.sqrt(((ys - my) ** 2 * e).sum() / w)
    return dict(maxabs=float(np.abs(d).max()), rms=float(d.std()),
                xy_spread=float(sx / (sy + 1e-9)), worst_x=int(x0 + ix),
                aligned_ok=True)
