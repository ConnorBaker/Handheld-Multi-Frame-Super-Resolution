"""Foliage / patch-size / robustness diagnostics (synthetic, ground-truth).

(a) patch_size_foliage: with the generalized L2 box filter, sweep tile size on a
    high-frequency (foliage-like) synthetic scene and measure recovery vs GT.
    Tests whether smaller alignment patches recover fine texture better.

(b) robustness_discrimination: build a burst whose CENTER sub-region carries
    inconsistent (independently 'moving') content across frames while the rest is
    static, then check the accumulated-robustness mask is LOW in the moving region
    and HIGH elsewhere. Tests whether robustness actually rejects motion (vs being
    uniformly ~1 simply because real test bursts are static).
"""

import numpy as np
import cv2

from . import groundtruth as gtmod
from . import pipeline_runner as pr
from . import metrics as met
from handheld_super_resolution.super_resolution import main


def patch_size_foliage(gt, tiles=(8, 12, 16, 24, 32), n=8, alpha=1e-4):
    burst = gtmod.make_burst(gt, downsample="area", n=n, alpha=alpha)
    H2, W2 = gt.shape[:2]
    rows = {}
    for ts in tiles:
        last = max(ts // 2, 4)
        ov = {"block_matching.tuning.tile_size": ts,
              "block_matching.tuning.tile_sizes": [ts, ts, ts, last]}
        try:
            out, _ = pr.run(burst, {**ov, "scale": 2})
            rows[ts] = gtmod._metrics(out, gt)
        except Exception as exc:
            rows[ts] = {"error": str(exc)[:120]}
    return rows


def robustness_discrimination(gt, n=8, alpha=1e-4, seed=0, region_frac=0.33):
    """Inject independent motion into a center sub-region; inspect the mask."""
    rng = np.random.default_rng(seed)
    # base static burst
    burst = gtmod.make_burst(gt, downsample="area", n=n, alpha=alpha, seed=seed)
    ref = burst["ref_raw"]
    H, W = ref.shape
    rs = int(min(H, W) * region_frac)
    ry, rx = (H - rs) // 2, (W - rs) // 2
    H2, W2, _ = gt.shape
    cfa = burst["CFA"]
    # overwrite the comp frames' center region with strongly, randomly shifted GT
    # content (mosaicked) -> inconsistent across frames == independent motion
    for k in range(burst["raw_comp"].shape[0]):
        dy, dx = rng.uniform(-30, 30, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sh = cv2.warpAffine(gt, M, (W2, H2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        lr = cv2.resize(sh, (W, H), interpolation=cv2.INTER_AREA)
        moving = np.zeros((H, W), np.float32)
        for ii in range(2):
            for jj in range(2):
                moving[ii::2, jj::2] = lr[ii::2, jj::2, cfa[ii, jj]]
        burst["raw_comp"][k, ry:ry + rs, rx:rx + rs] = moving[ry:ry + rs, rx:rx + rs]

    cfg = pr.build_config(burst, {"scale": 1})
    cfg.robustness.save_mask = True
    cfg.accumulated_robustness_denoiser.enabled = False
    _, dbg = main(burst["ref_raw"], burst["raw_comp"], cfg)
    accr = np.asarray(dbg["accumulated robustness"])  # (H,W), sum over comp frames
    ah, aw = accr.shape
    sy, sx = int(ry * ah / H), int(rx * aw / W)
    ss = int(rs * ah / H)
    moving_mean = float(np.nanmean(accr[sy:sy + ss, sx:sx + ss]))
    mask = np.ones(accr.shape, bool)
    mask[sy:sy + ss, sx:sx + ss] = False
    static_mean = float(np.nanmean(accr[mask]))
    nframes = burst["raw_comp"].shape[0]
    return dict(n_comp=nframes, moving_region_accepted=moving_mean,
                static_region_accepted=static_mean,
                discriminates=moving_mean < 0.7 * static_mean, accr=accr,
                region=(sy, sx, ss))
