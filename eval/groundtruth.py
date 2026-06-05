"""Independent ground truth + forward-model sensitivity.

Two ways `eval/synthetic.py` could be circular / too easy:
  1. GT = the Canon *JPEG*, which carries Canon's demosaic + sharpening.
  2. The synthetic forward model (one downsample kernel + mosaic) might happen to
     suit the pipeline.

This module addresses both:
  - `canon_gt` builds GT from the Canon *RAW* (CR2) demosaiced by rawpy -- an
    independent, un-sharpened reference.
  - `make_burst` takes a selectable downsample kernel; `sensitivity` runs the
    pipeline + baselines across several kernels and checks the conclusion
    (multi > single > bilinear vs GT) is robust to the forward-model choice.
"""

import numpy as np
import cv2
import rawpy
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr

from handheld_super_resolution.fast_monte_carlo import run_fast_MC
from . import pipeline_runner as pr
from . import metrics as met

_CFA = {"GRBG": np.array([[1, 0], [2, 1]]), "RGGB": np.array([[0, 1], [1, 2]])}

_DOWNSAMPLE = {
    "area": lambda im, W, H: cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA),
    "bicubic": lambda im, W, H: cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC),
    "lanczos": lambda im, W, H: cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4),
    "gaussian": lambda im, W, H: cv2.resize(cv2.GaussianBlur(im, (0, 0), 1.0), (W, H),
                                            interpolation=cv2.INTER_NEAREST),
}


def canon_gt(cr2_path, crop=None):
    """Demosaic the Canon RAW (independent of the JPEG) -> linear RGB [0,1]."""
    with rawpy.imread(cr2_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, gamma=(1, 1),
                              output_bps=16, half_size=False, user_flip=0)
    gt = rgb.astype(np.float32) / 65535.0
    if crop is not None:
        y, x, s = crop
        gt = gt[y:y + s, x:x + s]
    H, W, _ = gt.shape
    gt = gt[: H - H % 4, : W - W % 4]
    # normalize exposure to a sane range (RAW linear is often dark)
    gt = gt / max(np.percentile(gt, 99.5), 1e-6)
    return np.clip(gt, 0, 1)


def make_burst(gt, downsample="area", n=8, factor=2, max_shift=2.0, alpha=1e-4,
               beta=0.0, seed=0, cfa_str="GRBG"):
    """Synthetic burst with a SELECTABLE downsample forward model."""
    rng = np.random.default_rng(seed)
    ds = _DOWNSAMPLE[downsample]
    H2, W2, _ = gt.shape
    H, W = H2 // factor, W2 // factor
    cfa = _CFA[cfa_str]
    frames = []
    for i in range(n):
        dy, dx = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-max_shift, max_shift, 2))
        M = np.float32([[1, 0, dx * factor], [0, 1, dy * factor]])
        sh = cv2.warpAffine(gt, M, (W2, H2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        lr = ds(sh, W, H)
        bayer = np.zeros((H, W), np.float32)
        for ii in range(2):
            for jj in range(2):
                bayer[ii::2, jj::2] = lr[ii::2, jj::2, cfa[ii, jj]]
        if alpha > 0:
            sigma = np.sqrt(np.maximum(alpha * bayer + beta, 0)).astype(np.float32)
            bayer = bayer + rng.normal(0, 1, bayer.shape).astype(np.float32) * sigma
        frames.append(np.clip(bayer, 0, 1).astype(np.float32))
    std_curve, diff_curve = run_fast_MC(max(alpha, 1e-6), beta)
    ref = frames[0]
    SNR = float(np.mean(ref)) / std_curve[round(1000 * float(np.mean(ref)))]
    return dict(path="synthetic", ref_raw=ref, raw_comp=np.array(frames[1:]), ISO=100,
                tags=None, CFA=cfa, cfa_str=cfa_str, xyz2cam=None, wb=[1.0, 1.0, 1.0],
                ref_path=None, alpha=max(alpha, 1e-6), beta=beta,
                std_curve=std_curve, diff_curve=diff_curve, SNR=SNR)


def _metrics(out, gt):
    out = np.clip(out, 0, 1).astype(np.float32)
    return dict(PSNR=float(_psnr(gt, out, data_range=1.0)),
                SSIM=float(_ssim(gt, out, channel_axis=2, data_range=1.0)),
                DISTS=float(met.tiled_metrics(out, gt, tile=256, max_tiles=80)["DISTS"]),
                sharp_ratio=met.sharpness(out) / max(met.sharpness(gt), 1e-6))


def sensitivity(gt, overrides=None, downsamples=("area", "bicubic", "lanczos", "gaussian"),
                n=8, alpha=1e-4):
    """Run multi/single/bilinear vs GT across forward models; report robustness."""
    rows = {}
    for ds in downsamples:
        burst = make_burst(gt, downsample=ds, n=n, alpha=alpha)
        ov = dict(overrides or {}); ov["scale"] = 2
        multi, _ = pr.run(burst, ov)
        single, _ = pr.run_single(burst, overrides)
        bil = pr.run_bilinear(burst)
        H2, W2 = gt.shape[:2]
        up = lambda im: cv2.resize(im, (W2, H2), interpolation=cv2.INTER_LINEAR)
        rows[ds] = dict(multi=_metrics(multi, gt), single=_metrics(up(single), gt),
                        bilinear=_metrics(up(bil), gt))
    return rows
