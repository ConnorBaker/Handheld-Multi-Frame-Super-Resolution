"""Image-quality metrics: perceptual (LPIPS, DISTS) + classical (PSNR, SSIM) +
local sharpness (Laplacian variance), computed globally and per-tile.

Per-tile metrics drive both the sweep objective (mean DISTS over tiles) and the
automated failure detection (worst tiles by DISTS / sharpness regression).

LPIPS expects [0,1] with normalize=True (scaled to [-1,1] internally); DISTS
expects [0,1]. PSNR/SSIM via skimage.
"""

import numpy as np
import torch
import cv2
import lpips as _lpips_mod
from DISTS_pytorch import DISTS as _DISTS
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

_LPIPS = None
_DISTSM = None


def _models():
    global _LPIPS, _DISTSM
    if _LPIPS is None:
        _LPIPS = _lpips_mod.LPIPS(net="alex", verbose=False).cuda().eval()
        _DISTSM = _DISTS().cuda().eval()
    return _LPIPS, _DISTSM


def _batch(tiles):
    arr = np.stack(tiles).transpose(0, 3, 1, 2).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)).cuda()


def sharpness(rgb):
    """Laplacian variance on the luma (detail proxy)."""
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def tile_coords(shape, valid=None, tile=256, stride=None, min_valid=0.97, max_tiles=80):
    """Grid of (y, x) tile origins fully inside the valid mask, capped at max_tiles."""
    H, W = shape[:2]
    stride = stride or tile
    coords = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            if valid is None or valid[y:y + tile, x:x + tile].mean() >= min_valid:
                coords.append((y, x))
    if len(coords) > max_tiles:
        idx = np.linspace(0, len(coords) - 1, max_tiles).round().astype(int)
        coords = [coords[i] for i in dict.fromkeys(idx)]  # unique, ordered
    return coords


def tiled_metrics(mine, canon, valid=None, tile=256, max_tiles=80, batch=16, coords=None):
    """Per-tile LPIPS/DISTS/PSNR/SSIM/sharpness over the valid overlap.

    Returns a dict with per-tile arrays, tile coords, and global means.
    Lower LPIPS/DISTS = better; higher PSNR/SSIM = better.
    Pass ``coords`` (from tile_coords) to score on a FIXED grid across configs.
    """
    if coords is None:
        coords = tile_coords(mine.shape, valid, tile=tile, max_tiles=max_tiles)
    if not coords:
        raise RuntimeError("no fully-valid tiles in overlap")
    mt = [mine[y:y + tile, x:x + tile] for y, x in coords]
    ct = [canon[y:y + tile, x:x + tile] for y, x in coords]
    lp, ds = _models()

    lpv, dsv = [], []
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            mb = _batch(mt[i:i + batch])
            cb = _batch(ct[i:i + batch])
            lpv.append(lp(mb, cb, normalize=True).squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy())
            dsv.append(ds(mb, cb, batch_average=False).cpu().numpy())
    lpv = np.concatenate(lpv).astype(np.float64)
    dsv = np.concatenate(dsv).astype(np.float64)
    psnrv = np.array([_psnr(ct[i], mt[i], data_range=1.0) for i in range(len(coords))])
    ssimv = np.array([_ssim(ct[i], mt[i], channel_axis=2, data_range=1.0) for i in range(len(coords))])
    shp_m = np.array([sharpness(t) for t in mt])
    shp_c = np.array([sharpness(t) for t in ct])

    return dict(
        coords=coords, tile=tile,
        lpips=lpv, dists=dsv, psnr=psnrv, ssim=ssimv,
        sharp_mine=shp_m, sharp_canon=shp_c,
        LPIPS=float(lpv.mean()), DISTS=float(dsv.mean()),
        PSNR=float(psnrv.mean()), SSIM=float(ssimv.mean()),
        sharp_ratio=float((shp_m / np.maximum(shp_c, 1e-6)).mean()),
    )
