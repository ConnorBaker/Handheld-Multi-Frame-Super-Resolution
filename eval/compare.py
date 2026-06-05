"""Scriptable comparison pipeline + automated failure detection.

Two complementary failure signals:

  (A) vs Canon ground truth (perceptual): register the output into the Canon
      frame, compute per-tile DISTS/LPIPS; worst tiles = perceptual failures.
      Caveat: a single homography can't model parallax/non-planar foreground, so
      some "failures" here are registration error -- DISTS is alignment-robust,
      and (B) cross-checks without registration.

  (B) vs same-FOV baseline (clean): multi-frame vs single-frame at scale 1 (no
      registration needed). Tiles where multi sharpness < single sharpness are
      genuine algorithm regressions (e.g. moving foliage being averaged).

`prepare_target` + `score` cache the registration geometry so a sweep can score
hundreds of configs without re-running SIFT or re-reading the DNG.
"""

import os
import json

import numpy as np
import cv2
import imageio.v3 as iio

from . import pipeline_runner as pr
from . import registration as reg
from . import metrics as met


def _gamma(linear):
    return np.clip(linear, 0, 1) ** (1 / 2.2)


# --------------------------------------------------------------------------
# Cached-registration fast path (for the sweep)
# --------------------------------------------------------------------------
def prepare_target(burst, canon_rgb, ref_overrides=None, tile=256, max_tiles=80, work=1600):
    """Register a reference run to Canon and cache geometry for fast scoring."""
    mine, _ = pr.run(burst, ref_overrides)
    H, info = reg.estimate_homography(_gamma(mine), canon_rgb, work=work)
    _, valid = reg.warp_into(_gamma(mine), canon_rgb.shape, H)
    ys, xs = np.where(valid)
    pad = 8
    bbox = (ys.min() + pad, ys.max() - pad, xs.min() + pad, xs.max() - pad)
    y0, y1, x0, x1 = bbox
    canon_crop = canon_rgb[y0:y1, x0:x1]
    valid_crop = valid[y0:y1, x0:x1]
    coords = met.tile_coords(canon_crop.shape, valid_crop, tile=tile, max_tiles=max_tiles)
    return dict(H=H, info=info, bbox=bbox, canon_full_shape=canon_rgb.shape,
                canon_crop=canon_crop, valid_crop=valid_crop, coords=coords, tile=tile)


def score(burst, overrides, target):
    """Fast perceptual score for a config using cached registration geometry.

    Returns (scalars dict, full tiled-metrics dict, registered output crop).
    """
    mine, _ = pr.run(burst, overrides)
    warped, _ = reg.warp_into(_gamma(mine), target["canon_full_shape"], target["H"])
    y0, y1, x0, x1 = target["bbox"]
    mc = reg.color_match(warped[y0:y1, x0:x1], target["canon_crop"], target["valid_crop"])
    M = met.tiled_metrics(mc, target["canon_crop"], target["valid_crop"],
                          tile=target["tile"], coords=target["coords"])
    scalars = {k: M[k] for k in ["LPIPS", "DISTS", "PSNR", "SSIM", "sharp_ratio"]}
    return scalars, M, mc


# --------------------------------------------------------------------------
# Standalone evaluation + failure detection
# --------------------------------------------------------------------------
def evaluate(burst, canon_rgb, overrides=None, tile=256, max_tiles=300):
    """Full standalone evaluation of one config vs Canon (fresh registration)."""
    mine, cfg = pr.run(burst, overrides)
    R = reg.register(_gamma(mine), canon_rgb)
    M = met.tiled_metrics(R["mine"], R["canon"], R["valid"], tile=tile, max_tiles=max_tiles)
    return dict(overrides=overrides or {}, scale=int(cfg.scale), reg=R["info"],
                metrics={k: M[k] for k in ["LPIPS", "DISTS", "PSNR", "SSIM", "sharp_ratio"]},
                tiles=M, registered=R)


def detect_regressions(burst, overrides=None, tile=256, thresh=0.90, max_tiles=400):
    """Clean (registration-free) algorithm-failure detector.

    Compares multi-frame vs single-frame at scale 1 over the same FOV. Returns
    per-tile sharpness ratio (multi/single); tiles below `thresh` are regions
    where multi-frame fusion DESTROYS detail relative to a single frame.
    """
    ov = dict(overrides or {})
    ov["scale"] = 1
    multi, _ = pr.run(burst, ov)
    single, _ = pr.run_single(burst, overrides)
    coords = met.tile_coords(multi.shape, None, tile=tile, max_tiles=max_tiles)
    rows = []
    for (y, x) in coords:
        sm = met.sharpness(multi[y:y + tile, x:x + tile])
        ss = met.sharpness(single[y:y + tile, x:x + tile])
        rows.append(dict(y=int(y), x=int(x), sharp_multi=sm, sharp_single=ss,
                         ratio=float(sm / max(ss, 1e-6))))
    rows.sort(key=lambda r: r["ratio"])
    fails = [r for r in rows if r["ratio"] < thresh]
    return dict(rows=rows, fails=fails, n_fail=len(fails), n_tiles=len(rows),
                worst=rows[:10], multi=multi, single=single, tile=tile)


def detect_failures(burst, canon_rgb, overrides=None, tile=256, max_tiles=150):
    """Canon-anchored: does multi-frame help or hurt vs a single frame, per tile?

    Registers BOTH the configured (multi-frame) output and the single-frame
    baseline into the Canon frame, then compares per-tile DISTS. delta = DISTS
    (multi) - DISTS(single); delta > 0 means multi-frame made that tile
    perceptually WORSE than a single frame (a genuine fusion failure, not just
    denoising). Unlike `detect_regressions`, this is anchored to ground truth.
    """
    multi, _ = pr.run(burst, overrides)
    single, _ = pr.run_single(burst, overrides)
    Hm, _ = reg.estimate_homography(_gamma(multi), canon_rgb)
    Hs, _ = reg.estimate_homography(_gamma(single), canon_rgb)
    wm, vm = reg.warp_into(_gamma(multi), canon_rgb.shape, Hm)
    ws, vs = reg.warp_into(_gamma(single), canon_rgb.shape, Hs)
    valid = vm & vs
    ys, xs = np.where(valid)
    pad = 8
    y0, y1, x0, x1 = ys.min() + pad, ys.max() - pad, xs.min() + pad, xs.max() - pad
    cc, vc = canon_rgb[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    mc = reg.color_match(wm[y0:y1, x0:x1], cc, vc)
    sc = reg.color_match(ws[y0:y1, x0:x1], cc, vc)
    coords = met.tile_coords(cc.shape, vc, tile=tile, max_tiles=max_tiles)
    Mm = met.tiled_metrics(mc, cc, vc, tile=tile, coords=coords)
    Ms = met.tiled_metrics(sc, cc, vc, tile=tile, coords=coords)
    rows = [dict(y=int(coords[i][0]), x=int(coords[i][1]),
                 dists_multi=float(Mm["dists"][i]), dists_single=float(Ms["dists"][i]),
                 delta=float(Mm["dists"][i] - Ms["dists"][i])) for i in range(len(coords))]
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return dict(rows=rows, worst=rows[:10],
                n_worse=int(sum(1 for r in rows if r["delta"] > 0)), n_tiles=len(rows),
                DISTS_multi=float(Mm["DISTS"]), DISTS_single=float(Ms["DISTS"]))


# --------------------------------------------------------------------------
# Reporting (heatmaps, worst-tile montages, JSON)
# --------------------------------------------------------------------------
def _heatmap(coords, values, shape, tile, invert=False):
    """Paint per-tile values onto a canvas, colormapped (JET). Higher=worse(red)."""
    H, W = shape[:2]
    canvas = np.full((H, W), np.nan, np.float32)
    for (y, x), v in zip(coords, values):
        canvas[y:y + tile, x:x + tile] = v
    finite = np.isfinite(canvas)
    if finite.any():
        lo, hi = np.nanpercentile(canvas, 2), np.nanpercentile(canvas, 98)
        norm = np.clip((canvas - lo) / (hi - lo + 1e-9), 0, 1)
        if invert:
            norm = 1 - norm
    else:
        norm = np.zeros((H, W), np.float32)
    norm = np.nan_to_num(norm)
    hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def save_report(burst, canon_rgb, overrides, outdir, tile=256, n_worst=6):
    """Run evaluate + detect_regressions and write heatmaps, montages, JSON."""
    os.makedirs(outdir, exist_ok=True)
    ev = evaluate(burst, canon_rgb, overrides, tile=tile)
    reg_ = detect_regressions(burst, overrides, tile=tile)
    M, R = ev["tiles"], ev["registered"]

    def _u8(x, wide=1200):
        h, w = x.shape[:2]
        s = wide / w
        return (np.clip(cv2.resize(x, (wide, round(h * s))), 0, 1) * 255).astype(np.uint8)

    # DISTS heatmap over the registered overlap (red = worst)
    hm = _heatmap(M["coords"], M["dists"], R["canon"].shape, tile)
    blend = (0.55 * R["canon"] * 255 + 0.45 * hm).astype(np.uint8)
    iio.imwrite(os.path.join(outdir, "dists_heatmap.png"), _u8(blend / 255.0))

    # worst-DISTS tile montage: mine (top) vs canon (bottom)
    order = np.argsort(M["dists"])[::-1][:n_worst]
    cols = []
    for i in order:
        y, x = M["coords"][i]
        mt = (np.clip(R["mine"][y:y + tile, x:x + tile], 0, 1) * 255).astype(np.uint8)
        ct = (np.clip(R["canon"][y:y + tile, x:x + tile], 0, 1) * 255).astype(np.uint8)
        lbl = np.zeros((18, tile, 3), np.uint8)
        cv2.putText(lbl, f"D={M['dists'][i]:.3f}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cols.append(np.vstack([lbl, mt, ct]))
    if cols:
        iio.imwrite(os.path.join(outdir, "worst_dists_tiles_mine_vs_canon.png"), np.hstack(cols))

    # regression heatmap (multi/single sharpness; red = multi lost detail)
    rc = [(r["y"], r["x"]) for r in reg_["rows"]]
    rv = [r["ratio"] for r in reg_["rows"]]
    hm2 = _heatmap(rc, rv, reg_["multi"].shape, tile, invert=True)
    blend2 = (0.55 * _gamma(reg_["multi"]) * 255 + 0.45 * hm2).astype(np.uint8)
    iio.imwrite(os.path.join(outdir, "regression_heatmap.png"), _u8(blend2 / 255.0))

    summary = dict(
        overrides=overrides or {}, scale=ev["scale"], registration=ev["reg"],
        metrics=ev["metrics"],
        regression={"n_fail": reg_["n_fail"], "n_tiles": reg_["n_tiles"],
                    "worst_tiles": reg_["worst"]},
    )
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
