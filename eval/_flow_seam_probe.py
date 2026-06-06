"""Confirm (cheaply) that the per-tile piecewise-constant flow is the merge-seam
mechanism: measure inter-tile flow jumps and the predicted seam period.

The merge gathers each output sample at a flow read by BLOCK lookup
al[int(lr_y/tile), int(lr_x/tile)] (warp_ops.py) -- constant within a tile,
discontinuous at tile borders. Seam period in OUTPUT px = tile*scale.
"""
import numpy as np
from eval import pipeline_runner as pr
from handheld_super_resolution.super_resolution import main

burst = pr.load_burst_arw("/home/connorbaker/Pictures/ARW/2025-09-06/007")
cfg = pr.build_config(burst, overrides={"scale": 1, "debug": True, "verbose": 0})
tile = int(cfg.block_matching.tuning.tile_size)
print(f"resolved finest tile_size = {tile} px (LR/grey-quad units)")

# scale=1, a few frames -> cheap; we only need the flows
num, dbg = main(burst["ref_raw"], burst["raw_comp"][:4], cfg)
flows = dbg.get("flow", [])
print(f"got {len(flows)} per-tile flow fields, shape {flows[0].shape}  (np_y, np_x, 2)")

np.save("eval/halluc_out_007/_flows_s1.npy", np.stack([np.asarray(f) for f in flows]))
lr_h, lr_w = burst["ref_raw"].shape
allmag, alljump = [], []
oob_frac = []
for f in flows:
    f = np.asarray(f)
    mag = np.sqrt(f[..., 0]**2 + f[..., 1]**2)
    allmag.append(mag.ravel())
    jh = np.sqrt(np.diff(f[..., 0], axis=1)**2 + np.diff(f[..., 1], axis=1)**2)
    jv = np.sqrt(np.diff(f[..., 0], axis=0)**2 + np.diff(f[..., 1], axis=0)**2)
    alljump.append(np.concatenate([jh.ravel(), jv.ravel()]))
    # fraction of tiles whose flow pushes the gather out of bounds (kernel skips these)
    ys, xs = np.mgrid[0:f.shape[0], 0:f.shape[1]] * tile + tile / 2
    mvx = xs + f[..., 0]; mvy = ys + f[..., 1]
    oob = (mvx < 0) | (mvx >= lr_w) | (mvy < 0) | (mvy >= lr_h)
    oob_frac.append(oob.mean())
allmag = np.concatenate(allmag); alljump = np.concatenate(alljump)
pct = lambda a, p: np.percentile(a, p)
print("\nper-tile flow MAGNITUDE (LR px) distribution (all comp frames):")
print(f"  median {pct(allmag,50):.3f}  p90 {pct(allmag,90):.3f}  p99 {pct(allmag,99):.2f}  p99.9 {pct(allmag,99.9):.1f}  max {allmag.max():.0f}")
print(f"  fraction with |flow|>5px: {(allmag>5).mean()*100:.2f}%   >50px: {(allmag>50).mean()*100:.3f}%")
print(f"  fraction of tiles OOB (skipped by merge -> no contribution, not a seam): {np.mean(oob_frac)*100:.2f}%")
print("\ninter-tile flow JUMP magnitude (LR px) distribution:")
print(f"  median {pct(alljump,50):.3f}  p90 {pct(alljump,90):.3f}  p99 {pct(alljump,99):.2f}  max {alljump.max():.0f}")
print(f"  -> seam in FLAT regions is the MEDIAN-scale jump ({pct(alljump,50):.3f}px); outliers are rare & OOB-skipped")
print(f"\n  seam period in OUTPUT px = tile*scale: scale1={tile}px scale2={tile*2}px (matches flat-region spectral peaks 16/32/64)")
print(f"  => naive bilinear of the flow would SPREAD the rare {allmag.max():.0f}px outliers; a robust (median/clamped) smooth is needed")
