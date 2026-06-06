"""Drive the real-vs-hallucinated analysis end to end and regenerate every number
and image from one command:

    python -m eval._halluc_run --burst /path/to/DxO_dngs --out eval/halluc_out \
        [--merge auto|PATH.npy] [--quick]

--merge auto runs our multi-frame pipeline on the burst's ARW frames (so the dir
must hold the matching .ARW files) to cross-check DxO's detail against a prior-free
reconstruction; --merge PATH.npy loads a precomputed linear-RGB merge; omit it to
run the DxO-only analysis. All analysis lives in eval/hallucination.py; this file
is presentation + orchestration only.
"""
import argparse
import glob
import os

import numpy as np
import cv2

from . import hallucination as H
from .calibrate_optics import read_dng


# ----------------------------- rendering ----------------------------------

def _stretch(a, lo=1, hi=99):
    p1, p99 = np.percentile(a, lo), np.percentile(a, hi)
    return (np.clip((a - p1) / (p99 - p1 + 1e-6), 0, 1) * 255).astype(np.uint8)


def _label(img, txt, color=(0, 255, 0)):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 20), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def display_rgb(path, gamma=2.2):
    """Gray-world WB + gamma a DxO linear DNG to a viewable BGR uint8-ish float."""
    rgb = read_dng(path)
    gains = rgb.reshape(-1, 3).mean(0)
    gains = gains.mean() / np.maximum(gains, 1e-6)
    return (np.clip(rgb * gains, 0, 1) ** (1 / gamma))[..., ::-1]  # BGR


def render_overview(path, out, width=1600, grid=500):
    disp = display_rgb(path)
    Hh, Ww = disp.shape[:2]
    sc = width / Ww
    img = cv2.resize((disp * 255).astype(np.uint8), (width, int(Hh * sc)))
    for x in range(0, Ww, grid):
        xs = int(x * sc); cv2.line(img, (xs, 0), (xs, img.shape[0]), (0, 255, 0), 1)
        cv2.putText(img, str(x), (xs + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    for y in range(0, Hh, grid):
        ys = int(y * sc); cv2.line(img, (0, ys), (img.shape[1], ys), (0, 255, 0), 1)
        cv2.putText(img, str(y), (2, ys + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(out, img)


def render_heatmap(sm, ref_path, out, width=1600):
    """Overlay per-tile real_fraction on the scene (green=real, red=invented,
    opacity proportional to detail level)."""
    base = (display_rgb(ref_path) * 255).astype(np.uint8)
    Hh, Ww = base.shape[:2]
    sc = width / Ww
    base = cv2.resize(base, (width, int(Hh * sc)))
    over = base.copy()
    smax = np.nanpercentile(sm["grid_single"], 95) + 1e-9
    for iy, cy in enumerate(sm["ys"]):
        for ix, cx in enumerate(sm["xs"]):
            r = sm["grid_rf"][iy, ix]
            if not np.isfinite(r):
                continue
            det = min(sm["grid_single"][iy, ix] / smax, 1.0)
            g = min(r / 0.8, 1.0)
            col = (0, int(255 * g), int(255 * (1 - g)))  # BGR red->green
            s = sm["size"]
            y0, y1 = int((cy - s // 2) * sc), int((cy + s // 2) * sc)
            x0, x1 = int((cx - s // 2) * sc), int((cx + s // 2) * sc)
            a = 0.15 + 0.5 * det
            over[y0:y1, x0:x1] = (over[y0:y1, x0:x1] * (1 - a) + np.array(col) * a).astype(np.uint8)
            if det > 0.25:
                cv2.putText(over, f"{r:.2f}", (x0 + 3, y0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.imwrite(out, cv2.addWeighted(base, 0.35, over, 0.65, 0))


def render_montage(stack, merge_luma, cy, cx, size, out, tag, frames=None):
    """3 independent DxO single renders + their HF (top), consensus(real) | our
    merge | inconsistency(hallucination) map (bottom)."""
    rois, info = H.extract_aligned_roi(stack, cy, cx, size=size, search=130)
    d = H.consensus_decomp(rois, info)
    cons = d["consensus"]
    N = len(rois)
    fa, fb, fc = frames if frames else (0, N // 2, N - 1)   # spread across the burst, N-aware

    def hp_vis(a, k=6):
        hp = H._highpass(a, 1.6)
        return (np.clip(0.5 + k * hp / (a.std() + 1e-6), 0, 1) * 255).astype(np.uint8)

    top = np.hstack([_label(_stretch(rois[fa]), f"DxO frame {fa}"),
                     _label(_stretch(rois[fb]), f"DxO frame {fb}"),
                     _label(_stretch(rois[fc]), f"DxO frame {fc}")])
    mid = np.hstack([_label(hp_vis(rois[fa]), f"HF frame {fa}"),
                     _label(hp_vis(rois[fb]), f"HF frame {fb}"),
                     _label(hp_vis(rois[fc]), f"HF frame {fc}")])
    if merge_luma is not None:
        mg = H.align_to(rois[0], merge_luma, cy, cx, size)
        mg = (mg - mg.mean()) / (mg.std() + 1e-6) * cons.std() + cons.mean()
        mid_panel = _label(_stretch(mg), "OUR multi-frame merge")
    else:
        mid_panel = _label(_stretch(d["real_map"]), "consistent detail (REAL)")
    hmap = d["halluc_map"]
    hmap = cv2.applyColorMap((np.clip(hmap / (np.percentile(hmap, 99) + 1e-9), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    bot = np.hstack([_label(_stretch(cons), "DxO consensus = REAL (median of N)"),
                     mid_panel,
                     _label(hmap, "per-frame inconsistency = HALLUCINATION", (255, 255, 255))])
    gap = np.full((6, top.shape[1], 3), 40, np.uint8)
    full = _label(np.vstack([top, gap, mid, gap, bot]),
                  f"  {tag}: 3 INDEPENDENT DxO renders of the SAME scene  "
                  f"(real detail repeats; invented detail differs)  real_fraction={d['real_fraction']:.2f}",
                  (0, 255, 255))
    cv2.imwrite(out, full)
    return d["real_fraction"]


# ----------------------------- merge --------------------------------------

def compute_merge_luma(burst_dir):
    """Run our prior-free multi-frame pipeline on the burst's ARW frames ->
    gamma luma on the DxO grid (lazy import: pulls in cupy/torch/Warp)."""
    from . import pipeline_runner as pr
    burst = pr.load_burst_arw(burst_dir)
    merge_lin, _ = pr.run(burst)
    return H.rgb_to_luma(merge_lin)


# ----------------------------- driver -------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burst", required=True, help="dir of DxO linear-DNG renders")
    ap.add_argument("--out", default="eval/halluc_out", help="output dir")
    ap.add_argument("--pattern", default="DSC*_DxO-reference.dng", help="DNG glob (e.g. 'DSC*_DxO.dng')")
    ap.add_argument("--arw", default=None, help="dir of .ARW frames for --merge auto (default: --burst)")
    ap.add_argument("--merge", default=None, help="'auto' (run pipeline on .ARW) or a linear-RGB .npy")
    ap.add_argument("--quick", action="store_true", help="coarser scene map (faster)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.burst, args.pattern)))
    assert paths, f"no {args.pattern} in {args.burst}"
    print(f"{len(paths)} DxO renders from {args.burst}")

    stack = H.load_luma_stack(paths, cache=os.path.join(args.out, "luma_stack.npy"))
    print(f"stack {stack.shape}")

    # 1. burst geometry
    drift = H.measure_drift(stack)
    dxs = np.array([d["mean_dx"] for d in drift]); dys = np.array([d["mean_dy"] for d in drift])
    th = np.abs([d["theta_deg"] for d in drift]).max(); spr = np.max([d["spread_px"] for d in drift])
    print(f"[drift] pan span {np.hypot(np.ptp(dys), np.ptp(dxs)):.0f}px  max rotation {th:.2f}deg  "
          f"max corner spread {spr:.0f}px (large spread + ~0 rotation => PARALLAX => per-ROI registration)")

    # 2. scene-wide real-vs-hallucinated map (cached: re-runs are expensive)
    stride = 384 if args.quick else 256
    map_cache = os.path.join(args.out, "scene_map.npz")
    if os.path.exists(map_cache):
        z = np.load(map_cache)
        sm = {k: z[k] for k in z.files}
        for k in ("size", "n_tiles"):
            sm[k] = int(sm[k])
        for k in ("global_real_fraction", "weighted_real_fraction"):
            sm[k] = float(sm[k])
        print(f"[scene] reused {map_cache} (delete it to recompute)")
    else:
        sm = H.scene_map(stack, stride=stride,
                         progress=lambda i, n: print(f"  scene map row {i}/{n}", flush=True))
        np.savez(map_cache, **{k: (v if isinstance(v, np.ndarray) else np.array(v))
                               for k, v in sm.items()})
    render_overview(paths[0], os.path.join(args.out, "overview_grid.png"))
    render_heatmap(sm, paths[0], os.path.join(args.out, "realfraction_heatmap.png"))
    print(f"[scene] GLOBAL real_fraction (HF-power-weighted, {sm['n_tiles']} tiles) = "
          f"{sm['global_real_fraction']:.3f} -> {100*sm['global_real_fraction']:.0f}% of DxO HF is REAL, "
          f"{100*(1-sm['global_real_fraction']):.0f}% per-frame (invented + noise)")

    # 3. controls / structure on auto-selected detail + flat tiles
    single = sm["grid_single"]; peakok = sm["grid_peak"] > 0.5
    hf = np.where(peakok, single, np.nan)
    iy_d, ix_d = np.unravel_index(np.nanargmax(hf), hf.shape)
    target = np.nanquantile(hf, 0.15)                         # representative low-detail (not dead-flat) tile
    iy_f, ix_f = np.unravel_index(np.nanargmin(np.abs(hf - target)), hf.shape)
    det_yx = (int(sm["ys"][iy_d]), int(sm["xs"][ix_d]))
    flat_yx = (int(sm["ys"][iy_f]), int(sm["xs"][ix_f]))
    base = H.extract_aligned_roi(stack, *det_yx, size=256, search=130)[0][0]
    print(f"[control] resampling floor on the highest-detail tile {det_yx} = "
          f"{H.resampling_floor(base):.3f} (=> low real_fraction is genuine, not interpolation)")
    for tag, yx in [("detail " + str(det_yx), det_yx), ("flat " + str(flat_yx), flat_yx)]:
        rois = H.extract_aligned_roi(stack, *yx, size=256, search=130)[0]
        s = H.inconsistency_structure(rois)
        ac = s["lag1_autocorr"]
        kind = ("negligible HF (too flat to characterize)" if not np.isfinite(ac)
                else "white NOISE (residual)" if ac < 0.15 else "structured INVENTED texture")
        print(f"[structure] {tag}: lag-1 autocorr {ac:.2f} -> {kind} (peak scale ~{s['peak_scale_px']:.1f}px)")

    # 4. merge cross-check (optional)
    merge_luma = None
    if args.merge == "auto":
        arw_dir = args.arw or args.burst
        print(f"[merge] running our pipeline on the .ARW burst in {arw_dir} ...")
        merge_luma = compute_merge_luma(arw_dir)
        np.save(os.path.join(args.out, "merge_linear_luma.npy"), merge_luma)
    elif args.merge:
        lin = np.load(args.merge)
        merge_luma = H.rgb_to_luma(lin) if lin.ndim == 3 else lin

    # 5. montages on detail + flat (+ scene median) tiles
    iy_m, ix_m = np.unravel_index(np.argsort(single, axis=None)[len(single.ravel()) // 2], single.shape)
    mid_yx = (int(sm["ys"][iy_m]), int(sm["xs"][ix_m]))
    for tag, yx in [("detail", det_yx), ("mid", mid_yx), ("flat", flat_yx)]:
        rf = render_montage(stack, merge_luma, *yx, 384,
                            os.path.join(args.out, f"montage_{tag}.png"), tag)
        print(f"[montage] {tag} {yx} -> montage_{tag}.png (real_fraction {rf:.2f})")

    # 6. edge-width (ESF) cross-check where the scene has clean step edges
    med = np.median(stack[::5], 0).astype(np.float32)
    edges = H.find_step_edges(med)
    if edges and merge_luma is not None:
        res = H.compare_edge_sharpness(stack, merge_luma, edges)
        print(f"[edges] {res['conclusion']}")
    else:
        print(f"[edges] {len(edges)} clean step edges found"
              + ("" if merge_luma is not None else " (merge not supplied)")
              + " -> ESF check N/A on this scene" * (not edges))
    print(f"\nartifacts in {args.out}/ : realfraction_heatmap.png, montage_*.png, scene_map.npz")


if __name__ == "__main__":
    main()
