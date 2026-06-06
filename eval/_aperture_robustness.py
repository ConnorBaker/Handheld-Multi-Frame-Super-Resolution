"""STEP 3 (adversarial) -- NO-BLINDING and NO-GHOSTING checks for the opt-in
flow-regularization recipe, on the REAL 007 burst with INJECTED motion.

The "park burst" the task references (wind/vegetation motion) is not present on
this machine; only the rigid 007 facade is. To exercise the recipe on genuine
independent motion we INJECT a strongly, randomly shifted copy of the scene into a
center sub-region of every comparison frame (the eval/diagnostics.py
robustness_discrimination pattern), keeping the rest the real facade. That gives a
controlled moving region whose ground-truth answer is known: the robustness mask
MUST reject it (low accepted weight), and the merge MUST NOT show the per-frame
motion (no ghosting).

THE RISK (research_aperture_problem.md Q4): compute_s keys the robustness mask on
the 3x3 flow-discontinuity SPAN. If the flow median smooths that span at the
moving-region boundary, it flips s1 (reject) -> s2 (accept) and BLINDS the mask,
re-introducing the ghosts it was meant to catch. The consistency-span gate in
flow_vector_median_cupy is supposed to prevent exactly this (it does not median
where the span is large). This script tests whether it actually does.

Reports (per eval/CLAUDE.md, distributions where applicable):
  1. flow-span signal change in the moving region (the compute_s input).
  2. realized accumulated-robustness mask: moving vs static, default vs recipe.
  3. ghosting: moving-region merge inconsistency vs the static reference.

Run:  PYTHONPATH=. micromamba run -n handheld python eval/_aperture_robustness.py
"""
import sys
import numpy as np
import cupy as cp
import cv2

sys.path.insert(0, ".")
import eval.pipeline_runner as pr
from eval import _aperture_common as AC
from handheld_super_resolution.super_resolution import main

RECIPE = {
    "block_matching.tuning.flow_tikhonov_lambda": 1e-3,
    "block_matching.tuning.flow_median": {"window": 5, "outlier_thresh": 20.0,
                                          "guide_sigma": 0.0},
}
# an AGGRESSIVE variant with NO gate (median applied to EVERY tile) -- the recipe
# that SHOULD blind the mask, included to prove the gate is what protects it
# (a positive control for the no-blinding test).
NO_GATE = {
    "block_matching.tuning.flow_median": {"window": 5, "consistency_span": None,
                                          "outlier_thresh": None, "guide_sigma": 0.0},
}


def inject_motion(burst, region_frac=0.33, seed=0, amp=24, distinct=True):
    """Overwrite each comp frame's center sub-region with INDEPENDENT, per-frame
    content -> 'moving object' the robustness mask must reject.

    Returns (region (ry,rx,rs)). Mutates burst['raw_comp'] in place.

    ``distinct``: when True the injected content is a FAR-AWAY, per-frame-shifted
    crop of the scene (NOT a copy of the local content), so its local colour/mean
    DIFFERS from the reference -> the colour-distance robustness term ``d`` can
    fire (the realized-mask no-blinding test is only meaningful when the baseline
    actually discriminates; a shifted copy of the SAME facade is colour-matched
    and defeats ``d`` regardless of flow -- NOTES.md item 3). When False, a shifted
    copy of the reference (tests the flow-discontinuity term in isolation)."""
    rng = np.random.default_rng(seed)
    ref = burst["ref_raw"]
    H, W = ref.shape
    rs = int(min(H, W) * region_frac)
    ry, rx = (H - rs) // 2, (W - rs) // 2
    # a source crop from a DIFFERENT part of the frame (distinct content/brightness)
    src_y, src_x = 8, 8
    for k in range(burst["raw_comp"].shape[0]):
        dy, dx = rng.uniform(-amp, amp, 2)
        if distinct:
            # take a per-frame-shifted crop from the top-left scene region; its
            # content is unrelated to the center, so local means differ -> d fires
            sy = int(np.clip(src_y + dy, 0, H - rs))
            sx = int(np.clip(src_x + dx, 0, W - rs))
            patch = ref[sy:sy + rs, sx:sx + rs]
        else:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            sh = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
            patch = sh[ry:ry + rs, rx:rx + rs]
        burst["raw_comp"][k, ry:ry + rs, rx:rx + rs] = patch
    return ry, rx, rs


def run_with_mask(burst, overrides, region, want_flow=False):
    """Run main() with save_mask; return (accumulated_robustness, num_rgb[, flows])."""
    cfg = pr.build_config(burst, {**overrides, "scale": 1})
    cfg.robustness.save_mask = True
    cfg.accumulated_robustness_denoiser.enabled = False
    if want_flow:
        cfg.debug = True
    num, dbg = main(burst["ref_raw"], burst["raw_comp"], cfg)
    accr = np.asarray(dbg["accumulated robustness"])
    rgb = np.clip(np.nan_to_num(cp.asnumpy(num)), 0, 1)
    if want_flow:
        return accr, rgb, dbg["flow"], cfg
    return accr, rgb


def section0_compute_s_flip(burst, region):
    """DIRECT no-blinding test: does the recipe change the compute_s signal at the
    moving-region boundary, flipping s1 (reject) -> s2 (accept)?"""
    from handheld_super_resolution.cupy_ops import compute_s_cupy
    print("\n=== 0. compute_s flow-discontinuity signal at the moving boundary ===")
    print("  compute_s -> s1 (REJECT) where 3x3 flow span > Mt, else s2 (ACCEPT). "
          "Blinding = the recipe turns s1 tiles into s2 in the moving region.",
          flush=True)
    cfg0 = pr.build_config(burst, {"scale": 1})
    Mt = float(cfg0.robustness.tuning.Mt)
    s1 = float(cfg0.robustness.tuning.s1)
    s2 = float(cfg0.robustness.tuning.s2)
    ts = int(cfg0.block_matching.tuning.tile_sizes[0])
    ry, rx, rs = region
    # tile-grid index range covering the moving region + a 1-tile boundary ring
    ty0, ty1 = ry // ts - 1, (ry + rs) // ts + 1
    tx0, tx1 = rx // ts - 1, (rx + rs) // ts + 1

    for tag, ov in [("default", {}), ("recipe", RECIPE), ("no-gate(ctrl)", NO_GATE)]:
        _, _, flows, _ = run_with_mask(burst, ov, region, want_flow=True)
        s1_total = 0
        s1_in_region = 0
        ntiles = 0
        for fl in flows:
            S = cp.asnumpy(compute_s_cupy(cp.asarray(fl), Mt, s1, s2))
            is_s1 = (S <= s1 + 1e-6)            # rejecting tiles
            sub = is_s1[ty0:ty1, tx0:tx1]
            s1_total += int(is_s1.sum())
            s1_in_region += int(sub.sum())
            ntiles = sub.size
        n = len(flows)
        print(f"  {tag:<14}: s1(reject) tiles in moving region/frame "
              f"{s1_in_region/n:7.1f}/{ntiles}  ({100*s1_in_region/(n*ntiles):.1f}%); "
              f"whole-frame s1/frame {s1_total/n:.0f}", flush=True)
    print("  (recipe should keep ~as many s1 reject-tiles in the moving region as "
          "default; the no-gate control should LOSE them = blinding.)", flush=True)


def region_means(accr, region, ref_shape):
    ry, rx, rs = region
    H, W = ref_shape
    ah, aw = accr.shape
    sy, sx = int(ry * ah / H), int(rx * aw / W)
    ss_y, ss_x = int(rs * ah / H), int(rs * aw / W)
    mov = float(np.nanmean(accr[sy:sy + ss_y, sx:sx + ss_x]))
    m = np.ones(accr.shape, bool)
    m[sy:sy + ss_y, sx:sx + ss_x] = False
    sta = float(np.nanmean(accr[m]))
    return mov, sta


def main_run():
    burst = pr.load_burst_arw(AC.ARW_DIR)
    nframes = burst["raw_comp"].shape[0]
    region = inject_motion(burst, seed=0)
    print(f"007 burst, {nframes} comp frames; injected moving center region "
          f"{region} (real facade surround)", flush=True)

    section0_compute_s_flip(burst, region)

    print("\n=== 1+2. realized accumulated-robustness mask (moving vs static) ===")
    print("  (mask is summed over comp frames; max = n_comp = {}. LOWER in the "
          "moving region = better rejection. Blinding => moving goes UP.)"
          .format(nframes), flush=True)
    rows = {}
    for tag, ov in [("default", {}), ("recipe", RECIPE), ("no-gate(ctrl)", NO_GATE)]:
        accr, rgb = run_with_mask(burst, ov, region)
        mov, sta = region_means(accr, region, burst["ref_raw"].shape)
        rows[tag] = dict(mov=mov, sta=sta, accr=accr, rgb=rgb)
        print(f"  {tag:<14}: moving accepted {mov:6.3f}/{nframes}  "
              f"static accepted {sta:6.3f}/{nframes}  "
              f"ratio mov/sta {mov/(sta+1e-9):.3f}  "
              f"discriminates={mov < 0.7*sta}", flush=True)
    dmov = rows["recipe"]["mov"] - rows["default"]["mov"]
    print(f"\n  RECIPE vs default: moving-region accepted weight delta {dmov:+.3f} "
          f"(POSITIVE => recipe BLINDED the mask / accepts more motion = BAD)",
          flush=True)
    print(f"  no-gate control delta {rows['no-gate(ctrl)']['mov']-rows['default']['mov']:+.3f} "
          f"(this is what blinding looks like)", flush=True)

    print("\n=== 3. GHOSTING: moving-region merge inconsistency vs the static ref ===")
    print("  RMS(merge - reference) in the moving region (reference is the true "
          "static content). HIGHER => more per-frame motion leaked into the merge "
          "(ghosting).", flush=True)
    # reference luma at scale 1: a single-frame merge of the (clean) reference
    ref_rgb = pr.run_single(burst, overrides={"scale": 1})[0]
    ry, rx, rs = region
    s = 1  # scale
    rsl = (slice(ry, ry + rs), slice(rx, rx + rs))
    ref_crop = ref_rgb[rsl].mean(-1)
    for tag in ("default", "recipe", "no-gate(ctrl)"):
        mg = rows[tag]["rgb"][rsl].mean(-1)
        # level-match then RMS difference (ghosting = structured residual)
        a = (mg - mg.mean()) / (mg.std() + 1e-9) * ref_crop.std() + ref_crop.mean()
        rms = float(np.sqrt(np.mean((a - ref_crop) ** 2)))
        print(f"  {tag:<14}: moving-region RMS(merge-ref) {rms:.4f}", flush=True)
    print("  (default is the Wronski backstop; recipe should be ~equal, NOT higher.)",
          flush=True)


if __name__ == "__main__":
    main_run()
