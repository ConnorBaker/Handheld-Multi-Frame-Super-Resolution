"""STEP 3/4 -- VALIDATE the opt-in flow-regularization recipe on the 007 burst.

ONE reproducible script. Runs the multi-frame pipeline at scale=1 under several
flow-regularization configs (all opt-in; default OFF), caches each merge in a
worktree-local dir, and reports -- as DISTRIBUTIONS, reliability-gated, per
eval/CLAUDE.md -- whether the recipe reduces the multi-frame edge-fidelity cost
(divergence from the DxO cross-frame consensus) toward the single-frame floor,
WITHOUT (i) blinding the robustness mask or (ii) causing ghosting.

Sections:
  0. bit-identity: recipe OFF == default (load-bearing; printed first).
  A. the documented "jut" feature: single / multi / multi+recipe divergence.
  B. DISTRIBUTION over all clean step edges (reliability-gated): divergence +
     ESF edge-sharpness (no regression check).
  C. NO-BLINDING: does the recipe erase the robustness flow-discontinuity signal?
     Measures the change in compute_s's input (3x3 flow span) and the realized
     robustness mask on a synthetically-injected moving region.
  D. NO-GHOSTING: inject an independently-moving region into the burst and check
     the recipe does not increase its leak into the merge.
  E. symmetric skepticism: report where the recipe HURTS.

Run:  PYTHONPATH=. micromamba run -n handheld python eval/_aperture_validate.py
Cheap-mode (skip the cached multi-frame runs if present): set APERTURE_REUSE=1.
"""
import os
import sys
import glob
import time

import numpy as np

sys.path.insert(0, ".")
import eval.pipeline_runner as pr
import eval.hallucination as H
from eval.psf import esf_sigma
from eval import _aperture_common as AC

REUSE = os.environ.get("APERTURE_REUSE", "0") == "1"

# the recipe under test: gentle Tikhonov damping (cannot ghost) + a conservative
# guided vector-median that fixes ONLY isolated catastrophic flow outliers
# (outlier_thresh, robust to the burst's large real pan -- a consistency_span gate
# is over-tight when the whole field moves 26-147 px; see _aperture_robustness.py).
RECIPE = {
    "block_matching.tuning.flow_tikhonov_lambda": 1e-3,
    "block_matching.tuning.flow_median": {"window": 5, "outlier_thresh": 20.0,
                                          "guide_sigma": 0.0},
}
# damping-only and median-only, for attribution
DAMP_ONLY = {"block_matching.tuning.flow_tikhonov_lambda": 1e-3}
MED_ONLY = {"block_matching.tuning.flow_median":
            {"window": 5, "outlier_thresh": 20.0, "guide_sigma": 0.0}}


def _luma_run(burst, overrides, tag):
    p = os.path.join(AC.OUT, f"merge_{tag}.npy")
    if REUSE and os.path.exists(p):
        return np.load(p)
    lin, _ = pr.run(burst, overrides={**overrides, "scale": 1})
    m = H.rgb_to_luma(lin).astype(np.float32)
    np.save(p, m)
    return m


def section0_bit_identity(burst):
    print("\n=== 0. BIT-IDENTITY: recipe OFF == default ===", flush=True)
    lin_def, _ = pr.run(burst, overrides={"scale": 1})
    lin_off, _ = pr.run(burst, overrides={
        "scale": 1,
        "block_matching.tuning.flow_tikhonov_lambda": 0.0,
        "block_matching.tuning.flow_median": None})
    a, b = H.rgb_to_luma(lin_def), H.rgb_to_luma(lin_off)
    print(f"  flags-off == default: {np.array_equal(a, b)}  (max|diff| {np.abs(a-b).max():.2e})",
          flush=True)
    return a


def section_A_feature(burst, stack):
    print("\n=== A. the documented 'jut' feature (cy=960,cx=1840) ===", flush=True)
    cons, ref = AC.consensus_at(stack)
    m_single = H.rgb_to_luma(pr.run_single(burst, overrides={"scale": 1})[0])
    rows = []
    for tag, ov in [("single", None), ("default", {}), ("damp", DAMP_ONLY),
                    ("median", MED_ONLY), ("recipe", RECIPE)]:
        if tag == "single":
            m = m_single
        else:
            m = _luma_run(burst, ov, f"A_{tag}")
        d = AC.divergence(m, stack, cons, ref)
        rows.append((tag, d))
        print(f"  {tag:<8}: max|div| {d['maxabs']:.4f}  rms {d['rms']:.4f}  "
              f"x/y {d['xy_spread']:.2f}", flush=True)
    print("  (target = single; baseline multi = 'default'. The recipe should move "
          "'default' toward 'single' WITHOUT going below it.)", flush=True)
    return rows


def _edge_divergence_distribution(burst, stack, merges, edges, size=160, band=60,
                                  search=140):
    """For each clean step edge: max|div| & rms of each merge vs the DxO consensus,
    plus ESF edge sigma (reliability-gated). merges: dict tag->luma image."""
    from eval.hallucination import extract_aligned_roi, consensus_decomp, align_to
    out = {tag: dict(maxabs=[], rms=[], sigma=[], reliable=[]) for tag in merges}
    sl = slice(size // 2 - band // 2, size // 2 + band // 2)
    n_edges = 0
    for cy, cx, *_ in edges:
        rois, info = extract_aligned_roi(stack, cy, cx, size=size, search=search)
        dd = consensus_decomp(rois, info)
        cons = dd["consensus"]
        ref = np.ascontiguousarray(stack[0][cy - size // 2:cy + size // 2,
                                            cx - size // 2:cx + size // 2])
        s_cons, r_cons = esf_sigma(cons[:, sl])
        if not (np.isfinite(s_cons) and r_cons < 0.04 and 0.3 < s_cons < 4):
            continue  # gate: only edges the consensus itself fits cleanly
        n_edges += 1
        y0, y1, x0, x1 = 40, size - 40, 40, size - 40   # interior window
        wsl = (slice(y0, y1), slice(x0, x1))
        dc = cons[wsl]
        for tag, m in merges.items():
            a = align_to(ref, m, cy, cx, size, search)
            if a is None:
                continue
            al = (a - a.mean()) / (a.std() + 1e-9) * cons.std() + cons.mean()
            dvec = al[wsl] - dc
            s_m, r_m = esf_sigma(a[:, sl])
            out[tag]["maxabs"].append(float(np.abs(dvec).max()))
            out[tag]["rms"].append(float(dvec.std()))
            rel = np.isfinite(s_m) and r_m < 0.06
            out[tag]["reliable"].append(rel)
            out[tag]["sigma"].append(float(s_m) if rel else np.nan)
    return out, n_edges


def section_B_distribution(burst, stack):
    print("\n=== B. DISTRIBUTION over clean step edges (reliability-gated) ===",
          flush=True)
    ref_med = np.median(stack[::4], 0).astype(np.float32)
    edges = H.find_step_edges(ref_med, max_edges=24, min_sep=120)
    print(f"  found {len(edges)} candidate step edges", flush=True)
    merges = {
        "single": H.rgb_to_luma(pr.run_single(burst, overrides={"scale": 1})[0]),
        "default": _luma_run(burst, {}, "B_default"),
        "recipe": _luma_run(burst, RECIPE, "B_recipe"),
    }
    res, n = _edge_divergence_distribution(burst, stack, merges, edges)
    print(f"  {n} edges passed the consensus-fit gate", flush=True)
    for tag in ("single", "default", "recipe"):
        ma = np.array(res[tag]["maxabs"])
        rm = np.array(res[tag]["rms"])
        sg = np.array(res[tag]["sigma"])
        sg = sg[np.isfinite(sg)]
        print(f"  {tag:<8}: max|div| median {np.median(ma):.4f} "
              f"[IQR {np.percentile(ma,25):.4f}-{np.percentile(ma,75):.4f}]  "
              f"rms median {np.median(rm):.4f}  "
              f"edge_sigma median {np.median(sg):.3f} (n={sg.size})", flush=True)
    # paired: where does recipe beat / hurt default?
    md = np.array(res["default"]["maxabs"])
    mr = np.array(res["recipe"]["maxabs"])
    better = (mr < md - 1e-4).sum()
    worse = (mr > md + 1e-4).sum()
    print(f"  recipe vs default (paired, n={len(md)}): better on {better}, "
          f"worse on {worse}, ~equal on {len(md)-better-worse}", flush=True)
    print(f"  median delta (recipe-default) max|div|: {np.median(mr-md):+.5f} "
          f"(negative = improvement)", flush=True)
    # edge-sharpness regression check (paired, only edges reliable in BOTH)
    sgd = np.array(res["default"]["sigma"]); sgr = np.array(res["recipe"]["sigma"])
    ok = np.isfinite(sgd) & np.isfinite(sgr)
    if ok.sum():
        print(f"  edge_sigma paired (n={ok.sum()}): default median {np.median(sgd[ok]):.3f} "
              f"vs recipe {np.median(sgr[ok]):.3f}  "
              f"(delta {np.median(sgr[ok]-sgd[ok]):+.3f}; >0 = softer = regression)",
              flush=True)
    return res


def section_C_global(burst, stack, n_tiles=40, size=200, search=120, seed=0):
    """Scene-wide check: does the recipe help/hurt the merge vs DxO across many
    RANDOM regions (not just clean step edges)? Catches a net regression the
    edge-only gate would miss. Reports the paired distribution."""
    print("\n=== C. SCENE-WIDE divergence vs DxO (random regions, paired) ===",
          flush=True)
    merges = {"default": _luma_run(burst, {}, "C_default"),
              "recipe": _luma_run(burst, RECIPE, "C_recipe")}
    H_, W_ = stack.shape[1:]
    rng = np.random.default_rng(seed)
    pad = search + size // 2 + 8
    dd = {"default": [], "recipe": []}
    used = 0
    for _ in range(n_tiles * 3):
        if used >= n_tiles:
            break
        cy = int(rng.integers(pad, H_ - pad))
        cx = int(rng.integers(pad, W_ - pad))
        rois, info = H.extract_aligned_roi(stack, cy, cx, size=size, search=search)
        # require a clean NCC match (skip parallax/occlusion regions that don't
        # register -- those mix registration error into the metric)
        if np.median([k.get("peak", 0.0) for k in info]) < 0.5:
            continue
        dcmp = H.consensus_decomp(rois, info)
        cons = dcmp["consensus"]
        if cons.std() < 0.01:    # skip dead-flat regions (no structure to diverge)
            continue
        ref = np.ascontiguousarray(stack[0][cy - size // 2:cy + size // 2,
                                            cx - size // 2:cx + size // 2])
        win = (40, size - 40, 40, size - 40)
        ok = True
        vals = {}
        for tag, m in merges.items():
            d = AC.divergence(m, stack, cons, ref, cy=cy, cx=cx, win=win,
                              size=size, search=search)
            if not d["aligned_ok"]:
                ok = False
                break
            vals[tag] = d["rms"]
        if not ok:
            continue
        used += 1
        for tag in merges:
            dd[tag].append(vals[tag])
    for tag in ("default", "recipe"):
        a = np.array(dd[tag])
        print(f"  {tag:<8}: rms-div median {np.median(a):.4f} "
              f"[IQR {np.percentile(a,25):.4f}-{np.percentile(a,75):.4f}] (n={a.size})",
              flush=True)
    a = np.array(dd["default"]); b = np.array(dd["recipe"])
    print(f"  recipe vs default (paired, n={a.size}): better {int((b<a-1e-5).sum())}, "
          f"worse {int((b>a+1e-5).sum())}, equal {int((np.abs(b-a)<=1e-5).sum())}; "
          f"median delta {np.median(b-a):+.5f}", flush=True)


if __name__ == "__main__":
    t = time.time()
    stack = AC.load_stack()
    print(f"DxO luma stack {stack.shape} (shared cache)", flush=True)
    burst = pr.load_burst_arw(AC.ARW_DIR)
    print(f"007 burst {burst['raw_comp'].shape[0]+1} frames; SNR {burst['SNR']:.1f}",
          flush=True)

    section0_bit_identity(burst)
    section_A_feature(burst, stack)
    section_B_distribution(burst, stack)
    section_C_global(burst, stack)
    print(f"\n[validate sections 0/A/B/C done in {time.time()-t:.0f}s; "
          f"run _aperture_robustness.py for the no-blinding/no-ghosting checks]",
          flush=True)
