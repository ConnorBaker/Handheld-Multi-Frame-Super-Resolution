"""Shared helpers for the PARK scene (DSC00034-DSC00074) field-PSF re-test.

This scene reaches the BOTTOM field edge (foreground grass fills both bottom
corners, r_norm -> 1.0), which the 007 oblique facade could not (edges only to
~0.55-0.80 r_norm). It lets us re-ask the OPEN question from 007: is the
merge->DxO deconv GAP field-dependent? The decisive quantity is the GAP
(merge_sigma - DxO_sigma) vs field radius -- what a field-varying deconv would
have to vary -- not raw sigma (which is confounded by content/depth).

Everything here is read-only w.r.t. the package + configs. Heavy artifacts
(DxO luma stack, multi-frame merge luma) are cached under halluc_out_034/
(gitignored). Run nothing in this module directly; it is imported by the
_psf_field_park_* scripts.

Reuses, unmodified:
  * eval.calibrate_optics.read_dng / eval.hallucination.read_luma,load_luma_stack
    -- DxO linear-DNG -> gamma-luma (same space the 007 work used).
  * eval.pipeline_runner.load_burst_arw / run / run_single -- the merge (lens MTF
    still in, NO DxO field correction), exactly the reconstruction we compare.
  * eval.hallucination.rgb_to_luma -- merge RGB -> the same gamma-luma space.
  * eval.psf.esf_sigma + eval._psf_field_measure (edge finder, field map, bins).

VERIFY-EVERYTHING note (repo CLAUDE.md S1): nothing here is trusted from prose;
the _psf_field_park_validate driver re-derives every number on this GB10.
"""

import os
import glob
import time

import numpy as np

SCENE = "/home/connorbaker/Pictures/DSC00034-DSC00074"
OUT = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_034"

DXO_STACK = f"{OUT}/luma_stack.npy"          # (N,4000,6000) gamma-luma DxO-reference
MERGE_MULTI = f"{OUT}/_merge_multi_luma.npy"  # multi-frame merge gamma-luma
MERGE_SINGLE = f"{OUT}/_merge_single_luma.npy"  # single-frame baseline merge

os.makedirs(OUT, exist_ok=True)


def dxo_paths():
    """Sorted per-frame DxO-reference DNGs (the *_DxO-reference.dng, one per ARW)."""
    ps = sorted(glob.glob(f"{SCENE}/DSC*_DxO-reference.dng"))
    assert ps, f"no *_DxO-reference.dng in {SCENE}"
    return ps


def ensure_dxo_stack():
    """(N,H,W) float32 gamma-luma DxO stack, shared-WB, memmap-cached."""
    import eval.hallucination as H
    return H.load_luma_stack(dxo_paths(), cache=DXO_STACK, wb="shared")


def ensure_merge(kind="multi", n_frames=None, calm_frames=None):
    """Fresh pipeline merge -> gamma-luma (lens MTF still in, NO DxO correction).

    kind="multi": full multi-frame merge (robustness mask on -> moving grass is
    down-weighted, the wind mitigation). kind="single": reference-frame-only
    baseline (no temporal averaging at all -> a wind-free control: any grass gap
    here is NOT motion blur). calm_frames: optional explicit ARW indices to use
    (calmest subset); overrides n_frames. Cached per kind.
    """
    import eval.pipeline_runner as pr
    import eval.hallucination as H
    cache = MERGE_MULTI if kind == "multi" else MERGE_SINGLE
    tag = kind if calm_frames is None else f"{kind}_calm{len(calm_frames)}"
    cache = cache.replace(".npy", f"_{tag}.npy") if calm_frames is not None else cache
    if os.path.exists(cache):
        return np.load(cache)
    t = time.time()
    burst = pr.load_burst_arw(SCENE)
    if calm_frames is not None:
        # reorder so calmest frame is reference, then its calm neighbours as comp
        idx = list(calm_frames)
        allf = [burst["ref_raw"]] + list(burst["raw_comp"])
        burst = dict(burst)
        burst["ref_raw"] = allf[idx[0]]
        burst["raw_comp"] = np.array([allf[i] for i in idx[1:]]) if len(idx) > 1 else np.empty((0,)+allf[0].shape, np.float32)
    if kind == "single":
        rgb, _ = pr.run_single(burst, overrides={"scale": 1})
    else:
        rgb, _ = pr.run(burst, overrides={"scale": 1}, n_frames=n_frames)
    m = H.rgb_to_luma(rgb).astype(np.float32)
    np.save(cache, m)
    print(f"[ensure_merge {tag}] {m.shape} built in {time.time()-t:.0f}s -> {cache}")
    return m


# ---------------------------------------------------------------------------
# Robust global merge<->DxO registration, VERIFIED across the field.
# The merge grid (~4024x6048) is slightly larger than DxO (4000x6000); the 007
# work used a single central NCC translation. This scene is a near-static
# landscape (small handheld jitter, no big pan) so a global translation should
# suffice -- but we VERIFY it by checking the offset agrees in widely separated
# field windows (TL/TR/BL/BR/center) before trusting one number anywhere.
# ---------------------------------------------------------------------------

def merge_dxo_offset(dxo, mg, ph=900, verify=True):
    """(oy, ox) integer offset s.t. dxo[y,x] ~ mg[y+oy, x+ox], from a central NCC.

    If verify, also NCC five field windows and return per-window offsets so the
    caller can confirm a single translation is adequate (max corner-vs-center
    disagreement small) -- a field-varying offset would otherwise masquerade as a
    field-varying gap.
    """
    import cv2
    H, W = dxo.shape

    def _off(cy, cx, half):
        tmpl = np.ascontiguousarray(dxo[cy - half:cy + half, cx - half:cx + half])
        pad = 48
        win = np.ascontiguousarray(mg[cy - half - pad:cy + half + pad,
                                      cx - half - pad:cx + half + pad])
        if win.shape[0] < 2 * half + 2 * pad or win.shape[1] < 2 * half + 2 * pad:
            return None
        cc = cv2.matchTemplate(win, tmpl, cv2.TM_CCORR_NORMED)
        _, pk, _, loc = cv2.minMaxLoc(cc)
        return (loc[1] - pad, loc[0] - pad, float(pk))

    cy0, cx0 = H // 2, W // 2
    main = _off(cy0, cx0, ph // 2)
    oy, ox, pk = main
    field = {}
    if verify:
        m = 250 + ph // 2
        wins = dict(TL=(m, m), TR=(m, W - m), BL=(H - m, m), BR=(H - m, W - m),
                    C=(cy0, cx0))
        for nm, (cy, cx) in wins.items():
            r = _off(cy, cx, min(ph, 700) // 2)
            if r is not None:
                field[nm] = (r[0], r[1], r[2])
    return oy, ox, pk, field
