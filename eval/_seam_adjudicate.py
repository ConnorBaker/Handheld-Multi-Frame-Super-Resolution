"""Adjudicate the 'horizontal tile artifact' on the EXACT crop the user saw
(validate_deconv_cubic.png region), with the decisive control: DxO has NO tiles.
If DxO shows the same horizontal structure at the same grid-fold ratio as ours,
the period is coincidental real content (louvers ~20px ~ tile 16px), not tiling.

Renders ours-raw | ours+deconv | DxO consensus, shared contrast + cubic, 16px tile
grid overlaid on ours; prints the validated grid-fold ratio per panel.
"""
import numpy as np, glob, cv2, torch
from eval import hallucination as H, deconv as D, calibrate_deconv as CAL
from eval.psf import esf_sigma

OUT = "eval/halluc_out_007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')
m0 = np.load(f"{OUT}/merge_s1_luma.npy")
sig = CAL.auto_merge_sigma(m0)[0] or 2.0
ds = D.partial_gaussian_sigma(sig, 0.7 * sig)
dec = D.deblur_gated(torch.as_tensor(m0, device='cuda'), ds, iters=5, denoise='bilateral').clamp(0, 1).cpu().numpy()
print(f"scale=1 merge {m0.shape}, auto sigma {sig:.2f}, deconv done")

# the same edge set _validate_deconv_rejection used, same middle pick
edges = H.find_step_edges(np.median(STACK[::4], 0).astype(np.float32))
refs = []
for cy, cx, _, _ in edges:
    rois, info = H.extract_aligned_roi(STACK, cy, cx, size=160, search=140)
    s, r = esf_sigma(H.consensus_decomp(rois, info)['consensus'][:, 50:110])
    if np.isfinite(s) and r < 0.04 and 0.5 < s < 6:
        refs.append((cy, cx))
cy, cx = refs[len(refs) // 2]
print(f"user's crop center ~ (cy={cy}, cx={cx}); {len(refs)} reference edges")

ref = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])
cons = H.consensus_decomp(*H.extract_aligned_roi(STACK, cy, cx, size=160, search=140))['consensus']
our_raw = H.align_to(ref, m0, cy, cx, 160, 140)
our_dec = H.align_to(ref, dec, cy, cx, 160, 140)


def fold_ratio(region, period=16, axis=0, ctrl=(11, 13, 23, 27, 29)):
    prof = region.mean(axis=1 - axis).astype(np.float64)
    def fa(p):
        n = (prof.size // p) * p
        return np.ptp(prof[:n].reshape(-1, p).mean(0)) if n >= 3 * p else np.nan
    return fa(period) / (np.nanmean([fa(p) for p in ctrl]) + 1e-12)


print("\n=== horizontal grid-fold ratio at the 16px tile period (DxO has NO tiles) ===")
for a, nm in ((our_raw, "ours raw"), (our_dec, "ours+deconv"), (cons, "DxO consensus")):
    an = (a - a.mean()) / (a.std() + 1e-9)
    print(f"  {nm:<16} ratio {fold_ratio(an):.2f}   (>>1 only if structure locks to a 16px grid)")
print("  -> if DxO ~ ours, the horizontal bars are louver CONTENT (no tiles in DxO), not a seam")

# render: shared contrast, cubic 3x; overlay 16px grid (red) on the two 'ours' panels
lo, hi = np.percentile(cons, 2), np.percentile(cons, 98)
def panel(a, lab, grid):
    an = (a - a.mean()) / (a.std() + 1e-9) * cons.std() + cons.mean()
    v = (np.clip((an - lo) / (hi - lo), 0, 1) ** (1/2.2) * 255).astype(np.uint8)
    u = cv2.cvtColor(cv2.resize(v, (160*3, 160*3), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    if grid:
        for k in range(160 // 16 + 1):
            yy = int((16 * k) * 3)
            if 0 <= yy < u.shape[0]:
                u[yy:yy+1, :, 2] = 255
    cv2.rectangle(u, (0, 0), (u.shape[1]-1, 18), (0, 0, 0), -1)
    cv2.putText(u, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return u
g = np.full((480, 6, 3), 80, np.uint8)
cv2.imwrite(f"{OUT}/seam_adjudicate.png", np.hstack([
    panel(our_raw, "ours raw +16px grid", True), g,
    panel(our_dec, "ours+deconv +16px grid", True), g,
    panel(cons, "DxO (no tiles)", False)]))
print(f"\nsaved {OUT}/seam_adjudicate.png")
