"""Investigate the LOCALIZED 'sharp edge jutting right from the blind' the user sees
just below the 3rd red line from the bottom (~zoom y=336 -> crop y~112 -> abs y~992)
in the seam_adjudicate region (scale-1 merge crop centered cy=960,cx=1840).

Decisive control: DxO has no tiles. If the jut is in DxO too -> real content. If it
is ONLY in ours (and localized to a tile) -> bad tile fusion. Render ours-raw |
ours+deconv | DxO at high zoom over the bottom band, plus (ours+deconv - DxO)
difference to localize divergence; report the worst-divergence pixel + its tile.
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

edges = H.find_step_edges(np.median(STACK[::4], 0).astype(np.float32))
refs = []
for cy, cx, _, _ in edges:
    rois, info = H.extract_aligned_roi(STACK, cy, cx, size=160, search=140)
    s, r = esf_sigma(H.consensus_decomp(rois, info)['consensus'][:, 50:110])
    if np.isfinite(s) and r < 0.04 and 0.5 < s < 6:
        refs.append((cy, cx))
cy, cx = refs[len(refs) // 2]
ref = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])
cons = H.consensus_decomp(*H.extract_aligned_roi(STACK, cy, cx, size=160, search=140))['consensus']
our_raw = H.align_to(ref, m0, cy, cx, 160, 140)
our_dec = H.align_to(ref, dec, cy, cx, 160, 140)
print(f"crop center (cy={cy},cx={cx}); 160x160 crop spans abs y[{cy-80},{cy+80}) x[{cx-80},{cx+80})")

# TIGHT window around the feature (worst divergence was crop y=113,x=77)
y0, y1, x0, x1 = 98, 130, 50, 105
band = slice(y0, y1); xband = slice(x0, x1)
# level-match ours to DxO for a fair difference
def lm(a):
    return (a - a.mean()) / (a.std() + 1e-9) * cons.std() + cons.mean()
dr, dd, dc = lm(our_raw)[band, xband], lm(our_dec)[band, xband], cons[band, xband]
diff_raw = dr - dc      # merge vs DxO
diff_dec = dd - dc      # merge+deconv vs DxO
iy, ix = np.unravel_index(np.argmax(np.abs(diff_dec)), diff_dec.shape)
print(f"\nworst (ours+deconv - DxO): {diff_dec[iy,ix]:+.3f} at crop (y={y0+iy}, x={x0+ix})")
print(f"  same pixel: ours-RAW - DxO = {diff_raw[iy,ix]:+.3f}   (if ~same -> it's in the MERGE, not deconv)")
print(f"  RAW merge max|div| {np.abs(diff_raw).max():.3f}   DECONV max|div| {np.abs(diff_dec).max():.3f}")
# does the feature align to a tile boundary? worst-row profile across x
wr = np.abs(diff_dec).max(0)
print("\nmax|ours+deconv - DxO| per column (x in crop coords; tile borders at abs x%16==0):")
for i, x in enumerate(range(x0, x1)):
    bar = "#" * int(wr[i] * 200)
    bnd = " <-TILE BORDER" if (cx - 80 + x) % 16 == 0 else ""
    if wr[i] > 0.02 or bnd:
        print(f"  x={x:>3} (abs {cx-80+x}) {wr[i]:.3f} {bar}{bnd}")

lo, hi = np.percentile(dc, 2), np.percentile(dc, 98)
Z = 14
def panel(a, lab, grid=True, isdiff=False):
    if isdiff:
        v = np.clip(0.5 + 5 * a, 0, 1)
    else:
        v = np.clip((a - lo) / (hi - lo), 0, 1) ** (1 / 2.2)
    u = cv2.cvtColor((cv2.resize(v.astype(np.float32), (a.shape[1] * Z, a.shape[0] * Z), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if grid:
        for gy in range(y0, y1 + 1):
            if (cy - 80 + gy) % 16 == 0:
                yy = (gy - y0) * Z
                if 0 <= yy < u.shape[0]:
                    u[yy:yy+1, :, 2] = 255
        for gx in range(x0, x1 + 1):
            if (cx - 80 + gx) % 16 == 0:
                xx = (gx - x0) * Z
                if 0 <= xx < u.shape[1]:
                    u[:, xx:xx+1, 2] = 200
    cv2.rectangle(u, (0, 0), (u.shape[1]-1, 16), (0, 0, 0), -1)
    cv2.putText(u, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return u
g = np.full(((y1-y0) * Z, 5, 3), 80, np.uint8)
cv2.imwrite(f"{OUT}/jut_investigate.png", np.hstack([
    panel(dr, "ours raw"), g, panel(dd, "ours+deconv"), g,
    panel(dc, "DxO (no tiles)"), g, panel(diff_dec, "5x(deconv - DxO)", grid=False, isdiff=True)]))
print(f"\nsaved {OUT}/jut_investigate.png ({Z}x, red=16px tile grid)")
