"""Beat DxO -- MATCHED physical edge resolution (same physical edges across all
variants; reliability-gated on the DxO fit), plus a fair 2x-output visual. Avoids
the non-matched-sample trap (eval/CLAUDE.md)."""
import numpy as np, glob, torch, cv2
from eval import hallucination as H, deconv as D, calibrate_deconv as CAL
from eval.psf import esf_sigma

OUT = "eval/halluc_out_007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')
s1 = np.load(f'{OUT}/merge_s1_luma.npy'); s2 = np.load(f'{OUT}/merge_s2_luma.npy')


def gd(luma, sigma, tf=0.7, it=5):
    ds = D.partial_gaussian_sigma(sigma, tf * sigma)
    return D.deblur_gated(torch.as_tensor(luma, device='cuda'), ds, iters=it, denoise='bilateral').clamp(0, 1).cpu().numpy()
sig1 = CAL.auto_merge_sigma(s1)[0] or 2.0; sig2 = CAL.auto_merge_sigma(s2)[0] or 4.0
s1d, s2d = gd(s1, sig1), gd(s2, sig2)

edges = H.find_step_edges(np.median(STACK[::4], 0).astype(np.float32))  # reference set in DxO 1x grid
print(f"=== MATCHED physical edge sigma (sensor px) over {len(edges)} DxO edges; gated on DxO fit rms<0.04 ===")
print(f"{'edge':>13}{'DxO':>8}{'s1':>8}{'s1+dc':>8}{'s2':>8}{'s2+dc':>8}")
rows = []
for cy, cx, _, _ in edges:
    ref = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])
    rois, info = H.extract_aligned_roi(STACK, cy, cx, size=160, search=140)
    sdxo, rdxo = esf_sigma(H.consensus_decomp(rois, info)['consensus'][:, 50:110])
    if not (np.isfinite(sdxo) and rdxo < 0.04 and 0.5 < sdxo < 6):
        continue
    a1 = esf_sigma(H.align_to(ref, s1, cy, cx, 160, 140)[:, 50:110])[0]
    a1d = esf_sigma(H.align_to(ref, s1d, cy, cx, 160, 140)[:, 50:110])[0]
    tmpl2 = cv2.resize(ref, (320, 320), interpolation=cv2.INTER_CUBIC)            # 1x template -> 2x
    a2 = esf_sigma(H.align_to(tmpl2, s2, 2 * cy, 2 * cx, 320, 280)[:, 100:220])[0] / 2   # -> physical
    a2d = esf_sigma(H.align_to(tmpl2, s2d, 2 * cy, 2 * cx, 320, 280)[:, 100:220])[0] / 2
    rows.append((sdxo, a1, a1d, a2, a2d))
    print(f"  ({cy:>4},{cx:>4}){sdxo:>8.2f}{a1:>8.2f}{a1d:>8.2f}{a2:>8.2f}{a2d:>8.2f}")
a = np.array(rows)
print(f"\n  MEDIAN (n={len(a)}):  DxO {np.median(a[:,0]):.2f}  s1 {np.median(a[:,1]):.2f}  "
      f"s1+dc {np.median(a[:,2]):.2f}  s2 {np.median(a[:,3]):.2f}  s2+dc {np.median(a[:,4]):.2f}")
beat1 = (a[:,2] < a[:,0]).mean()*100; beat2 = (a[:,4] < a[:,0]).mean()*100
print(f"  s1+deconv sharper than DxO on {beat1:.0f}% of matched edges; s2+deconv on {beat2:.0f}%")
print(f"  -> verdict: scale=2+deconv {'BEATS' if np.median(a[:,4])<np.median(a[:,0]) else 'does NOT beat'} DxO physical edge sigma (matched)")

# ---- fair 2x-output visual: DxO has no SR (bicubic up) vs our scale=2 SR ----
cy, cx = edges[np.argmax([e[3] for e in edges])][:2] if False else (1920, 4240)  # a clean column edge
ref = np.ascontiguousarray(STACK[0][cy - 64:cy + 64, cx - 64:cx + 64])
rois, info = H.extract_aligned_roi(STACK, cy, cx, size=128, search=140); cons = H.consensus_decomp(rois, info)['consensus']
dxo_up = cv2.resize(cons, (256, 256), interpolation=cv2.INTER_CUBIC)             # DxO at 2x output = bicubic (no SR)
tmpl2 = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_CUBIC)
s2c = H.align_to(tmpl2, s2d, 2 * cy, 2 * cx, 256, 280)                            # our scale=2 SR+deconv
s2c = (s2c - s2c.mean()) / (s2c.std() + 1e-9) * cons.std() + cons.mean()          # level-match
lo, hi = np.percentile(dxo_up, 1), np.percentile(dxo_up, 99)
def panel(a, t):
    u = cv2.cvtColor((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    u = cv2.resize(u, (a.shape[1] * 2, a.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(u, (0, 0), (u.shape[1] - 1, 18), (0, 0, 0), -1); cv2.putText(u, t, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return u
sheet = np.hstack([panel(dxo_up, 'DxO @2x output (bicubic, no SR)'), np.full((512, 6, 3), 80, np.uint8),
                   panel(s2c, 'our scale=2 SR + deconv')])
cv2.imwrite(f'{OUT}/beat_dxo_2x.png', sheet); print(f"\nsaved {OUT}/beat_dxo_2x.png (fair, shared contrast)")
