"""Validate the two surviving HF levers against the DxO consensus (the cross-frame-
REAL reference), on the 007 burst:
  (A) measured-PSF gated deconvolution -- can it reach DxO's edge sigma from REAL
      detail, at <= DxO's overshoot + acceptable noise?
  (B) robustness rejection (robustness.t up) -- does rejecting the unalignable tail
      preserve/recover texture HF vs the default?
Matched edges, reliability-gated, deslanted overshoot -- no convenient samples."""
import numpy as np, glob, torch, cv2, time
from eval import hallucination as H, deconv as D, calibrate_deconv as CAL, pipeline_runner as pr
from eval.psf import esf_sigma, _erf_model
from scipy.ndimage import gaussian_filter1d

OUT = "eval/halluc_out_007"
ARW = "/home/connorbaker/Pictures/ARW/2025-09-06/007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')

burst = pr.load_burst_arw(ARW)
merges = {}
for t in (0.12, 0.30):                       # default vs aggressive rejection (fidelity-sweep found ~0.31)
    lin, _ = pr.run(burst, overrides={"scale": 1, "robustness.tuning.t": t})
    merges[t] = H.rgb_to_luma(lin)
    print(f"merge robustness.t={t} done", flush=True)
m0 = merges[0.12]
sig = CAL.auto_merge_sigma(m0)[0] or 2.0
print(f"auto merge sigma = {sig:.2f}")


def gd(luma, target_frac, iters=5):
    ds = D.partial_gaussian_sigma(sig, target_frac * sig)
    return D.deblur_gated(torch.as_tensor(luma, device='cuda'), ds, iters=iters, denoise='bilateral').clamp(0, 1).cpu().numpy()


def deslant_overshoot(roi, band=slice(40, 120)):
    g = np.abs(np.gradient(gaussian_filter1d(roi.astype(np.float64), 2.0, axis=0), axis=1)) ** 2
    sub = roi[:, band]; gg = g[:, band]; cen = (gg * np.arange(sub.shape[1])).sum(1) / (gg.sum(1) + 1e-9)
    prof = np.array([np.interp(np.arange(sub.shape[1]), np.arange(sub.shape[1]) - (cen[i] - sub.shape[1] / 2), sub[i]) for i in range(sub.shape[0])]).mean(0)
    b = np.median(np.sort(prof)[-8:]); d = np.median(np.sort(prof)[:8]); return max(0, prof.max() - b) / (b - d + 1e-9)


# reference edges + DxO consensus sigma
edges = H.find_step_edges(np.median(STACK[::4], 0).astype(np.float32))
refs, dxo_sig = {}, []
for cy, cx, _, _ in edges:
    ref = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])
    rois, info = H.extract_aligned_roi(STACK, cy, cx, size=160, search=140)
    s, r = esf_sigma(H.consensus_decomp(rois, info)['consensus'][:, 50:110])
    if np.isfinite(s) and r < 0.04 and 0.5 < s < 6:
        refs[(cy, cx)] = ref; dxo_sig.append(s)
DXO = np.median(dxo_sig)
print(f"\nDxO consensus edge sigma (median, n={len(dxo_sig)}) = {DXO:.2f}px  [the target]")


def matched_sigma(img):
    return np.median([esf_sigma(H.align_to(refs[k], img, k[0], k[1], 160, 140)[:, 50:110])[0] for k in refs])
def flat_noise(img, y=200, x=2600, s=200):
    c = img[y:y + s, x:x + s].astype(np.float32); return float((c - cv2.GaussianBlur(c, (0, 0), 2.0)).std())
def overshoot(img):
    return float(np.median([deslant_overshoot(H.align_to(refs[k], img, k[0], k[1], 160, 140)) for k in refs]))

# (A) deconv sweep: reach DxO sigma from the (default-rejection) merge
print("\n=== (A) DECONVOLUTION: reach DxO edge sigma from real detail ===")
print(f"{'variant':<22}{'edge sigma':>11}{'overshoot':>11}{'flat noise':>12}")
print(f"{'DxO consensus':<22}{DXO:>11.2f}{overshoot(STACK[0]):>11.3f}{flat_noise(STACK[0]):>12.4f}")
print(f"{'merge (no deconv)':<22}{matched_sigma(m0):>11.2f}{overshoot(m0):>11.3f}{flat_noise(m0):>12.4f}")
best = None
for tf in (0.75, 0.65, 0.55):
    out = gd(m0, tf)
    s, o, n = matched_sigma(out), overshoot(out), flat_noise(out)
    flag = '  <- ~DxO' if abs(s - DXO) < 0.12 else ''
    print(f"{'merge+deconv tf=' + str(tf):<22}{s:>11.2f}{o:>11.3f}{n:>12.4f}{flag}")
    if best is None or abs(s - DXO) < abs(best[1] - DXO):
        best = (out, s, o, n, tf)

# (B) rejection: texture HF + edge sigma, t=0.12 vs t=0.30
print("\n=== (B) ROBUSTNESS REJECTION (t 0.12 vs 0.30): texture HF + edges ===")
def tex_hf(img, y=1500, x=3850, s=300):   # a textured facade region
    c = img[y:y + s, x:x + s].astype(np.float32); return float((c - cv2.GaussianBlur(c, (0, 0), 1.6)).std())
for t in (0.12, 0.30):
    print(f"  t={t}: edge sigma {matched_sigma(merges[t]):.2f}  texture-HF {tex_hf(merges[t]):.4f}  flat-noise {flat_noise(merges[t]):.4f}")
print("  (rejection helps most on misaligned/foliage tiles; this facade is largely rigid -> expect a small effect here)")

# verdict + fair visual on a column edge
out, s, o, n = best[0], best[1], best[2], best[3]
print(f"\nVERDICT: merge+deconv reaches edge sigma {s:.2f} (DxO {DXO:.2f}) at overshoot {o:.3f} (DxO {overshoot(STACK[0]):.3f}), "
      f"flat-noise {n:.4f} (DxO {flat_noise(STACK[0]):.4f})")
cy, cx = list(refs)[len(refs) // 2]
def panel(a, lab, lo, hi):
    u = cv2.cvtColor((np.clip((H.align_to(refs[(cy, cx)], a, cy, cx, 160, 140) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    u = cv2.resize(u, (160 * 3, 160 * 3), interpolation=cv2.INTER_NEAREST); cv2.rectangle(u, (0, 0), (u.shape[1] - 1, 18), (0, 0, 0), -1); cv2.putText(u, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return u
cons = H.consensus_decomp(*H.extract_aligned_roi(STACK, cy, cx, size=160, search=140))['consensus']
lo, hi = np.percentile(cons, 1), np.percentile(cons, 99)
oa = (out - out.mean()) / (out.std() + 1e-9) * cons.std() + cons.mean()
cv2.imwrite(f'{OUT}/validate_deconv.png', np.hstack([panel(m0, 'merge raw', lo, hi), np.full((480, 6, 3), 80, np.uint8), panel(oa, f'merge+deconv s{s:.2f}', lo, hi), np.full((480, 6, 3), 80, np.uint8), cv2.resize(cv2.cvtColor((np.clip((cons - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (480, 480), interpolation=cv2.INTER_NEAREST)]))
print(f"saved {OUT}/validate_deconv.png")
