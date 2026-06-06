"""Out-resolve DxO at <=1x? Downscale our de-aliased scale=2 merge and DxO's actual
single-frame render to 1.0/0.75/0.5x native, compare physical edge sigma (matched,
reliability-gated) + anti-aliasing on the near-Nyquist louvers.

Hypothesis: at downscaled output the multi-frame source is de-aliased (sub-pixel
fusion) while a single DxO frame carries baked-in aliasing/moiré; edge sigma may be
filter-limited (wash) but ours should be cleaner / more real near Nyquist."""
import numpy as np, glob, cv2
from eval import hallucination as H
from eval.psf import esf_sigma

OUT = "eval/halluc_out_007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')
ours_src = np.load(f'{OUT}/merge_s2_flow_bilinear.npy')   # 2x native, de-aliased, block-free
dxo_src = np.ascontiguousarray(STACK[0])                  # DxO's ACTUAL single-frame output, 1x native
print(f"ours_src(2x) {ours_src.shape}, dxo_src(1x) {dxo_src.shape}")

# FAIR: deconvolve our downscaled output (DxO's output is already deconvolved). Comparing
# our RAW merge to DxO's deconvolved render wrongly favours DxO; with our deconv it's parity.
import torch
from eval import deconv as D, calibrate_deconv as CAL
def _gated_deconv(luma):
    sig = CAL.auto_merge_sigma(luma)[0] or 2.0
    ds = D.partial_gaussian_sigma(sig, 0.7 * sig)
    return D.deblur_gated(torch.as_tensor(luma, device='cuda'), ds, iters=5, denoise='bilateral').clamp(0, 1).cpu().numpy()


def area_to(img, fy, fx):
    return cv2.resize(img, (int(round(img.shape[1] * fx)), int(round(img.shape[0] * fy))), interpolation=cv2.INTER_AREA)


def edge_sigmas(dxo_t, ours_t, t):
    """Matched physical edge sigma over edges found in the DxO-downscaled image."""
    edges = H.find_step_edges(dxo_t)
    dl, ol = [], []
    for cy, cx, _, _ in edges:
        if not (80 <= cy < dxo_t.shape[0] - 80 and 80 <= cx < dxo_t.shape[1] - 80):
            continue
        ref = np.ascontiguousarray(dxo_t[cy - 80:cy + 80, cx - 80:cx + 80])
        sd, rd = esf_sigma(ref[:, 50:110])
        a = H.align_to(ref, ours_t, cy, cx, 160, 60)
        so, ro = esf_sigma(a[:, 50:110])
        if np.isfinite(sd) and rd < 0.04 and np.isfinite(so) and ro < 0.05:
            dl.append(sd / t); ol.append(so / t)          # -> physical (sensor) px
    return np.array(dl), np.array(ol)


print("\n=== physical edge sigma (sensor px) vs output scale; matched, reliability-gated ===")
print(f"{'output':>8}{'DxO':>9}{'ours':>9}{'n':>4}  (ours<DxO => out-resolve)")
for t in (1.0, 0.75, 0.5):
    dxo_t = area_to(dxo_src, t, t)                        # 1x -> t (DxO output, already deconvolved)
    ours_t = _gated_deconv(area_to(ours_src, t / 2, t / 2))  # 2x -> t, then OUR deconv (fair)
    if t <= 0.5:
        print(f"{t:>7.2f}x   edge sigma filter-limited at <=0.5x (output edges ~1px; esf unreliable) -- skipped")
        continue
    d, o = edge_sigmas(dxo_t, ours_t, t)
    if d.size:
        print(f"{t:>7.2f}x{np.median(d):>9.2f}{np.median(o):>9.2f}{d.size:>4}  "
              f"{'OURS SHARPER' if np.median(o) < np.median(d) else 'DxO sharper'} (by {np.median(d)-np.median(o):+.2f}px)")
    else:
        print(f"{t:>7.2f}x   no reliably-matched edges")

# anti-aliasing demonstration at 0.5x on the near-Nyquist louver region
t = 0.5
dxo_h = area_to(dxo_src, t, t); ours_h = area_to(ours_src, t / 2, t / 2)
ly, lx = int(3088 * t), int(5392 * t)                    # louver tile (1x coords) -> 0.5x
W = 120
dl = dxo_h[ly - W:ly + W, lx - W:lx + W]
ref = np.ascontiguousarray(dl)
ol = H.align_to(ref, ours_h, ly, lx, 2 * W, 80)
# residual near-Nyquist energy (aliasing/moiré + grain) after a proper anti-alias prefilter:
def near_nyq(a):
    return float((a - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.2)).std())
print(f"\n0.5x louver region near-Nyquist energy (lower=cleaner/less aliasing+grain): "
      f"DxO {near_nyq(dl):.4f}  ours {near_nyq(ol):.4f}")
ol = (ol - ol.mean()) / (ol.std() + 1e-9) * dl.std() + dl.mean()
lo, hi = np.percentile(dl, 1), np.percentile(dl, 99)
def panel(a, tt):
    u = cv2.cvtColor((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    u = cv2.resize(u, (a.shape[1] * 3, a.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(u, (0, 0), (u.shape[1] - 1, 18), (0, 0, 0), -1); cv2.putText(u, tt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return u
cv2.imwrite(f'{OUT}/resolution_05x.png', np.hstack([panel(dl, 'DxO single-frame @0.5x'), np.full((W*6, 6, 3), 80, np.uint8), panel(ol, 'our multi-frame @0.5x')]))
print(f"saved {OUT}/resolution_05x.png")
