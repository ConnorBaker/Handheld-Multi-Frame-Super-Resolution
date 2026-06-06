"""Sweep measured-PSF deconvolutions of the merge to match DxO's edge sharpness.

Measures, per setting, the median edge sigma over the gate-passing step edges, the
edge overshoot (halo), and the flat-region noise -- so we pick the setting that
reaches DxO's edge sigma WITHOUT haloing or amplifying noise. Deconvolution is done
in the gamma-luma space the edges were measured in (and that we'd ship as a display
postproc), targeting the measured merge PSF (Gaussian sigma~2.1)."""
import numpy as np, glob, torch, cv2
from eval import hallucination as H
from eval.psf import esf_sigma
from eval import deconv as D

stack = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')),
                          cache='eval/halluc_out_007/luma_stack.npy')
merge_g = np.load('eval/halluc_out_007/merge_linear_luma.npy')           # gamma luma
mt = torch.as_tensor(merge_g, device='cuda')
RELIABLE = [(640, 4000), (2160, 4080), (1920, 4240)]
SL = slice(50, 110); bandf = lambda im: im[:, SL]
MERGE_SIGMA, DXO_SIGMA = 2.1, 1.87

# reference rois (DxO ref roi for aligning the merge; DxO consensus sigma/overshoot)
refrois = {}; dxo_sig = []; dxo_over = []
def overshoot(b):
    col = b.mean(0); lo, hi = np.median(col[:6]), np.median(col[-6:]); a, c = min(lo, hi), max(lo, hi)
    return float((max(0, col.max() - c) + max(0, a - col.min())) / (abs(c - a) + 1e-9))
for cy, cx in RELIABLE:
    rois, info = H.extract_aligned_roi(stack, cy, cx, size=160, search=140); d = H.consensus_decomp(rois, info)
    refrois[(cy, cx)] = rois[0]
    s, _ = esf_sigma(bandf(d['consensus'])); dxo_sig.append(s); dxo_over.append(overshoot(bandf(d['consensus'])))

# flat sky crops (on each grid) for noise: high-pass std
def flat_noise(img2d, y, x):
    c = img2d[y:y+200, x:x+200].astype(np.float32)
    return float((c - cv2.GaussianBlur(c, (0, 0), 2.0)).std())
dxo_noise = flat_noise(stack[0], 150, 2600)
merge_noise0 = flat_noise(merge_g, 150, 2600)

def measure(img2d):
    sg, ov = [], []
    for cy, cx in RELIABLE:
        a = H.align_to(refrois[(cy, cx)], img2d, cy, cx, 160, 140)
        s, _ = esf_sigma(bandf(a)); sg.append(s); ov.append(overshoot(bandf(a)))
    return np.median(sg), np.median(ov), flat_noise(img2d, 150, 2600)

print(f"{'method':<26}{'edge sigma':>11}{'overshoot':>11}{'flat noise':>12}")
print(f"{'DxO consensus (target)':<26}{np.median(dxo_sig):>11.2f}{np.median(dxo_over):>11.3f}{dxo_noise:>12.4f}")
print(f"{'our merge (raw)':<26}{MERGE_SIGMA:>11.2f}{'-':>11}{merge_noise0:>12.4f}")
print('-' * 60)

sig_partial = D.partial_gaussian_sigma(MERGE_SIGMA, DXO_SIGMA)   # ~0.955 -> targets 2.1->1.87
configs = []
for nsr in (3e-2, 1e-2, 3e-3):
    configs.append((f"Wiener full(s2.1) nsr={nsr:g}", lambda nsr=nsr: D.wiener_fft(mt, D.gaussian_psf(MERGE_SIGMA), nsr)))
for nsr in (1e-2, 3e-3, 1e-3):
    configs.append((f"Wiener partial(s{sig_partial:.2f}) nsr={nsr:g}",
                    lambda nsr=nsr: D.wiener_fft(mt, D.gaussian_psf(sig_partial), nsr)))
for it in (2, 4, 8):
    configs.append((f"RL full(s2.1) iters={it}", lambda it=it: D.richardson_lucy_fft(mt, D.gaussian_psf(MERGE_SIGMA), it)))
for it in (5, 15, 40):
    configs.append((f"RL partial(s{sig_partial:.2f}) it={it}", lambda it=it: D.richardson_lucy_fft(mt, D.gaussian_psf(sig_partial), it)))

best = None
for name, fn in configs:
    out = fn().clamp(0, 1).cpu().numpy()
    s, o, n = measure(out)
    flag = ''
    if abs(s - DXO_SIGMA) < 0.08:
        flag = '  <- matches DxO sigma'
        if best is None or o < best[2]:
            best = (name, s, o, n, out)
    print(f"{name:<26}{s:>11.2f}{o:>11.3f}{n:>12.4f}{flag}")

if best:
    name, s, o, n, out = best
    cv2.imwrite('eval/halluc_out_007/merge_deconv.png',
                (np.clip(out[1700:2200, 4000:4500], 0, 1) * 255).astype(np.uint8))
    print(f"\nBEST match: {name} -> sigma {s:.2f} (DxO {DXO_SIGMA}), overshoot {o:.3f} (DxO {np.median(dxo_over):.3f}), "
          f"noise {n:.4f} (DxO {dxo_noise:.4f}, raw merge {merge_noise0:.4f})")
