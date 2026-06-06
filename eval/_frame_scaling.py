"""Adversarial test of 'more frames -> cleaner not sharper' (which was measured at
scale=1, where there is NO finer grid). At scale=2, separate the two regimes vs
frame count N:
  - OPTICS-limited edges: matched physical edge sigma vs DxO. Expect FLAT in N
    (the lens attenuated those freqs in EVERY frame; fusion can't recover them).
  - SAMPLING-limited texture: SR-band FFT power in a fine-texture region. Expect
    RISE then SATURATE (sub-pixel fusion fills the finer grid, up to the
    sampling-ratio ceiling ~N=4).
  - FLAT region SR-band power: expect DROP ~1/sqrt(N) (that is just noise).
If edges are flat while texture rises-then-saturates and flat-noise drops, the
honest answer is: frames sharpen SAMPLING-limited detail (saturating), never
OPTICS-limited edges -- only deconvolution (up to the noise floor) touches those.
"""
import numpy as np, glob, cv2
from eval import pipeline_runner as pr, hallucination as H
from eval.psf import esf_sigma

OUT = "eval/halluc_out_007"
ARW = "/home/connorbaker/Pictures/ARW/2025-09-06/007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')
burst = pr.load_burst_arw(ARW)
edges = H.find_step_edges(np.median(STACK[::4], 0).astype(np.float32))

# DxO consensus edge sigma (the optics-limited target) over reliable edges
refs, dxo_sig = {}, []
for cy, cx, _, _ in edges:
    ref = np.ascontiguousarray(STACK[0][cy-80:cy+80, cx-80:cx+80])
    rois, info = H.extract_aligned_roi(STACK, cy, cx, size=160, search=140)
    s, r = esf_sigma(H.consensus_decomp(rois, info)['consensus'][:, 50:110])
    if np.isfinite(s) and r < 0.04 and 0.5 < s < 6:
        refs[(cy, cx)] = ref; dxo_sig.append(s)
DXO = float(np.median(dxo_sig))


def srband_power(region):
    """FFT power in the SR band (0.25-0.5 cyc/output-px = above native Nyquist at x2)."""
    a = region.astype(np.float64); a = a - a.mean()
    win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    F = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    h, w = a.shape; cy, cx = h // 2, w // 2
    fy = (np.arange(h) - cy) / h; fx = (np.arange(w) - cx) / w
    R = np.hypot(fy[:, None], fx[None, :])
    return float(F[(R >= 0.25) & (R < 0.5)].mean())


def matched_edge_sigma(m2):
    """median physical edge sigma over DxO edges (m2 is the scale-2 merge)."""
    out = []
    for (cy, cx), ref in refs.items():
        tmpl2 = cv2.resize(ref, (320, 320), interpolation=cv2.INTER_CUBIC)
        s = esf_sigma(H.align_to(tmpl2, m2, 2*cy, 2*cx, 320, 280)[:, 100:220])[0] / 2
        if np.isfinite(s) and 0.4 < s < 6:
            out.append(s)
    return float(np.median(out)), len(out)


# fine-texture region and flat region (scale-2 coords)
TEX = (6200, 2900); FLAT = (200, 8300); SZ = 384
print(f"DxO consensus edge sigma (optics-limited target) = {DXO:.2f}px   n_edges={len(refs)}")
print(f"\n{'N':>4}{'edge sigma':>12}{'tex SR-pow':>13}{'flat SR-pow':>13}{'flat/N0':>9}")
rows = []
flat0 = None
for N in (2, 4, 8, 16, 28):
    cache = f"{OUT}/_fs_merge_s2_N{N}.npy"
    try:
        m2 = np.load(cache)
    except FileNotFoundError:
        lin, _ = pr.run(burst, overrides={"scale": 2}, n_frames=N-1)  # N-1 comp + ref = N
        m2 = H.rgb_to_luma(lin).astype(np.float32); np.save(cache, m2)
    es, ne = matched_edge_sigma(m2)
    tex = srband_power(m2[TEX[0]:TEX[0]+SZ, TEX[1]:TEX[1]+SZ])
    flat = srband_power(m2[FLAT[0]:FLAT[0]+SZ, FLAT[1]:FLAT[1]+SZ])
    if flat0 is None:
        flat0 = flat
    print(f"{N:>4}{es:>12.2f}{tex:>13.2e}{flat:>13.2e}{flat/flat0:>9.2f}", flush=True)
    rows.append((N, es, tex, flat))

a = np.array(rows)
print(f"\n  edge sigma vs N: {a[:,1].min():.2f}..{a[:,1].max():.2f} (spread {a[:,1].max()-a[:,1].min():.2f}px) "
      f"-> {'FLAT (optics-limited; fusion cannot sharpen)' if a[:,1].max()-a[:,1].min() < 0.15 else 'varies'}")
print(f"  texture SR-power N2->N28: x{a[-1,2]/a[0,2]:.2f}; N4->N28 x{a[-1,2]/a[1,2]:.2f} "
      f"-> {'RISES then SATURATES (sampling SR)' if a[-1,2] > 1.15*a[0,2] else 'flat'}")
print(f"  flat SR-power N2->N28: x{a[-1,3]/a[0,3]:.2f} (sqrt(28/2)={np.sqrt(28/2):.1f}x drop expected if noise)")
