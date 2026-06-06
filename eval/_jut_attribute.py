"""Attribute the feature distortion: single-frame (reference-only merge, SAME kernel,
NO comp-frame alignment) vs multi-frame merge, both vs DxO at the feature.
If single-frame matches DxO (compact) but multi-frame smears -> comp-frame FLOW is
the cause (alignment). If both smear -> the steerable KERNEL. If single-frame also
diverges the same way -> it is sensor/demosaic content, not our fusion.
"""
import numpy as np, glob, cv2
from eval import hallucination as H
from eval import pipeline_runner as pr

OUT = "eval/halluc_out_007"
STACK = H.load_luma_stack(sorted(glob.glob('/home/connorbaker/Pictures/DNG/2025-09-06/007/DSC*_DxO.dng')), cache=f'{OUT}/luma_stack.npy')
m_multi = np.load(f"{OUT}/merge_s1_luma.npy")
burst = pr.load_burst_arw("/home/connorbaker/Pictures/ARW/2025-09-06/007")
rgb_single, _ = pr.run_single(burst, overrides={"scale": 1})
m_single = H.rgb_to_luma(rgb_single)
print(f"single-frame merge {m_single.shape}, multi-frame {m_multi.shape}")

cy, cx = 960, 1840
ref = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])
cons = H.consensus_decomp(*H.extract_aligned_roi(STACK, cy, cx, size=160, search=140))['consensus']
a_multi = H.align_to(ref, m_multi, cy, cx, 160, 140)
a_single = H.align_to(ref, m_single, cy, cx, 160, 140)
single_dxo = np.ascontiguousarray(STACK[0][cy - 80:cy + 80, cx - 80:cx + 80])  # one raw DxO frame (no multi-frame)

y0, y1, x0, x1 = 98, 130, 50, 105
sl = (slice(y0, y1), slice(x0, x1))
def lm(a, ref=cons):
    return (a - a.mean()) / (a.std() + 1e-9) * ref.std() + ref.mean()
dc = cons[sl]
for nm, im in (("multi-frame merge", a_multi), ("single-frame merge", a_single), ("one DxO frame", single_dxo)):
    d = lm(im)[sl] - dc
    iy, ix = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    print(f"  {nm:<20} vs DxO-consensus: max|div| {np.abs(d).max():.3f}  rms {d.std():.3f}  (worst at x={x0+ix})")
print("\n  -> if single-frame << multi-frame: comp-frame FLOW smears it (alignment).")
print("     if single-frame ~ multi-frame: the steerable KERNEL. if 'one DxO frame' also")
print("     diverges similarly: it is just frame-to-frame content/noise, not our fusion.")

# SMEARING DIRECTION: second-moment spread of the |div| energy in x vs y.
# horizontal smear (jut to the right) -> x-spread > y-spread, and MORE so for multi-frame.
print("\nsmearing direction (second-moment spread of |ours - DxO| energy, in px):")
for nm, im in (("single-frame", a_single), ("multi-frame", a_multi)):
    d = np.abs(lm(im)[sl] - dc); d = np.clip(d - np.percentile(d, 60), 0, None)  # keep the feature
    ys, xs = np.mgrid[0:d.shape[0], 0:d.shape[1]]; w = d.sum() + 1e-12
    mx, my = (xs * d).sum() / w, (ys * d).sum() / w
    sx = np.sqrt(((xs - mx) ** 2 * d).sum() / w); sy = np.sqrt(((ys - my) ** 2 * d).sum() / w)
    print(f"  {nm:<13} x-spread {sx:.2f}  y-spread {sy:.2f}  x/y {sx/(sy+1e-9):.2f}  (>1 = horizontally elongated)")

# visual: single | multi | DxO-consensus | one-DxO-frame, tight zoom
lo, hi = np.percentile(dc, 2), np.percentile(dc, 98); Z = 14
def panel(a, lab):
    v = np.clip((lm(a)[sl] - lo) / (hi - lo), 0, 1) ** (1/2.2)
    u = cv2.cvtColor((cv2.resize(v.astype(np.float32), ((x1-x0)*Z, (y1-y0)*Z), interpolation=cv2.INTER_CUBIC)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(u, (0, 0), (u.shape[1]-1, 16), (0, 0, 0), -1); cv2.putText(u, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return u
g = np.full(((y1-y0)*Z, 5, 3), 80, np.uint8)
cv2.imwrite(f"{OUT}/jut_attribute.png", np.hstack([panel(a_single, "single-frame"), g, panel(a_multi, "multi-frame"), g,
                                                   panel(cons, "DxO consensus"), g, panel(single_dxo, "one DxO frame")]))
print(f"\nsaved {OUT}/jut_attribute.png")
