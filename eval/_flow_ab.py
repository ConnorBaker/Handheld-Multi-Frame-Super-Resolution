"""A/B the block artifact against the fixed-tile discretization: scale=2 merge with
flow_upscale_mode 'nearest' (piecewise-constant per-tile flow) vs 'bicubic'
(continuous flow). Their DIFFERENCE isolates the tile-grid artifact."""
import numpy as np, cv2, time
from eval import pipeline_runner as pr
from eval import hallucination as H

burst = pr.load_burst_arw("/home/connorbaker/Pictures/ARW/2025-09-06/007")
out = {}
for mode in ("nearest", "bicubic"):
    t = time.time()
    lin, _ = pr.run(burst, overrides={"scale": 2, "block_matching.tuning.flow_upscale_mode": mode})
    out[mode] = H.rgb_to_luma(lin)
    np.save(f"eval/halluc_out_007/merge_s2_flow_{mode}.npy", out[mode].astype(np.float32))
    print(f"flow={mode}: merge {out[mode].shape} in {time.time()-t:.0f}s", flush=True)

n, b = out["nearest"], out["bicubic"]
cy, cx, W = 3840, 8480, 200
dn, db = n[cy-W:cy+W, cx-W:cx+W], b[cy-W:cy+W, cx-W:cx+W]
diff = dn - db
print(f"\nnearest-vs-bicubic difference over the artifact region: RMS {diff.std():.5f}  maxabs {np.abs(diff).max():.4f}")

def tile_period(a):
    win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    F = np.abs(np.fft.fftshift(np.fft.fft2(a * win))); m = a.shape[0]; c = m // 2; F[c-2:c+3, c-2:c+3] = 0
    pk = np.unravel_index(np.argmax(F), F.shape); per = m / max(np.hypot(pk[0]-c, pk[1]-c), 1e-9)
    return per, float(F.max()/(F.mean()+1e-9))
# the DIFFERENCE's periodicity = the flow-tile artifact (real louvers cancel in the diff)
pd, rd = tile_period(diff)
print(f"DIFFERENCE dominant period {pd:.1f}px peak/mean {rd:.1f}  (a sharp peak at ~tile-size => nearest flow tiling is the artifact)")
# whole-frame difference energy
print(f"whole-frame nearest-vs-bicubic RMS {(n-b).std():.5f}")

# visual: nearest | bicubic | 10x difference
def vis(a, amp=1, center=0.0):
    v = np.clip(center + amp * a, 0, 1) if center else np.clip((a - np.percentile(a, 2)) / (np.percentile(a, 98) - np.percentile(a, 2) + 1e-6), 0, 1)
    return cv2.cvtColor((cv2.resize(v.astype(np.float32), (a.shape[1]*2, a.shape[0]*2), interpolation=cv2.INTER_NEAREST)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
def lbl(u, t): cv2.rectangle(u, (0, 0), (u.shape[1]-1, 18), (0, 0, 0), -1); cv2.putText(u, t, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return u
g = np.full((W*4, 6, 3), 80, np.uint8)
sheet = np.hstack([lbl(vis(dn), "flow=nearest (per-tile)"), g, lbl(vis(db), "flow=bicubic (continuous)"),
                   g, lbl(vis(diff, amp=20, center=0.5), "20x difference (the tiling)")])
cv2.imwrite("eval/halluc_out_007/flow_ab.png", sheet)
print("saved flow_ab.png")
