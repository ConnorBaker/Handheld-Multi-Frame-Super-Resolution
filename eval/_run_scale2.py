"""Run the multi-frame pipeline at scale=1 and scale=2 on the 007 ARW burst,
save linear-RGB + gamma-luma for the 'beat DxO at every resolution' analysis."""
import numpy as np, time, os
from eval import pipeline_runner as pr
from eval import hallucination as H

OUT = "eval/halluc_out_007"
ARW = "/home/connorbaker/Pictures/ARW/2025-09-06/007"
os.makedirs(OUT, exist_ok=True)

t = time.time()
burst = pr.load_burst_arw(ARW)
print(f"loaded burst {burst['raw_comp'].shape[0]+1} frames in {time.time()-t:.0f}s; ref {burst['ref_raw'].shape}", flush=True)

for scale in (1, 2):
    t = time.time()
    lin, cfg = pr.run(burst, overrides={"scale": scale})
    luma = H.rgb_to_luma(lin)
    np.save(f"{OUT}/merge_s{scale}_linear.npy", lin.astype(np.float32))
    np.save(f"{OUT}/merge_s{scale}_luma.npy", luma.astype(np.float32))
    print(f"scale={scale}: merge {lin.shape} in {time.time()-t:.0f}s, "
          f"range {lin.min():.3f}/{lin.max():.3f} -> saved", flush=True)
print("DONE scale=1 and scale=2 merges saved", flush=True)
