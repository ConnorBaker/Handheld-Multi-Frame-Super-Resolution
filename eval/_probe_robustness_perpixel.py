"""Light probes for the per-pixel-vs-global robustness question.

Probe 1: demonstrate the t-cancellation on well-aligned content (all frames
         agree -> equal R -> t shift cancels in num/den), and that t DOES bite
         when frames differ in alignment.
Probe 2: confirm the robustness mask R = clip(S*exp(-d^2/sigma^2) - t, 0, 1)
         varies PER PIXEL via d^2 and S, while t/s1/s2/Mt are scalars.
"""
import cupy as cp
import numpy as np

from handheld_super_resolution.cupy_ops import robustness_threshold_cupy


def merge_1d(values, weights):
    """Scalar kernel-regression merge: sum(w*v)/sum(w)."""
    num = float((weights * values).sum())
    den = float(weights.sum())
    return num / den if den != 0 else float("nan")


def probe1_cancellation():
    print("=== Probe 1: does t cancel? ===")
    rng = np.random.default_rng(0)
    N = 12  # frames
    # ---- Case A: WELL-ALIGNED. Every frame has the SAME color distance d to ref
    # (so identical pre-threshold R). Sample values are the (noisy) pixel values.
    d_sq = np.full(N, 0.04, dtype=np.float32)     # equal small distance, all frames
    sigma_sq = np.float32(0.05)
    S = np.float32(2.0)                           # s2 (flow-consistent)
    base = np.exp(-d_sq / sigma_sq) * S           # pre-threshold, equal across frames
    samples = (0.5 + 0.02 * rng.standard_normal(N)).astype(np.float32)
    for t in (0.12, 0.43):
        R = np.clip(base - t, 0.0, 1.0).astype(np.float32)
        out = merge_1d(samples, R)
        print(f"  aligned  t={t:.2f}: R(all)={R[0]:.4f}  merged={out:.6f}")

    # ---- Case B: MIXED alignment at the MARGIN. 8 aligned frames (small d) +
    # 4 PARTIALLY-misaligned frames whose distance puts their pre-threshold R in
    # the band that t straddles -- so t actually changes how much they contribute.
    d_sq = np.concatenate([np.full(8, 0.04), np.full(4, 0.13)]).astype(np.float32)
    base = np.exp(-d_sq / sigma_sq) * S
    aligned_val = 0.5 + 0.02 * rng.standard_normal(8)
    ghost_val = 0.5 + 0.30 + 0.02 * rng.standard_normal(4)   # misaligned -> biased
    samples = np.concatenate([aligned_val, ghost_val]).astype(np.float32)
    truth = float(aligned_val.mean())
    for t in (0.12, 0.43):
        R = np.clip(base - t, 0.0, 1.0).astype(np.float32)
        out = merge_1d(samples, R)
        print(f"  margin   t={t:.2f}: R(aln)={R[0]:.4f} R(ghost)={R[8]:.4f} "
              f"merged={out:.6f}  (aligned-only truth={truth:.6f}, "
              f"err={abs(out-truth):.6f})")


def probe2_perpixel_variation():
    print("\n=== Probe 2: does the mask vary per pixel? ===")
    H, W, tile = 64, 64, 16
    rng = cp.random.RandomState(1)
    # spatially-varying color distance: low in left half (flat/aligned),
    # high in right half (misaligned texture)
    d_sq = cp.zeros((H, W), dtype=cp.float32)
    d_sq[:, W // 2:] = 0.5
    d_sq += 0.02 * rng.standard_normal((H, W)).astype(cp.float32)
    d_sq = cp.clip(d_sq, 0, None)
    sigma_sq = cp.full((H, W), 0.05, dtype=cp.float32)
    # S per TILE (flow-discontinuity penalty), 2.0 or 12.0
    nt = H // tile
    S = cp.full((nt, nt), 2.0, dtype=cp.float32)
    S[0, 0] = 12.0  # one flow-irregular tile flagged
    for t in (0.12, 0.43):
        R = robustness_threshold_cupy(d_sq, sigma_sq, S, t, tile)
        R = cp.asnumpy(R)
        print(f"  t={t:.2f}: R min={R.min():.4f} max={R.max():.4f} "
              f"mean={R.mean():.4f}  std(spatial)={R.std():.4f}  "
              f"frac at 0 (rejected)={(R==0).mean():.3f}")
        # spatial structure: left (aligned) vs right (misaligned)
        print(f"           left-half mean={R[:, :W//2].mean():.4f}  "
              f"right-half mean={R[:, W//2:].mean():.4f}")


if __name__ == "__main__":
    probe1_cancellation()
    probe2_perpixel_variation()
