"""STEP 1 -- CENSUS (free): per-tile structure-tensor eigenvalues for the 007
reference, and the fraction of tiles that are aperture-ambiguous.

The per-tile LK normal matrix == the structure tensor == compute_hessian_cupy's
output (utils_image.py): H = [[Sgx2, Sgxgy],[Sgxgy, Sgy2]] summed over each tile.
We reuse the EXACT pipeline path (grey reference -> Sobel grads -> Hessian at the
finest pyramid level, tile_size resolved from SNR) so the eigenvalues are the ones
ica_iterate_cupy actually inverts.

Aperture-ambiguity (Shi-Tomasi "Good Features to Track", CVPR 1994; Bigun et al.
TPAMI 1991): one large eigenvalue, one ~0, large l1 == a 1-D (edge) tile where the
along-edge flow is unconstrained. We classify:
  - lambda_min  = (tr - sqrt(tr^2 - 4det))/2   (the constrained-direction energy)
  - coherence   = (l1 - l2)/(l1 + l2)          (1 == perfectly 1-D edge)
and tie the threshold to the NOISE MODEL: a Sobel gradient [-1,0,1] of a pixel with
noise std sigma_n has variance 2*sigma_n^2, so a tile of N pixels accumulates
E[Sgx2_noise] ~ N * 2 * sigma_n^2 from noise alone. An eigenvalue at/below that is
indistinguishable from noise -- not a reliable constraint. tau_noise = that floor.

Reports the DISTRIBUTION (median + spread + quantiles), per eval/CLAUDE.md.
"""
import sys
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from numba import cuda

sys.path.insert(0, ".")
from eval import pipeline_runner as pr
from eval import _aperture_common as AC
from handheld_super_resolution.utils_image import compute_grey_images, compute_hessian_cupy
from handheld_super_resolution.ICA import SOBEL_X, SOBEL_Y


def census():
    burst = pr.load_burst_arw(AC.ARW_DIR)
    cfg = pr.build_config(burst, overrides={"scale": 1})
    # finest pyramid level: factor 1, tile_sizes[0]
    tile_size = int(cfg.block_matching.tuning.tile_sizes[0])
    print(f"tile_size (finest level) = {tile_size}; SNR {burst['SNR']:.1f}")

    # reproduce the pipeline grey reference + gradients + Hessian. The pipeline's
    # finest level is the grey at factor 1 (build_gaussian_pyramid downsample by 1
    # is a gaussian-blur identity on a torch tensor); the Sobel grads + Hessian
    # here match init_ica exactly. compute_grey_images(FFT) returns a numba device
    # array -> wrap zero-copy as a torch cuda tensor for conv2d.
    ref = burst["ref_raw"].astype(np.float32)
    cuda_ref = cuda.to_device(ref)
    grey_nb = compute_grey_images(cuda_ref, cfg.grey_method)  # numba device array
    grey = torch.as_tensor(cp.asarray(grey_nb), device="cuda")
    gradx = F.conv2d(grey[None, None], SOBEL_X, padding="same").squeeze()
    grady = F.conv2d(grey[None, None], SOBEL_Y, padding="same").squeeze()
    H_t = compute_hessian_cupy(gradx, grady, tile_size)  # (ny, nx, 2, 2) cupy

    A = cp.asarray(H_t, dtype=cp.float64)
    a, b = A[..., 0, 0], A[..., 0, 1]
    d = A[..., 1, 1]
    tr = a + d
    disc = cp.sqrt(cp.clip(tr * tr - 4.0 * (a * d - b * b), 0, None))
    l1 = (tr + disc) / 2.0   # large eigenvalue (across-edge energy)
    l2 = (tr - disc) / 2.0   # small eigenvalue (along-edge energy = lambda_min)
    coh = (l1 - l2) / (l1 + l2 + 1e-30)
    det = a * d - b * b

    l1 = cp.asnumpy(l1).ravel()
    l2 = cp.asnumpy(l2).ravel()
    coh = cp.asnumpy(coh).ravel()
    det = cp.asnumpy(det).ravel()
    n = l1.size

    # --- noise-tied threshold ---
    # grey is a luma in raw units; estimate per-pixel grey noise std from the
    # noise model at the reference brightness. The grey is ~mean of the 4 bayer
    # quads, so its noise std ~ sigma_raw / 2 (4-sample average); use the raw
    # std_curve at the mean grey level as a conservative (slightly high) floor.
    grey_np = grey.detach().cpu().numpy()
    mean_grey = float(np.mean(np.clip(grey_np, 0, 1)))
    std_curve = np.asarray(cfg.noise_model.std_curve)
    sigma_raw = float(std_curve[min(len(std_curve) - 1, round(1000 * mean_grey))])
    # grey FFT-lowpass + 4-quad effect: be conservative, use sigma_raw directly
    sigma_n = sigma_raw
    Npix = tile_size * tile_size
    tau_noise = Npix * 2.0 * sigma_n * sigma_n  # E[Sg^2] from pure noise per tile
    print(f"mean grey {mean_grey:.4f}; sigma_raw {sigma_raw:.5f}; "
          f"per-tile noise eigen floor tau_noise = {tau_noise:.4g}")

    def q(x):
        return np.percentile(x, [5, 25, 50, 75, 95])

    print("\n--- structure-tensor eigenvalue distribution (per tile, N={}) ---".format(n))
    print(f"  l1 (large) quantiles [5/25/50/75/95]: {q(l1)}")
    print(f"  l2 (small=lambda_min)              : {q(l2)}")
    print(f"  coherence (l1-l2)/(l1+l2)          : {q(coh)}")

    # aperture-ambiguous == strong edge (l1 well above noise) BUT along-edge
    # constraint l2 at/below the noise floor (1-D structure).
    strong = l1 > tau_noise                 # tile has a real gradient at all
    flat = l1 <= tau_noise                  # no usable structure (sky/smooth)
    ambiguous = strong & (l2 <= tau_noise)  # 1-D: edge but along-edge unconstrained
    well_cond = strong & (l2 > tau_noise)   # 2-D: corner/texture, both dirs ok
    # coherence-based cross-check (Bigun): high coherence + strong l1
    coh_amb = strong & (coh > 0.9)

    print("\n--- tile classification (noise-tied tau) ---")
    print(f"  flat (l1<=tau, no structure)         : {flat.mean()*100:5.1f}%  ({flat.sum()})")
    print(f"  aperture-AMBIGUOUS (l1>tau, l2<=tau) : {ambiguous.mean()*100:5.1f}%  ({ambiguous.sum()})")
    print(f"  well-conditioned (l1>tau, l2>tau)    : {well_cond.mean()*100:5.1f}%  ({well_cond.sum()})")
    print(f"  [x-check] high-coherence (>0.9)&strong: {coh_amb.mean()*100:5.1f}%  ({coh_amb.sum()})")
    print(f"  among STRONG tiles, fraction ambiguous: "
          f"{ambiguous.sum()/max(1,strong.sum())*100:5.1f}%")

    # the solvable guard the pipeline already uses (|det| >= 1e-10): how many fail?
    unsolvable = np.abs(det) < 1e-10
    print(f"\n  pipeline |det|<1e-10 (unsolvable, dx=dy=0): {unsolvable.mean()*100:5.2f}% ({unsolvable.sum()})")
    print(f"  (these get NO ICA update today -- they keep the block-match prior)")

    return dict(tau_noise=tau_noise, ambiguous_frac=float(ambiguous.mean()),
                strong_frac=float(strong.mean()), well_cond_frac=float(well_cond.mean()),
                flat_frac=float(flat.mean()))


if __name__ == "__main__":
    census()
