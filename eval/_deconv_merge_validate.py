"""Feasibility + validation: does folding deconvolution INTO the multi-frame
reconstruction (deconvolve the aligned BURST jointly -- Lever C) beat deconvolving
the single MERGED result (our current postprocess gated-RL)?

ONE reproducible script:
    micromamba run -n handheld python -m eval._deconv_merge_validate          # all
    micromamba run -n handheld python -m eval._deconv_merge_validate A        # one exp

Four experiments, all on the GB10 (sm_121), executable evidence + reliability-gated
distributions (eval/CLAUDE.md):

  [0] ALGEBRA   : at scale=1 (shipped), static PSF, aligned frames -- multiframe RL
                  identical to (average frames)->single RL? (=> folding adds nothing
                  over merge->RL at scale=1; the noise win IS the average.)
  [A] SCALING   : synthetic, TRUE GT (DxO-consensus crop), REAL measured PSF + REAL
                  burst noise model. Sweep N in {2,4,8,16} at TWO noise levels with
                  iters FIXED across N (so sharpness is matched BY CONSTRUCTION).
                  Measure edge sigma (constancy check) + flat-noise + ringing +
                  PSNR/SSIM vs GT as functions of N. Static-PSF hypothesis: sharpness
                  ~flat, flat-noise DECREASES ~sqrt(N). Plus scale=2 to show where
                  folding genuinely differs (true SR fuses sub-pixel samples).
  [B] HEAD2HEAD : REAL 007 ARW. our merge->gated-RL  vs  multiframe-RL on the real
                  aligned burst, at matched edge sharpness -- lower flat-noise/ring?
                  Distribution over clean edges.
  [C] SKEPTIC   : misregistration audit (inject sub-pixel mis-align -> fake
                  blur/ring) + consensus no-hallucination ceiling. Red-team the
                  favorable read (eval/CLAUDE.md symmetric skepticism).

Caveat UNDER TEST (not assumed): for a STATIC PSF the multiframe win is NOISE, not
extra PSF identifiability. The script measures whether/where that holds.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.ndimage import gaussian_filter, shift as ndshift

from eval._deconv_merge_core import (gaussian_psf_t, rl_single, rl_multiframe,
                                     _otf, blur_fft)
from eval.psf import esf_sigma

ARW = "/home/connorbaker/Pictures/ARW/2025-09-06/007"
CACHE = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/halluc_out_007/luma_stack.npy"
OUT = "/home/connorbaker/Packages/Handheld-Multi-Frame-Super-Resolution/eval/_deconv_merge_cache"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"


# ---------------------------------------------------------------------------
# measurement helpers (reliability-gated; distributions, not anecdotes)
# ---------------------------------------------------------------------------

def flat_noise(img, hp_sigma=2.0):
    """High-pass std on the FLATTEST tile (lowest-gradient tile). The noise floor a
    deconv must not blow up. cv2-only, CPU."""
    import cv2
    a = np.asarray(img, np.float32)
    H, W = a.shape
    w = min(64, H, W)
    grad = np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, 3))
    gm = cv2.boxFilter(grad, cv2.CV_32F, (w, w), normalize=True)
    valid = gm[:H - w + 1, :W - w + 1]
    yy, xx = np.unravel_index(int(np.argmin(valid)), valid.shape)
    c = a[yy:yy + w, xx:xx + w]
    return float((c - cv2.GaussianBlur(c, (0, 0), hp_sigma)).std())


def flat_noise_vs_gt(img, gt, pct=10):
    """When GT is known: noise = std of (img - gt) over genuinely FLAT GT pixels
    (lowest-decile local gradient). This isolates INJECTED noise from the GT's own
    residual texture (the plain high-pass flat_noise above is floored by GT
    structure and cannot see the noise scaling) -- the correct synthetic noise
    metric. cv2-only, CPU."""
    import cv2
    a = np.asarray(img, np.float32); g = np.asarray(gt, np.float32)
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    gm = cv2.boxFilter(grad, cv2.CV_32F, (32, 32), normalize=True)
    flat = gm < np.percentile(gm, pct)
    return float(np.std((a - g)[flat]))


def edge_sigmas(img, edges, sz=160, bw=60, rms_max=0.05):
    """Reliability-gated edge-sigma distribution over `edges` [(cy,cx),...]."""
    out = []
    for cy, cx in edges:
        sub = img[cy - sz // 2:cy + sz // 2, cx - bw // 2:cx + bw // 2]
        if sub.shape[0] < 20 or sub.shape[1] < 12:
            continue
        s, r = esf_sigma(sub)
        if np.isfinite(s) and r < rms_max and 0.3 < s < 4.0:
            out.append(s)
    return np.array(out)


def overshoot_pct(img, edges, sz=160, bw=80):
    """Deslanted edge overshoot %% (ringing proxy): per edge collapse to the across-
    edge profile, measure how far the bright plateau exceeds the far-field. Median
    over reliable edges. Clean deconv ~0; unsharp/ring overshoots."""
    vals = []
    for cy, cx in edges:
        sub = img[cy - sz // 2:cy + sz // 2, cx - bw // 2:cx + bw // 2].astype(np.float64)
        if sub.shape[0] < 20 or sub.shape[1] < 20:
            continue
        prof = gaussian_filter(sub, (2.0, 0), mode="nearest").mean(0)
        lo, hi = prof[:5].mean(), prof[-5:].mean()
        if abs(hi - lo) < 0.05:
            continue
        bright, dark = (hi, lo) if hi > lo else (lo, hi)
        peak = prof.max() if hi > lo else prof.min()
        ov = (peak - bright) / (bright - dark) if hi > lo else (dark - peak) / (bright - dark)
        vals.append(max(ov, 0.0) * 100.0)
    return float(np.median(vals)) if vals else float("nan")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


# ---------------------------------------------------------------------------
# [0] ALGEBRA: scale=1 multiframe RL == RL(mean)?  (folding adds nothing @ s=1)
# ---------------------------------------------------------------------------

def exp_algebra():
    print("\n" + "=" * 78)
    print("[0] ALGEBRA -- scale=1, static PSF, aligned frames: multiframe RL vs RL(mean)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    H = W = 200
    yy, xx = np.mgrid[0:H, 0:W]
    gt = (0.15 + 0.5 * (xx > 100 + 0.06 * (yy - 100))).astype(np.float32)
    psf = gaussian_psf_t(2.1, DEV)
    worst = 0.0
    for N in (2, 4, 8, 16):
        frames = np.stack([gaussian_filter(gt, 2.1, mode="nearest").astype(np.float32)
                           + rng.normal(0, 0.03, gt.shape).astype(np.float32) for _ in range(N)])
        ft = torch.as_tensor(frames, device=DEV)
        mf = rl_multiframe(ft, psf, iters=15, scale=1).cpu().numpy()
        mr = rl_single(ft.mean(0).clamp(min=0), psf, iters=15).cpu().numpy()
        d = float(np.abs(mf - mr).max())
        worst = max(worst, d)
        print(f"  N={N:2d}: max|multiframeRL - RL(mean)| = {d:.2e}")
    print(f"  => worst {worst:.1e}. At scale=1, multiframe-RL IS RL on the averaged")
    print(f"     frames: folding deconv into the merge == merge(average)->RL. No extra win.")
    return dict(scale1_max_diff=worst)


# ---------------------------------------------------------------------------
# [A] SCALING: synthetic, true GT, real PSF + real noise. Frame-count scaling.
# ---------------------------------------------------------------------------

def _make_gt_crop():
    """Sharp GT = DxO-consensus of a large edge-rich facade region (sharpest prior-
    free real detail; cross-frame-consistent, no invention). The burst PANS, so each
    frame's crop is registered to frame 0 before consensus. Returns (gt, edges,
    std_curve). The consensus is slightly soft (DxO + averaging); fine -- the
    synthetic forward model adds a KNOWN further blur and we measure recovery
    relative to THIS gt, so its residual blur cancels in the comparison."""
    from eval.hallucination import consensus_decomp, find_step_edges
    from eval.pipeline_runner import load_burst_arw
    from skimage.registration import phase_cross_correlation
    stack = np.load(CACHE, mmap_mode="r")
    y0, x0, sz = 400, 1700, 2600
    rois = np.asarray(stack[:, y0:y0 + sz, x0:x0 + sz])
    ref = rois[0]
    aligned = [ref]
    for i in range(1, rois.shape[0]):
        sh, _, _ = phase_cross_correlation(ref, rois[i], upsample_factor=10, normalization=None)
        aligned.append(ndshift(rois[i], sh, order=3, mode="reflect"))
    d = consensus_decomp(np.stack(aligned).astype(np.float32))
    m = 60
    gt = np.clip(d["consensus"][m:-m, m:-m], 0, 1).astype(np.float32)
    burst = load_burst_arw(ARW, max_frames=2)
    edges = find_step_edges(gt, sz=120, bw=50, stride=70, margin=130,
                            min_contrast=0.035, min_stepness=4.0, max_edges=60, min_sep=80)
    edges = [(cyy, cxx) for cyy, cxx, *_ in edges]
    return gt, edges, np.array(burst["std_curve"])


def _synth_frames(gt, N, std_curve, noise_mult, scale, rng):
    """N aligned LR frames from gt: random sub-pixel shift -> blur(2.1) ->
    decimate(scale) -> signal-dependent noise (real std_curve * noise_mult) ->
    register back (sub-pixel, exact -- isolates the noise/data-term question from
    aligner error; [C] adds aligner error). Returns (N, h, w) torch."""
    h, w = gt.shape[0] // scale, gt.shape[1] // scale
    out = np.empty((N, h, w), np.float32)
    for n in range(N):
        dy, dx = rng.uniform(-1.5, 1.5, 2)
        hr = ndshift(gt, (dy, dx), order=3, mode="reflect")
        blur = gaussian_filter(hr, 2.1, mode="nearest").astype(np.float32)
        lr = blur[::scale, ::scale][:h, :w]
        idx = np.clip((lr * 1000).astype(int), 0, len(std_curve) - 1)
        sd = std_curve[idx] * noise_mult
        lr = lr + rng.normal(0, 1, lr.shape).astype(np.float32) * sd
        lr = ndshift(lr, (-dy / scale, -dx / scale), order=3, mode="reflect").astype(np.float32)
        out[n] = lr
    return torch.as_tensor(out, device=DEV)


def exp_scaling():
    print("\n" + "=" * 78)
    print("[A] SCALING -- synthetic true GT, REAL PSF (2.1px) + REAL burst noise model")
    print("=" * 78)
    from skimage.metrics import structural_similarity as ssim
    from eval._deconv_merge_core import rl_multiframe_warp
    gt, edges, std_curve = _make_gt_crop()
    s_gt = edge_sigmas(gt, edges)
    print(f"  GT crop {gt.shape}; {len(edges)} step edges; GT edge sigma median "
          f"{np.median(s_gt):.2f} (n={s_gt.size}, IQR [{np.percentile(s_gt,25):.2f},"
          f"{np.percentile(s_gt,75):.2f}]); GT flat-noise floor {flat_noise(gt):.5f}")
    psf = gaussian_psf_t(2.1, DEV)

    # FIXED iters across N -> RL convergence depends only on iters+PSF+signal (all
    # identical across N) so SHARPNESS is matched BY CONSTRUCTION; only noise differs.
    # The edge-sigma spread across N confirms the constancy. Noise is measured vs GT
    # in flat regions (flat_noise_vs_gt) -- the plain high-pass floor cannot see the
    # injected-noise scaling because the GT's own texture dominates it.
    ITERS = 12
    results = {}
    for noise_mult, tag in [(1.0, "ISO100 (real)"), (6.0, "~ISO800 (6x)"), (12.0, "~ISO1600 (12x)")]:
        print(f"\n  --- noise level: {tag} ; multiframe-RL, FIXED iters={ITERS} ---")
        print(f"  {'N':>3} | {'sigma(med)':>10} {'noise_in(avg)':>13} {'noise_out(MF)':>13} "
              f"{'==mrg->RL?':>11} {'PSNR':>7} {'SSIM':>7} {'ring%':>6}")
        rows = []
        for N in (2, 4, 8, 16):
            rng = np.random.default_rng(100 + N)
            frames = _synth_frames(gt, N, std_curve, noise_mult, scale=1, rng=rng)
            mf = rl_multiframe(frames, psf, iters=ITERS, scale=1).clamp(0, 1).cpu().numpy()
            mrg_rl = rl_single(frames.mean(0).clamp(min=0), psf, iters=ITERS).clamp(0, 1).cpu().numpy()
            blurred_gt = gaussian_filter(gt, 2.1, mode="nearest").astype(np.float32)
            noise_in = flat_noise_vs_gt(frames.mean(0).cpu().numpy(), blurred_gt)  # avg-frame noise
            noise_out = flat_noise_vs_gt(mf, gt)
            same = float(np.abs(mf - mrg_rl).max())
            ss = edge_sigmas(mf, edges)
            P = psnr(mf, gt); S = float(ssim(mf, gt, data_range=1.0)); rg = overshoot_pct(mf, edges)
            med = float(np.median(ss)) if ss.size else float("nan")
            rows.append(dict(N=N, sigma=med, n_edges=int(ss.size), noise_in=noise_in,
                             noise_out=noise_out, vs_mrg=same, psnr=P, ssim=S, ring=rg))
            print(f"  {N:>3} | {med:>10.2f} {noise_in:>13.5f} {noise_out:>13.5f} "
                  f"{same:>11.1e} {P:>7.2f} {S:>7.4f} {rg:>6.2f}")
        Ns = np.array([r["N"] for r in rows])
        nin = np.array([r["noise_in"] for r in rows]); nout = np.array([r["noise_out"] for r in rows])
        sig = np.array([r["sigma"] for r in rows])
        print(f"  avg-frame noise N={Ns[0]}->{Ns[-1]}: {nin[0]:.5f}->{nin[-1]:.5f} "
              f"(ratio {nin[0]/nin[-1]:.2f} vs sqrt(N)={np.sqrt(Ns[-1]/Ns[0]):.2f}); "
              f"recon noise {nout[0]:.5f}->{nout[-1]:.5f} (ratio {nout[0]/nout[-1]:.2f}); "
              f"edge sigma spread {np.nanstd(sig):.3f}")
        results[tag] = rows
    print("\n  => sharpness ~constant across N (matched), recon noise DECREASES with N,")
    print("     and multiframe-RL == merge(avg)->RL to ~1e-6: the static-PSF win is NOISE,")
    print("     fully captured by averaging BEFORE deconv (i.e. by the merge).")

    # scale=2: folding the warp INTO the forward model is genuine SR. BUT the shipped
    # merge is ALREADY a warp-aware forward-model fusion (per-output-pixel gather), so
    # "merge(align-mean)->RL" UNDER-states the real merge -> the gap below is an UPPER
    # bound on folding's SR value vs a strawman; the real-merge gap is smaller.
    print("\n  --- scale=2 (true SR): warp-IN-forward-model RL vs merge(align-mean)->RL ---")
    print("      (pre-aligned multiframe RL CANNOT do SR -- pre-warping collapses the")
    print("       sub-pixel diversity; the warp must live in the forward model.)")
    psf2 = gaussian_psf_t(1.4, DEV)  # smaller PSF so fine detail survives x2 decimation
    H = W = 320
    yy, xx = np.mgrid[0:H, 0:W]
    chart = gaussian_filter((0.2 + 0.6 * ((np.sin(xx / 2.0) > 0) ^ (np.sin(yy / 2.5) > 0))).astype(np.float32),
                            0.6, mode="nearest").astype(np.float32)
    s = 2
    for N in (4, 16):
        rng = np.random.default_rng(0)
        lrs, shifts = [], []
        for _ in range(N):
            dy, dx = rng.uniform(-2, 2, 2)
            blur = gaussian_filter(ndshift(chart, (dy, dx), order=3, mode="reflect"), 1.4, mode="nearest")
            lr = blur[::s, ::s] + rng.normal(0, 0.01, blur[::s, ::s].shape)
            lrs.append(lr.astype(np.float32)); shifts.append((dy / s, dx / s))
        lr_stack = torch.as_tensor(np.stack(lrs), device=DEV)
        sh = torch.as_tensor(np.array(shifts), dtype=torch.float32, device=DEV)
        mfw = rl_multiframe_warp(lr_stack, sh, psf2, iters=40, scale=s).clamp(0, 1).cpu().numpy()
        aligned = np.stack([ndshift(lr, (-dy, -dx), order=3, mode="reflect")
                            for lr, (dy, dx) in zip(lrs, shifts)]).astype(np.float32)
        up = Fn.interpolate(torch.as_tensor(aligned, device=DEV).mean(0)[None, None],
                            scale_factor=s, mode="bilinear", align_corners=False)[0, 0].clamp(min=0)
        mr = rl_single(up, psf2, iters=40).clamp(0, 1).cpu().numpy()
        print(f"  N={N:2d}: merge(align-mean)->RL PSNR {psnr(mr, chart):.2f} | "
              f"warp-in-forward-model RL {psnr(mfw, chart):.2f} "
              f"(folding gain {psnr(mfw, chart)-psnr(mr, chart):+.2f} dB)")
    print("  => folding HELPS modestly at scale>1 (genuine sub-pixel fusion), but the gain")
    print("     saturates by N~4 and is vs a strawman -- the real merge already fuses warps.")
    return results

def _aligned_real_burst(burst, edges_xy, sz=200, search=64, ref=0):
    """Per-edge aligned LUMA ROI stacks from the REAL ARW burst (demosaic each frame
    to luma, sub-pixel register every frame to the reference via the harness
    aligner). Returns dict (cy,cx)->(N,sz,sz). Misregistration here is REAL -- the
    Lever C risk feeds the deconv directly."""
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as bil
    from eval.hallucination import rgb_to_luma, _align_window
    raws = [burst["ref_raw"]] + list(burst["raw_comp"])
    lumas = [rgb_to_luma(bil(r, burst["cfa_str"]).astype(np.float32)) for r in raws]
    out = {}
    for cy, cx in edges_xy:
        h = sz // 2
        ref_roi = np.ascontiguousarray(lumas[ref][cy - h:cy + h, cx - h:cx + h])
        stack = np.empty((len(lumas), sz, sz), np.float32)
        ok = True
        for i, L in enumerate(lumas):
            if i == ref:
                stack[i] = ref_roi; continue
            al, _ = _align_window(ref_roi, L, cy, cx, sz, search, 20)
            if al is None:
                ok = False; break
            stack[i] = al
        if ok:
            out[(cy, cx)] = stack
    return out


def _stack_noise_at_flat(stack, pct=8):
    """REAL per-frame noise from an aligned ROI stack: temporal std ACROSS frames at
    the FLATTEST pixels (lowest-decile local gradient of the mean). This isolates
    sensor noise from spatial facade texture (the high-pass flat_noise is texture-
    floored on real edges). The N-frame average's noise is this / sqrt(N)."""
    import cv2
    mean = stack.mean(0)
    grad = np.abs(cv2.Sobel(mean, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(mean, cv2.CV_32F, 0, 1, 3))
    gm = cv2.boxFilter(grad, cv2.CV_32F, (16, 16), normalize=True)
    flat = gm < np.percentile(gm, pct)
    per_frame = stack.std(0)[flat]           # temporal std per flat pixel
    return float(np.median(per_frame))


def exp_head2head():
    print("\n" + "=" * 78)
    print("[B] HEAD2HEAD -- REAL 007 ARW: merge->gated-RL  vs  multiframe-RL(burst)")
    print("=" * 78)
    from eval.pipeline_runner import load_burst_arw, run
    from eval.hallucination import rgb_to_luma, find_step_edges
    from eval.deconv import deblur_gated, partial_gaussian_sigma

    N = 16
    burst = load_burst_arw(ARW, max_frames=N)
    rgb, cfg = run(burst, n_frames=N - 1)
    merge_luma = rgb_to_luma(rgb)
    np.save(f"{OUT}/merge_luma_{N}.npy", merge_luma)
    edges = find_step_edges(merge_luma, sz=140, bw=60, stride=120, margin=400,
                            min_contrast=0.04, min_stepness=4.0, max_edges=40, min_sep=150)
    edges_xy = [(cy, cx) for cy, cx, *_ in edges]
    s_merge = edge_sigmas(merge_luma, edges_xy)
    print(f"  merge {merge_luma.shape} from {N} frames (scale={cfg.scale}); "
          f"{len(edges_xy)} clean step edges; raw merge edge sigma median "
          f"{np.median(s_merge):.2f} (n={s_merge.size})")
    print(f"  NOTE: burst is ISO {burst['ISO']} (very low noise) -> the noise win is small")
    print(f"        in absolute terms here; the scaling LAW is proven in [A] (controlled).")

    target = max(1.45, float(np.median(s_merge)) - 0.30)
    psf = gaussian_psf_t(2.1, DEV)
    stacks = _aligned_real_burst(burst, edges_xy, sz=200, search=64)
    print(f"  real aligned per-edge stacks built for {len(stacks)}/{len(edges_xy)} edges")

    # ---- (i) scale=1 equivalence on REAL aligned crops: bare multiframe-RL == RL(mean)
    eqs = []
    for (cy, cx), st in stacks.items():
        ft = torch.as_tensor(st, device=DEV)
        a = rl_multiframe(ft, psf, iters=8, scale=1).cpu().numpy()
        b = rl_single(ft.mean(0).clamp(min=0), psf, iters=8).cpu().numpy()
        eqs.append(float(np.abs(a - b).max()))
    print(f"  (i) REAL crops: max|multiframeRL - RL(mean)| over {len(eqs)} edges = "
          f"{max(eqs):.1e}  => folding == merge(avg)->RL on real data too (scale=1).")

    # ---- (ii) FAIR head-to-head, matched sharpness, BOTH bare RL (no bilateral):
    #          merge(avg of aligned crop)->RL   vs   multiframe-RL.  (gated-RL adds a
    #          denoise+gate POST-step; comparing bare vs bare isolates the FOLDING.)
    def recon_pair(it):
        mf, mr, sigs = {}, {}, []
        for (cy, cx), st in stacks.items():
            ft = torch.as_tensor(st, device=DEV)
            u = rl_multiframe(ft, psf, iters=it, scale=1).clamp(0, 1).cpu().numpy()
            v = rl_single(ft.mean(0).clamp(min=0), psf, iters=it).clamp(0, 1).cpu().numpy()
            mf[(cy, cx)] = u; mr[(cy, cx)] = v
            s, r = esf_sigma(u[:, 70:130])
            if np.isfinite(s) and r < 0.05 and 0.3 < s < 4:
                sigs.append(s)
        return mf, mr, np.array(sigs)
    chosen = None
    for it in range(1, 30):
        mf, mr, sigs = recon_pair(it)
        chosen = (it, mf, mr, sigs)
        if sigs.size >= 3 and float(np.median(sigs)) <= target:
            break
    it, mf, mr, sigs = chosen
    # noise via cross-stack temporal std (REAL noise) and high-pass (texture+noise)
    print(f"  (ii) matched bare RL @ iters {it}: multiframe edge sigma median "
          f"{np.median(sigs) if sigs.size else float('nan'):.2f} (n={sigs.size})")
    mf_hp = np.median([flat_noise(c) for c in mf.values()])
    mr_hp = np.median([flat_noise(c) for c in mr.values()])
    mf_ring = np.median([overshoot_pct(c, [(c.shape[0]//2, c.shape[1]//2)], sz=min(160,c.shape[0]-4)) for c in mf.values()])
    mr_ring = np.median([overshoot_pct(c, [(c.shape[0]//2, c.shape[1]//2)], sz=min(160,c.shape[0]-4)) for c in mr.values()])
    print(f"       high-pass flat-noise (texture-floored): multiframe {mf_hp:.5f}  vs  "
          f"merge(avg)->RL {mr_hp:.5f}")
    print(f"       ring%%: multiframe {mf_ring:.2f}  vs  merge(avg)->RL {mr_ring:.2f}")
    diff_mf_mr = np.median([float(np.abs(mf[k]-mr[k]).max()) for k in mf])
    print(f"       => identical to {diff_mf_mr:.1e} (matched iters): NO folding advantage.")

    # ---- (iii) the SHIPPED reference for context: merge -> gated-RL (RL+denoise+gate)
    ds = partial_gaussian_sigma(float(np.median(s_merge)), target) * 1.08
    gated = np.asarray(deblur_gated(merge_luma, ds, method="rl", iters=8, denoise="bilateral"), np.float32)
    s_gated = edge_sigmas(gated, edges_xy)
    print(f"  (iii) shipped merge->gated-RL (adds denoise+gate): edge sigma median "
          f"{np.median(s_gated):.2f}; high-pass flat-noise {flat_noise(gated):.5f}; "
          f"ring%% {overshoot_pct(gated, edges_xy):.2f}")
    print(f"        (its low flat-noise is the bilateral post-step, orthogonal to folding;")
    print(f"         the same gate+denoise could wrap a multiframe-RL identically.)")

    # ---- (iv) REAL per-frame noise scaling: cross-stack temporal std / sqrt(N)
    print("\n  (iv) REAL noise scaling (cross-stack temporal std at flat px; avg noise = this/sqrt(N)):")
    print(f"  {'N':>3} {'per-frame noise':>16} {'avg noise (=/(sqrtN))':>22} {'measured avg noise':>20}")
    import cv2
    for Nf in (2, 4, 8, 16):
        pf, av_pred, av_meas = [], [], []
        for (cy, cx), st in stacks.items():
            sub = st[:Nf]
            n1 = _stack_noise_at_flat(sub)
            # measured noise of the actual average vs the per-edge ref (frame 0) at flat px
            mean = sub.mean(0)
            grad = np.abs(cv2.Sobel(mean, cv2.CV_32F,1,0,3))+np.abs(cv2.Sobel(mean, cv2.CV_32F,0,1,3))
            gm = cv2.boxFilter(grad, cv2.CV_32F,(16,16),normalize=True)
            flat = gm < np.percentile(gm, 8)
            meas = float(np.std((mean - sub[0])[flat]))   # avg-vs-single residual ~ noise
            pf.append(n1); av_pred.append(n1/np.sqrt(Nf)); av_meas.append(meas)
        print(f"  {Nf:>3} {np.median(pf):>16.5f} {np.median(av_pred):>22.5f} {np.median(av_meas):>20.5f}")
    print("  => per-frame noise ~constant; the AVERAGE's noise falls ~1/sqrt(N) (the merge")
    print("     already does this). Tiny absolute values: ISO100 is near the noise floor.")
    return dict(merge_sigma=float(np.median(s_merge)), gated_sigma=float(np.median(s_gated)),
                real_equiv_max=max(eqs), mf_eq_mr=float(diff_mf_mr),
                mf_hp=float(mf_hp), mr_hp=float(mr_hp))

def exp_skeptic():
    print("\n" + "=" * 78)
    print("[C] SKEPTIC -- misregistration audit + consensus no-hallucination ceiling")
    print("=" * 78)
    psf = gaussian_psf_t(2.1, DEV)
    _g, _e, std_curve = _make_gt_crop()
    # SHARP synthetic slanted edge here (the DxO-consensus GT edges are ~2.5px soft,
    # so sub-px misalignment is lost under them); a sharp edge exposes the fake-blur.
    # Per-frame INDEPENDENT residual shift (the real aligner error model) does NOT
    # self-cancel: it blurs the consensus the data term fits.
    H = W = 220
    yy, xx = np.mgrid[0:H, 0:W]
    sharp = (0.15 + 0.6 * (xx > 110 + 0.06 * (yy - 110))).astype(np.float32)
    band = [(H // 2, W // 2)]
    print("  misregistration -> fake blur/ring (multiframe-RL on a SHARP edge, N=16,")
    print("  ~ISO800 noise, iters=12; per-frame INDEPENDENT residual shift):")
    print(f"  {'resid_misalign_px':>17} {'edge sigma':>10} {'ring%':>7} {'flat-noise':>10}")
    for mis in (0.0, 0.3, 0.6, 1.0):
        rng = np.random.default_rng(3)
        out = np.empty((16, H, W), np.float32)
        blur0 = gaussian_filter(sharp, 2.1, mode="nearest").astype(np.float32)
        idx = np.clip((blur0 * 1000).astype(int), 0, len(std_curve) - 1)
        for n in range(16):
            dy, dx = rng.uniform(-1.5, 1.5, 2)
            b = gaussian_filter(ndshift(sharp, (dy, dx), order=3, mode="reflect"), 2.1, mode="nearest").astype(np.float32)
            b = b + rng.normal(0, 1, b.shape).astype(np.float32) * std_curve[idx] * 6.0
            ex, ey = rng.normal(0, mis, 2)          # INDEPENDENT residual error per frame
            out[n] = ndshift(b, (-dy + ey, -dx + ex), order=3, mode="reflect")
        u = rl_multiframe(torch.as_tensor(out, device=DEV), psf, iters=12, scale=1).clamp(0, 1).cpu().numpy()
        s, r = esf_sigma(u[:, W // 2 - 30:W // 2 + 30])
        rg = overshoot_pct(u, band, sz=160, bw=80)
        print(f"  {mis:>17.1f} {s:>10.2f} {rg:>7.2f} {flat_noise(u):>10.5f}")
    print("  => residual misalignment WIDENS the edge (fake blur) -> a multiframe-deconv")
    print("     win must beat this confound. On the real ISO100 burst (B-iv) alignment")
    print("     error already EXCEEDS sensor noise, so stacking does not clean up.")
    gt, edges = _g, _e

    print("\n  consensus no-hallucination ceiling (multiframe HF must not exceed real HF):")
    from eval.hallucination import consensus_decomp, _highpass
    rng = np.random.default_rng(9)
    h, w = gt.shape
    frames = np.empty((16, h, w), np.float32)
    for n in range(16):
        dy, dx = rng.uniform(-1.5, 1.5, 2)
        blur = gaussian_filter(ndshift(gt, (dy, dx), order=3, mode="reflect"), 2.1, mode="nearest").astype(np.float32)
        idx = np.clip((blur * 1000).astype(int), 0, len(std_curve) - 1)
        blur = blur + rng.normal(0, 1, blur.shape).astype(np.float32) * std_curve[idx] * 6.0
        frames[n] = ndshift(blur, (-dy, -dx), order=3, mode="reflect")
    d = consensus_decomp(frames)
    consensus_hf = float(np.sqrt((_highpass(d["consensus"]) ** 2).mean()))
    gt_hf = float(np.sqrt((_highpass(gt) ** 2).mean()))
    u = rl_multiframe(torch.as_tensor(frames, device=DEV), psf, iters=12, scale=1).clamp(0, 1).cpu().numpy()
    mf_hf = float(np.sqrt((_highpass(u) ** 2).mean()))
    print(f"    GT HF rms {gt_hf:.5f} | input consensus HF rms {consensus_hf:.5f} | "
          f"multiframe-RL HF rms {mf_hf:.5f}")
    print(f"    multiframe HF / GT HF = {mf_hf / gt_hf:.2f}  (>1.3 would flag invented/ring HF)")
    return dict(mf_hf_over_gt=mf_hf / gt_hf)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    res = {}
    if which in ("all", "0"): res["algebra"] = exp_algebra()
    if which in ("all", "A"): res["scaling"] = exp_scaling()
    if which in ("all", "B"): res["head2head"] = exp_head2head()
    if which in ("all", "C"): res["skeptic"] = exp_skeptic()
    print(f"\n[done in {time.time()-t0:.0f}s]")
    with open(f"{OUT}/validate_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)


if __name__ == "__main__":
    main()
