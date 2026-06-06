"""Optuna sweep against SYNTHETIC ground-truth fidelity, over MULTIPLE scenes.

Fixes the two flaws of eval/sweep.py:
  - objective is fidelity vs a KNOWN ground truth (perfect registration), so it
    cannot be gamed by over-sharpening (inventing high-freq that isn't in GT
    raises the loss). Objective = mean over scenes of 0.5*(1-SSIM) + 0.5*DISTS
    vs GT. SSIM/DISTS both anchored to truth.
  - averages over several scenes (different Canon crops) so the result is less
    overfit to one image.

Sweeps the full pipeline: the alignment / robustness / merge params AND the
integrated post-processing components -- Stage A Polyblur ``deblur`` and Stage B
``aberration`` (per-tile deblur + learned defringer). A ``post_mode`` selector
picks none / deblur / aberration (mutually exclusive, as in the live pipeline),
and the chosen stage's response params are tuned. The objective scores the
*post-processed* output, so a component only wins if it raises fidelity to the
known GT (over-sharpening is penalised); the ``default`` reference now also runs
the shipped post-processing, so "best" is measured against the real pipeline.

A fourth ``post_mode='deconv_rl'`` sweeps the MEASURED-PSF deconvolution
(``eval.deconv``), the one component whose blur kernel is DERIVED, not blind. The
synthetic forward model injects a KNOWN Gaussian PSF (``OPTICS.psf_sigma``), and
the merge inherits ~that blur, so the deconv sigma is computed, not searched:
``deconv_sigma = fudge * partial_gaussian_sigma(merge_sigma_est, target)`` where
``merge_sigma_est`` is the injected psf at the merge's scale and ``target`` a
fraction of it (``DECONV_TARGET_FRAC``). Only the REGULARIZERS are swept (iters,
method, denoise, a narrow ``fudge``). To stop the optimizer winning by
over-sharpening / amplifying noise -- the failure that broke the gameable
Canon-DISTS sweep -- the objective is made ONE-SIDED for this mode: the
synthetic-GT fidelity anchor (SSIM+DISTS) PLUS penalties that only ever *add* loss
for exceeding a deterministic edge probe's target sharpness, the merge's flat
noise, or the merge's edge overshoot (``_deconv_penalties``).

The synthetic forward model injects a known optical PSF + lateral CA (``OPTICS``,
applied per frame in ``synthetic.make_synthetic_burst``) so the deblur/defringe
have a real degradation to invert -- without it the merge already removes
everything the GT comparison can see and post-processing can only amplify noise.

PER-BURST mode (``--dxo-ref``): use NATIVE-RES crops of a DxO reference as the GT
instead of the Canon JPEGs -- content-matched to a real shot, parallax-free, and
professionally lens-corrected. Crops (not downscaled) + the burst's REAL sensor
noise (``--burst``) keep the scale/SNR/displacement regime identical to the real
burst, so the tuned params transfer (downscaling would change the SNR). Defaults to
``factor=1`` (synthetic LR at the real native resolution). Same Optuna study/sampler.

Usage:
    python -m eval.sweep_synthetic --trials 120 --outdir results/eval/sweep_synth_v2
    python -m eval.sweep_synthetic --dxo-ref ref.dng --burst <burst_dir> --trials 120
"""

import argparse
import csv
import json
import os

import numpy as np
import cv2
import imageio.v3 as iio
import optuna
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr

from . import synthetic as syn
from . import pipeline_runner as pr
from . import metrics as met
from . import deconv as dc
from .psf import esf_sigma

# A few varied scenes (building, foliage, mixed) from different bursts' Canon JPEGs.
SCENES = [
    ("test_burst/Samsung_dataset/0003/Canon/im.JPG", 1000, 2400, 1024),
    ("test_burst/Samsung_dataset/0001/Canon/im.JPG", 1200, 2600, 1024),
    ("test_burst/Samsung_dataset/0050/Canon/im.JPG", 1200, 2600, 1024),
]

# Optical degradation injected into the synthetic forward model so the deblur/
# aberration stages have a real PSF + lateral CA to invert (the clean-GT comparison
# can then reward them). CALIBRATED from the a6300 + lens via DxO's lens-measured
# corrections (eval.calibrate_optics on the DxO baseline/CA/sharp DNGs of DSC00034):
# a mild ~0.54px Gaussian PSF and ASYMMETRIC lateral CA (R ~+1.35px at the corner,
# B ~0). psf_sigma_corner == psf_sigma here -- field variation wasn't measured (one
# global sigma); refine if a per-region PSF is calibrated later.
OPTICS = dict(psf_sigma=0.543, psf_sigma_corner=0.543, ca_R=1.35, ca_B=-0.03)

# --- measured-PSF deconvolution sweep (post_mode='deconv_rl') ---------------
# The deconv sigma is DERIVED from the KNOWN injected PSF, not swept. The merge
# inherits ~the injected per-frame blur; at SR scale=factor the merge grid is
# `factor`x finer, so a sigma measured in full-res (GT) px maps to ~factor*sigma in
# merge px. We deconvolve toward a TARGET that is a fraction of the merge sigma (an
# external sharpness anchor, per eval/NOTES.md -- the optimizer cannot move it):
#   merge_sigma_est = factor * OPTICS['psf_sigma']
#   deconv_sigma    = fudge * partial_gaussian_sigma(merge_sigma_est, TARGET_FRAC*merge_sigma_est)
DECONV_TARGET_FRAC = 0.6        # deconvolve the merge edge to 60% of its sigma
# One-sided objective weights (post_mode='deconv_rl' only). Each penalty is >=0 and
# fires only when the deconv exceeds a baseline measured on the SAME image/probe:
#   + W_SIGMA * |edge_sigma - target|              (target a sharpness, not "sharper")
#   + W_NOISE * max(0, flat_noise - merge_flat_noise)   (no extra flat-region noise)
#   + W_OVER  * max(0, overshoot  - merge_overshoot)    (no extra edge halo/ringing)
# Scaled so a ~0.3 px sigma miss / a doubling of flat noise / a 0.1 overshoot bump
# each cost ~0.01-0.03 -- the order of a real DISTS/SSIM fidelity move, so fidelity
# still leads and the penalties veto over-sharpening rather than dominate.
W_SIGMA, W_NOISE, W_OVER = 0.05, 4.0, 0.15


def load_scenes(scenes=SCENES):
    gts = []
    for path, y, x, s in scenes:
        if not os.path.exists(path):
            continue
        img = iio.imread(path).astype(np.float32) / 255.0
        gts.append(img[y:y + s, x:x + s].copy())
    if not gts:
        raise RuntimeError("no scenes found")
    return gts


def load_dxo_scenes(dxo_ref_path, n=3, size=1024, margin=0.1):
    """Extract ``n`` high-detail, non-overlapping NATIVE-RES crops from a DxO
    reference, to use as per-burst GT scenes (content-matched to the real shot).

    Native-res CROPS -- deliberately NOT downscaled: downscaling averages pixels,
    which raises the SNR and so shifts the SNR-resolved params (k_detail/k_denoise/
    D_th, robustness) and convergence (ica.n_iter) into a regime that won't transfer
    to the real full-res burst. A crop preserves scale, SNR and displacement; the
    real per-burst noise is injected separately (run_sweep's alpha/beta).
    """
    from .calibrate_optics import read_dng
    img = read_dng(str(dxo_ref_path))
    H, W = img.shape[:2]
    m = int(min(H, W) * margin)
    grad = np.hypot(*np.gradient(img.mean(2).astype(np.float64)))
    cands = sorted(((float(grad[y:y + size, x:x + size].mean()), y, x)
                    for y in range(m, H - size - m, size // 2)
                    for x in range(m, W - size - m, size // 2)), reverse=True)
    chosen, taken = [], []
    for _, y, x in cands:
        if all(abs(y - ty) >= size or abs(x - tx) >= size for ty, tx in taken):  # non-overlapping
            chosen.append(np.ascontiguousarray(img[y:y + size, x:x + size]))
            taken.append((y, x))
            if len(chosen) >= n:
                break
    if not chosen:
        raise RuntimeError(f"no {size}px crops fit in {dxo_ref_path} ({H}x{W})")
    return chosen


def suggest(trial):
    ov = {
        "ica.tuning.n_iter": trial.suggest_int("ica_n_iter", 1, 8),
        "ica.tuning.sigma_blur": trial.suggest_float("ica_sigma_blur", 0.0, 1.5),
        "robustness.tuning.t": trial.suggest_float("rob_t", 0.0, 0.5),
        "robustness.tuning.s1": trial.suggest_float("rob_s1", 0.5, 4.0),
        "robustness.tuning.s2": trial.suggest_float("rob_s2", 4.0, 20.0),
        "robustness.tuning.Mt": trial.suggest_float("rob_Mt", 0.2, 2.0),
        "merging.kernel": trial.suggest_categorical("kernel", ["steerable", "iso"]),
        "merging.tuning.k_detail": trial.suggest_float("k_detail", 0.18, 0.5),
        "merging.tuning.k_denoise": trial.suggest_float("k_denoise", 0.5, 5.0),
        "merging.tuning.D_th": trial.suggest_float("D_th", 0.0, 1.0),
        "merging.tuning.D_tr": trial.suggest_float("D_tr", 0.5, 2.0),
        "merging.tuning.k_stretch": trial.suggest_float("k_stretch", 1.0, 6.0),
        "merging.tuning.k_shrink": trial.suggest_float("k_shrink", 1.0, 4.0),
    }
    ov.update(_suggest_post(trial))
    return ov


def _suggest_post(trial):
    """Post-processing component selection + its response parameters.

    Stage A (Polyblur ``deblur``), Stage B (``aberration`` = per-tile deblur +
    learned defringer) and ``deconv_rl`` (measured-PSF deconvolution, ``eval.deconv``)
    are mutually exclusive, matching ``raw2rgb.postprocess`` precedence (aberration >
    deblur > sharpen; deconv_rl is the eval-only measured-PSF path). ``none`` keeps
    the bare merge as a baseline. For deblur/aberration we sweep the deconvolution
    response (alpha, beta), the blur-model (c, sigma_b -- the lever that sets how
    much blur is estimated/inverted), and the on/off sub-stage toggles. Fixed:
    patch_size, overlap, ker_size (structural).

    ``deconv_rl`` is different on purpose: its blur kernel is DERIVED from the known
    injected PSF (see DECONV_* above), so we sweep ONLY the regularizers -- the
    deconv sigma is never suggested. Its params live under the non-config ``_deconv``
    prefix and are popped off before the pipeline runs (``score_config``).
    """
    mode = trial.suggest_categorical("post_mode", ["none", "deblur", "aberration", "deconv_rl"])
    ov = {
        "postprocessing.aberration.enabled": mode == "aberration",
        "postprocessing.deblur.enabled": mode == "deblur",
        "postprocessing.sharpening.enabled": False,
    }
    if mode == "deconv_rl":
        # ONLY regularizers -- NOT the deconv sigma (it is derived in _apply_post).
        # method: 'rl' (preferred -- gentler, less halo/noise) or 'wiener'. NOTE
        # eval.deconv.deblur has no 'tv_rl', so 'wiener' is the second regularized
        # inverse offered here (a TV-RL variant would slot in once deconv adds it).
        ov["_deconv.enabled"] = True
        ov["_deconv.iters"] = trial.suggest_int("dc_iters", 2, 8)
        ov["_deconv.method"] = trial.suggest_categorical("dc_method", ["rl", "wiener"])
        ov["_deconv.denoise"] = trial.suggest_categorical("dc_denoise", [True, False])
        # narrow band: the ~8% non-Gaussian-edge correction, NOT a free knob to hit
        # the target (the one-sided penalties veto over-sharpening if it strays).
        ov["_deconv.fudge"] = trial.suggest_float("dc_fudge", 0.95, 1.20)
    elif mode == "deblur":
        ov["postprocessing.deblur.c"] = trial.suggest_float("db_c", 0.2, 0.6)
        ov["postprocessing.deblur.sigma_b"] = trial.suggest_float("db_sigma_b", 0.2, 0.9)
        ov["postprocessing.deblur.alpha"] = trial.suggest_float("db_alpha", 0.0, 8.0)
        ov["postprocessing.deblur.beta"] = trial.suggest_float("db_beta", 0.0, 8.0)
        ov["postprocessing.deblur.n_iter"] = trial.suggest_int("db_n_iter", 1, 3)
        ov["postprocessing.deblur.remove_halo"] = trial.suggest_categorical("db_remove_halo", [True, False])
        ov["postprocessing.deblur.prefiltering"] = trial.suggest_categorical("db_prefiltering", [True, False])
    elif mode == "aberration":
        ov["postprocessing.aberration.c"] = trial.suggest_float("ab_c", 0.2, 0.6)
        ov["postprocessing.aberration.sigma_b"] = trial.suggest_float("ab_sigma_b", 0.2, 0.9)
        ov["postprocessing.aberration.alpha"] = trial.suggest_float("ab_alpha", 0.0, 8.0)
        ov["postprocessing.aberration.beta"] = trial.suggest_float("ab_beta", 0.0, 8.0)
        ov["postprocessing.aberration.do_deblur"] = trial.suggest_categorical("ab_do_deblur", [True, False])
        ov["postprocessing.aberration.do_defringe"] = trial.suggest_categorical("ab_do_defringe", [True, False])
    return ov


def fidelity(out, gt):
    out = np.clip(out, 0, 1).astype(np.float32)
    s = float(_ssim(gt, out, channel_axis=2, data_range=1.0))
    d = float(met.tiled_metrics(out, gt, tile=256, max_tiles=64)["DISTS"])
    return 0.5 * (1 - s) + 0.5 * d, s, d


def _apply_post(out, post):
    """Apply the active post-processing component (aberration > deblur) on the
    linear-RGB merge output, mirroring ``raw2rgb.postprocess``'s precedence.

    The synthetic burst has no DNG, so the color/devignette/tonemap/gamma steps
    (which need it) are skipped -- this isolates the swept deblur/aberration
    components, which act on linear [0,1] RGB exactly as they do live.
    """
    from handheld_super_resolution import raw2rgb
    out = np.clip(out, 0, 1).astype(np.float32)
    ab = post.get("aberration", None)
    db = post.get("deblur", None)
    if ab is not None and ab.get("enabled", False):
        return np.clip(raw2rgb._aberration(out, ab), 0, 1).astype(np.float32)
    if db is not None and db.get("enabled", False):
        return np.clip(raw2rgb._deblur(out, db), 0, 1).astype(np.float32)
    return out


def _merge_sigma_est(factor):
    """The merge's edge sigma in merge px, derived from the KNOWN injected PSF. The
    merge inherits ~the per-frame optical blur; at SR scale=factor the output grid is
    ``factor``x finer, so a full-res-px sigma scales to ~factor*sigma in merge px."""
    return float(factor) * float(OPTICS["psf_sigma"])


def _apply_deconv(out, dp, factor):
    """Apply measured-PSF deconvolution to the linear-RGB merge (per channel).

    Per-channel (not luma-only) is the simpler correct choice here: the injected CA
    means R/G/B carry slightly different edge content, and deconvolving luma then
    recombining would smear chroma across the shared luma kernel. The sigma is
    DERIVED (``_merge_sigma_est`` -> ``partial_gaussian_sigma`` toward a target
    fraction), only ``dp``'s regularizers (iters/method/denoise/fudge) vary.
    ``deblur_gated`` (global sigma per channel) is used per trial -- cheap; the
    field-varying ``deblur_field_varying`` is the FINAL-REFINE step (too slow to
    sweep on full frames). Returns (out_rgb, deconv_sigma)."""
    out = np.clip(out, 0, 1).astype(np.float32)
    m = _merge_sigma_est(factor)
    ds = float(dp.get("fudge", 1.08)) * dc.partial_gaussian_sigma(m, DECONV_TARGET_FRAC * m)
    method = dp.get("method", "rl")
    iters = int(dp.get("iters", 5))
    denoise = "bilateral" if dp.get("denoise", True) else None
    res = np.empty_like(out)
    for c in range(3):
        oc = dc.deblur_gated(out[..., c], ds, method=method, iters=iters,
                             denoise=denoise, gate=True)
        res[..., c] = oc if isinstance(oc, np.ndarray) else oc.cpu().numpy()
    return np.clip(res, 0, 1).astype(np.float32), ds


def _edge_probe(merge_sigma, n=200, contrast=0.6, base=0.2):
    """A deterministic slanted step edge (clean) + the same edge blurred to
    ``merge_sigma`` (the merge-equivalent baseline). Gives a guaranteed clean edge to
    measure sharpness/overshoot on -- natural GT crops often have NO clean step edge
    (verified: 2/3 of the Canon scenes have none), so probing the output's own edges
    is unreliable. Deconvolving this probe with the trial's operator reproduces the
    sharpness/halo the trial's deconv yields on a real edge of that blur."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    clean = (base + contrast * (xx > n / 2 + 0.06 * (yy - n / 2))).astype(np.float32)
    blur = cv2.GaussianBlur(clean, (0, 0), float(max(merge_sigma, 1e-3)))
    return clean, blur


_PROBE_BAND = slice(70, 130)


def _overshoot(luma, band=_PROBE_BAND):
    """Edge over/undershoot: the largest excursion past the edge's two plateaus
    (98th/2nd pct), normalized by edge contrast. Ringing raises it; a plain blur is
    ~0. (Deslant-free but honest on this near-vertical probe; trust the direction.)"""
    roi = luma[:, band]
    hi, lo = np.percentile(roi, 98), np.percentile(roi, 2)
    return float(max(roi.max() - hi, lo - roi.min()) / max(hi - lo, 1e-6))


def _deconv_penalties(dp, factor):
    """One-sided penalties for post_mode='deconv_rl', measured on the edge probe.

    Returns (penalty_total, terms) where penalty = W_SIGMA*|sigma-target|
    + W_NOISE*max(0, flat_noise - merge_flat_noise) + W_OVER*max(0, overshoot -
    merge_overshoot). Baselines are the BLURRED probe (the merge's own sharpness /
    noise / overshoot), so the optimizer can only reach the target sharpness without
    exceeding the merge's noise floor or halo -- it cannot win by over-sharpening."""
    m = _merge_sigma_est(factor)
    target = DECONV_TARGET_FRAC * m
    _clean, blur = _edge_probe(m)
    out, ds = _apply_deconv(np.repeat(blur[..., None], 3, axis=2), dp, factor)
    out_l = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    s_out, _ = esf_sigma(out_l[:, _PROBE_BAND])
    if not np.isfinite(s_out):
        s_out = m  # fit failed -> treat as "no sharpening" (no sigma credit/penalty)
    n_merge = dc._flat_noise_std(blur)
    n_out = dc._flat_noise_std(out_l)
    o_merge = _overshoot(blur)
    o_out = _overshoot(out_l)
    p_sig = W_SIGMA * abs(s_out - target)
    p_noise = W_NOISE * max(0.0, n_out - n_merge)
    p_over = W_OVER * max(0.0, o_out - o_merge)
    terms = dict(deconv_sigma=ds, edge_sigma=float(s_out), target=float(target),
                 flat_noise=float(n_out), merge_flat_noise=float(n_merge),
                 overshoot=float(o_out), merge_overshoot=float(o_merge),
                 p_sigma=float(p_sig), p_noise=float(p_noise), p_overshoot=float(p_over))
    return float(p_sig + p_noise + p_over), terms


def _pop_deconv(overrides):
    """Split off the ``_deconv.*`` (non-config) keys so they never reach OmegaConf.

    Returns (clean_overrides, deconv_params_or_None). deconv_params has the bare
    regularizer names (iters/method/denoise/fudge) iff post_mode='deconv_rl'."""
    ov = dict(overrides or {})
    dp = {k.split(".", 1)[1]: ov.pop(k) for k in list(ov) if k.startswith("_deconv.")}
    return ov, (dp if dp.get("enabled") else None)


def score_config(gts, bursts, overrides, factor=2):
    """Mean fidelity loss over scenes for one config (bursts precomputed).

    Runs the merge at ``scale=factor``, then the swept post-processing
    (deblur/aberration, or measured-PSF ``deconv_rl``), then scores the
    *post-processed* output vs GT -- so the new components are in the objective.
    PSNR is tracked alongside (it penalizes the noise amplification / over-sharpening
    that SSIM+DISTS are softer on). For ``deconv_rl`` a ONE-SIDED penalty (computed
    once on a derived edge probe, scene-independent) is added to the mean fidelity
    loss. Returns (loss, SSIM, DISTS, sharp_ratio, PSNR, deconv_terms_or_None)."""
    ov_clean, dp = _pop_deconv(overrides)
    losses, ssims, dists, sharps, psnrs = [], [], [], [], []
    for gt, burst in zip(gts, bursts):
        ov = dict(ov_clean)
        ov["scale"] = factor
        out, cfg = pr.run(burst, ov)
        if dp is not None:
            out, _ds = _apply_deconv(out, dp, factor)   # measured-PSF deconv path
        else:
            out = _apply_post(out, cfg.postprocessing)  # shipped deblur/aberration
        loss, s, d = fidelity(out, gt)
        losses.append(loss); ssims.append(s); dists.append(d)
        psnrs.append(float(_psnr(np.clip(gt, 0, 1), np.clip(out, 0, 1), data_range=1.0)))
        sharps.append(met.sharpness(out) / max(met.sharpness(gt), 1e-6))
    fid = float(np.mean(losses))
    dterms = None
    if dp is not None:
        penalty, dterms = _deconv_penalties(dp, factor)
        fid += penalty
    return (fid, float(np.mean(ssims)), float(np.mean(dists)),
            float(np.mean(sharps)), float(np.mean(psnrs)), dterms)


def run_sweep(n_trials=60, outdir="results/eval/sweep_synth", study_name="synth_fidelity", seed=0,
              gts=None, alpha=1e-4, beta=0.0, factor=2, n_frames=8):
    os.makedirs(outdir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if gts is None:
        gts = load_scenes()
    # precompute synthetic bursts once (same noisy LR frames for every trial -> fair).
    # OPTICS injects a known PSF + lateral CA so deblur/aberration have a real target;
    # alpha/beta are the (real, per-burst) sensor noise so the SNR regime matches.
    bursts = [syn.make_synthetic_burst(gt, n=n_frames, factor=factor, alpha=alpha, beta=beta,
                                       seed=i, **OPTICS)[0] for i, gt in enumerate(gts)]
    print(f"[scenes] {len(gts)} GT crops | factor={factor} n_frames={n_frames} "
          f"alpha={alpha:.2e} beta={beta:.2e}")

    base_loss, base_s, base_d, base_sh, base_psnr, _ = score_config(gts, bursts, None, factor)
    print(f"[ref] default (shipped postproc): loss={base_loss:.4f} SSIM={base_s:.4f} "
          f"DISTS={base_d:.4f} PSNR={base_psnr:.2f} sharp/gt={base_sh:.3f}")

    def objective(trial):
        try:
            loss, s, d, sh, ps, dterms = score_config(gts, bursts, suggest(trial), factor)
        except Exception as exc:
            trial.set_user_attr("error", str(exc)[:200])
            return 1.0
        if not np.isfinite(loss):
            return 1.0
        trial.set_user_attr("SSIM", s)
        trial.set_user_attr("DISTS", d)
        trial.set_user_attr("sharp_ratio", sh)
        trial.set_user_attr("PSNR", ps)
        if dterms is not None:                       # one-sided deconv penalty breakdown
            for k, v in dterms.items():
                trial.set_user_attr(k, v)
        return loss

    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=f"sqlite:///{os.path.join(outdir, 'study.db')}",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.set_user_attr("default_loss", base_loss)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bt = study.best_trial
    best_nested = {}
    for k, v in suggest(_Frozen(bt)).items():
        d = best_nested
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    best_nested["scale"] = factor
    OmegaConf.save(OmegaConf.create(best_nested), os.path.join(outdir, "best_config.yaml"))
    out = dict(default=dict(loss=base_loss, SSIM=base_s, DISTS=base_d, PSNR=base_psnr, sharp_ratio=base_sh),
               best=dict(loss=study.best_value, **{k: bt.user_attrs.get(k) for k in ("SSIM", "DISTS", "PSNR", "sharp_ratio")}),
               best_params=study.best_params, n_trials=len(study.trials))
    with open(os.path.join(outdir, "best.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[best] loss={study.best_value:.4f} SSIM={bt.user_attrs.get('SSIM'):.4f} "
          f"DISTS={bt.user_attrs.get('DISTS'):.4f} PSNR={bt.user_attrs.get('PSNR'):.2f} "
          f"sharp/gt={bt.user_attrs.get('sharp_ratio'):.3f}")
    print("[best] params:", json.dumps(study.best_params))
    return out


class _Frozen:
    def __init__(self, trial):
        self._p = trial.params
    def suggest_int(self, n, *a, **k):
        return self._p[n]
    def suggest_float(self, n, *a, **k):
        return self._p[n]
    def suggest_categorical(self, n, *a, **k):
        return self._p[n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--outdir", default="results/eval/sweep_synth")
    ap.add_argument("--dxo-ref", help="DxO reference DNG: use its native-res crops as GT "
                                      "(per-burst, content-matched)")
    ap.add_argument("--burst", help="raw burst dir: take its real sensor noise (alpha,beta)")
    ap.add_argument("--frames", type=int, default=8, help="synthetic burst frame count")
    ap.add_argument("--factor", type=int, default=None,
                    help="SR factor; default 2 for the Canon scenes, 1 for --dxo-ref "
                         "(keeps the synthetic LR at the real burst's native resolution)")
    args = ap.parse_args()

    if args.dxo_ref:
        gts = load_dxo_scenes(args.dxo_ref)
        alpha, beta = 1e-4, 0.0
        if args.burst:
            from . import pipeline_runner as pr
            b = pr.load_burst_arw(args.burst, max_frames=2)
            alpha, beta = float(b["alpha"]), float(b["beta"])
        run_sweep(n_trials=args.trials, outdir=args.outdir, study_name="dxo_fidelity",
                  gts=gts, alpha=alpha, beta=beta, factor=args.factor or 1, n_frames=args.frames)
    else:
        run_sweep(n_trials=args.trials, outdir=args.outdir, factor=args.factor or 2,
                  n_frames=args.frames)
