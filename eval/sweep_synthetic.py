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

The synthetic forward model injects a known optical PSF + lateral CA (``OPTICS``,
applied per frame in ``synthetic.make_synthetic_burst``) so the deblur/defringe
have a real degradation to invert -- without it the merge already removes
everything the GT comparison can see and post-processing can only amplify noise.

Usage:
    python -m eval.sweep_synthetic --trials 120 --outdir results/eval/sweep_synth_v2
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

# A few varied scenes (building, foliage, mixed) from different bursts' Canon JPEGs.
SCENES = [
    ("test_burst/Samsung_dataset/0003/Canon/im.JPG", 1000, 2400, 1024),
    ("test_burst/Samsung_dataset/0001/Canon/im.JPG", 1200, 2600, 1024),
    ("test_burst/Samsung_dataset/0050/Canon/im.JPG", 1200, 2600, 1024),
]

# Known optical degradation injected into the synthetic forward model (px at GT
# 2x scale) so the deblur/aberration stages have a real PSF + lateral CA to invert
# and the clean-GT comparison can reward them. Field-varying blur (sharper centre,
# blurrier corner) + radial R/B chromatic shift. Provisional, moderate values --
# calibrate from a real lens (e.g. DxO lens-measured corrections) when available.
OPTICS = dict(psf_sigma=1.2, psf_sigma_corner=2.0, ca_px=2.5)


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

    Stage A (Polyblur ``deblur``) and Stage B (``aberration`` = per-tile deblur +
    learned defringer) are mutually exclusive, matching ``raw2rgb.postprocess``
    precedence (aberration > deblur > sharpen). ``none`` keeps the bare merge as a
    baseline. Swept: the deconvolution response (alpha, beta), the blur-model
    (c, sigma_b -- the lever that sets how much blur is estimated/inverted), and
    the on/off sub-stage toggles. Fixed: patch_size, overlap, ker_size (structural).
    """
    mode = trial.suggest_categorical("post_mode", ["none", "deblur", "aberration"])
    ov = {
        "postprocessing.aberration.enabled": mode == "aberration",
        "postprocessing.deblur.enabled": mode == "deblur",
        "postprocessing.sharpening.enabled": False,
    }
    if mode == "deblur":
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


def score_config(gts, bursts, overrides):
    """Mean fidelity loss over scenes for one config (bursts precomputed).

    Runs the merge, then the swept post-processing (deblur/aberration), then
    scores the *post-processed* output vs GT -- so the new components are in the
    objective. PSNR is tracked alongside (it penalizes the noise amplification /
    over-sharpening that SSIM+DISTS are softer on).
    """
    losses, ssims, dists, sharps, psnrs = [], [], [], [], []
    for gt, burst in zip(gts, bursts):
        ov = dict(overrides or {})
        ov["scale"] = 2
        out, cfg = pr.run(burst, ov)
        out = _apply_post(out, cfg.postprocessing)
        loss, s, d = fidelity(out, gt)
        losses.append(loss); ssims.append(s); dists.append(d)
        psnrs.append(float(_psnr(np.clip(gt, 0, 1), np.clip(out, 0, 1), data_range=1.0)))
        sharps.append(met.sharpness(out) / max(met.sharpness(gt), 1e-6))
    return (float(np.mean(losses)), float(np.mean(ssims)), float(np.mean(dists)),
            float(np.mean(sharps)), float(np.mean(psnrs)))


def run_sweep(n_trials=60, outdir="results/eval/sweep_synth", study_name="synth_fidelity", seed=0):
    os.makedirs(outdir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    gts = load_scenes()
    # precompute synthetic bursts once (same noisy LR frames for every trial -> fair).
    # OPTICS injects a known PSF + lateral CA so deblur/aberration have a real target.
    bursts = [syn.make_synthetic_burst(gt, n=8, alpha=1e-4, seed=i, **OPTICS)[0]
              for i, gt in enumerate(gts)]
    print(f"[scenes] {len(gts)} GT crops")

    base_loss, base_s, base_d, base_sh, base_psnr = score_config(gts, bursts, None)
    print(f"[ref] default (shipped postproc): loss={base_loss:.4f} SSIM={base_s:.4f} "
          f"DISTS={base_d:.4f} PSNR={base_psnr:.2f} sharp/gt={base_sh:.3f}")

    def objective(trial):
        try:
            loss, s, d, sh, ps = score_config(gts, bursts, suggest(trial))
        except Exception as exc:
            trial.set_user_attr("error", str(exc)[:200])
            return 1.0
        if not np.isfinite(loss):
            return 1.0
        trial.set_user_attr("SSIM", s)
        trial.set_user_attr("DISTS", d)
        trial.set_user_attr("sharp_ratio", sh)
        trial.set_user_attr("PSNR", ps)
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
    best_nested["scale"] = 2
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--outdir", default="results/eval/sweep_synth")
    args = ap.parse_args()
    run_sweep(n_trials=args.trials, outdir=args.outdir)
