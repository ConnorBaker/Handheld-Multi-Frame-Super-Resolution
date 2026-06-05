"""Optuna parameter sweep: find a config minimizing DISTS vs the Canon ground
truth (perceptual, alignment-robust).

Registration geometry is cached once (``compare.prepare_target``) so each trial
only runs the pipeline + warp + tiled metrics (~5 s at scale 2). DISTS is the
objective (more robust to residual registration error than LPIPS); LPIPS / PSNR /
SSIM are recorded as user attrs.

Usage:
    python -m eval.sweep --burst test_burst/Samsung_dataset/0003/Samsung \
        --canon test_burst/Samsung_dataset/0003/Canon/im.JPG --trials 80
"""

import argparse
import csv
import json
import os

import numpy as np
import imageio.v3 as iio
import optuna
from omegaconf import OmegaConf

from . import pipeline_runner as pr
from . import compare


# Search space: (override_key, optuna suggest spec). k_detail floored at 0.18 to
# avoid the demosaic-support collapse (-> NaN) seen below ~0.15.
def suggest_overrides(trial, scale):
    return {
        "scale": scale,
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


def make_objective(burst, target, scale):
    def objective(trial):
        ov = suggest_overrides(trial, scale)
        try:
            scalars, _, _ = compare.score(burst, ov, target)
        except Exception as exc:  # a bad config (e.g. NaN output) -> penalty
            trial.set_user_attr("error", str(exc)[:200])
            return 1.0
        d = scalars["DISTS"]
        if not np.isfinite(d):
            return 1.0
        for k in ("LPIPS", "PSNR", "SSIM", "sharp_ratio"):
            trial.set_user_attr(k, float(scalars[k]))
        return float(d)
    return objective


def run_sweep(burst_path, canon_path, n_trials=80, scale=2,
              outdir="results/eval/sweep", study_name="handheld_0003", seed=0):
    os.makedirs(outdir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    burst = pr.load_burst(burst_path)
    canon = iio.imread(canon_path).astype(np.float32) / 255.0
    target = compare.prepare_target(burst, canon, ref_overrides={"scale": scale},
                                    tile=256, max_tiles=80)

    # reference points: default multi-frame config and single-frame baseline.
    # The single frame is scale 1, so it needs its OWN registration (not the
    # cached scale-2 homography).
    import eval.registration as reg
    import eval.metrics as met
    base, _, _ = compare.score(burst, {"scale": scale}, target)
    single, _ = pr.run_single(burst)
    Rs = reg.register(np.clip(single, 0, 1) ** (1 / 2.2), canon)
    single_dists = met.tiled_metrics(Rs["mine"], Rs["canon"], Rs["valid"],
                                     tile=target["tile"], max_tiles=80)["DISTS"]
    print(f"[ref] default multi DISTS={base['DISTS']:.4f}  single-frame DISTS={single_dists:.4f}")

    storage = f"sqlite:///{os.path.join(outdir, 'study.db')}"
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.set_user_attr("default_DISTS", base["DISTS"])
    study.set_user_attr("single_DISTS", single_dists)
    study.optimize(make_objective(burst, target, scale), n_trials=n_trials,
                   show_progress_bar=False)

    # persist leaderboard CSV + best config yaml
    rows = sorted(study.trials, key=lambda t: (t.value if t.value is not None else 9))
    with open(os.path.join(outdir, "trials.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "DISTS", "LPIPS", "PSNR", "SSIM", "sharp_ratio"] + list(rows[0].params))
        for t in rows:
            w.writerow([t.number, t.value,
                        t.user_attrs.get("LPIPS"), t.user_attrs.get("PSNR"),
                        t.user_attrs.get("SSIM"), t.user_attrs.get("sharp_ratio")]
                       + [t.params.get(k) for k in rows[0].params])

    best_ov = suggest_overrides(_FrozenTrial(study.best_trial), scale)
    OmegaConf.save(OmegaConf.create(_nest(best_ov)), os.path.join(outdir, "best_config.yaml"))
    result = dict(default_DISTS=base["DISTS"], single_DISTS=single_dists,
                  best_DISTS=study.best_value, best_params=study.best_params,
                  best_user_attrs=dict(study.best_trial.user_attrs), n_trials=len(study.trials))
    with open(os.path.join(outdir, "best.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[best] DISTS={study.best_value:.4f}  (default {base['DISTS']:.4f}, single {single_dists:.4f})")
    print("[best] params:", json.dumps(study.best_params))
    return study, result


class _FrozenTrial:
    """Adapter so suggest_overrides can read a completed trial's params back."""
    def __init__(self, trial):
        self._p = trial.params
    def suggest_int(self, name, *a, **k):
        return self._p[name]
    def suggest_float(self, name, *a, **k):
        return self._p[name]
    def suggest_categorical(self, name, *a, **k):
        return self._p[name]


def _nest(flat):
    """Turn dotted-key overrides into a nested dict for a YAML config."""
    out = {}
    for key, val in flat.items():
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", required=True)
    ap.add_argument("--canon", required=True)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--outdir", default="results/eval/sweep")
    ap.add_argument("--study-name", default="handheld_0003")
    args = ap.parse_args()
    run_sweep(args.burst, args.canon, n_trials=args.trials, scale=args.scale,
              outdir=args.outdir, study_name=args.study_name)
