# `eval/` — comparison + parameter-sweep harness

Automated quality evaluation of the handheld pipeline against the **Canon
ground-truth** images in `test_burst/Samsung_dataset/<id>/Canon/`, plus an
**Optuna** parameter sweep that optimises a perceptual metric, and automated
**failure detection** so we can discover where the algorithm / a given config
breaks down.

## Install

```bash
micromamba run -n handheld pip install -r eval/requirements.txt
```

(`optuna`, `lpips`, `DISTS-pytorch`. Note: installing these may upgrade torch —
see `NOTES.md`.)

## Modules

| module | what it does |
|---|---|
| `pipeline_runner.py` | Decode a burst once (`load_burst`); run the full pipeline (`run`), single-frame baseline (`run_single`), bilinear baseline (`run_bilinear`); `postprocess` for display. All return float32 RGB in [0,1]. |
| `registration.py` | Register a pipeline output into the Canon frame (SIFT + RANSAC homography at a downscaled working res, rescaled to full res), crop the overlap, color-match. |
| `metrics.py` | LPIPS, DISTS, PSNR, SSIM, Laplacian sharpness — globally and per 256² tile (fixed-grid via `coords` for cross-config consistency). |
| `compare.py` | `evaluate` (one config vs Canon), `detect_failures` (Canon-anchored multi-vs-single per tile), `detect_regressions` (registration-free multi-vs-single sharpness), `save_report` (heatmaps + worst-tile montages + JSON), and the cached `prepare_target`/`score` fast path for sweeps. |
| `sweep.py` | Optuna study minimising DISTS vs Canon over alignment / robustness / merge params. |

## Usage

Parameter sweep (writes `study.db`, `trials.csv`, `best_config.yaml`, `best.json`):

```bash
python -m eval.sweep \
  --burst test_burst/Samsung_dataset/0003/Samsung \
  --canon test_burst/Samsung_dataset/0003/Canon/im.JPG \
  --trials 80 --scale 2
```

One-off comparison report (heatmaps + worst tiles + summary) for a config:

```python
from eval import pipeline_runner as pr, compare
import imageio.v3 as iio, numpy as np
b = pr.load_burst("test_burst/Samsung_dataset/0003/Samsung")
canon = iio.imread("test_burst/Samsung_dataset/0003/Canon/im.JPG").astype(np.float32)/255
compare.save_report(b, canon, {"scale": 2}, "results/eval/baseline")
```

Run the pipeline with the swept best config:

```bash
python run_handheld.py --impath <burst> --outpath out.png --config results/eval/sweep/best_config.yaml
```

## Metric & registration notes

- **DISTS is the primary objective** — it is texture-aware and tolerant of small
  residual misalignment, which matters because Canon vs Samsung differ in
  sensor / lens / viewpoint and a single homography cannot model parallax on
  non-planar foreground. LPIPS/PSNR/SSIM are recorded but are more
  alignment-sensitive.
- Outputs are **color-matched** (per-channel mean/std) to Canon before scoring,
  so camera-color differences don't dominate the metric.
- The sweep **caches the registration homography** from a reference run and
  reuses it for every trial (the output geometry barely changes across configs),
  so each trial is just pipeline + warp + tiled metrics.

## Failure detection (two complementary signals)

1. **Canon-anchored** (`detect_failures`): per-tile DISTS of the multi-frame
   output *and* the single-frame baseline vs Canon. `delta = multi − single > 0`
   ⇒ multi-frame made that tile perceptually **worse** than one frame.
2. **Registration-free** (`detect_regressions`): per-tile sharpness ratio
   multi/single at scale 1 (same FOV, no homography). Fast, runs on any burst,
   but conflates benign denoising of flat/noisy regions with real detail loss —
   use (1) to confirm.

Result images/JSON land in `results/eval/` (git-ignored; large).
