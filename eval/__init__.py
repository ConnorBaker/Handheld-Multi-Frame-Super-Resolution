"""Evaluation + parameter-sweep harness for the handheld pipeline.

Modules:
    pipeline_runner  -- decode a burst once, build configs, run the pipeline / baselines
    registration     -- register the Canon ground truth to a pipeline output, crop, color-match
    metrics          -- LPIPS / DISTS / PSNR / SSIM (+ local sharpness)
    compare          -- scriptable comparison + automated failure detection
    sweep            -- Optuna parameter search minimizing a perceptual metric vs Canon

See eval/README.md and eval/NOTES.md.
"""
