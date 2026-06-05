"""Post-sweep: render default vs best config vs Canon and report the gains.

Run after eval.sweep has produced results/eval/sweep/best_config.yaml:

    python -m eval.finalize \
        --burst test_burst/Samsung_dataset/0003/Samsung \
        --canon test_burst/Samsung_dataset/0003/Canon/im.JPG
"""

import argparse
import json
import os

import numpy as np
import cv2
import imageio.v3 as iio
from omegaconf import OmegaConf

from . import pipeline_runner as pr
from . import registration as reg
from . import metrics as met


def _flatten(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _gamma(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", required=True)
    ap.add_argument("--canon", required=True)
    ap.add_argument("--best", default="results/eval/sweep/best_config.yaml")
    ap.add_argument("--outdir", default="results/eval/final")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    b = pr.load_burst(args.burst)
    canon = iio.imread(args.canon).astype(np.float32) / 255.0
    best_ov = _flatten(OmegaConf.to_container(OmegaConf.load(args.best), resolve=True))
    best_ov.setdefault("scale", 2)

    configs = {"default": {"scale": 2}, "best": best_ov}
    summary = {}
    rendered = {}
    for name, ov in configs.items():
        mine, cfg = pr.run(b, ov)
        R = reg.register(_gamma(mine), canon)
        M = met.tiled_metrics(R["mine"], R["canon"], R["valid"], tile=256, max_tiles=200)
        summary[name] = {k: M[k] for k in ["LPIPS", "DISTS", "PSNR", "SSIM", "sharp_ratio"]}
        rendered[name] = (R, mine, cfg)
        # full processed image
        iio.imwrite(os.path.join(args.outdir, f"{name}_full.png"),
                    (pr.postprocess(b, mine, cfg) * 255).astype(np.uint8))

    # worst default tile -> compare default vs best vs canon there
    Rd = rendered["default"][0]
    Md = met.tiled_metrics(Rd["mine"], Rd["canon"], Rd["valid"], tile=256, max_tiles=200)
    wi = int(np.argmax(Md["dists"]))
    y, x = Md["coords"][wi]
    Rb = rendered["best"][0]
    panels = []
    for img, lab in [(Rd["mine"], "default"), (Rb["mine"], "best"), (Rd["canon"], "canon")]:
        t = (np.clip(img[y:y + 256, x:x + 256], 0, 1) * 255).astype(np.uint8)
        bar = np.zeros((18, 256, 3), np.uint8)
        cv2.putText(bar, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.vstack([bar, t]))
    iio.imwrite(os.path.join(args.outdir, "worst_tile_default_vs_best_vs_canon.png"), np.hstack(panels))

    summary["best_config"] = best_ov
    with open(os.path.join(args.outdir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("default", "best")}, indent=2))
    print("best config:", json.dumps(best_ov))


if __name__ == "__main__":
    main()
