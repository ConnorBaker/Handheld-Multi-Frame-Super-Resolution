"""CLI: run our pipeline on a raw burst and compare per-axis to a DxO reference.

Reproduces the ours-vs-DxO comparison in one command: load the burst (Sony ARW or
DNG), run the merge (+ optional post-processing), register/colour-match to the DxO
reference, and print the per-axis table from ``compare_dxo`` (chromatic residual,
sharpness + overshoot/halo, flat-patch noise, plus DISTS/PSNR/SSIM correlates).

Examples:
    python -m eval.compare_burst <burst_dir> DSC00034_DxO-reference.dng
    python -m eval.compare_burst <burst_dir> ref.dng --postproc none --frames 10
    python -m eval.compare_burst <burst_dir> ref.dng --scale 2 --postproc aberration

The reference is a DxO export of the SAME reference frame -- a linear .dng
(read demosaiced via rawpy) or a .tif/.png. Export with distortion OFF so the
geometry matches ours (near-identity homography). See ``dxo-reference-eval``.
"""

import argparse
from pathlib import Path

from . import pipeline_runner as pr
from . import compare_dxo as cmp
from .calibrate_optics import _load


def run_comparison(burst_dir, reference, scale=1, postproc="aberration", frames=None):
    """Run the pipeline on ``burst_dir`` and return ``compare_dxo.compare`` vs ``reference``."""
    burst_dir = Path(burst_dir)
    if any(burst_dir.glob("*.[Aa][Rr][Ww]")):
        burst = pr.load_burst_arw(burst_dir, max_frames=frames)
        ours, cfg = pr.run(burst, {"scale": scale})
    else:
        burst = pr.load_burst(burst_dir)
        ours, cfg = pr.run(burst, {"scale": scale}, n_frames=frames)
    if postproc != "none":
        from handheld_super_resolution import raw2rgb
        post = cfg.postprocessing
        ours = (raw2rgb._aberration(ours, post.aberration) if postproc == "aberration"
                else raw2rgb._deblur(ours, post.deblur))
    return cmp.compare(ours, _load(str(reference)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst", help="burst directory (.ARW or .dng frames)")
    ap.add_argument("reference", help="DxO reference of the same frame (.dng / .tif / .png)")
    ap.add_argument("--scale", type=int, default=1, help="SR factor (default 1 = native res)")
    ap.add_argument("--postproc", choices=["none", "deblur", "aberration"], default="aberration",
                    help="post-processing applied to our merge before comparing (default aberration)")
    ap.add_argument("--frames", type=int, default=None, help="use only the first N frames")
    args = ap.parse_args()

    print(f"[compare_burst] {args.burst}  vs  {args.reference}  "
          f"(scale={args.scale}, postproc={args.postproc}, frames={args.frames or 'all'})")
    cmp._print(run_comparison(args.burst, args.reference, args.scale, args.postproc, args.frames))


if __name__ == "__main__":
    main()
