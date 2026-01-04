import os
import glob

import gc
import time
from itertools import product
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torchvision import io
from torchvision.transforms.v2 import functional as F

from handheld_super_resolution.super_resolution import setup, run_pipeline
from handheld_super_resolution.utils import getTime


def run(ref_path, ref_raw, raw_comp, xyz2cam, tags, config):
    tuning = config.robustness.tuning

    outpath = Path() / f"t-{tuning.t:.2f}-s1-{tuning.s1:.2f}-s2-{tuning.s2:02}-Mt-{tuning.Mt:.2f}.jpg"

    handheld_output, debug_dict = run_pipeline(
        ref_path, ref_raw, raw_comp, xyz2cam, tags, config
    )
    handheld_output = handheld_output.nan_to_num()
    handheld_output = handheld_output.clip(0.0, 1.0)

    #### Save images ####
    if config.verbose >= 3:
        print("\nSaving images ---------\n")
        t1 = time.perf_counter()

    # Exepcted tensor should have channels first
    # NOTE: NVJPEG won't work with images as large as we're processing, so move to CPU first.
    handheld_output = (
        F.to_dtype(handheld_output, dtype=torch.uint8, scale=True)
        .permute([2, 0, 1])
        .cpu()
    )
    io.write_jpeg(handheld_output, str(outpath))

    if config.verbose >= 3:
        t1 = getTime(t1, "- Super-resolved image saved")

    if (
        config.robustness.save_mask
        and debug_dict.get("accumulated robustness", None) is not None
    ):
        rob = torch.tensor(debug_dict["accumulated robustness"]) / len(raw_comp)
        # rob = np.repeat(rob[..., None], 3, axis=-1)
        # Upscale NN to output scale
        # handheld_output is now CHW
        # Make sure rob is 1HW
        # rob = F.resize(rob[None, ...], [handheld_output.shape[1], handheld_output.shape[2]])
        rob = F.to_dtype(rob[None, ...], dtype=torch.uint8, scale=True).cpu()
        io.write_jpeg(rob, str(outpath.with_suffix(".rob.jpg")))
        # rob = cv2.resize(rob, (handheld_output.shape[1], handheld_output.shape[0]), interpolation=cv2.INTER_NEAREST)
        # imsave(Path(args.outpath).with_suffix('.rob.jpg'), img_as_ubyte(rob))

    if config.verbose >= 3:
        t1 = getTime(t1, "- Accumulated robustness saved")


def main():
    # Load the default and do re-usable setup
    config = OmegaConf.load("configs/ica.yaml")

    t1 = time.perf_counter()

    with torch.no_grad():
        ref_path, ref_raw, raw_comp, xyz2cam, tags, config = setup(
            str(Path.home() / "Pictures" / "ARW" / "2025-09-06" / "011"), config
        )
        t1 = getTime(t1, "Ran re-usable setup")

        for t, s1, s2, Mt in product(
            (0.04 * (n + 1) for n in range(8)),
            (0.25 * (n + 1)  for n in range(8)),
            (n + 6 for n in range(8)),
            (0.2 * (n + 1) for n in range(8)),
        ):
            OmegaConf.update(config, "robustness.tuning.t", t)
            OmegaConf.update(config, "robustness.tuning.s1", s1)
            OmegaConf.update(config, "robustness.tuning.s2", s2)
            OmegaConf.update(config, "robustness.tuning.Mt", Mt)
            run(ref_path, ref_raw, raw_comp, xyz2cam, tags, config)
            t1 = getTime(t1, f"Ran test with {config.robustness.tuning}")
            gc.collect()
            torch.cuda.empty_cache()

    # robustness:
    #     enabled: true
    #     save_mask: true
    #     tuning:
    #         t: 0.12
    #         s1: 2
    #         s2: 12
    #         Mt: 0.8


if __name__ == "__main__":
    main()
