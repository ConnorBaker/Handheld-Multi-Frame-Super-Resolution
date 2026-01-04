from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import exifread
import rawpy
from torch import Tensor
import torch
from jaxtyping import Float32
import warnings
from typing import Any, Self

from . import raw2rgb
from .utils import DEFAULT_NUMPY_FLOAT_TYPE
from .utils_dng import PHOTO_INTER, SUPPORTED


def read_raw_file(raw_path: str) -> npt.NDArray[np.uint8]:
    """Helper function to read a single raw file."""
    with rawpy.imread(raw_path) as rawObject:
        # return rawObject.raw_image.copy()
        return rawObject.raw_image_visible.copy()


def load_burst_parallel(
    burst_path: str, glob: str
) -> tuple[str, rawpy.RawPy, npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Parallel version of load_burst using multiprocessing.

    Parameters
    ----------
    burst_path : Path or str
        Path of the folder containing the globbed files
    """

    raw_path_list = [str(p) for p in Path(burst_path).glob(glob)]
    assert len(raw_path_list) != 0, (
        f"At least one file matching {glob} must be present in the burst folder."
    )

    # Reference image (still read separately for metadata extraction)
    # Use the first frame as the reference.
    raw_path = raw_path_list[0]
    raw: rawpy.RawPy = rawpy.imread(raw_path)
    ref_raw: npt.NDArray[np.uint8] = raw.raw_image_visible.copy()

    # Read companion images in parallel
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as pool:
        raw_comp: npt.NDArray[np.uint8] = np.array(
            list(pool.map(read_raw_file, raw_path_list[1:]))
        )

    return raw_path, raw, ref_raw, raw_comp


@dataclass(kw_only=True, slots=True)
class Burst:
    """
    Represents a burst of photos.
    """

    raw: rawpy.RawPy
    """The reference image as a RawPy handle"""

    ref_raw: npt.NDArray[np.uint8]
    """The reference image"""

    raw_comp: npt.NDArray[np.uint8]
    """All frames in the burst, excluding the reference image"""

    iso: int
    """The ISO of the reference frame"""

    tags: dict[str, Any]
    """The EXIF tags of the reference frame"""

    cfa: npt.NDArray[np.uint8]
    """The Color Filter Array of the reference frame"""

    xyz2cam: npt.NDArray[np.float32]
    """The xyz2cam of the reference frame"""

    white_balance: npt.NDArray[np.float32]
    """The white balance of the reference frame"""

    ref_path: str
    """The path to the reference frame"""

    cam2rgb: Float32[Tensor, "3 3"]
    """The cam2rgb of the reference frame"""

    @classmethod
    def load(cls, burst_path: str, glob: str) -> Self:
        raw_path, raw, ref_raw, raw_comp = load_burst_parallel(burst_path, glob)

        #### Reading tags of the reference image
        xyz2cam: npt.NDArray[np.float32] = np.array(raw.rgb_xyz_matrix[:3])

        # reading exifs for white level, black leve and CFA
        with open(raw_path, "rb") as raw_file:
            tags = exifread.process_file(raw_file)

        if "Image PhotometricInterpretation" in tags.keys():
            photo_inter = tags["Image PhotometricInterpretation"].values[0]
            if photo_inter not in SUPPORTED:
                warnings.warn(
                    "The input images have a photometric interpretation "
                    'of type "{}", but only {} are supprted.'.format(
                        PHOTO_INTER[photo_inter],
                        str([PHOTO_INTER[i] for i in SUPPORTED]),
                    )
                )

        else:
            warnings.warn(
                "PhotometricInterpretation could not be found in image tags. "
                "Please ensure that it is one of {}".format(
                    str([PHOTO_INTER[i] for i in SUPPORTED])
                )
            )

        white_level = int(raw.white_level)  # there is only one white level
        # exifread method is inconsistent because camera manufacters can put
        # this under many different tags.

        black_levels: npt.NDArray[np.float32] = np.array(raw.black_level_per_channel)

        white_balance: npt.NDArray[np.float32] = np.array(raw.camera_whitebalance)

        # copying to ensure contiguity of the array
        cfa: npt.NDArray[np.uint8] = raw.raw_pattern.copy()
        # Rawpy gives channel 3 to the second green channel. Setting both greens to 1
        cfa[cfa == 3] = 1

        if "EXIF ISOSpeedRatings" in tags.keys():
            iso = int(str(tags["EXIF ISOSpeedRatings"]))
        elif "Image ISOSpeedRatings" in tags.keys():
            iso = int(str(tags["Image ISOSpeedRatings"]))
        else:
            raise AttributeError(
                "ISO value could not be found in both EXIF and Image type."
            )

        # Clipping ISO to 100 from below
        iso = max(100, min(3200, iso))

        #### Performing whitebalance
        assert ref_raw.dtype == raw_comp.dtype, (
            f"Reference and comp images should have the same data type, got {ref_raw.dtype} and {raw_comp.dtype}."
        )

        if np.issubdtype(ref_raw.dtype, np.integer):
            ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
            raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
            for i in range(2):
                for j in range(2):
                    channel: np.uint8 = cfa[i, j]
                    k: np.float32 = white_balance[channel] / white_balance[1]

                    ref_raw[i::2, j::2] -= black_levels[channel]
                    ref_raw[i::2, j::2] /= white_level - black_levels[channel]
                    ref_raw[i::2, j::2] *= k

                    raw_comp[:, i::2, j::2] -= black_levels[channel]
                    raw_comp[:, i::2, j::2] /= white_level - black_levels[channel]
                    raw_comp[:, i::2, j::2] *= k
        else:
            warnings.warn(
                "Input DNG images are not in integer format: is the input valid RAW data?"
            )

        cam2rgb = raw2rgb.get_cam2rgb(torch.as_tensor(xyz2cam, device="cuda"))

        return cls(
            raw=raw,
            ref_raw=ref_raw,
            raw_comp=raw_comp,
            iso=iso,
            tags=tags,
            cfa=cfa,
            xyz2cam=xyz2cam,
            white_balance=white_balance,
            ref_path=raw_path,
            cam2rgb=cam2rgb,
        )
