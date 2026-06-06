"""Noise-profile lookup (darktable a/b -> pipeline alpha/beta).

Verifies the bundled a6300 profile resolves to the values measured by hand from
darktable's data, scales monotonically with ISO, interpolates between profiles,
clamps out of range, and yields a sane run_fast_MC curve.
"""

import numpy as np
import pytest

from eval import noise_profiles as npf


def test_a6300_iso200_matches_darktable():
    # averaged per-channel a/b for ILCE-6300 @ ISO200 (darktable noiseprofiles.json)
    alpha, beta = npf.lookup("ILCE-6300", 200)
    assert alpha == pytest.approx(8.319e-6, rel=0.01), alpha
    assert beta == pytest.approx(7.130e-9, rel=0.01), beta


def test_noise_increases_with_iso():
    a200 = npf.lookup("ILCE-6300", 200)[0]
    a800 = npf.lookup("ILCE-6300", 800)[0]
    a3200 = npf.lookup("ILCE-6300", 3200)[0]
    assert a200 < a800 < a3200


def test_iso_interpolates_between_profiles():
    a200 = npf.lookup("ILCE-6300", 200)[0]
    a250 = npf.lookup("ILCE-6300", 250)[0]
    a225 = npf.lookup("ILCE-6300", 225)[0]
    assert a200 < a225 < a250  # 225 has no profile -> interpolated


def test_iso_clamped_out_of_range():
    assert npf.lookup("ILCE-6300", 50) == npf.lookup("ILCE-6300", 100)      # below min
    assert npf.lookup("ILCE-6300", 999999) == npf.lookup("ILCE-6300", 51200)  # above max


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        npf.lookup("NOPE-9000", 200)


def test_params_make_a_sane_noise_curve():
    from handheld_super_resolution.fast_monte_carlo import run_fast_MC
    alpha, beta = npf.lookup("ILCE-6300", 200)
    std_curve, _ = run_fast_MC(alpha, beta)
    sc = np.asarray(std_curve)
    assert np.isfinite(sc).all()
    assert sc.min() >= 0 and 0 < sc.max() < 0.1   # clean APS-C ISO200: well under 10% full-scale
