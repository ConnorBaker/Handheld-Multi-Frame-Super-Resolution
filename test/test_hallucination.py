"""Guards for eval.hallucination's real-vs-hallucinated decomposition.

All synthetic + CPU (no DNGs, no GPU): we CONSTRUCT stacks with a known split of
cross-frame-consistent (REAL) vs per-frame-independent (INVENTED) high-frequency
content and assert the decomposition recovers it -- the property the whole DxO
analysis rests on. Thresholds are set from the analytic expectation with margin,
per the verify-everything policy: a test that could pass on a broken estimator is
worthless.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from eval.hallucination import (
    pcc_shift, consensus_decomp, resampling_floor, inconsistency_structure,
    extract_aligned_roi,
)


def _field(shape, smooth, rms, rng):
    """Band-limited random field (smoothed white noise) normalized to `rms`."""
    f = gaussian_filter(rng.standard_normal(shape).astype(np.float32), smooth)
    return (f / (f.std() + 1e-9) * rms).astype(np.float32)


def test_pcc_shift_roundtrip():
    """Recovered shift, applied via ndshift, registers mov back onto ref."""
    rng = np.random.default_rng(0)
    ref = _field((400, 400), 2.0, 0.15, rng) + 0.3   # rms/level of a real gamma luma
    mov = ndshift(ref, (3.4, -2.1), order=3, mode="reflect")
    dy, dx, err = pcc_shift(ref, mov, upsample=20)
    assert abs(dy - (-3.4)) < 0.1 and abs(dx - 2.1) < 0.1   # correction = -applied shift
    s = slice(40, -40)
    re = ndshift(mov, (dy, dx), order=3, mode="reflect")
    resid = np.abs(re[s, s] - ref[s, s]).mean()
    uncorrected = np.abs(mov[s, s] - ref[s, s]).mean()
    assert resid < 0.01                       # double-cubic interpolation floor on a luma-level field
    assert resid < 0.1 * uncorrected          # and alignment removes >90% of the displacement error


def test_all_consistent_is_all_real():
    """Identical frames (no per-frame component) -> real_fraction ~ 1."""
    rng = np.random.default_rng(1)
    one = _field((160, 160), 1.0, 0.1, rng) + 0.3
    rois = np.repeat(one[None], 24, axis=0)
    assert consensus_decomp(rois)["real_fraction"] > 0.99


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)])
def test_recovers_known_consistent_invented_split(a, b):
    """frame_f = C + I_f, C shared (REAL), I_f independent per frame (INVENTED),
    drawn from the same spectrum so the high-pass scales both equally. Then
    real_fraction must match the injected power ratio a^2/(a^2+b^2)."""
    rng = np.random.default_rng(2)
    N, sz = 64, 160
    C = _field((sz, sz), 1.0, a, rng)
    rois = np.stack([C + _field((sz, sz), 1.0, b, rng) for _ in range(N)])
    expected = a * a / (a * a + b * b)
    rf = consensus_decomp(rois)["real_fraction"]
    # finite N leaks b^2/N of the invented power into the consistent mean
    assert abs(rf - expected) < 0.05, f"real_fraction {rf:.3f} vs expected {expected:.3f}"


def test_resampling_floor_is_high():
    """Resampling-only control (one image, random sub-pixel round-trips) must score
    near 1: the low real_fractions on real data are genuine, not interpolation."""
    rng = np.random.default_rng(3)
    base = _field((256, 256), 1.5, 0.1, rng) + 0.3
    assert resampling_floor(base, n=41, seed=0) > 0.9


def test_inconsistency_separates_noise_from_texture():
    """The inconsistent component's lag-1 autocorrelation tells residual NOISE
    (white -> ~0) from synthesized TEXTURE (structured -> >0)."""
    rng = np.random.default_rng(4)
    flat = 0.3 + np.zeros((128, 128), np.float32)
    white = np.stack([flat + _field((128, 128), 0.01, 0.05, rng) for _ in range(20)])
    struct = np.stack([flat + _field((128, 128), 2.5, 0.05, rng) for _ in range(20)])
    ac_white = inconsistency_structure(white)["lag1_autocorr"]
    ac_struct = inconsistency_structure(struct)["lag1_autocorr"]
    assert ac_white < 0.15, f"white noise autocorr {ac_white:.2f} should be ~0"
    assert ac_struct > 0.25, f"structured texture autocorr {ac_struct:.2f} should be >0"
    assert ac_struct > ac_white + 0.2


def test_extract_aligned_roi_registers_known_shifts():
    """A stack of known-shifted copies registers back to agreement (the per-ROI
    matchTemplate + sub-pixel path), far tighter than the unaligned crop."""
    rng = np.random.default_rng(5)
    base = _field((600, 600), 2.0, 0.12, rng) + 0.3
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-3, 3, 2)) for _ in range(11)]
    stack = np.stack([ndshift(base, s, order=3, mode="reflect") for s in shifts]).astype(np.float32)
    rois, info = extract_aligned_roi(stack, 300, 300, size=200, search=40)
    s = slice(30, -30)
    aligned_std = rois[:, s, s].std(0).mean()
    naive = stack[:, 200:400, 200:400]            # same box, no per-frame alignment
    naive_std = naive[:, s, s].std(0).mean()
    assert aligned_std < 0.2 * naive_std          # alignment collapses the cross-frame spread
    assert all(d["peak"] > 0.9 for d in info)     # clean NCC match on every frame
