"""Validation for the DxO reference-eval tooling (calibrate_optics + compare_dxo).

Un-gameable round-trips against KNOWN injected degradation: calibrate must recover
the PSF / lateral-CA we put in via synthetic.apply_optics, and the per-axis compare
metrics must order known degradations correctly. No external files needed -- the
"scene" is the deterministic band-limited GT from test_reconstruction_accuracy.

Tolerances are set from observed recovery (the 1-param fits recover pure injected
degradation near-exactly; combined CA+blur leaves a little cross-talk).
"""

import numpy as np
import pytest

from eval import synthetic as syn
from eval import calibrate_optics as cal
from eval import compare_dxo as cmp
import test.test_reconstruction_accuracy as T


@pytest.fixture(scope="module")
def sharp():
    return T._ground_truth(size=512)  # deterministic clean RGB in [0.1, 0.9]


def test_calibrate_recovers_injected_ca(sharp):
    for inj in (2.0, 3.5):
        plain = syn.apply_optics(sharp, psf_sigma=0.0, ca_px=inj)
        r = cal.calibrate(plain, sharp)
        assert abs(r["ca_R"] - inj) < 0.4, f"ca_R {r['ca_R']} != +{inj}"
        assert abs(r["ca_B"] + inj) < 0.4, f"ca_B {r['ca_B']} != -{inj}"
        assert abs(r["ca_px"] - inj) < 0.4, f"ca_px {r['ca_px']} != {inj}"


def test_calibrate_recovers_injected_blur(sharp):
    for inj in (1.0, 2.5):
        plain = syn.apply_optics(sharp, psf_sigma=inj, ca_px=0.0)
        r = cal.calibrate(plain, sharp)
        assert abs(r["psf_sigma"] - inj) < 0.4, f"psf_sigma {r['psf_sigma']} != {inj}"


def test_calibrate_combined_no_confound(sharp):
    """CA and blur injected together are each recovered (the fits are ~independent)."""
    plain = syn.apply_optics(sharp, psf_sigma=2.0, ca_px=3.0)
    r = cal.calibrate(plain, sharp)
    assert abs(r["ca_px"] - 3.0) < 0.6, f"ca_px {r['ca_px']} != 3.0"
    assert abs(r["psf_sigma"] - 2.0) < 0.6, f"psf_sigma {r['psf_sigma']} != 2.0"


def test_chromatic_residual_orders_ca(sharp):
    none = cmp.chromatic_residual(sharp)["global"]
    mild = cmp.chromatic_residual(syn.apply_optics(sharp, ca_px=2.0))["global"]
    strong = cmp.chromatic_residual(syn.apply_optics(sharp, ca_px=5.0))["global"]
    assert none < mild < strong, f"CA residual not monotone: {none:.4f} {mild:.4f} {strong:.4f}"


def test_chromatic_residual_grows_with_radius(sharp):
    """Lateral CA is radial -> the outer ring must show more fringing than the inner."""
    radial = cmp.chromatic_residual(syn.apply_optics(sharp, ca_px=5.0), n_radial=4)["radial"]
    assert radial[-1] > radial[0], f"outer ring not > inner: {radial}"


def test_flat_noise_detects_noise(sharp):
    rng = np.random.default_rng(0)
    noisy = np.clip(sharp + rng.normal(0, 0.03, sharp.shape).astype(np.float32), 0, 1)
    assert np.mean(cmp.flat_noise(noisy)) > np.mean(cmp.flat_noise(sharp)) + 0.005


def test_edge_overshoot_detects_halo(sharp):
    from skimage.filters import unsharp_mask
    haloed = unsharp_mask(sharp, radius=3, amount=3.0, channel_axis=2, preserve_range=True)
    assert cmp.edge_overshoot(haloed) > cmp.edge_overshoot(sharp)


def test_compare_runs_aligned(sharp):
    """End-to-end compare on a same-grid pair (ours degraded vs a clean reference)."""
    ours = syn.apply_optics(sharp, psf_sigma=1.5, ca_px=3.0)
    res = cmp.compare(ours, sharp, register_first=False)
    assert {"ours", "dxo", "correlates"} <= set(res)
    assert res["ours"]["chromatic_residual"] > res["dxo"]["chromatic_residual"]
    assert np.isfinite(res["ours"]["sharpness"])
    assert np.isfinite(res["ours"]["flat_noise"]).all()
