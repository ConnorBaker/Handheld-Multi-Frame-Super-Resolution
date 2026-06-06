"""Integration guards for the opt-in measured-PSF RL deconvolution path.

Two opt-in, default-off features wired into the SHIPPED pipeline:
  (#1) ``merging.expose_edge_strength`` -> ``main`` returns the merge's
       structure-tensor ``l1`` edge field in ``debug_dict['edge_strength']`` at
       output resolution, WITHOUT changing the merge output;
  (#2) ``postprocessing.aberration.deblur_method == 'measured_rl'`` -> a measured-PSF
       Richardson-Lucy deconvolution, GATED by that field, replaces the bounded
       Polyblur deblur (the defringe still runs).

These assert the two invariants that make the change safe to ship: the default path
is byte-for-byte unchanged, and the new path runs / is well-formed / actually gates
(flat region ~untouched, edge sharpened). GPU (the pipeline + aberration model).
"""

from pathlib import Path

import numpy as np
import cupy as cp
import pytest
from omegaconf import OmegaConf

import test.test_reconstruction_accuracy as T
from eval.synthetic import make_synthetic_burst
from eval import pipeline_runner as pr
from handheld_super_resolution.super_resolution import main
from handheld_super_resolution import raw2rgb
from handheld_super_resolution.deconv import edge_gate_from_l1

_DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def merged():
    """A real multi-frame merge + its exposed edge-strength field (flag ON), and the
    same merge with the flag OFF, from one synthetic burst."""
    gt = T._ground_truth()
    burst, gt, _ = make_synthetic_burst(
        gt, n=T._N_FRAMES, factor=T._FACTOR, alpha=T._ALPHA, seed=T._SEED)
    cfg_off = pr.build_config(burst, {"scale": T._FACTOR})
    num_off, dbg_off = main(burst["ref_raw"], burst["raw_comp"], cfg_off)
    num_off = cp.asnumpy(num_off).copy()

    cfg_on = pr.build_config(burst, {"scale": T._FACTOR})
    cfg_on.merging.expose_edge_strength = True
    num_on, dbg_on = main(burst["ref_raw"], burst["raw_comp"], cfg_on)
    num_on = cp.asnumpy(num_on).copy()
    img = np.clip(np.nan_to_num(num_on), 0, 1).astype(np.float32)
    return dict(num_off=num_off, num_on=num_on, dbg_off=dbg_off, dbg_on=dbg_on, img=img)


def _ab_cfg(**ov):
    c = OmegaConf.load(_DEFAULT_CFG).postprocessing.aberration
    return OmegaConf.merge(c, dict(enabled=True, **ov))


# ---------------------------------------------------------------------------
# #1 expose_edge_strength
# ---------------------------------------------------------------------------

def test_edge_strength_absent_by_default(merged):
    assert "edge_strength" not in merged["dbg_off"]


def test_edge_strength_present_and_well_formed_when_on(merged):
    es = merged["dbg_on"]["edge_strength"]
    out_shape = merged["num_on"].shape[:2]
    assert es.shape == out_shape, f"edge field {es.shape} != output {out_shape}"
    assert np.isfinite(es).all(), "edge field must be finite (border inf sanitized)"
    assert es.min() >= 0.0


def test_exposing_edge_strength_does_not_change_merge(merged):
    """The flag must be observational only -- the merged image is identical."""
    assert np.array_equal(merged["num_off"], merged["num_on"])


# ---------------------------------------------------------------------------
# #2 measured_rl path
# ---------------------------------------------------------------------------

def test_polyblur_path_byte_for_byte_unchanged(merged):
    """Default (deblur_method absent) == explicit 'polyblur', and passing a gate
    does NOT change the Polyblur output (gate is for measured_rl only)."""
    img, es = merged["img"], merged["dbg_on"]["edge_strength"]
    default = raw2rgb._aberration(img, _ab_cfg())
    explicit = raw2rgb._aberration(img, _ab_cfg(deblur_method="polyblur"))
    with_gate = raw2rgb._aberration(img, _ab_cfg(), edge_strength=es)
    assert np.array_equal(default, explicit)
    assert np.array_equal(default, with_gate)


def test_measured_rl_runs_and_is_well_formed(merged):
    img, es = merged["img"], merged["dbg_on"]["edge_strength"]
    out = raw2rgb._aberration(img, _ab_cfg(deblur_method="measured_rl", sigma=1.1, iters=3),
                              edge_strength=es)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_measured_rl_unknown_method_raises(merged):
    with pytest.raises(ValueError):
        raw2rgb._aberration(merged["img"], _ab_cfg(deblur_method="bogus"))


@pytest.fixture(scope="module")
def merged_flat_plus_edge():
    """A merge from a GT that HAS a genuine flat plateau (left half, constant) and a
    sharp interior step edge (right half) -- the broadband recon GT has no flat
    region, so this fixture is needed to show the gate leaves a flat untouched."""
    from scipy.ndimage import gaussian_filter
    size = 1024
    a = np.empty((size, size), np.float32)
    a[:, : size // 2] = 0.30                                       # FLAT plateau
    fld = gaussian_filter(np.random.default_rng(0).standard_normal((size, size // 2)).astype(np.float32), 3.0)
    fld = (fld - fld.min()) / (fld.max() - fld.min())
    a[:, size // 2:] = 0.30 + 0.45 * fld                          # textured right half
    a[200:820, 760:765] = 0.85                                    # a crisp interior edge
    gt = np.ascontiguousarray(np.repeat(a[..., None], 3, 2)).astype(np.float32)
    burst, gt, _ = make_synthetic_burst(gt, n=20, factor=2, alpha=1e-3, seed=0)
    cfg = pr.build_config(burst, {"scale": 2})
    cfg.merging.expose_edge_strength = True
    num, dbg = main(burst["ref_raw"], burst["raw_comp"], cfg)
    img = np.clip(np.nan_to_num(cp.asnumpy(num)), 0, 1).astype(np.float32)
    return img, dbg["edge_strength"]


def test_measured_rl_gates_to_edges(merged_flat_plus_edge):
    """The deconv (gated by the pipeline's OWN l1) leaves a FLAT plateau essentially
    unchanged while SHARPENING an edge -- the whole point of reusing the structure
    tensor as the gate."""
    import cv2
    from eval.psf import esf_sigma
    img, es = merged_flat_plus_edge
    gate = edge_gate_from_l1(es).numpy()
    # deblur ONLY (no defringe) so we measure the deconv itself
    deb = raw2rgb._aberration(
        img, _ab_cfg(deblur_method="measured_rl", sigma=1.1, iters=5, do_defringe=False),
        edge_strength=es)
    assert np.isfinite(deb).all() and 0.0 <= deb.min() and deb.max() <= 1.0

    # FLAT plateau deep in the left half (cols 100..400): gate ~0, pixels unmoved
    flat = (slice(300, 700), slice(100, 400))
    assert gate[flat].mean() < 0.05, f"flat-plateau gate {gate[flat].mean():.4f} should be ~0"
    flat_delta = np.abs(deb[flat] - img[flat]).max()
    assert flat_delta < 5e-3, f"flat plateau moved by {flat_delta:.5f}"

    # the interior step edge near col ~762: gate fires, and the edge is sharpened
    band = slice(300, 700)
    sub = img[band, 740:785].mean(2)
    ex = 740 + int(np.argmax(np.abs(np.diff(sub.mean(0)))))
    b = slice(ex - 22, ex + 22)
    assert gate[band, b].mean() > 0.3, f"edge gate {gate[band, b].mean():.3f} too low"
    sb, _ = esf_sigma(img[band, b].mean(2))
    sa, _ = esf_sigma(deb[band, b].mean(2))
    assert sa < sb, f"edge should sharpen ({sb:.3f} -> {sa:.3f})"
