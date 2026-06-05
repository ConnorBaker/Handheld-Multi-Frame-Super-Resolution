"""Strong, un-gameable reconstruction-accuracy test for multi-frame SR.

Unlike ``test_pipeline.py`` (``corr > 0.99`` vs the reference frame -- a weak,
partly-gameable bound), this starts from a KNOWN high-resolution ground truth
``GT`` and a forward image-formation model, then asserts a *per-pixel error
budget* (PSNR + percentile abs-error) against GT. Because GT is known exactly the
metric cannot be gamed.

Forward model (REUSED from ``eval/synthetic.py`` so the test matches the eval
harness): GT --(sub-pixel shift)--> low-pass + 2x decimate (sensor/optics) -->
Bayer mosaic --> (+ Poisson-Gaussian noise). We run the pipeline at ``scale=2``
to super-resolve back to GT resolution and compare on an interior crop (borders
excluded to avoid warp/decimation edge effects).

The real proof multi-frame SR *works*: it beats the single-frame and bilinear
baselines (``run_single`` / ``run_bilinear``) on the SAME per-pixel metric.

GT design: a band-limited random field (gaussian-blurred white noise kept below
the LR Nyquist so SR can actually recover it) + crisp geometric edges -- dense,
broadband, NON-periodic high-frequency content. Deliberately NO periodic
sinusoidal gratings (a grating with period ~ the block-matching tile/search size
is adversarial: many equally-good matches one period apart). Higher sensor noise
(``_ALPHA`` = 1e-3) is used because multi-frame SR's core strength is denoising by
averaging aligned frames, so the win over single-frame/bilinear is clearest there.

DETERMINISM: the pipeline is bit-deterministic for fixed input. (A Warp merge
cross-stream race previously made the output wobble ~0.18/px run-to-run -- the
merge's read-modify-write into num/den ran on Warp's private CUDA stream and
raced default-stream work on the shared accumulators. Fixed in ``warp_ops.py``;
see ``REFACTORING_PLAN.md``.) ``test_reconstruction_deterministic`` guards against
that race returning. Thresholds are empirical with margin: observed interior PSNR
~37.2 dB multi-frame, ~32.2 single / bilinear (margin ~+5 dB); abs-error
mean ~0.012, p99 ~0.044.
"""

import numpy as np
import cv2
import pytest
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as _psnr

# Reuse the eval harness: same forward model + same ready-made baselines.
from eval.synthetic import make_synthetic_burst
from eval import pipeline_runner as pr

# --- synthetic-GT / forward-model parameters (fixed; keep the test fast) ---
_GT_SIZE = 1024          # GT side; LR = 512 (the coarsest pyramid level needs this)
_FACTOR = 2              # SR / decimation factor (pipeline scale=2)
_N_FRAMES = 20           # burst size
_ALPHA = 1e-3            # sensor noise (Poisson-Gaussian); noisy enough that multi-frame's
                         # denoising win dominates (see docstring)
_SEED = 0                # fixed seed -> reproducible GT, shifts, noise
_FIELD_SIGMA = 2.0       # blur of the random field: keeps energy below the LR Nyquist
_CROP = 48               # interior crop (px) to exclude border/warp/decimation artifacts
_N_DETERMINISM_RUNS = 3  # repeated reconstructions -- must be bit-identical (race guard)

# --- empirical assertion thresholds (see module docstring for observed values) ---
_PSNR_MIN = 34.5         # observed multi-frame ~37.2 dB; ~2.7 dB cushion
_MARGIN_DB = 2.0         # observed best-margin ~+5 dB vs single/bilinear
_AE_P99_MAX = 0.07       # observed p99 ~0.044
_AE_MEAN_MAX = 0.018     # observed mean ~0.012


def _ground_truth(seed=_SEED, size=_GT_SIZE):
    """A known high-res GT with genuine, *recoverable* high-frequency content.

    = band-limited random field (gaussian-blurred white noise, energy kept below
    the LR Nyquist so SR can actually recover it; being broadband it contains a
    rich, NON-periodic spread of frequencies) + crisp geometric edges (a
    rectangle and a disk -> broadband edges, the demosaic/SR stress). Returned as
    float32 RGB in [0.1, 0.9] (a sane, in-gamut linear range).

    DELIBERATELY no periodic sinusoidal gratings: a grating with period ~ the
    block-matching tile/search size is ADVERSARIAL for block matching (many
    equally-good matches one period apart -> ambiguous alignment). The broadband
    field already supplies dense, edge-rich, non-periodic high-frequency detail.
    """
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), _FIELD_SIGMA)
    field = (field - field.min()) / (field.max() - field.min())

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    edges = np.zeros((size, size), np.float32)
    edges[200:480, 150:520] += 0.30                                   # rectangle
    edges[(yy - 700) ** 2 + (xx - 640) ** 2 < 150 ** 2] += 0.28       # disk

    img = 0.7 * field + edges
    img = (img - img.min()) / (img.max() - img.min())
    img = 0.1 + 0.8 * img
    return np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2)).astype(np.float32)


def _interior_psnr(out, gt, m=_CROP):
    s = slice(m, -m)
    return float(_psnr(gt[s, s].clip(0, 1), out[s, s].clip(0, 1), data_range=1.0))


def _interior_abs_error(out, gt, m=_CROP):
    s = slice(m, -m)
    e = np.abs(gt[s, s].clip(0, 1) - out[s, s].clip(0, 1))
    return float(e.mean()), float(np.percentile(e, 99)), float(e.max())


def _upscale(img, size=_GT_SIZE):
    """Bilinearly upscale a native-res baseline to GT resolution for comparison."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def reconstruction():
    """Build the synthetic burst once and run multi (xN, must be identical) + baselines."""
    gt = _ground_truth()
    burst, gt, _shifts = make_synthetic_burst(
        gt, n=_N_FRAMES, factor=_FACTOR, alpha=_ALPHA, seed=_SEED)

    # The pipeline is deterministic (merge race fixed), so these runs MUST be
    # bit-identical; we keep all of them for the determinism regression guard.
    outs = [pr.run(burst, {"scale": _FACTOR})[0] for _ in range(_N_DETERMINISM_RUNS)]
    out = outs[0]

    # Baselines (deterministic): the same steerable-kernel single-frame merge, and
    # a plain bilinear demosaic of the reference frame -- both at native res, then
    # bilinearly upscaled to GT resolution for a fair per-pixel comparison.
    single, _ = pr.run_single(burst)
    bilinear = pr.run_bilinear(burst)

    return dict(
        gt=gt, outs=outs, out=out,
        psnr=_interior_psnr(out, gt),
        ae=_interior_abs_error(out, gt),
        psnr_single=_interior_psnr(_upscale(single), gt),
        psnr_bilinear=_interior_psnr(_upscale(bilinear), gt),
    )


def test_reconstruction_deterministic(reconstruction):
    """The multi-frame output must be BIT-IDENTICAL for fixed input.

    Regression guard for the Warp merge cross-stream race (warp_ops.py): a nonzero
    diff means the merge's accumulation is again racing default-stream work.
    """
    outs = reconstruction["outs"]
    for i in range(1, len(outs)):
        d = float(np.abs(outs[i] - outs[0]).max())
        assert d == 0.0, (
            f"multi-frame run {i} differs from run 0 by {d:.2e} for identical input "
            f"-- the Warp merge stream race may have returned (see warp_ops.py "
            f"merge_accumulate_cupy / merge_ref_accumulate_cupy)")


def test_multiframe_psnr_meets_budget(reconstruction, capsys):
    """Per-pixel error budget vs KNOWN GT: PSNR floor + abs-error budget."""
    r = reconstruction
    mean_ae, p99_ae, max_ae = r["ae"]
    with capsys.disabled():
        print(f"\n[reconstruction-accuracy] interior PSNR vs GT (crop={_CROP}px, "
              f"alpha={_ALPHA}, N={_N_FRAMES}):")
        print(f"  multi-frame (deterministic): {r['psnr']:.2f} dB")
        print(f"  single-frame baseline      : {r['psnr_single']:.2f} dB")
        print(f"  bilinear   baseline        : {r['psnr_bilinear']:.2f} dB")
        print(f"  abs-error: mean={mean_ae:.4f} p99={p99_ae:.4f} max={max_ae:.4f}")

    assert r["psnr"] > _PSNR_MIN, (
        f"multi-frame PSNR {r['psnr']:.2f} dB <= floor {_PSNR_MIN} dB "
        f"(empirical observed ~37.2 dB)")
    assert p99_ae < _AE_P99_MAX, (
        f"99th-pct abs error {p99_ae:.4f} >= budget {_AE_P99_MAX} (observed ~0.044)")
    assert mean_ae < _AE_MEAN_MAX, (
        f"mean abs error {mean_ae:.4f} >= budget {_AE_MEAN_MAX} (observed ~0.012)")


def test_multiframe_beats_single_and_bilinear(reconstruction, capsys):
    """The real proof multi-frame SR works: it beats both baselines on PSNR vs GT."""
    r = reconstruction
    margin_single = r["psnr"] - r["psnr_single"]
    margin_bilinear = r["psnr"] - r["psnr_bilinear"]
    with capsys.disabled():
        print(f"\n[reconstruction-accuracy] multi-frame margin: "
              f"+{margin_single:.2f} dB vs single, +{margin_bilinear:.2f} dB vs bilinear")

    assert margin_single > _MARGIN_DB, (
        f"multi-frame beats single-frame by only {margin_single:.2f} dB "
        f"(< {_MARGIN_DB} dB); multi-frame SR is not adding detail. "
        f"multi={r['psnr']:.2f} single={r['psnr_single']:.2f}")
    assert margin_bilinear > _MARGIN_DB, (
        f"multi-frame beats bilinear by only {margin_bilinear:.2f} dB (< {_MARGIN_DB} dB). "
        f"multi={r['psnr']:.2f} bilinear={r['psnr_bilinear']:.2f}")


def test_multiframe_output_well_formed(reconstruction):
    """Output is finite, in-gamut and at the super-resolved shape."""
    r = reconstruction
    out, gt = r["out"], r["gt"]
    assert out.shape == gt.shape == (_GT_SIZE, _GT_SIZE, 3)
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0
