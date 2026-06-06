"""StreamingMerge == main(): the incremental / arbitrary-scale merge refactor.

StreamingMerge separates main()'s monolithic per-frame-analyse-then-accumulate into
a scale-free build (align + robustness + kernels per frame) and a scale-dependent
render (replay the Warp gathers + divide). These tests assert it is a behaviour-
preserving refactor:
  - render(scale) reproduces main()'s linear output at that scale (the merge is the
    same ops in the same frame order),
  - streaming A-then-B is bit-identical to building [A, B] (additive accumulation),
  - one representation renders at arbitrary scale,
  - the persisted artifact round-trips exactly,
  - the (additivity-breaking) accumulated-robustness denoiser is refused.

main()'s num has NaN at border pixels (den == 0); both paths produce the SAME NaN
mask, so equivalence is asserted on the finite pixels and the masks are checked equal.
"""

from pathlib import Path

import numpy as np
import cupy as cp
import pytest
from omegaconf import OmegaConf
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

from handheld_super_resolution.params import update_snr_config, sanitize_config
from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.streaming import StreamingMerge

H = W = 512
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _scene():
    np.random.seed(0)
    base = cp.asarray(gaussian_filter(cp.asarray(np.random.rand(H, W).astype(np.float32)), 3.0))
    base = (base - base.min()) / (base.max() - base.min()) * 0.45 + 0.05
    return cp.asnumpy(base).astype(np.float32)


def _shift(img, dx, dy):
    yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij")
    warped = map_coordinates(cp.asarray(img), cp.stack([yy - dy, xx - dx], 0), order=1, mode="reflect")
    return cp.asnumpy(warped).astype(np.float32)


def _config(**overrides):
    c = OmegaConf.load(DEFAULT_CONFIG)
    c.verbose = 0
    c.noise_model.alpha = 1e-4
    c.noise_model.beta = 1e-5
    update_snr_config(c, 30.0)
    c.noise_model.std_curve = np.linspace(1e-3, 5e-2, 1001).tolist()
    c.noise_model.diff_curve = np.linspace(1e-3, 5e-2, 1001).tolist()
    c.exif = OmegaConf.create({"cfa_pattern": [[0, 1], [1, 2]], "iso": 100, "white_balance": [1.0, 1.0, 1.0]})
    c.accumulated_robustness_denoiser.enabled = False
    for key, value in overrides.items():
        OmegaConf.update(c, key, value)
    sanitize_config(c, (H, W))
    return c


def _burst(n=3):
    ref = _scene()
    shifts = [(0.4, -0.3), (-0.6, 0.5), (0.2, 0.7), (-0.4, -0.5)][:n]
    comps = np.stack([_shift(ref, dx, dy) for dx, dy in shifts]).astype(np.float32)
    return ref, comps


def _smerge(ref, comps, cfg):
    sm = StreamingMerge(ref, cfg)
    for c in comps:
        sm.add_frame(c)
    return sm


def _same_nan_mask(a, b):
    return np.array_equal(np.isfinite(a), np.isfinite(b))


@pytest.mark.parametrize("scale", [1, 2])
def test_streaming_equals_main(scale):
    """render(scale) reproduces main()'s linear num at that scale (same merge,
    same frame order). Border-NaN mask matches; finite pixels are tightly equal."""
    ref, comps = _burst()
    num_main = cp.asnumpy(main(ref, comps, _config(scale=scale))[0])
    num_sm = cp.asnumpy(_smerge(ref, comps, _config(scale=scale)).render(scale))
    assert _same_nan_mask(num_main, num_sm), "border-NaN (den==0) masks differ"
    m = np.isfinite(num_main)
    assert np.allclose(num_main[m], num_sm[m], rtol=1e-4, atol=1e-5), \
        f"max|diff|={np.abs(num_main[m] - num_sm[m]).max():.2e}"


def test_streaming_additive():
    """Streaming: add A, render; add B, render == build [A, B] and render. The merge
    accumulates (num/den += per frame) in the same order, so this is bit-identical."""
    ref, comps = _burst(n=2)
    sm = StreamingMerge(ref, _config(scale=2))
    sm.add_frame(comps[0])
    _ = sm.render(2)                      # render mid-stream must not disturb state
    sm.add_frame(comps[1])
    streamed = cp.asnumpy(sm.render(2))
    batched = cp.asnumpy(_smerge(ref, comps, _config(scale=2)).render(2))
    assert _same_nan_mask(streamed, batched)
    m = np.isfinite(streamed)
    assert np.abs(streamed[m] - batched[m]).max() == 0.0, "streaming != batched (not additive)"


def test_streaming_arbitrary_scale():
    """One representation renders at any scale; re-rendering is stable (no drift)."""
    ref, comps = _burst()
    sm = _smerge(ref, comps, _config())
    for s in (1, 2):
        out = cp.asnumpy(sm.render(s))
        assert out.shape == (round(s * H), round(s * W), 3)
    a = cp.asnumpy(sm.render(2)); b = cp.asnumpy(sm.render(2))
    assert _same_nan_mask(a, b) and np.abs(np.nan_to_num(a) - np.nan_to_num(b)).max() == 0.0


def test_streaming_persist_round_trip(tmp_path):
    """save() the scale-free artifact -> load() -> render(scale) is bit-identical
    (no re-analysis); and a fresh scale renders from the loaded artifact."""
    ref, comps = _burst()
    sm = _smerge(ref, comps, _config(scale=2))
    path = sm.save(str(tmp_path / "burst.npz"))
    before = cp.asnumpy(sm.render(2))
    loaded = StreamingMerge.load(path)
    after = cp.asnumpy(loaded.render(2))
    assert _same_nan_mask(before, after)
    m = np.isfinite(before)
    assert np.abs(before[m] - after[m]).max() == 0.0, "persist round-trip changed the render"
    assert cp.asnumpy(loaded.render(1)).shape == (H, W, 3)  # arbitrary scale after load


def test_streaming_rejects_accumulated_denoiser():
    """The accumulated-robustness denoiser's overwrite branch breaks additive
    streaming, so StreamingMerge must refuse it rather than silently mis-merge."""
    ref, _ = _burst()
    cfg = _config()
    cfg.accumulated_robustness_denoiser.enabled = True
    with pytest.raises(ValueError):
        StreamingMerge(ref, cfg)
