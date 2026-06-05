"""Property-based (Hypothesis) tests for the CuPy / NVIDIA-Warp kernel ports.

These complement the example-based equivalence tests in
``test/test_kernel_equivalence.py``. Two flavours of property are exercised:

(A) INVARIANTS that must hold no matter whether any reference implementation is
    correct (e.g. den >= 0, symmetric/PSD covariances, robustness in [0, 1],
    block-matching optimality). These are strictly stronger than equivalence:
    they cannot share a bug with the reference because they don't consult it.

(B) EQUIVALENCE between the production port and the preserved reference kernel,
    fuzzed across the valid input space with Hypothesis shrinking, so a failure
    minimises to the smallest offending shape/values rather than the one happy
    path a parametrize tuple happened to pick.

Conventions/tolerances mirror ``test_kernel_equivalence.py``: the GRBG CFA, the
``M Mᵀ + εI`` SPD covariance trick (``_pd_covs``), rtol=1e-3/atol=1e-4 for the
floating-point merge/upscale equivalences, and the cov grid is grey-resolution
(``lr//2`` in bayer mode, full ``lr`` in grey mode). Each GPU launch is ~ms, so
``max_examples`` is kept small and arrays tiny (LR up to ~48x48).
"""

import numpy as np
import cupy as cp
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

import handheld_super_resolution.cupy_ops as ops
from . import reference_kernels as ref

CFA = np.array([[1, 0], [2, 1]])  # GRBG, same constant as test_kernel_equivalence

RTOL, ATOL = 1e-3, 1e-4

# GPU kernel launches dominate runtime (~ms each) and Hypothesis' deadline /
# data-generation health checks aren't meaningful for that cost model, so we
# disable both and cap the example count. function_scoped_fixture isn't used
# here but the filter keeps the settings reusable.
PROP_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)
EQ_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Helpers (shared with the example tests' conventions)
# ---------------------------------------------------------------------------
def _pd_covs_from(M):
    """Per-pixel symmetric positive-definite 2x2 cov grid from a draw matrix.

    Same ``M·Mᵀ + 0.5·I`` trick as ``test_kernel_equivalence._pd_covs`` (the
    ``+0.5 I`` keeps it well-conditioned / invertible), but the entries of M are
    fuzzed by Hypothesis instead of seeded RNG.
    """
    M = cp.asarray(M, dtype=cp.float32)
    cov = cp.matmul(M, cp.swapaxes(M, -1, -2)) + 0.5 * cp.eye(2, dtype=cp.float32)
    return cp.ascontiguousarray(cov.astype(cp.float32))


# bounded, finite, non-pathological scalar floats (value itself is the fuzz axis)
def _floats(lo, hi):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


# Array *content* is built from a Hypothesis-drawn seed via a seeded RNG rather
# than drawing thousands of floats element-by-element: per-element st.lists is
# both unsupported above ~8k elements and pathologically slow, whereas a single
# drawn int still varies content across examples (and shrinks) while the
# meaningful, bug-finding axes -- shapes and the discrete config knobs
# (bayer/iso/scale/tile/law/rden) -- are drawn directly so Hypothesis minimises
# them on failure. Each kernel gets its own RNG stream off the same seed.
def _seeds():
    return st.integers(min_value=0, max_value=2**31 - 1)


def _u(rng, shape, lo, hi):
    """Contiguous float32 cupy array of `shape`, uniform in [lo, hi)."""
    a = (rng.random(shape).astype(np.float32) * np.float32(hi - lo) + np.float32(lo))
    return cp.ascontiguousarray(cp.asarray(a.astype(np.float32)))


def _normal(rng, shape):
    """Contiguous float32 cupy array of `shape`, standard normal."""
    return cp.ascontiguousarray(cp.asarray(rng.standard_normal(shape).astype(np.float32)))


# ---------------------------------------------------------------------------
# Composite strategies producing VALID, self-consistent kernel inputs
# ---------------------------------------------------------------------------
@st.composite
def merge_accumulate_inputs(draw, force_bayer=None, single_site=False):
    """Valid inputs for merge_accumulate_cupy / reference_merge_accumulate.

    Returns a dict of cupy arrays + scalars. Shapes are kept small. The cov grid
    is grey-resolution (lr//2 bayer, lr grey). Flows are bounded to (-2, 2) so
    both in-bounds and near-OOB gather paths are exercised. Robustness ``r`` is
    in [0, 1] (so den increments are non-negative). When ``single_site`` the comp
    image is zero everywhere except one interior pixel set to 1 (for the Bayer
    channel-routing invariant).
    """
    bayer = draw(st.booleans()) if force_bayer is None else force_bayer
    iso = draw(st.booleans()) if bayer else False  # grey is always steerable path here
    scale = draw(st.sampled_from([1, 2]))
    tile = draw(st.sampled_from([4, 8]))
    npy = draw(st.integers(min_value=2, max_value=5))
    npx = draw(st.integers(min_value=2, max_value=5))
    lr_h, lr_w = npy * tile, npx * tile
    rng = np.random.default_rng(draw(_seeds()))

    if single_site:
        si = draw(st.integers(min_value=2, max_value=lr_h - 3))
        sj = draw(st.integers(min_value=2, max_value=lr_w - 3))
        comp = cp.zeros((lr_h, lr_w), dtype=cp.float32)
        comp[si, sj] = cp.float32(1.0)
        al = cp.zeros((npy, npx, 2), dtype=cp.float32)  # zero flow: clean mapping
        r = cp.ones((lr_h, lr_w), dtype=cp.float32)
        site = (int(si), int(sj))
    else:
        comp = _u(rng, (lr_h, lr_w), 0.0, 1.0)
        al = _u(rng, (npy, npx, 2), -2.0, 2.0)  # exercises in-bounds + near-OOB gather
        r = _u(rng, (lr_h, lr_w), 0.0, 1.0)  # robustness in [0,1] -> den increments >= 0
        site = None

    cov_h = lr_h // 2 if bayer else lr_h
    cov_w = lr_w // 2 if bayer else lr_w
    covs = _pd_covs_from(_normal(rng, (cov_h, cov_w, 2, 2)))

    hr_h, hr_w = lr_h * scale, lr_w * scale
    # den0 non-negative so the den>=0 / additivity invariants are meaningful.
    den0 = _u(rng, (hr_h, hr_w, 3), 0.0, 1.0)
    num0 = _u(rng, (hr_h, hr_w, 3), -1.0, 1.0)

    return dict(comp=comp, al=al, covs=covs, r=r, bayer=int(bayer), iso=int(iso),
                scale=float(scale), tile=tile, num0=num0, den0=den0, site=site)


@st.composite
def merge_ref_inputs(draw):
    """Valid inputs for merge_ref_accumulate_cupy / reference, robustness-denoise
    randomly on or off (with a self-consistent acc_rob grid when on)."""
    bayer = draw(st.booleans())
    iso = draw(st.booleans()) if bayer else False
    scale = draw(st.sampled_from([1, 2]))
    lr_h = draw(st.integers(min_value=8, max_value=24))
    lr_w = draw(st.integers(min_value=8, max_value=24))
    rng = np.random.default_rng(draw(_seeds()))
    ref_img = _u(rng, (lr_h, lr_w), 0.0, 1.0)

    cov_h = lr_h // 2 if bayer else lr_h
    cov_w = lr_w // 2 if bayer else lr_w
    covs = _pd_covs_from(_normal(rng, (cov_h, cov_w, 2, 2)))

    hr_h, hr_w = lr_h * scale, lr_w * scale
    den0 = _u(rng, (hr_h, hr_w, 3), 0.0, 1.0)
    num0 = _u(rng, (hr_h, hr_w, 3), -1.0, 1.0)

    rden = draw(st.booleans())
    if rden:
        acc_rob = _u(rng, (lr_h, lr_w), 0.0, 8.0)  # straddles max_frame_count=2
        mfc, rmax, mult = 2, 2, 8.0
    else:
        acc_rob, mfc, rmax, mult = None, 0, 0, 0.0

    return dict(ref_img=ref_img, covs=covs, bayer=int(bayer), iso=int(iso),
                scale=float(scale), num0=num0, den0=den0, acc_rob=acc_rob,
                rden=int(rden), mfc=mfc, rmax=rmax, mult=float(mult))


@st.composite
def gradient_inputs(draw):
    """Arbitrary (gh, gw, 2) gradient grid for estimate_cov_cupy.

    Deliberately *not* restricted to gradients of a real image -- the symmetry /
    PSD invariants must hold for any input. A random per-grid magnitude scale
    exercises both near-isotropic and strongly anisotropic structure tensors.
    """
    gh = draw(st.integers(min_value=3, max_value=12))
    gw = draw(st.integers(min_value=3, max_value=12))
    rng = np.random.default_rng(draw(_seeds()))
    scale = draw(_floats(0.0, 5.0))  # magnitude axis: near-isotropic .. anisotropic
    grads = _normal(rng, (gh, gw, 2)) * cp.float32(scale)
    # selection law + denoise/detail params, all valid ranges
    law = draw(st.sampled_from([ops.SEL_HARD_THRESHOLD, ops.SEL_LINEAR]))
    return dict(grads=cp.ascontiguousarray(grads.astype(cp.float32)),
                k_detail=draw(_floats(0.1, 1.0)), k_denoise=draw(_floats(1.0, 5.0)),
                D_th=draw(_floats(0.0, 0.5)), D_tr=draw(_floats(0.01, 1.0)),
                k_stretch=draw(_floats(1.0, 4.0)), k_shrink=draw(_floats(1.0, 4.0)),
                law=int(law))


@st.composite
def block_matching_inputs(draw):
    """Valid inputs for block_matching_l1_cupy / reference_block_matching_l1.

    ref/moving are a little larger than the patch grid (as in the example test)
    so interior tiles have full in-bounds support. Base flow in (-3, 3)."""
    tile = draw(st.sampled_from([4, 8]))
    npy = draw(st.integers(min_value=3, max_value=6))
    npx = draw(st.integers(min_value=3, max_value=6))
    R = draw(st.sampled_from([1, 2, 4]))
    H, W = npy * tile, npx * tile
    mh = H + draw(st.integers(0, 5))
    mw = W + draw(st.integers(0, 5))
    rng = np.random.default_rng(draw(_seeds()))
    ref_img = _u(rng, (mh, mw), 0.0, 1.0)
    mov = _u(rng, (mh, mw), 0.0, 1.0)
    al = _u(rng, (npy, npx, 2), -3.0, 3.0)
    return dict(ref_img=ref_img, mov=mov, al=al, tile=tile, R=R)


@st.composite
def upscale_inputs(draw):
    """Valid inputs for upscale_warp_stats_cupy / reference_dogson_upscale.

    Random bayer (=> x2) / grey (=> x1) and ref (no flow) / warped (per-tile
    flow grid sized to cover the HR grid). Flow in (-2, 2)."""
    bayer = draw(st.booleans())
    warp = draw(st.booleans())
    nc = 3 if bayer else 1
    lny = draw(st.integers(min_value=8, max_value=24))
    lnx = draw(st.integers(min_value=8, max_value=24))
    rng = np.random.default_rng(draw(_seeds()))
    ls = _u(rng, (nc, lny, lnx), 0.0, 1.0)
    if warp:
        s = 2 if bayer else 1
        ts = draw(st.sampled_from([8, 16]))
        npy, npx = (lny * s) // ts + 2, (lnx * s) // ts + 2
        flow = _u(rng, (npy, npx, 2), -2.0, 2.0)
    else:
        ts, flow = None, None
    return dict(ls=ls, bayer=bayer, ts=ts, flow=flow)


@st.composite
def robustness_threshold_inputs(draw):
    """Valid inputs for robustness_threshold_cupy.

    sigma_sq is kept strictly positive (the kernel divides d_sq/sigma_sq), d_sq
    non-negative (it is a squared distance), and S per-tile in a generous range
    incl. >1 to confirm the clip-to-1 actually bites. ``t`` is a small threshold.
    """
    tile = draw(st.sampled_from([4, 8, 16]))
    npy = draw(st.integers(min_value=2, max_value=6))
    npx = draw(st.integers(min_value=2, max_value=6))
    H, W = npy * tile, npx * tile
    rng = np.random.default_rng(draw(_seeds()))
    d_sq = _u(rng, (H, W), 0.0, 5.0)
    sigma_sq = _u(rng, (H, W), 1e-2, 5.0)  # strictly positive: kernel divides by it
    S = _u(rng, (npy, npx), 0.0, 3.0)  # incl. >1 so the clip-to-1 actually bites
    t = draw(_floats(0.0, 1.0))
    return dict(d_sq=d_sq, sigma_sq=sigma_sq, S=S, t=t, tile=tile)


# ===========================================================================
# merge_accumulate_cupy
# ===========================================================================
@PROP_SETTINGS
@given(p=merge_accumulate_inputs())
def test_merge_accumulate_den_nonneg_and_additive(p):
    """INVARIANT: with den0 >= 0, the accumulated den is (i) elementwise >= 0 and
    (ii) >= den0 elementwise (additivity/monotonicity), and (iii) all outputs are
    finite for finite PD inputs. den only ever has non-negative increments
    (w·local_r with w=exp(-z/2)>=0, local_r in [0,1]), so this holds regardless
    of the reference."""
    args = (p["comp"], p["al"], p["covs"], p["r"], p["bayer"], p["iso"],
            p["scale"], p["tile"], CFA)
    n, d = ops.merge_accumulate_cupy(*args, p["num0"].copy(), p["den0"].copy())
    assert bool(cp.isfinite(n).all()), "num has non-finite entries"
    assert bool(cp.isfinite(d).all()), "den has non-finite entries"
    assert bool((d >= 0).all()), f"den has negative entries (min={float(d.min())})"
    # additivity: accumulating a frame can only grow den (within fp tolerance)
    assert bool((d >= p["den0"] - ATOL).all()), \
        f"den decreased below den0 (min delta={float((d - p['den0']).min())})"


@PROP_SETTINGS
@given(p=merge_accumulate_inputs(force_bayer=True, single_site=True))
def test_merge_accumulate_bayer_channel_routing(p):
    """INVARIANT (Bayer channel routing): a comp image that is non-zero at a
    single site contributes only to that site's CFA channel in num. With zero
    flow the comp pixel's parity (si&1, sj&1) selects the channel, so only
    channel CFA[si&1, sj&1] of num may change; the other channels stay at num0.
    """
    si, sj = p["site"]
    expected = int(CFA[si & 1, sj & 1])
    args = (p["comp"], p["al"], p["covs"], p["r"], p["bayer"], p["iso"],
            p["scale"], p["tile"], CFA)
    n, _ = ops.merge_accumulate_cupy(*args, p["num0"].copy(), p["den0"].copy())
    delta = n - p["num0"]
    for ch in range(3):
        changed = float(cp.abs(delta[..., ch]).max())
        if ch == expected:
            # the single bright site must actually deposit somewhere in num
            assert changed > 0.0, f"expected channel {ch} unchanged for site {p['site']}"
        else:
            assert changed == 0.0, \
                f"channel {ch} changed (max|delta|={changed}) but only {expected} should"


@EQ_SETTINGS
@given(p=merge_accumulate_inputs())
def test_merge_accumulate_equiv_reference(p):
    """EQUIVALENCE: Warp port == reference RawModule accumulate, fuzzed."""
    args = (p["comp"], p["al"], p["covs"], p["r"], p["bayer"], p["iso"],
            p["scale"], p["tile"], CFA)
    n_new, d_new = ops.merge_accumulate_cupy(*args, p["num0"].copy(), p["den0"].copy())
    n_old, d_old = ref.reference_merge_accumulate(*args, p["num0"].copy(), p["den0"].copy())
    assert cp.allclose(n_new, n_old, rtol=RTOL, atol=ATOL), "num mismatch vs reference"
    assert cp.allclose(d_new, d_old, rtol=RTOL, atol=ATOL), "den mismatch vs reference"


# ===========================================================================
# merge_ref_accumulate_cupy
# ===========================================================================
@PROP_SETTINGS
@given(p=merge_ref_inputs())
def test_merge_ref_finite_and_den_nonneg(p):
    """INVARIANT: outputs are finite, and on the purely additive path (no
    robustness-denoise overwrite) den is >= den0 >= 0 (den increments are
    sums of exp(-y/2) >= 0). When robustness-denoise overwrites a pixel it
    *replaces* den with a sum of non-negative weights, so den stays >= 0 there
    too; we only assert the >= den0 monotonicity on the additive path."""
    args = (p["ref_img"], p["covs"], p["bayer"], p["iso"], p["scale"], CFA)
    tail = (p["acc_rob"], p["rden"], p["mfc"], p["rmax"], p["mult"])
    n, d = ops.merge_ref_accumulate_cupy(*args, p["num0"].copy(), p["den0"].copy(), *tail)
    assert bool(cp.isfinite(n).all()), "num has non-finite entries"
    assert bool(cp.isfinite(d).all()), "den has non-finite entries"
    assert bool((d >= 0).all()), f"den has negative entries (min={float(d.min())})"
    if not p["rden"]:
        assert bool((d >= p["den0"] - ATOL).all()), \
            f"den decreased below den0 on additive path (min delta={float((d - p['den0']).min())})"


@EQ_SETTINGS
@given(p=merge_ref_inputs())
def test_merge_ref_equiv_reference(p):
    """EQUIVALENCE: Warp port == reference accumulate_ref, fuzzed incl.
    robustness-denoise on/off."""
    args = (p["ref_img"], p["covs"], p["bayer"], p["iso"], p["scale"], CFA)
    tail = (p["acc_rob"], p["rden"], p["mfc"], p["rmax"], p["mult"])
    n_new, d_new = ops.merge_ref_accumulate_cupy(*args, p["num0"].copy(), p["den0"].copy(), *tail)
    n_old, d_old = ref.reference_merge_ref_accumulate(*args, p["num0"].copy(), p["den0"].copy(), *tail)
    assert cp.allclose(n_new, n_old, rtol=RTOL, atol=ATOL), "num mismatch vs reference"
    assert cp.allclose(d_new, d_old, rtol=RTOL, atol=ATOL), "den mismatch vs reference"


# ===========================================================================
# estimate_cov_cupy
# ===========================================================================
@PROP_SETTINGS
@given(p=gradient_inputs())
def test_estimate_cov_symmetric_and_psd(p):
    """INVARIANT: the steerable covariance is (i) symmetric (cov[..,0,1] ==
    cov[..,1,0] exactly, it's assigned by aliasing) and (ii) positive
    semidefinite for ANY gradient input -- trace >= 0 and det >= -tol. A 2x2
    symmetric matrix is PSD iff trace>=0 and det>=0; covariances are built as
    k1²·e1e1ᵀ + k2²·e2e2ᵀ with k²>=0, so this must hold no matter the input."""
    cov = ops.estimate_cov_cupy(p["grads"], p["k_detail"], p["k_denoise"],
                                p["D_th"], p["D_tr"], p["k_stretch"], p["k_shrink"],
                                p["law"])
    assert bool(cp.isfinite(cov).all()), "cov has non-finite entries"
    sym = float(cp.abs(cov[..., 0, 1] - cov[..., 1, 0]).max())
    assert sym == 0.0, f"cov not symmetric (max asym={sym})"
    tr = cov[..., 0, 0] + cov[..., 1, 1]
    det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    assert bool((tr >= -ATOL).all()), f"negative trace (min={float(tr.min())})"
    assert bool((det >= -1e-3).all()), f"negative det (min={float(det.min())})"


# ===========================================================================
# block_matching_l1_cupy
# ===========================================================================
def _l1_cost_at_residual(ref_img, mov, base_flow, tile, sx, sy):
    """Per-tile L1 cost of `mov` against `ref` at integer residual (sx, sy) on
    top of the rounded base flow -- the exact quantity block_matching minimises.
    Returns an (n_patch_y, n_patch_x) cost grid. Mirrors the kernel's gather
    (OOB -> 0)."""
    npy, npx = base_flow.shape[:2]
    H, W = npy * tile, npx * tile
    r = ref_img[:H, :W]
    mh, mw = mov.shape
    base_x = cp.round(base_flow[..., 0])
    base_y = cp.round(base_flow[..., 1])
    fx = cp.repeat(cp.repeat(base_x, tile, 0), tile, 1).astype(cp.int32)
    fy = cp.repeat(cp.repeat(base_y, tile, 0), tile, 1).astype(cp.int32)
    yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.int32), cp.arange(W, dtype=cp.int32),
                         indexing="ij")
    gy = yy + fy + sy
    gx = xx + fx + sx
    inb = (gy >= 0) & (gy < mh) & (gx >= 0) & (gx < mw)
    sh = cp.where(inb, mov[cp.clip(gy, 0, mh - 1), cp.clip(gx, 0, mw - 1)], cp.float32(0))
    return cp.abs(r - sh).reshape(npy, tile, npx, tile).sum(axis=(1, 3))


@PROP_SETTINGS
@given(p=block_matching_inputs())
def test_block_matching_optimality(p):
    """INVARIANT (optimality): the L1 cost at the returned per-tile shift is
    <= the cost at the zero residual (the rounded base flow) on interior tiles.
    The brute-force search always evaluates the (0,0) residual and keeps it
    unless something strictly beats it, so this is true by construction and does
    not consult the reference -- it would catch an off-by-one or a broken argmin
    in the port."""
    ref_img, mov, al, t, R = p["ref_img"], p["mov"], p["al"], p["tile"], p["R"]
    flow = ops.block_matching_l1_cupy(ref_img, mov, al, t, R)
    cost0 = _l1_cost_at_residual(ref_img, mov, al, t, 0, 0)
    res_x = cp.asnumpy((flow[..., 0] - cp.round(al[..., 0])).astype(cp.int32))
    res_y = cp.asnumpy((flow[..., 1] - cp.round(al[..., 1])).astype(cp.int32))
    npy, npx = flow.shape[:2]
    cost0_np = cp.asnumpy(cost0)
    # interior tiles only (border tiles' rounded base flow can push the whole
    # tile OOB, where the landscape is degenerate); the property holds there too
    # but the spec asks for interior tiles.
    for iy in range(1, npy - 1):
        for ix in range(1, npx - 1):
            best = float(_l1_cost_at_residual(ref_img, mov, al, t,
                                              int(res_x[iy, ix]), int(res_y[iy, ix]))[iy, ix])
            assert best <= cost0_np[iy, ix] + 1e-3, (
                f"tile ({iy},{ix}) returned cost {best} > zero-shift cost "
                f"{cost0_np[iy, ix]} (residual=({int(res_x[iy, ix])},{int(res_y[iy, ix])}))")


@EQ_SETTINGS
@given(p=block_matching_inputs())
def test_block_matching_equiv_reference(p):
    """EQUIVALENCE: integer gather == reference map_coordinates(order=0), which
    must be bit-identical (both are exact integer shifts), fuzzed."""
    ref_img, mov, al, t, R = p["ref_img"], p["mov"], p["al"], p["tile"], p["R"]
    new = ops.block_matching_l1_cupy(ref_img, mov, al, t, R)
    old = ref.reference_block_matching_l1(ref_img, mov, al, t, R)
    assert cp.array_equal(new, old), "block-matching flow differs from reference"


# ===========================================================================
# robustness_threshold_cupy
# ===========================================================================
@PROP_SETTINGS
@given(p=robustness_threshold_inputs())
def test_robustness_threshold_in_unit_interval(p):
    """INVARIANT: R = clip(S·exp(-d_sq/sigma_sq) - t, 0, 1) is in [0, 1] and
    finite for any sigma_sq > 0, d_sq >= 0. The final clip guarantees the range;
    this catches a regression that dropped/changed the clip."""
    R = ops.robustness_threshold_cupy(p["d_sq"], p["sigma_sq"], p["S"], p["t"], p["tile"])
    assert bool(cp.isfinite(R).all()), "R has non-finite entries"
    assert bool((R >= 0.0).all()), f"R below 0 (min={float(R.min())})"
    assert bool((R <= 1.0).all()), f"R above 1 (max={float(R.max())})"


# ===========================================================================
# upscale_warp_stats_cupy
# ===========================================================================
@EQ_SETTINGS
@given(p=upscale_inputs())
def test_upscale_equiv_reference(p):
    """EQUIVALENCE: vectorized Dogson upscale == reference RawKernel. The OOB
    inf mask must match exactly and finite values allclose, fuzzed over
    bayer/grey and ref/warp."""
    new = ops.upscale_warp_stats_cupy(p["ls"], p["bayer"], p["ts"], p["flow"])
    old = ref.reference_dogson_upscale(p["ls"], p["bayer"], p["ts"], p["flow"])
    assert cp.array_equal(cp.isinf(new), cp.isinf(old)), "OOB inf mask differs"
    fin = ~cp.isinf(new)
    assert cp.allclose(new[fin], old[fin], rtol=RTOL, atol=ATOL), "finite values differ"
