# Refactoring plan

Turns the findings in `ADVERSARIAL_REVIEW.md` into ordered, PR-sized changes. Each
step names its **scope**, **method** (with the *verified* API to use), the **guard**
(test that protects it), and the **risk**. Sequencing matters: delete dead code
first (shrinks the surface), then collapse the backend, then do library rewrites
behind the equivalence oracle, then add types.

**Status:** PR 1 (dead-numba-kernel removal + `linalg.py` deletion) and the
import/crash bug-fixes from PR 0 are **done** — ~1,900 LoC removed, full suite
green. numba is **kept** as a dependency for array plumbing + the unported
frame-count denoisers (the "drop numba entirely" half of PR 1 is deferred — it
needs the PR 2 backend collapse). PR 2–4 are not started. Per `CLAUDE.md` §1,
every claimed win (perf, equivalence, API behavior) must be *benchmarked/verified
on this machine* before it lands — do not assume from docs or memory.

---

## PR 0 — bug fixes (no behavior change)

- `super_resolution.py`: `warnings.warns` → `warnings.warn` (353, currently turns
  a missing-orientation warning into an `AttributeError` crash); delete the
  duplicate `compute_grey_images_` binding (73/75); delete dead commented blocks
  (132, 245–252).
- `utils.py`: `np.int` → `np.int64` (34, 48) — `np.int` was removed in numpy ≥1.24,
  so `getSigned`/`isTypeInt` currently throw.
- `kernels.py`: delete the unused `GAT_` / `compute_grey_images_` timer wrappers
  (57–58); the live code calls the cupy fns directly.
- `alignment.py`: drop the unused `bilinear_interpolation` import; `utils.py` drop
  unused `float64`/`complex64`.

Guard: full suite green. Risk: ~none. One small PR.

## PR 1 — delete dead numba + drop the numba dependency

**Kernel removal: DONE** (the `@cuda.jit` compute kernels + launchers + `linalg.py`
are gone; imports fixed; suite green). **Remaining:** the live numba *plumbing*
(`cuda.to_device`/`stream`/`synchronize`) is still in place, so numba is still a
dependency — finish that in PR 2 (it's the same stream-ownership question).

Scope: remove all 46 `@cuda.jit` kernels and their Python launchers across
`merge.py`, `ICA.py`, `block_matching.py`, `kernels.py`, `robustness.py`,
`utils_image.py`, `utils.py`; delete `linalg.py` (100% dead — every function is
reached only by dead kernels). Replace the live numba *plumbing*:

- `cuda.to_device(x)` → `torch.as_tensor(x, device="cuda")` (or a cupy array).
- `cuda.stream()` / `device_array_like` / `synchronize` → torch streams or drop.
- `compute_grey_images(...,"FFT")`: return the torch tensor directly, killing the
  `cuda.as_cuda_array` → torch round-trip.

Also: update the two `@requires_numba` tests in `test_utils_image.py` that lazily
import deleted kernels (delete, or gate behind a "numba kernels present" check),
and remove `numba` from `environment.yaml` / `requirements.txt`.

Guard: `test_kernel_equivalence.py` + `test_kernel_properties.py` +
`test_pipeline.py` (none reference in-package numba — verified). Risk: medium — the
stream/sync replacement. First confirm which `cuda.synchronize()` calls were
guarding numba↔torch ordering; they become unnecessary once one library owns
memory. Net: ~2,000 LoC removed, one fewer GPU framework, PTX-incompat class gone.

## PR 2 — single array framework (torch + DLPack)

Make **torch** the array owner end-to-end; reach cupy via `cupy.from_dlpack` for
the `cupyx.scipy.ndimage` ops and Warp via `wp.from_dlpack` for the merge (already
so). Remove the now-redundant `cuda.synchronize()` barriers in `alignment.py` and
`main` (they exist only because numba + torch both managed memory).

Guard: pipeline + reconstruction tests; additionally diff the output against the
pre-change pipeline on a fixed burst (must match within fp tolerance). Risk: medium
(stream correctness) — verify with a deterministic run, don't assume.

## PR 3 — library rewrites (one op per PR, behind the equivalence oracle)

Each keeps `reference_kernels.py` as the oracle and must pass
`test_kernel_equivalence.py` + the property tests at rtol=1e-3/atol=1e-4. Verified
API caveats are inline (all checked on this box — see `CLAUDE.md`/`ADVERSARIAL_REVIEW.md`):

- **2×2 eig / inverse** (`estimate_cov_cupy`; merge cov-inverse): batched
  `torch.linalg.eigh` / `solve` over `(H,W,2,2)`. eigh returns **ascending**
  eigenvalues (reverse for λ1≥λ2); eigenvector sign is arbitrary but cancels in
  `Σ kᵢ²eᵢeᵢᵀ`. **Benchmark against the closed form** — for millions of 2×2s the
  closed form can win; if so this is readability-only, keep the faster one.
- **Sub-pixel warps** (`ica_iterate_cupy`; `map_coordinates`): `F.grid_sample`,
  **`align_corners=True`**, grid last-dim **(x, y)**, `g = 2·coord/(size−1) − 1`,
  `padding_mode='zeros'`.
- **Per-tile reductions** (`reshape(n,t,n,t).sum`, Hessian): `Tensor.unfold`×2 +
  `einsum('...pi,...pj->...ij')`.
- **Box sums** (`estimate_cov` `box2`): `avg_pool2d(..., divisor_override=1)`
  (verified = windowed sum) or `uniform_filter`.
- **Real FFTs** (grey high-pass, block-match correlation): `rfft2`/`irfft2`
  (pass `s=`; pair `fftshift`/`ifftshift`).
- **Elementwise chains** (`gat_cupy`, `apply_noise_model_cupy`,
  `robustness_threshold_cupy`): wrap in `cupy.fuse` (or port to torch +
  `torch.compile`) — correctness-neutral allocation/launch win.
- **Merge readability** (`warp_ops.py`): `wp.inverse(wp.mat22(...))` + `wp.outer`
  (both verified to compile) for the cov math; `wp.static` to compile-specialize
  `bayer`/`iso`. Keep it a **per-thread SIMT kernel** (tiles deadlock on its
  divergent branches). **No `wp.eig2` exists** — any 2×2 eig stays closed-form in
  a `@wp.func`.

Land each only if equivalence holds *and* perf ≥ current (benchmark, per policy).

## PR 4 — type annotations & API hygiene

The package has essentially no type hints. Strategy (Python 3.12 syntax — `X | None`,
builtin generics, `type` aliases):

- Define one canonical array alias: post-PR-2 `type Array = torch.Tensor`; before
  that, a `Protocol` exposing `__dlpack__` / `__cuda_array_interface__` (since
  funcs currently accept numba/cupy/torch arrays interchangeably).
- Annotate the public surface first: `process`, `main`, `align`, `init_alignment`,
  `merge` / `merge_ref`, `estimate_kernels`, `compute_robustness`, and the
  `cupy_ops` / `warp_ops` launchers (dtype/shape in the hint + docstring).
- Replace stringly-typed config access (`config.block_matching.tuning.tile_size`)
  with an **OmegaConf structured config**: a dataclass schema in `params.py`,
  `OmegaConf.merge(schema, yaml)`. Gives static checking, real defaults, and IDE
  completion in place of `config.<dotted>` lookups that fail only at runtime.
- Add a `py.typed` marker; wire `mypy`/`pyright` into dev deps + a CI lane (none
  today). Drop the `@author` / `Created on …` headers for one-line module docstrings.

Guard: type-checking is additive (no runtime change); suite stays green.

## Optional / later

- Unify the five "sample at grid+flow" gathers (ICA warp, merge cov-interp, Dogson
  upscale, L1 gather, patch extraction) behind one `grid_sample` helper (§C of the
  review).
- `cumsum` summed-area table to remove the L1 `(2R+1)²` shift loop (no cuCIM here).
- End-to-end differentiability (`wp.Tape` + torch autograd) to *learn* the tuning
  params against the synthetic-GT loss instead of Optuna-sweeping: a fixed-iteration
  ICA-LK is differentiable via Warp static loop-unroll; converge-to-tolerance needs
  implicit-function-theorem handling.

## Verification checklist (apply to every PR above)

1. `micromamba run -n handheld python -m pytest test/` green (skips unchanged).
2. For any rewrite: equivalence holds vs `reference_kernels.py` at the stated tol.
3. For any "faster"/"simpler" claim: a measured before/after on this machine.
4. For any API claim: a compiled/run probe, not a doc citation (CLAUDE.md §1).

## Known robustness issue (verified) — align/merge nondeterminism

The multi-frame path is **not deterministic for fixed input**: reconstructing the
*same* fixed-seed synthetic burst multiple times yields outputs that differ by up
to **~0.15–0.19 per pixel** (on a [0,1] scale) and ~0.8–1 dB PSNR — far above
float-rounding noise (~1e-5). The single-frame and bilinear baselines are exactly
reproducible, so the source is the alignment/merge, not the inputs. Mechanism is
most likely a nondeterministic reduction/conv order flipping a near-tie argmin in
block matching (a *discrete* alignment change that then propagates). It is **not**
fixed by `torch.use_deterministic_algorithms` (the wobble is in the cupy/Warp ops,
not just torch). `test/test_reconstruction_accuracy.py` documents and tolerates it
(best-of-N for the capability claim + a median floor to keep it visible).
The PR 3 alignment work should investigate: deterministic tie-breaking in
`block_matching_l1_cupy`'s argmin, and deterministic reductions, then assert
run-to-run output equality on a fixed input as a new regression test.
