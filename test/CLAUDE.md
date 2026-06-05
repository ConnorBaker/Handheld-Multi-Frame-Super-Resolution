# test/CLAUDE.md

How the pipeline is tested and the conventions to keep when adding tests. Run:
`micromamba run -n handheld python -m pytest test/`. See the repo-root `CLAUDE.md`
for the verification policy this suite embodies.

## Layers

- **Equivalence** (`test_kernel_equivalence.py`): every rewritten kernel is
  asserted equal to its preserved ORIGINAL in `reference_kernels.py`, swept across
  the config space (bayer/grey × iso/steerable × scale × robustness). Float ops are
  compared at a tight rtol/atol; integer ops bit-exact.
- **Properties** (`test_kernel_properties.py`, Hypothesis): invariants that consult
  no reference — den ≥ 0 and additive, Bayer channel routing, covariance symmetric
  + PSD, L1 argmin optimality, robustness ∈ [0, 1] — plus fuzzed equivalence.
  Invariants are *stronger* than equivalence: they cannot share a bug with the
  reference.
- **Alignment** (`test_alignment.py`): recovers known synthetic shifts (L1 exact,
  ICA sub-pixel).
- **End-to-end** (`test_pipeline.py`, `test_reconstruction_accuracy.py`): synthetic
  known-GT reconstruction. Prefer a per-pixel budget (PSNR) plus a baseline
  comparison (multi-frame must beat single-frame and bilinear) over a loose
  correlation bound.
- **utils_image** (`test_utils_image.py`): cupy ports vs torch / numpy references.

`reference_kernels.py` holds the original RawModule / `map_coordinates`
implementations — the regression oracle.

## Practices

- **Preserve-the-oracle.** Rewriting a kernel (numba → cupy/torch/Warp) means: keep
  the original in `reference_kernels.py`, add or extend an equivalence test, and
  only then change production. The reference is how we know the rewrite is faithful
  — **do not delete it when you change a kernel.**
- **numba-comparison tests auto-skip here** via the `requires_numba` marker (they
  import numba kernels lazily, so collection still succeeds). Those skips are
  expected on this hardware, not failures.
- **Honesty over green.** Fixed seeds, small arrays. Set end-to-end thresholds from
  *observed* values with margin — never guess, never tune until a test passes
  vacuously. If a property or equivalence genuinely fails, surface it (`xfail` with
  the counterexample); do not weaken the assertion to hide a real bug.
- **Hypothesis cost model.** GPU launches dominate runtime, so deadlines and
  data-size health checks are disabled and the example count is kept modest; array
  *content* is drawn via a seeded RNG off a Hypothesis-drawn seed, while shapes and
  the discrete config knobs are drawn directly so they shrink on failure.
- **New invariants are cheap and valuable.** When you touch a kernel, prefer adding
  a property (something true of *any* correct implementation) over only extending
  the example sweep.
