# Adversarial review: simplification & library modernization

Scope: the `handheld_super_resolution/` package (≈5,200 LoC). The pipeline was
written in 2022 against numba `@cuda.jit`, then partially migrated to
CuPy/torch/Warp so it runs on the GB10 (numba can't JIT here). This review asks:
what can we delete, what abstractions are missing, and where are we re-rolling
things the libraries now do for us?

Findings are grouped by leverage. (A)–(C) and (F)–(G) are grounded in the code
and verified by grep; (D)–(E) map hand-rolled patterns to library APIs.

---

## A. ~40% of the package is dead numba code (biggest single win)

The migration kept every original numba kernel *in place* next to its live
CuPy/torch/Warp replacement. There are **46 `@cuda.jit` kernels**; none can JIT
on this hardware and none are on the live code path. Verified: the live
`main()`/`align()`/`compute_robustness()`/`estimate_kernels()`/`merge()` call
only the `*_cupy` / torch / Warp functions.

| file | dead numba (approx) | live replacement | of total |
|---|---|---|---|
| `linalg.py` | **234 (100%)** | inlined into `cupy_ops`/`warp_ops` math | 234 |
| `robustness.py` | ~514 | `*_cupy` in `cupy_ops.py` | 648 |
| `ICA.py` | ~418 (`ica_kernel_8/16/32/64`, `compute_hessian`) | `ica_iterate_cupy`, `compute_hessian_cupy` | 458 |
| `merge.py` | ~297 (`accumulate`, `accumulate_ref`) | Warp kernels in `warp_ops.py` | 411 |
| `block_matching.py` | ~240 (`cuda_L1_local_search16/32/64`) | `block_matching_l1_cupy` | 343 |
| `kernels.py` | ~106 (`cuda_estimate_kernel`, `compute_k`, …) | `estimate_cov_cupy` | 217 |
| `utils_image.py` | ~250 (`cuda_GAT`, `cuda_decimate_to_grey`, frame-count denoisers, dogson device fns, `fft_lowpass`) | `gat_cupy`, `*_decimate_cupy`, torch | 545 |
| `utils.py` | ~50 (`cuda_divide`, `cuda_add`, device `clamp`) | `divide`/`add` (cupy) | 126 |

`linalg.py` is **entirely dead**: every function (`invert_2x2`, `quad_mat_prod`,
`interpolate_cov`, `get_eigen_*`, `solve_2x2`, `bilinear_interpolation`) is
referenced only by dead numba kernels, and `bilinear_interpolation` is imported
into `alignment.py` but never called.

**Test impact: none.** The kernel-equivalence tests compare the live ports
against the `RawModule` references in `test/reference_kernels.py`, *not* against
the in-package numba kernels. The only tests that touch in-package numba
(`test_*_matches_numba`) are `@requires_numba` (skipped here) and import the
kernels lazily *inside* the test function, so collection is safe.

**Recommendation.** Delete the dead numba kernels and `linalg.py` (git preserves
them). This also lets us **drop numba as a dependency** (see B). If cross-hardware
numba parity is a real product goal, quarantine them under
`handheld_super_resolution/_legacy_numba/` behind an explicit dispatch and add a
CI lane on numba-capable hardware — but today they are untested-and-dead, the
worst of both.

---

## B. Collapse three GPU frameworks (numba + cupy + torch) into one + DLPack

`main()` moves data with **numba** (`cuda.to_device`, `cuda.stream`,
`cuda.device_array_like`), computes with **torch** (pyramid, gradients, FFT,
block-matching) and **cupy** (robustness, kernels, divide), and merges with
**Warp**. Because numba uses the legacy default stream while torch uses
per-thread default streams, the code must sprinkle manual barriers — e.g.
`alignment.py` has 5 `cuda.synchronize()` calls with the comment *"Without this
barrier, PyTorch could read 'alignment' before the kernel writes finish."* That
hazard exists **only because** numba and torch are both managing device memory.

The live numba surface is now just array plumbing: `cuda.to_device` (×6),
`cuda.stream`, `cuda.device_array_like`, `cuda.synchronize` (×11 across the
package), and one `cuda.as_cuda_array`. All of it has a torch/cupy equivalent.
There is even a pointless round-trip: `compute_grey_images(..., "FFT")` returns
`cuda.as_cuda_array(torch_tensor.real)` only for `init_alignment` to wrap it back
into a torch tensor.

**Recommendation.** Make **torch** the single array owner (it already owns the
heaviest stages), reach cupy via `cupy.from_dlpack` for the `cupyx.scipy.ndimage`
ops, and Warp via `wp.from_dlpack` for the merge (already done). Result: numba
dependency gone, the PTX-version incompatibility surface gone, and almost every
`cuda.synchronize()` removable (one stream/owner). Zero-copy throughout; drop the
`cp.asnumpy`/`cuda.as_cuda_array` bounces.

---

## C. The missing abstraction: tile-size generalization (and launcher boilerplate)

The most repeated smell is **one algorithm hand-specialized per tile size and
thread-block layout**:

- `ICA.py`: `ica_kernel_8`, `_16`, `_32`, `_64` — ~370 lines that differ only in
  `TILE_SIZE`, shared-memory dims, and warp-reduction layout.
- `block_matching.py`: `cuda_L1_local_search16/32/64` — same story for SAD search.

The CuPy ports already prove the right abstraction: **one size-generic function**
(`ica_iterate_cupy`, `block_matching_l1_cupy`) handles all tile sizes by
expressing the per-tile reduction as `x.reshape(n, t, n, t).sum((1,3))`. The
specialization was a manual-CUDA optimization; vectorized array ops make it
unnecessary. Finishing the migration deletes the variants outright.

A second instance is **launcher boilerplate**: `robustness.py` has 8
`(host wrapper + @cuda.jit kernel)` pairs, each repeating the
`threadsperblock`/`blockspergrid`/bounds-check ceremony (~30–40 lines each). The
cupy ports collapse each to a ~10-line array expression.

Third, several ops are the *same primitive* — "sample an array at `grid + flow`":
the ICA warp, the merge covariance interpolation, `upscale_warp_stats`, the L1
integer gather, and `extract_flow_patches`. These could share **one resampling
helper** (see D, `grid_sample`) instead of four bespoke gathers.

---

## D. Hand-rolled → library primitive (Torch / CuPy / Warp)

> Rows verified against current sources (PyTorch 2.12, CuPy 14.1, Warp 1.14).
> Caveats that matter for a faithful port are called out under the table —
> several "obvious" swaps are traps (§H).

| hand-rolled today | library expression | notes |
|---|---|---|
| ICA bilinear warp (numba) / `map_coordinates` in `ica_iterate_cupy` | `torch.nn.functional.grid_sample` (batched over tiles, `align_corners=True`) | one call, autodiff-able, removes manual lerp; unifies with the other gathers |
| per-tile reductions `x.reshape(n,t,n,t).sum((1,3))` (Hessian, L1 SAD) | `F.unfold` / `Tensor.unfold(d,t,t)` ×2, or `avg_pool2d`×t² | L2 box filter already does this (`avg_pool2d`); apply uniformly |
| closed-form 2×2 eigen-decomp + sign conventions (`estimate_cov_cupy`, `linalg.get_eigen_*`) | `torch.linalg.eigh`/`eigvalsh` batched over `(H,W,2,2)` | drops the bespoke eigenvector sign logic; numerically robust |
| merge covariance inverse (manual `1/det` arithmetic, 3 sites) | `torch.linalg.inv`/`solve` batched, or **Warp `wp.inverse(wp.mat22)`** inside the kernel | Warp `mat22`/`vec2`/`wp.determinant` would make `warp_ops` read like the math |
| local mean/var (`compute_local_stats`) | `cupyx.scipy.ndimage.uniform_filter` | already done in port — good |
| local min / flow min-max (`local_min`, `compute_s`) | `minimum_filter`/`maximum_filter` | already done — good |
| grey high-pass via `fft2` then zeroing quadrants | `torch.fft.rfft2`/`irfft2` | real input ⇒ ~2× less work + half the spectrum to mask |
| `estimate_cov` 2×2-window structure tensor (`box2`) | `uniform_filter`/separable conv | replaces the manual padded shifts |
| Gaussian pyramid blur (`_gaussian_kernel1d` + 2 conv2d) | `cupyx`/torch separable gaussian, or `kornia` if a dep is acceptable | current is fine; mostly fine to keep |

Notably **already modern and worth keeping**: the L2 FFT cross-correlation +
`avg_pool2d` box filter in `block_matching.py`, the `unfold`-based tiling in
`init_alignment`, and the `cupyx.scipy.ndimage` filters in the robustness port.

**Port caveats (doc-verified — get these wrong and the equivalence tests fail):**
- `grid_sample`: the hand-rolled warps sample pixel **centers** in absolute
  coords, so a faithful port needs **`align_corners=True`** and normalized coords
  `g = 2·coord/(size−1) − 1`; the grid's last dim is **(x, y)**, and the default
  flipped `True→False` after torch 1.2, so always pass it explicitly.
- `torch.linalg.eigh`/`cupy.linalg.eigh` return eigenvalues **ascending**, but the
  code wants descending (`l1 ≥ l2`) — reverse. Eigenvector **sign is arbitrary**,
  which is *safe here* because the covariance is rebuilt as `Σ kᵢ²·eᵢeᵢᵀ` (outer
  product cancels sign) and `compute_k` depends only on `l1,l2`. The
  isotropic/degenerate special-case the code hand-guards is handled internally by
  `eigh`. **But**: for millions of 2×2 systems the closed form can be *faster*
  than a batched cuSOLVER call — this is a **readability** win, possibly a perf
  regression; benchmark before committing.
- `avg_pool2d` windowed **sum** (not mean): pass `divisor_override=1` (cleaner than
  the current `* t*t`); use `count_include_pad=False` to match `uniform_filter`'s
  edge behavior.
- `irfft2` needs the explicit `s=` (original length) for a correct inverse; pair
  `fftshift` with `ifftshift` (they differ on odd sizes — the current grey path is
  correct).

---

## E. Library features we're leaving on the table

- **`cupy.fuse`** (zero-risk, correctness-neutral) on the pure elementwise/
  reduction helpers that currently allocate a pile of temporaries —
  `apply_noise_model_cupy`, `gat_cupy`, `robustness_threshold_cupy`,
  `compute_dist_cupy`, the `compute_k` arithmetic. `fuse` collapses each ufunc
  chain (+ trailing reduction) into one kernel. (It can **not** fuse the
  gather-bearing functions — `block_matching_l1_cupy`, `ica_iterate_cupy`,
  `upscale_warp_stats_cupy`, merge — which is exactly why those stay custom.)
- **`torch.compile`** (`mode="reduce-overhead"` → CUDA graphs, good for the small
  per-frame tensors here) over the same elementwise-heavy chains once they're
  torch — recovers much of what the hand-written numba kernels were chasing.
- **Batched `torch.linalg`/`cupy.linalg`** (`eigh`/`inv`/`solve`) instead of
  per-pixel scalar formulas — the structure-tensor → covariance path is the prime
  candidate. Prefer `solve` over `inv` for the ICA system (faster + stabler; the
  `inv` docs say so explicitly).
- **`einsum`** for the `2×2` outer-product accumulations written out scalar-by-
  scalar (`Σ g⊗g` structure tensor; `Σ kᵢ²·eᵢeᵢᵀ` covariance recompose).
- **`rfft2`/`irfft2`** instead of full `fft2` everywhere the input is real (grey
  high-pass, block-matching) — ~2× less work, half the spectrum to mask.
- **`cumsum`-based summed-area table** (`cp.cumsum(cp.cumsum(x,0),1)` + 4-corner
  differences) is the dependency-free way to make the L1 block-matcher's per-shift
  box-SAD O(1) per tile, if the `(2R+1)²` Python shift loop ever needs optimizing
  (cuCIM's `integral_image`/`block_reduce` are **not** installed).
- **DLPack end-to-end** instead of `cp.asnumpy` / `cuda.as_cuda_array` hops (the
  existing `from_dlpack` bridges are already correct).
- **Warp built-ins for the merge's 2×2 math:** `wp.inverse(wp.mat22(...))`,
  `wp.determinant`, `wp.outer`, `wp.ddot` replace the inline `det = ixx*iyy − …;
  inv = 1/det` arithmetic in `warp_ops.py` for readability. **Caveat:** Warp 1.14
  has **no `eig2`/symmetric-2×2 eigensolve** (only `eig3`), so a structure-tensor
  eigendecomposition stays closed-form in a `@wp.func` (or `eig3` on a padded 3×3).
- **`wp.static` / static-`if`** to compile-time-specialize the merge on
  `bayer`/`iso` (today runtime `int` args branched per pixel) — dead-strips the
  unused channel-routing paths. This is the Warp analogue of the tile-size
  specialization in §C, but done by the compiler instead of by hand-copied kernels.
- **Warp tiles** (`tile_load` with automatic OOB zero-padding, `tile_sum` +
  `tile_atomic_add`, `tile_matmul`) pay off only for **dense regular reuse / GEMM /
  reductions** — candidate sites: the Hessian / structure-tensor accumulation, and
  windowed L1/L2 SAD/SSD (the matched-filter term maps to `tile_matmul`; docs show
  a tile reduction ~52× over naive atomics). The **merge stays a per-thread SIMT
  kernel**: it's a bounded, clamped, branch-heavy gather, and cooperative tile ops
  **deadlock under divergent branches**. (Verify reminder: `hasattr(wp,"tile_load")`
  is *always* False — tile ops are kernel-only builtins, not module attrs.)
- **End-to-end differentiability (not currently exploited).** torch
  (`grid_sample`/`linalg`/convs are autograd-able) and Warp (`wp.Tape`;
  `from_torch`/`to_torch` carry gradients, **DLPack does not**) can make the whole
  pipeline differentiable. A fixed-iteration ICA-LK is differentiable via Warp
  static loop-unrolling (`max_unroll`); a converge-to-tolerance LK needs implicit-
  function-theorem handling. This would let the tuning params
  (`k_detail`/`k_denoise`/`D_th`/…, robustness `t`/`s1`/`s2`) be *learned* against
  the synthetic-GT loss instead of Optuna-swept.

**Unifying conclusion (both research passes reached this independently):** the
biggest win isn't any single API — adopting `linalg` + `grid_sample` + `unfold` +
`einsum` + `fuse` lets the **entire `cupy_ops.py` port layer and the dead numba
kernels collapse into ~a dozen batched torch calls**, eliminating the
three-backend (numba + CuPy + torch) split that the GB10 ptxas incompatibility
forced. Only the genuine per-output-pixel scatter (the Warp `merge`) legitimately
stays a custom kernel.

---

## F. Bugs & cruft (quick, low-risk)

- **`warnings.warns` → `warnings.warn`** (`super_resolution.py:353`). On a missing
  `Image Orientation` EXIF tag this raises `AttributeError` instead of warning —
  a crash on the error path it was meant to soften.
- **`np.int`** in `utils.getSigned`/`isTypeInt` (`utils.py:34,48`) — removed in
  numpy ≥1.24, so `getSigned` (used by `computePSNR`/`computeRMSE`) throws. Use
  `np.int64`.
- **`compute_grey_images_` defined twice** (`super_resolution.py:73,75`) — first
  binding is dead.
- **Dead timer wrappers** `GAT_`, `compute_grey_images_` created in
  `kernels.py:57,58` but never called (the code calls the cupy fns directly).
- **Unused import** `bilinear_interpolation` in `alignment.py:5`; unused
  `float64`/`complex64` in `utils.py`.
- **Dead commented blocks** (`super_resolution.py:132, 245–252`).
- Typos: `cuda_uspcale_dogson`; the `align_lvl_block_matching_L1` error message
  references `config.block_matching.metric` (wrong path).
- The `timer()` helper hard-depends on `cuda.synchronize()` (numba) — fold into B.

---

## G. Suggested sequencing (each step stays green under the equivalence tests)

1. **Bug fixes (F)** — isolated, no behavior change.
2. **Delete dead numba + `linalg.py`; drop the numba dependency (A, B).** Remove
   the live numba plumbing in favor of torch+DLPack; delete the now-unnecessary
   `cuda.synchronize()` barriers. Update/remove the two `@requires_numba` tests
   that lazily import the deleted kernels.
3. **Unify resampling (C, D)** — introduce one `grid_sample`-based helper; route
   the ICA warp, `upscale_warp_stats`, and the L1 gather through it.
4. **Batch the 2×2 linear algebra (D, E)** — replace the closed-form eig/inverse
   with `torch.linalg` (or Warp `mat22`), behind the existing equivalence tests.
5. **Optional perf (E)** — `rfft2`, `torch.compile` on the fused chains.

Estimated net effect of steps 1–2 alone: ~2,000 LoC removed, one fewer GPU
framework, and the PTX-incompatibility class of bugs eliminated — with no change
to output (guarded by the 38 kernel-equivalence cases).

---

## H. Anti-patterns / traps (do NOT "simplify" these)

The research surfaced several swaps that *look* like simplifications but are wrong
here — recording them so a future pass doesn't regress correctness or perf:

- **Don't rewrite the Warp merge as `cupyx.jit`.** It was already tried and
  rejected: `threadIdx` is `uint32` and the merge's index math mixes it with
  `int32`/Python ints → `Cannot cast int32 to uint32`. Warp handles the typing
  cleanly; keep it.
- **Don't rewrite the merge as `scatter_add`.** The merge is a *gather* — each
  output pixel pulls its own bounded 3×3 neighborhood into `num/den` — so it needs
  **no atomics**; a per-tap scatter would only add contention. (`cupyx.scatter_add`
  *does* support float64 on this GPU — verified, sm_121 has native double
  `atomicAdd` — the point is simply that the merge shouldn't scatter at all.)
- **Don't use `sepfir2d` for the gradients.** It's odd-length **symmetric** only;
  the Sobel `[-1,0,1]` is anti-symmetric and the `[-0.5,0.5]` grad kernels are
  even-length — neither fits. Use `convolve2d`/`correlate2d` (mind conv-vs-xcorr
  flip) or just keep the torch convs.
- **Don't use `order≥2` `map_coordinates` / `RegularGridInterpolator` for the
  Dogson upscale.** Dogson's biquadratic is *not* a B-spline/Catmull-Rom, so the
  taps won't match `_dogson_q`; the equivalence test would fail.
- **Don't add cuCIM** for `block_reduce` / integral image — it isn't installed
  (and is Linux-only RAPIDS). The `reshape(n,t,n,t).sum((1,3))` idiom and a
  `cumsum` SAT cover both in CuPy core.
- **Don't drop to `cupy.fuse` on the gather functions** — it only fuses
  elementwise + reductions, not fancy-index gathers / `reshape` / `map_coordinates`.
- **Don't look for a Warp `eig2`/symmetric-2×2 eigensolve** — it doesn't exist
  (only `eig3`); use a closed-form `@wp.func`. And **don't force the merge into
  Warp tiles** — cooperative tile ops deadlock in its divergent per-pixel
  bounds/channel branches; the per-thread SIMT kernel is correct and faster.

## I. Deprecated torch APIs to flag if reintroduced

If any refactor reaches for the legacy spellings, use the `torch.linalg`
replacements (the olds are deprecated/removed): `torch.norm` → `linalg.norm`/
`vector_norm`/`matrix_norm`; `torch.symeig`/`torch.eig` → `linalg.eigh`/`eigvalsh`
(symmetric) ; `torch.solve(B,A)` → `linalg.solve(A,B)` (**argument order flipped**);
`torch.inverse` → `linalg.inv` (prefer `solve`); `torch.det` → `linalg.det`;
`torch.qr`/`torch.svd` → `linalg.qr`/`linalg.svd` (different `mode`/return). And
always pass `align_corners=` explicitly to `grid_sample`/`affine_grid`.
