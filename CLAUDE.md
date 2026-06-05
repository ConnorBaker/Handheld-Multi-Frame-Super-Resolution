# CLAUDE.md

Repo-wide guidance. These instructions override default behavior. Component
details live in nested files — read the one for the area you're touching:

- `handheld_super_resolution/CLAUDE.md` — source layout, module index, pipeline.
- `test/CLAUDE.md` — testing practices, properties, organization.
- `ADVERSARIAL_REVIEW.md` — code-simplification findings (what / why).
- `REFACTORING_PLAN.md` — planned, ordered refactors (how / when).
- `eval/NOTES.md` — offline-evaluation findings.

## 1. Verify everything; trust no finding you did not execute

This project runs on unusual hardware (NVIDIA GB10 / Grace Blackwell, sm_121)
where library behavior diverges from docs and from other GPUs. **Do not trust
claims about library behavior, source contents, or test results — your own, a
subagent's, or a doc's — until you have run code that demonstrates them on this
machine.**

This is not hypothetical: confident, source-citing research subagents have shipped
wrong conclusions here — a `hasattr` table inferred from grep that missed a
`from ... import *` wildcard, and a "float64 unsupported" claim that was true on
older hardware but false on sm_121. The only subagent output that proved correct
was *executable and self-checking* (tests that run).

- **Grep locates; it does not confirm.** A grep miss can be the wrong pattern, not
  absence. After a hit, read the surrounding file; settle presence / absence and
  behavior by running a probe.
- **Prefer executable evidence over prose.** Compile it, check the value, run the
  test. A passing test outranks a confident summary.
- **Validate on this env / hardware / version**, not from memory of another setup.
  (Warp `@wp.kernel`s can't be defined via `exec()` / heredoc — use a real file.)
- **Subagent results are inputs to verify, not conclusions.** Read what they wrote,
  re-run what they claim passed, re-derive their numbers.

**Keep these docs durable.** No line numbers, no test / case counts, no exact
version pins in any CLAUDE.md — they rot. Put point-in-time specifics in the
review / plan docs, or reference `environment.yaml`.

## 2. Environment

- Run everything through the micromamba env **`handheld`**:
  `micromamba run -n handheld python ...` / `micromamba run -n handheld python -m pytest test/`.
- The live GPU work runs on **torch + cupy + NVIDIA Warp**; exact pins are in
  `environment.yaml` / `requirements.txt`.
- **numba cannot JIT on this hardware** (ptxas rejects the PTX it emits). The numba
  `@cuda.jit` kernels in the package are dead here and kept only as reference — do
  not write new numba kernels. Details: `handheld_super_resolution/CLAUDE.md`.
