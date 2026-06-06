# eval/CLAUDE.md

Offline quality evaluation: the comparison + sweep harness, and the discipline
for *trusting a measurement*. See `eval/README.md` for how the harness is wired
and run, `eval/NOTES.md` for current findings, and the repo-root `CLAUDE.md` §1
for the verification policy this section specializes.

Root §1 says: trust no finding you did not execute. This file extends it to the
finding you *did* execute. Running the measurement is necessary, not sufficient —
a number you computed yourself can still be unrepresentative, unstable, or
flattering. Most eval mistakes here were honest measurements that were never
stress-tested. The fixes below each killed a real, recorded self-deception.

## Trust no measurement until you've stress-tested it

- **Verify generality, not just existence — measure the distribution.** A finding
  at one location is an anecdote. Sample many (edges, tiles, regions) and report
  **median AND spread**; a single "edge sharpness" number hid a field that varied
  ~3×. One number for a whole image is a bug until proven uniform.
- **Gate on reliability before you quote.** Attach a fit/confidence flag to every
  per-location measurement and exclude the bad ones *first*. A "we beat the
  reference" median reversed once curve-fit-contaminated edges were dropped — the
  win was in the noise, not the method.
- **Triangulate fragile metrics with ≥2 methods; reconcile before reporting.** If a
  number swings with how you compute it (overshoot read 0.008 / 2.6% / 15% by
  method), it is not a result. Compute it ≥2 ways; if they disagree, trust the
  *direction* and report the disagreement — never cherry-pick the flattering one.
- **Stratify by region/regime; never one number for the whole frame.** "Smoother
  than the reference" was true on textured surface and false on flat sky. Split
  flat vs textured (vs whatever regimes apply), measure each, and always state
  which region a claim is about.
- **Validate every visualization against the raw pixels.** Per-panel
  percentile-stretch + nearest-neighbor zoom fabricated "blockiness" out of
  sub-noise; the FFT showed no such structure. Render comparisons with a **shared
  contrast/stretch** and cubic (not nearest) zoom, and confirm any visible feature
  in the underlying statistics before believing your own eyes.

## Symmetric skepticism — red-team your own favorable claims

- **Scrutinize favorable numbers at least as hard as unfavorable ones.** The
  failures above all shared one root: rigorous skepticism aimed at the external
  reference, credulity aimed at our own measurements because they supported the
  goal. Asymmetric skepticism *is* the bug. When a result flatters the project,
  that is the cue to dig in harder, not to ship it.
- **Run an adversarial self-review BEFORE reporting a favorable result**, not after
  someone questions it. Every correction here came from an adversarial pass that
  should have run first: try to break your own claim — resample, re-tile, swap the
  estimator, drop the unreliable samples — and report only what survives.
- **Tuning a derived parameter to hit a target is a red flag, not a knob.** Nudging
  a value that *should be computed* (e.g. a fudge on `sigma_r`) until the output
  matches the target is silently fitting to the answer. Re-derive it from first
  principles; if measurement and target disagree, that gap is the finding — keep
  derived params derived, and tune only true hyperparameters against a one-sided
  objective that can't win by overshooting.
