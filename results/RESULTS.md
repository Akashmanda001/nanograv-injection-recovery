# Stage-2 Results: Monte Carlo Validation of PTA Correlation Estimators

**Date:** 2026-10-14 · **Runtime:** 1957 s (~33 min) · **Realizations:** 605
(8490 estimator rows) · **Seeds:** deterministic (`BASE_SEED=20261014`;
exact grid in `manifest.json`; raw data in `sweep_raw.npz`)

## What was swept

| Dimension | Values |
|---|---|
| Scenarios | GWB (HD-correlated) at log10 A ∈ {−15.0, −14.6, −14.3}; pure null; loud *uncorrelated* red noise in 2 pulsars; common monopole systematic; common dipole systematic (systematics at fixed log10 A = −14.6) |
| Intrinsic red noise | weak (−17.5…−16.0), fiducial (−17.0…−15.0), loud (−16.0…−14.0) |
| Array | 12 pulsars, NG15-mimicking synthetic schedules, 15-yr baseline |
| Estimators | raw Pearson, whitened Pearson, optimal statistic (Chamberlin+2015) |
| Noise models | oracle (truth) vs per-pulsar ML fits (nstarts=2, maxiter=150); joint array ML fit (global common term) on a reduced 2-cell × 10-realization grid |
| Selection | all pulsars vs Project-1-style RMS-reduction groups (Strong >50% / Moderate 10–50% / Weak <10%), grouping by oracle posterior-mean red-noise subtraction (the simulation analog of Project 1's raw-vs-published-whitened comparison) |
| Realizations | 30/cell (GWB), 35/cell (nulls), 10/cell (joint) |

Detection = HD-template-fit SNR > 3.0. Metrics: SNR mean ± std, detection
probability / false-positive rate, OS amplitude bias & RMSE vs injected A².

## Headline findings

### 1. The optimal statistic dominates detection; whitening helps Pearson modestly
Fiducial noise, all 12 pulsars, oracle models (detection probability at SNR>3):

| log10 A_gwb | raw Pearson | whitened Pearson | optimal statistic |
|---|---|---|---|
| −15.0 | 0.03 (SNR 0.6±1.2) | 0.07 (0.7±1.5) | **0.43** (2.8±3.6) |
| −14.6 | 0.50 (3.4±2.8) | 0.60 (4.3±3.5) | **0.88** (27.6±27.9) |
| −14.3 | 0.80 (5.7±4.1) | 0.80 (6.4±4.6) | **0.93** (91.4±91.8) |

Whitening before correlating is a consistent but small win over raw
Pearson — the same qualitative effect Project 1 saw on real data, now with
a measured size. The noise-weighted optimal statistic is in a different
league, as theory predicts. At log10 A = −15.0 even the OS detects less
than half the time: a 12-pulsar array is marginal there.

### 2. The noise-model degeneracy is quantified — and it is severe
Replacing oracle noise models with per-pulsar ML fits (the realistic case)
collapses every estimator (GWB −14.6, fiducial):

| Estimator | oracle Pdet | fitted Pdet |
|---|---|---|
| whitened Pearson | 0.60 | **0.03** |
| optimal statistic | 0.88 | **0.05** |

Mechanism (confirmed in the fits): the per-pulsar ML fit has no common
term, so common power is absorbed into "intrinsic" red noise — fitted
spectral indices slam into the γ=7 bound. The whitened residuals then have
the signal largely removed, and the OS's noise model is wrong in exactly
the signal-dominated modes. OS amplitude recovery tells the same story:
oracle overestimates mildly (relative bias +13% to +35% across noise
levels, RMSE ~0.9–1.5× truth), while fitted models *underestimate*
systematically (relative bias −60% to −75% at all three noise levels).

### 3. A joint array noise model (global common term) recovers most of the loss
On the reduced grid (GWB −14.6, fiducial, n=10): OS detection probability
**0.70** with the joint fit vs 0.88 oracle and 0.05 per-pulsar-fitted; null
FPR 0.00. The joint fit recovers the common amplitude as
log10 A = **−14.80 ± 0.09** (truth −14.60). This is the poor-man's version
of the common-process noise modeling real analyses use — and the sweep
shows it is close to necessary: single-pulsar noise modeling is not a
safe shortcut when a common signal may be present.

### 4. False positives: uncorrelated red noise is safe; monopole/dipole systematics are not
FPR at SNR > 3 (fiducial noise; loud-noise numbers in parentheses):

| Null scenario | raw Pearson | whitened Pearson | optimal statistic |
|---|---|---|---|
| pure null | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| loud uncorrelated red (2 pulsars) | 0.00 (0.06) | 0.00 (0.00) | 0.00 (0.00) |
| common monopole (clock-like) | 0.23 (0.26) | 0.37 (0.26) | **0.60** (0.49) |
| common dipole (ephemeris-like) | 0.40 (0.43) | 0.49 (0.34) | **0.77** (0.54) |

Two messages. First, loud *uncorrelated* red noise in individual pulsars
does not fake a Hellings–Downs detection for any estimator — the naive
"red noise mimics correlations" worry does not survive a template fit.
Second, common *systematics* do: a monopole or dipole common process
projects onto the HD template (mean OS SNR +7.2 / +20.0 under the null),
and the better the estimator, the more confidently it is fooled — the OS
has the highest FPR of the three. This is exactly why real PTA analyses
run monopole/dipole consistency checks alongside the HD search. (Fitted
noise models show lower FPR here only because they destroy all
sensitivity — finding 2.)

### 5. Selection by whitening response: no clear win, and the Strong group is unusable
Project-1-style groups (GWB −14.6, fiducial; median sizes
Strong 0 / Moderate 7 / Weak 5 of 12):

| Selection | raw P. SNR | whitened P. SNR | OS SNR |
|---|---|---|---|
| all (12) | 3.4±2.8 | 4.3±3.5 | 27.6±27.9 |
| Moderate-only | 3.6±3.4 | 4.3±4.0 | 30.6±32.6 |
| Weak-only | 0.2±1.1 | 0.3±1.5 | 1.0±4.0 |
| Strong-only | — (0 usable realizations) | — | — |

Dropping the white-noise-dominated "Weak" pulsars (Moderate-only) performs
about the same as the full array — within noise, no significant gain. The
Strong group (>50% RMS reduction) is too sparse to evaluate at these
thresholds (median 0 pulsars per realization; 0/40 usable). In this
simulation regime the absolute 50%/10% thresholds from Project 1 mostly
separate "some red noise" from "white-dominated" rather than identifying
an elite subset. Selection by whitening response neither clearly helps nor
clearly hurts — an honest null result for this dimension.

## Limitations (read before citing)

- **Residual-level injection.** No timing-model fit; real analyses absorb
  part of any common signal into the fit. Estimator performance here is an
  upper bound vs a full pipeline (DESIGN.md §2.2).
- **12 pulsars, synthetic schedules.** Smaller and more regular than NG15's
  68 pulsars; detection probabilities do not transfer directly.
- **ML noise fits, not Bayesian.** The degeneracy finding uses
  point-estimate ML; a full hierarchical Bayesian treatment (v2) could
  behave differently — though the direction (absorption of common power)
  is the known hard problem, not an artifact.
- **Reduced fitted-model settings** (nstarts=2, maxiter=150 vs Stage-1's
  4/300) and a **small joint grid** (n=10/cell) — the joint-fit numbers
  have wide error bars.
- **Strong selection group inconclusive** (too sparse); systematic
  amplitude fixed at a single value (−14.6).
- Fitted-model fallback fraction: 0.0000 — no silent oracle substitution.

## Bug fixes made during Stage 2 (reproducibility record)

1. `simulate_array` (loud_red): the loud pulsars' `Pulsar.log10_A_red /
   gamma_red` were never updated to the injected (−13.0, 4.0), so the
   "oracle" model mismatched the truth. Fixed — oracle now matches.
2. Selection grouping: Stage-1's `whitened_us` is L⁻¹r (dimensionless,
   unit RMS by construction), which makes RMS-reduction meaningless. Added
   `red_subtracted_residuals_us` (Wiener posterior-mean red-noise
   subtraction, real µs units) — the correct analog of NANOGrav's
   published whitened residuals — and grouped on that.

## What Stage 3 (write-up / portfolio) should say

- Lead with the method ladder result (finding 1) and the degeneracy
  quantification (finding 2): together they answer the project's research
  question — *which* methods recover the HD pattern is secondary to
  *how the noise model is built*.
- The joint-fit rescue (finding 3) is the constructive counterpart: it
  shows the way out and motivates the v2 Bayesian extension honestly.
- The systematics result (finding 4) connects the toy project to real
  PTA practice (monopole/dipole consistency checks) — good interview
  material.
- Report the selection null result (finding 5) plainly; do not oversell.
- Keep the honest framing: this is a reproducible comparison of specific
  analysis choices, not an invention of injection-recovery. All seeds,
  code, and raw outputs (`sweep_raw.npz`, `manifest.json`) are in
  `results/`.

## Figures

- `figures/fig1_snr_vs_amplitude.png` — detection SNR vs injected amplitude
- `figures/fig2_false_positive_rates.png` — FPR per null scenario/estimator
- `figures/fig3_amplitude_bias.png` — OS amplitude bias/RMSE vs noise strength
- `figures/fig4_oracle_vs_fitted.png` — oracle vs fitted vs joint noise models
- `figures/fig5_selection_groups.png` — SNR by pulsar selection group

Reproduce everything with:
`python3 stage2/run_sweep.py && python3 stage2/analyze.py`
