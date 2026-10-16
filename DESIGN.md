# DESIGN.md — PTA Injection-Recovery: Simulation-Based Validation of Correlation Estimators

## 1. Purpose and scope

Project 1 computed pairwise Pearson correlations on NANOGrav 15-year
residuals — matched approximately by observing time, before and after
NANOGrav's published whitened residuals — with pulsars grouped by
RMS-reduction (Strong >50% / Moderate 10–50% / Weak <10%), compared against
the Hellings–Downs (HD) curve. It could show *that* correlations changed
under whitening/selection, but not *why*: genuine common signal,
pulsar-specific noise, or a limitation of the analysis method.

This project closes that gap with controlled simulation: generate datasets
where the ground truth is known, then measure each estimator's recovery
rate, amplitude accuracy, and false-positive rate across noise strengths,
pulsar selections, and signal scenarios.

**What this is:** a reproducible comparison of the specific whitening and
selection choices from Project 1.
**What this is not:** an invention of injection-recovery. Injection-recovery
testing and posterior predictive checks are established PTA practice
(including NANOGrav's). The novelty, if any, is in the *specific*
comparison — not the method.

## 2. Design corrections (recorded explicitly)

### 2.1 The noise-weighted estimator IS the optimal statistic

The original brief described a ladder of "raw Pearson → whitened Pearson
→ noise-weighted method" with "a full PTA optimal statistic as a later
extension." **Correction:** the noise-weighted, HD-template matched filter
over pulsar pairs *is* the PTA optimal statistic (Chamberlin et al. 2015).
There is no separate "later" optimal statistic to build. The Stage-1 method
ladder is therefore:

1. **Raw Pearson** — pairwise Pearson r on epoch-matched residuals.
2. **Whitened Pearson** — whiten each pulsar by its intrinsic noise model,
   then epoch-matched Pearson.
3. **Optimal statistic** — inverse-noise-weighted HD matched filter.

The **full Bayesian hierarchical analysis** (joint posterior over noise +
common process, e.g. enterprise) is the genuine later extension, slated
for v2/Stage-3 — not the optimal statistic itself.

### 2.2 v1 injects at the residual level

The Stage-1 simulator generates timing *residuals* directly. It does **not**
simulate TOAs and fit a timing model. Real PTA analyses fit for spin,
astrometry, and binary parameters, and that fit absorbs some covariance —
including part of any common signal — especially at low frequencies.
Residual-level injection therefore *overstates* estimator performance
relative to a full pipeline. This is the central known limitation of v1,
flagged here and addressed by the v2 extension (§7).

## 3. Simulator spec (`src/simulate.py`)

### 3.1 Pulsar array

- **Pulsars:** `Pulsar` dataclass — name, sky position (RA/Dec), MJD epochs,
  per-TOA measurement uncertainties, EFAC/EQUAD white-noise scaling,
  intrinsic red-noise power-law parameters, and the residual time series.
- **Sky positions:** 8 well-known NANOGrav MSPs (J1909-3744, J1713+0747,
  J1744-1134, J1600-3053, J1918-0642, J1012+5307, J1640+2224, J2317+1439)
  with approximate public coordinates (used only for realistic angular
  separations); uniform-sky fallback beyond 8.

### 3.2 Observing epochs

- **Real data loader** (`load_ng15_epochs`): parses TEMPO2-format `.tim`
  files (MJD + TOA error columns) under a user-supplied directory
  (`NG15_DATA_DIR`). Raises `FileNotFoundError` if absent — the pipeline
  **never downloads data automatically**.
- **Synthetic fallback** (`synthetic_schedule`): mimics NANOGrav 15-yr
  practice with documented parameters —
  15-yr baseline; ~monthly cadence (±7 d jitter); ~15% random epoch dropout;
  30% of pulsars get a 14-day high-cadence final 2 yr; per-TOA errors
  lognormal (median 0.3 µs, 0.4 dex, clamped [0.05, 3] µs).

### 3.3 Noise model

- **White noise:** per-TOA Gaussian, σ_eff = √((EFAC·σ_TOA)² + EQUAD²),
  EFAC ∈ [1.0, 1.3], EQUAD ∈ [0, 0.3] µs.
- **Intrinsic red noise:** power-law Gaussian process,
  P(f) = A²/(12π²)·(f/f_yr)^{-γ}·f_yr^{-3}, realized through a Fourier
  basis (30 modes at k/Tspan). log10 A_red ∈ [-17, -15], γ ∈ [1.5, 5.5] —
  kept mostly below the GWB level, as for the best PTA pulsars.
- **Shared Fourier grid:** all pulsars use one Tspan (the array maximum),
  enterprise-style, so a common process has a single consistent
  representation and the optimal-statistic template is correctly
  normalized (Fourier weights carry the df/2 factor).

### 3.4 Signal injection

- **GWB:** HD-correlated red noise (γ = 13/3) via Cholesky of the HD
  spatial covariance (diagonal 1, off-diagonal from the standard
  Γ(ξ) = ½ + 1.5·x·ln(x) − x/4, x = (1−cos ξ)/2).
- **Null scenarios:**
  - `(a) null` — no common signal (intrinsic noise only);
  - `(b) loud_red` — two pulsars get loud *uncorrelated* red noise
    (log10 A = −13.0), testing false HD-like correlations;
  - `(c) monopole` — common clock-like systematic (Γ_ab = 1);
  - `(d) dipole` — common ephemeris-like systematic (Γ_ab = cos ξ).

## 4. Estimators (`src/estimators.py`)

### 4.1 Epoch matching

Project-1 style: greedy nearest-neighbor pairing within 10 days, each
epoch used once (`match_epochs`). Pairs with <10 matched epochs are
skipped (NaN).

### 4.2 Raw Pearson (`raw_pearson`)

Pairwise Pearson r on epoch-matched raw residuals. The Project-1 baseline.
Sensitive to everything — including red-noise leakage that mimics
correlations.

### 4.3 Whitened Pearson (`whitened_pearson`)

Each pulsar is whitened (Cholesky) by its **intrinsic-only** noise model
(white + intrinsic red, *no* common term), then epoch-matched Pearson.
Whitening by the intrinsic model keeps a common signal in the residuals,
so this is a fair detection comparison against raw Pearson; it tests
whether removing pulsar-specific red noise clarifies the HD pattern.
(`whitened_full_us` — whitened by the full model *including* the common
term, NANOGrav-style — is also computed; it removes the common signal by
construction.)

### 4.4 Optimal statistic (`optimal_statistic`)

The Chamberlin et al. (2015) noise-weighted HD matched filter:

A²_hat = Σ_{a<b} Γ_ab·M_ab / Σ_{a<b} Γ_ab²·T_ab,

with M_ab = r_aᵀP_a⁻¹S_abP_b⁻¹r_b and T_ab = tr(P_a⁻¹S_abP_b⁻¹S_ba),
S_ab the unit-amplitude HD template evaluated exactly in the Fourier
basis. Combining via the Γ-free M_ab/T_ab avoids dividing by small HD
coefficients (numerically stable near the HD zero-crossing).

**P_a formulation (weak-signal):** P_a = white + intrinsic red, *excluding*
the common process's auto-term. Verified during development: including the
GWB auto-term in P_a suppresses exactly the low-frequency modes carrying
the signal (SNR 9.1 → 0.35 on the same realization). The weak-signal OS
treats the common process as the signal, not the noise.

### 4.5 Noise-model fitters

- `true_noise_model(pulsar)` — **oracle** model from simulation truth.
  Stage-1 uses this so the estimator comparison is not confounded by
  noise-estimation error.
- `fit_noise_model(pulsar)` — naive per-pulsar ML (white fixed, intrinsic
  PL). Provided as a utility; documented as misspecified under a strong
  common process.
- `fit_array_noise_model(pulsars)` — joint ML with one global common
  amplitude + per-pulsar intrinsic red. Recovers the common amplitude well
  but the intrinsic/common decomposition is degenerate (see §6).

## 5. Metrics (`src/metrics.py`)

- **HD-template-fit detection SNR:** least-squares fit of pairwise
  estimates to the HD curve, ρ_ab = amp·Γ_HD(ξ_ab); SNR = amp/σ_amp.
  Weights: Fisher-z 1/(n−3) for Pearson; per-pair σ for the OS.
- **Amplitude bias / RMSE:** of the OS A²_gwb estimate vs the injected
  A²_gwb, over Monte Carlo realizations (Stage-2).
- **False-positive rate:** fraction of null-scenario realizations with
  detection SNR above a fixed threshold (default 3.0), reported per null
  scenario (a/b/c/d).
- **Detection probability:** same fraction under injected scenarios
  (power curve input for Stage-2).

## 6. Lessons from Stage-1 development (do not regress)

1. **HD formula:** the standard PTA form uses ln(x), not ln(x/2), with
   x = (1−cos ξ)/2. The ln(x/2) variant yields a non-positive-definite
   spatial covariance (Cholesky fails). Verified: HD(60°)≈−0.082,
   HD(90°)≈−0.145, HD(180°)=0.25.
2. **Unit discipline:** residuals are simulated in microseconds but the
   PSD model is in s²/Hz; all Fourier-domain math is done in seconds.
   A stray `/SEC_PER_DAY` once made all red noise 86400× too small —
   caught because single-pulsar fits returned flat likelihoods.
3. **Shared frequency grid required:** per-pulsar Tspan grids make the
   common-process template inconsistent across pulsars (wrong by ~df
   factors); one array-wide Tspan fixes OS amplitude normalization
   (verified: recovered A² within a factor ~2 of truth over 20 seeds).
4. **Noise-model identifiability is the hard problem.** Per-pulsar ML fits
   (with or without a common term) cannot separate intrinsic from common
   red noise when a strong GWB is present — common power is absorbed into
   "intrinsic" terms, biasing P_a and collapsing OS sensitivity (SNR
   20 → 0.8). This is a *finding*, not a bug, and a core Stage-2 topic.
5. **Direct OS combination:** combine via ΣΓM/ΣΓ²T, never via per-pair
   ρ_ab = M/(ΓT) (unstable at the HD zero-crossing).

## 7. Stage-2 Monte Carlo sweep plan

**Grid:**
- **Noise strengths:** white-noise scale ×{0.5, 1, 2}; intrinsic red
  amplitude range ×{weak, fiducial, loud}; EFAC/EQUAD on/off.
- **Selection strategies** (mirroring Project 1's RMS-reduction groups):
  all pulsars; Strong-only / Moderate-only / Weak-only subsets (by
  whitened-vs-raw RMS reduction); random subsets of size N (N = 5, 8, 15);
  "best-timer" subsets (lowest white noise).
- **Estimators:** the three of §4, each with oracle *and* fitted noise
  models (quantifying the §6.4 degradation explicitly).
- **Scenarios:** GWB at log10 A ∈ {−15.0, −14.6, −14.3}; nulls (a)–(d).
- **Realizations:** ≥200 per cell (fewer for exploratory cells).

**Outputs per cell:** detection-probability curves, amplitude bias/RMSE,
FPR per null scenario, and the ranking of estimators by each metric.
**Deliverable:** a figure + table set answering "which method, under which
noise/selection, recovers the HD pattern and at what false-positive cost."

### 7.1 Stage-2 execution record (2026-10-14)

Ran a reduced grid (see `results/manifest.json` for the exact record):
12 pulsars; GWB at log10 A ∈ {−15.0, −14.6, −14.3} × 3 intrinsic-noise
levels (30 realizations/cell); nulls — pure null (3 noise levels),
loud uncorrelated red, monopole, dipole systematics at fixed −14.6
(35 realizations/cell); oracle vs per-pulsar ML noise models (nstarts=2,
maxiter=150); joint array ML fit on 2 cells × 10 realizations; Project-1-
style RMS-reduction selection groups (Strong/Moderate/Weak) from oracle
posterior-mean red-noise subtraction. 605 realizations, 8490 estimator
rows, ~33 min. Headline results in `results/RESULTS.md`: (1) OS dominates
detection (Pdet 0.43/0.88/0.93 across amplitudes, fiducial, oracle);
whitened Pearson modestly beats raw Pearson. (2) Per-pulsar ML noise fits
collapse detection (OS Pdet 0.88→0.05) and bias amplitude low (−60–75%),
because common power is absorbed into "intrinsic" red noise. (3) Joint
array fit (global common term) recovers Pdet 0.70 and the common amplitude
(−14.80±0.09 vs truth −14.60). (4) Loud uncorrelated red noise gives ~zero
false HD detections, but monopole/dipole systematics do (OS FPR
0.60/0.77 fiducial) — the better the estimator, the more confidently
fooled. (5) Selection by whitening response: Moderate-only ≈ all pulsars;
Weak-only can't detect; Strong group too sparse to evaluate (null result
for this dimension). Two Stage-1 bugs fixed en route: loud_red pulsars'
oracle parameters now match the injection; selection grouping now uses
Wiener posterior-mean subtraction (real units) instead of dimensionless
L⁻¹r whitening.

## 8. v2 extension note (enterprise/libstempo-based)

Replace residual-level injection with a full simulation pipeline:
- Simulate TOAs with libstempo/TEMPO2 (design matrix from real par files),
  inject GWB + noise at the TOA level, run the full timing-model fit
  (enterprise), and extract post-fit residuals.
- Compare estimators on post-fit vs pre-fit residuals to quantify
  timing-fit absorption of the common signal (the §2.2 limitation).
- Graduate the noise model to full Bayesian estimation (enterprise) with
  noise-marginalized optimal statistic; revisit §6.4 with proper
  hierarchical modeling.
- Gate: only after Stage-2 results justify the complexity.

## 9. Reproducibility

- Fixed seeds throughout (`synthetic_schedule`, `simulate_array`,
  `inject_common_red_noise`); the demo uses seed 7 end-to-end.
- `requirements.txt`: numpy, scipy, matplotlib only. Python 3.10+.
- No data files are bundled or downloaded; no network access needed.
