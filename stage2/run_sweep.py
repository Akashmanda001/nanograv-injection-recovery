"""Stage-2 Monte Carlo sweep for the PTA injection-recovery project.

Grid (see results/manifest.json for the exact record):
  MAIN grid -- oracle vs per-pulsar-fitted noise models:
    * GWB injections: log10 A_gwb in {-15.0, -14.6, -14.3}
      x intrinsic red noise {weak, fiducial, loud}, 30 realizations each
    * null: noise {weak, fiducial, loud}, 35 realizations each
    * loud_red / monopole / dipole: noise {fiducial, loud}, 35 each
    * selection: all / strong / moderate / weak (Project-1-style
      RMS-reduction groups from oracle whitening), applied post-hoc
    * estimators: raw Pearson, whitened Pearson, optimal statistic
  JOINT grid (reduced) -- joint array ML noise fit (has a global common
    term): gwb(-14.6)/null x fiducial noise, 10 realizations each.
    Tests whether the joint fit fixes the per-pulsar-fit degeneracy.

Per-pulsar ML fits use nstarts=2/maxiter=150 for speed (documented in the
manifest; Stage-1 used nstarts=4/maxiter=300).  A single failed pulsar fit
falls back to that pulsar's oracle model and is counted (fallback_frac in
the manifest) rather than discarding the realization.

Outputs: results/sweep_raw.npz (structured per-row records), results/
manifest.json (exact grid, seeds, timings, code notes).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulate import synthetic_schedule, simulate_array  # noqa: E402
from estimators import (raw_pearson, whitened_pearson, optimal_statistic,  # noqa: E402
                        true_noise_model, fit_noise_model,
                        fit_array_noise_model,
                        red_subtracted_residuals_us)  # noqa: E402
from metrics import hd_template_fit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

N_PSR = 12
NFREQ = 30
THRESH = 3.0
BASE_SEED = 20261014

NOISE_LEVELS = {
    "weak": (-17.5, -16.0),
    "fiducial": (-17.0, -15.0),
    "loud": (-16.0, -14.0),
}
GWB_AMPS = [-15.0, -14.6, -14.3]

ROW_DTYPE = np.dtype([
    ("scenario", "U12"), ("log10_A_gwb", "f8"), ("noise", "U10"),
    ("real", "i4"), ("estimator", "U18"), ("model", "U10"),
    ("selection", "U10"), ("n_psr", "i4"),
    ("snr", "f8"), ("A2_hat", "f8"), ("sigma_A2", "f8"),
])
EXTRA_DTYPE = np.dtype([
    ("scenario", "U12"), ("log10_A_gwb", "f8"), ("noise", "U10"),
    ("real", "i4"), ("n_strong", "i4"), ("n_moderate", "i4"),
    ("n_weak", "i4"), ("fitted_fallbacks", "i4"),
    ("joint_log10_A_common", "f8"),
])


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def rms_reduction_groups(pulsars, common_amp):
    """Project-1-style groups from full-model-whitened RMS reduction.

    Mirrors Project 1, which grouped pulsars by RMS reduction between raw
    residuals and NANOGrav's published whitened residuals (best-estimate
    red noise subtracted, real time units).  Here each pulsar's residuals
    get the oracle posterior-mean red noise subtracted
    (``red_subtracted_residuals_us``: white + intrinsic red, plus the
    common auto-term when a common process exists):
    reduction = 1 - RMS(subtracted)/RMS(raw).
    Strong >50%, Moderate 10-50%, Weak <10%.
    Returns dict selection -> pulsar indices.
    """
    groups = {"strong": [], "moderate": [], "weak": []}
    for i, p in enumerate(pulsars):
        raw = _rms(p.residuals_us)
        sub = red_subtracted_residuals_us(p, nfreq=NFREQ,
                                          log10_A_common=common_amp)
        red = 1.0 - _rms(sub) / raw if raw > 0 else 0.0
        if red > 0.50:
            groups["strong"].append(i)
        elif red > 0.10:
            groups["moderate"].append(i)
        else:
            groups["weak"].append(i)
    return groups


def _pearson_snr(rows):
    return float(hd_template_fit(rows, kind="pearson")["snr"])


def _subset(pulsars, models, idx):
    return [pulsars[i] for i in idx], [models[i] for i in idx]


def run_realization(scenario, log10_A_gwb, noise, real, seed,
                    do_joint=False, common_amp=None):
    """One realization; returns (rows, extra).

    ``common_amp`` is the log10 amplitude of the common process used for
    the full-model oracle whitening that defines the Project-1-style
    selection groups (None when no common process exists).
    """
    rows, extra = [], None
    red_range = NOISE_LEVELS[noise]
    schedule = synthetic_schedule(n_psr=N_PSR, seed=seed)
    pulsars = simulate_array(
        schedule, scenario=scenario, seed=seed + 7,
        log10_A_gwb=log10_A_gwb, log10_A_red_range=red_range)

    oracle = [true_noise_model(p, nfreq=NFREQ) for p in pulsars]
    groups = rms_reduction_groups(pulsars, common_amp)

    # --- oracle-model estimators on every selection ---
    selections = {"all": list(range(N_PSR))}
    selections.update(groups)
    for sel, idx in selections.items():
        if len(idx) < 3:
            for est in ("raw_pearson", "whitened_pearson",
                        "optimal_statistic"):
                rows.append((scenario, log10_A_gwb, noise, real, est,
                             "oracle", sel, len(idx),
                             np.nan, np.nan, np.nan))
            continue
        sub, sub_oracle = _subset(pulsars, oracle, idx)
        raw_rows = raw_pearson(sub)
        rows.append((scenario, log10_A_gwb, noise, real, "raw_pearson",
                     "oracle", sel, len(idx), _pearson_snr(raw_rows),
                     np.nan, np.nan))
        wp_rows, _ = whitened_pearson(sub, noise_models=sub_oracle)
        rows.append((scenario, log10_A_gwb, noise, real,
                     "whitened_pearson", "oracle", sel, len(idx),
                     _pearson_snr(wp_rows), np.nan, np.nan))
        os_rows, os_sum = optimal_statistic(sub, noise_models=sub_oracle)
        rows.append((scenario, log10_A_gwb, noise, real,
                     "optimal_statistic", "oracle", sel, len(idx),
                     float(os_sum["snr"]), float(os_sum["A2_hat"]),
                     float(os_sum["sigma_A2"])))

    # --- per-pulsar fitted models ("all" selection only) ---
    fitted, fallbacks = [], 0
    for p in pulsars:
        try:
            fitted.append(fit_noise_model(p, nfreq=NFREQ, nstarts=2,
                                          maxiter=150))
        except Exception:
            fitted.append(true_noise_model(p, nfreq=NFREQ))
            fallbacks += 1
    wp_rows, _ = whitened_pearson(pulsars, noise_models=fitted)
    rows.append((scenario, log10_A_gwb, noise, real, "whitened_pearson",
                 "fitted", "all", N_PSR, _pearson_snr(wp_rows),
                 np.nan, np.nan))
    os_rows, os_sum = optimal_statistic(pulsars, noise_models=fitted)
    rows.append((scenario, log10_A_gwb, noise, real, "optimal_statistic",
                 "fitted", "all", N_PSR, float(os_sum["snr"]),
                 float(os_sum["A2_hat"]), float(os_sum["sigma_A2"])))

    joint_logA = np.nan
    if do_joint:
        try:
            joint = fit_array_noise_model(pulsars, nfreq=NFREQ,
                                          maxiter=150)
            joint_logA = float(joint[0]["log10_A_common"])
            _, os_sum = optimal_statistic(pulsars, noise_models=joint)
            rows.append((scenario, log10_A_gwb, noise, real,
                         "optimal_statistic", "joint", "all", N_PSR,
                         float(os_sum["snr"]), float(os_sum["A2_hat"]),
                         float(os_sum["sigma_A2"])))
        except Exception:
            rows.append((scenario, log10_A_gwb, noise, real,
                         "optimal_statistic", "joint", "all", N_PSR,
                         np.nan, np.nan, np.nan))

    extra = (scenario, log10_A_gwb, noise, real,
             len(groups["strong"]), len(groups["moderate"]),
             len(groups["weak"]), fallbacks, joint_logA)
    return rows, extra


def build_grid():
    """(cell list, joint cell list); cell = (scenario, log10_A, noise, n)."""
    cells, joint = [], []
    for amp in GWB_AMPS:
        for nz in NOISE_LEVELS:
            cells.append(("gwb", amp, nz, 30))
    for nz in NOISE_LEVELS:
        cells.append(("null", np.nan, nz, 35))
    for sc in ("loud_red", "monopole", "dipole"):
        for nz in ("fiducial", "loud"):
            # monopole/dipole systematics injected at a fixed moderate
            # amplitude (-14.6); null/loud_red have no common process.
            amp = -14.6 if sc in ("monopole", "dipole") else np.nan
            cells.append((sc, amp, nz, 35))
    joint.append(("gwb", -14.6, "fiducial", 10))
    joint.append(("null", np.nan, "fiducial", 10))
    return cells, joint


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    cells, joint_cells = build_grid()
    all_rows, all_extra = [], []
    n_real_total = 0
    manifest = {
        "n_psr": N_PSR, "nfreq": NFREQ, "snr_threshold": THRESH,
        "base_seed": BASE_SEED,
        "noise_levels": {k: list(v) for k, v in NOISE_LEVELS.items()},
        "gwb_amps": GWB_AMPS,
        "fitted_model_settings": {"nstarts": 2, "maxiter": 150,
                                  "note": "Stage-1 used nstarts=4/maxiter=300"},
        "joint_model_settings": {"maxiter": 150},
        "selection": "Project-1-style RMS-reduction groups from oracle "
                     "FULL-model whitening (white + intrinsic red + common "
                     "auto-term when a common process exists), mirroring "
                     "Project 1's raw-vs-published-whitened grouping. "
                     "Strong>50%, Moderate 10-50%, Weak<10%; "
                     "<3 pulsars -> NaN",
        "notes": [
            "v1 residual-level injection (no timing-model fit).",
            "fitted noise models reuse full-array per-pulsar fits on "
            "selection subsets (documented simplification).",
        ],
        "cells": [], "joint_cells": [],
    }
    for ci, (sc, amp, nz, n) in enumerate(cells):
        tc = time.time()
        manifest["cells"].append({"scenario": sc,
                                  "log10_A_gwb": amp, "noise": nz,
                                  "n_realizations": n})
        for r in range(n):
            seed = BASE_SEED + ci * 10000 + r
            common_amp = (None if sc in ("null", "loud_red") else amp)
            rows, extra = run_realization(sc, amp, nz, r, seed,
                                          common_amp=common_amp)
            all_rows.extend(rows)
            all_extra.append(extra)
            n_real_total += 1
        print(f"[{ci+1}/{len(cells)}] {sc} A={amp} noise={nz} x{n} "
              f"({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)",
              flush=True)
    for ci, (sc, amp, nz, n) in enumerate(joint_cells):
        tc = time.time()
        manifest["joint_cells"].append({"scenario": sc,
                                        "log10_A_gwb": amp, "noise": nz,
                                        "n_realizations": n})
        for r in range(n):
            seed = BASE_SEED + 900000 + ci * 10000 + r
            common_amp = None if sc == "null" else amp
            rows, extra = run_realization(sc, amp, nz, r, seed,
                                          do_joint=True,
                                          common_amp=common_amp)
            all_rows.extend(rows)
            all_extra.append(extra)
            n_real_total += 1
        print(f"[joint {ci+1}/{len(joint_cells)}] {sc} A={amp} "
              f"noise={nz} x{n} ({time.time()-tc:.0f}s, "
              f"total {time.time()-t0:.0f}s)", flush=True)

    rows_arr = np.array(all_rows, dtype=ROW_DTYPE)
    extra_arr = np.array(all_extra, dtype=EXTRA_DTYPE)
    np.savez(os.path.join(RESULTS_DIR, "sweep_raw.npz"),
             rows=rows_arr, extra=extra_arr)
    fb = float(np.mean([e[7] for e in all_extra]))
    manifest["fitted_fallback_fraction"] = fb
    manifest["n_realizations_total"] = n_real_total
    manifest["runtime_s"] = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Done: {n_real_total} realizations, {len(all_rows)} rows, "
          f"{time.time()-t0:.0f}s. fallback_frac={fb:.4f}")


if __name__ == "__main__":
    main()
