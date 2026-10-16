"""Detection and recovery metrics for the estimator comparison.

Single-realization metrics (used by the demo):
  * ``hd_template_fit`` -- least-squares fit of pairwise estimates to the
    Hellings-Downs template; returns the fitted correlation amplitude, its
    uncertainty, and the detection SNR = amplitude / uncertainty.
  * ``amplitude_bias`` / ``amplitude_rmse`` -- recovery accuracy of the
    optimal-statistic A_gwb^2 estimate against the injected value.

Ensemble metrics (used by the Stage-2 Monte Carlo sweep):
  * ``false_positive_rate`` -- fraction of null-scenario realizations whose
    detection SNR exceeds a fixed threshold.
  * ``detection_probability`` -- same fraction under injected scenarios.
"""

from __future__ import annotations

import numpy as np

from simulate import hellings_downs


def hd_template_fit(pair_rows, kind="pearson"):
    """Fit pairwise estimates to the Hellings-Downs template.

    Model: rho_ab = amp * Gamma_HD(xi_ab), Gamma from ``hellings_downs``.
    Weights: Fisher-z variance 1/(n-3) for Pearson methods ("pearson"),
    or the optimal-statistic's own per-pair sigma ("os").

    Returns dict(amp, sigma_amp, snr).
    """
    xi = np.radians([r["angsep_deg"] for r in pair_rows])
    gamma = hellings_downs(xi)
    rho = np.array([r["rho"] for r in pair_rows], dtype=float)
    ok = np.isfinite(rho) & np.isfinite(gamma)
    rho, gamma = rho[ok], gamma[ok]
    if len(rho) < 3:
        return {"amp": np.nan, "sigma_amp": np.inf, "snr": np.nan}
    if kind == "os":
        sig = np.array([r["sigma"] for r in pair_rows], dtype=float)[ok]
        w = np.where(np.isfinite(sig) & (sig > 0), 1.0 / sig ** 2, 0.0)
    else:  # pearson: Fisher-z variance of r is ~1/(n-3)
        n = np.array([r.get("n_matched", 30) for r in pair_rows],
                     dtype=float)[ok]
        w = np.clip(n - 3.0, 1.0, None)
    wsum = np.sum(w * gamma ** 2)
    if wsum <= 0:
        return {"amp": np.nan, "sigma_amp": np.inf, "snr": np.nan}
    amp = np.sum(w * gamma * rho) / wsum
    sigma_amp = 1.0 / np.sqrt(wsum)
    return {"amp": float(amp), "sigma_amp": float(sigma_amp),
            "snr": float(amp / sigma_amp)}


def amplitude_bias(estimates, true_value):
    """Mean signed error of recovered amplitudes vs the injected value."""
    estimates = np.asarray(estimates, dtype=float)
    ok = np.isfinite(estimates)
    if not np.any(ok):
        return np.nan
    return float(np.mean(estimates[ok] - true_value))


def amplitude_rmse(estimates, true_value):
    """Root-mean-square error of recovered amplitudes vs injected value."""
    estimates = np.asarray(estimates, dtype=float)
    ok = np.isfinite(estimates)
    if not np.any(ok):
        return np.nan
    return float(np.sqrt(np.mean((estimates[ok] - true_value) ** 2)))


def false_positive_rate(snrs, threshold=3.0):
    """Fraction of null-scenario realizations with SNR above threshold."""
    snrs = np.asarray(snrs, dtype=float)
    ok = np.isfinite(snrs)
    if not np.any(ok):
        return np.nan
    return float(np.mean(snrs[ok] > threshold))


def detection_probability(snrs, threshold=3.0):
    """Fraction of injected-scenario realizations with SNR above threshold."""
    return false_positive_rate(snrs, threshold)
