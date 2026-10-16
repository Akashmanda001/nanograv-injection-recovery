"""The three correlation estimators of the method ladder.

  1. ``raw_pearson`` -- pairwise Pearson correlations on epoch-matched
     residuals (the Project-1 baseline).
  2. ``whitened_pearson`` -- each pulsar is whitened by its fitted
     *intrinsic* (white + red, no common term) noise model first, then
     epoch-matched Pearson.  Whitening by the intrinsic model keeps a
     common signal in the residuals, so this is a fair detection
     comparison; whitening by the *full* model (``whitened_full_us``)
     removes the common signal, mirroring NANOGrav's published
     whitened residuals.
  3. ``optimal_statistic`` -- the noise-weighted Hellings-Downs matched
     filter of Chamberlin et al. (2015).  This *is* the PTA optimal
     statistic, not a later extension of it: pairwise cross-correlations
     weighted by inverse noise covariance and projected onto the HD
     template.  The full Bayesian hierarchical analysis remains the
     later (Stage-2+) extension.

All three operate on epoch-matched residuals where matching is required;
the optimal statistic needs no epoch matching because it works through
each pulsar's own Fourier design matrix.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from simulate import (fourier_basis, powerlaw_psd, white_noise_std_us,
                    angular_separation, hellings_downs)


# ---------------------------------------------------------------------------
# Epoch matching (Project-1 style: matched approximately by observing time)
# ---------------------------------------------------------------------------

def match_epochs(mjds_a, mjds_b, tol_days=10.0):
    """Greedy nearest-neighbor epoch matching within a tolerance.

    Returns index arrays (ia, ib) pairing each epoch of ``a`` with the
    nearest unused epoch of ``b`` closer than ``tol_days``.
    """
    ia, ib = [], []
    used = np.zeros(len(mjds_b), dtype=bool)
    order = np.argsort(mjds_a)
    for i in order:
        d = np.abs(mjds_b - mjds_a[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_days:
            ia.append(i)
            ib.append(j)
            used[j] = True
    return np.asarray(ia), np.asarray(ib)


def _pairwise_frame(pulsars):
    """Angular separations and name pairs for all pulsar pairs."""
    pairs = []
    for i in range(len(pulsars)):
        for j in range(i + 1, len(pulsars)):
            sep = angular_separation(pulsars[i].ra_deg, pulsars[i].dec_deg,
                                     pulsars[j].ra_deg, pulsars[j].dec_deg)
            pairs.append((i, j, np.degrees(sep)))
    return pairs


def raw_pearson(pulsars, tol_days=10.0, min_epochs=10):
    """Pairwise Pearson r on epoch-matched raw residuals."""
    out = []
    for i, j, sep_deg in _pairwise_frame(pulsars):
        ia, ib = match_epochs(pulsars[i].mjds, pulsars[j].mjds, tol_days)
        r = np.nan
        if len(ia) >= min_epochs:
            a = pulsars[i].residuals_us[ia]
            b = pulsars[j].residuals_us[ib]
            if np.std(a) > 0 and np.std(b) > 0:
                r = float(np.corrcoef(a, b)[0, 1])
        out.append({"pair": (pulsars[i].name, pulsars[j].name),
                    "angsep_deg": sep_deg, "rho": r,
                    "n_matched": len(ia)})
    return out


# ---------------------------------------------------------------------------
# Per-pulsar noise model: fit, then whiten
# ---------------------------------------------------------------------------

def true_noise_model(pulsar, nfreq=30, include_common=False,
                     log10_A_common=None, gamma_common=13.0 / 3.0):
    """Oracle noise model built from the simulation's true parameters.

    Stage-1 uses this for the estimator comparison so that noise-model
    *estimation* error does not confound the comparison of the estimators
    themselves.  (Per-pulsar ML fits in the presence of a strong common
    process are fundamentally degenerate -- the common power is absorbed
    into "intrinsic" red noise, biasing P_a and collapsing optimal-
    statistic sensitivity.  Quantifying that degradation is a Stage-2
    topic; see DESIGN.md.)

    If ``include_common``, the common auto-term at ``log10_A_common`` is
    added (for reference; the weak-signal OS uses the intrinsic-only P_a).
    """
    r, wstd, F, freqs = _prep_pulsar(pulsar, nfreq)
    df = freqs[1] - freqs[0]
    v = np.repeat(powerlaw_psd(pulsar.log10_A_red, pulsar.gamma_red,
                               freqs) * df / 2.0, 2)
    if include_common and log10_A_common is not None:
        v = v + (10.0 ** (2 * log10_A_common)) * np.repeat(
            powerlaw_psd(0.0, gamma_common, freqs) * df / 2.0, 2)
    C = np.diag(wstd ** 2) + (F * v) @ F.T + 1e-12 * np.eye(len(r))
    L = np.linalg.cholesky(C)
    return {"log10_A_red": pulsar.log10_A_red,
            "gamma_red": pulsar.gamma_red,
            "log10_A_common": log10_A_common,
            "whitened_us": (np.linalg.solve(L, r)) * 1e6,
            "noise_cov": C, "fourier_F": F, "fourier_freqs": freqs}


def _noise_nll(r, wstd, F, Phi_c, df, freqs, logA_red, gamma_red, logA_c):
    """Negative log marginal likelihood for white + red + common model."""
    v = (np.repeat(powerlaw_psd(logA_red, gamma_red, freqs) * df / 2.0, 2)
         + (10.0 ** (2 * logA_c)) * np.repeat(Phi_c, 2))
    C = np.diag(wstd ** 2) + (F * v) @ F.T + 1e-12 * np.eye(len(r))
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return 1e30, None
    w = np.linalg.solve(L, r)
    return 0.5 * (2.0 * np.sum(np.log(np.diag(L))) + w @ w), C


def _prep_pulsar(pulsar, nfreq):
    r = np.asarray(pulsar.residuals_us, dtype=float) * 1e-6  # us -> s
    wstd = white_noise_std_us(pulsar.toa_errors_us, pulsar.efac,
                              pulsar.equad_us) * 1e-6
    tspan = getattr(pulsar, "fourier_tspan", None)
    F, freqs = fourier_basis(pulsar.mjds, nfreq, tspan)
    return r, wstd, F, freqs


def fit_noise_model(pulsar, nfreq=30, nstarts=4, maxiter=300):
    """Naive single-pulsar ML fit of white + intrinsic power-law red noise.

    No common-process term: this is the simplest "whiten by your own noise
    model" choice.  It is misspecified when a common signal is present --
    the intrinsic term absorbs common power -- which is precisely one of
    the whitening choices Stage-2 quantifies.  Prefer
    ``fit_array_noise_model`` for the optimal statistic.

    ``nstarts``/``maxiter`` control the multi-start L-BFGS-B search
    (defaults reproduce Stage-1 behavior; the Stage-2 sweep uses fewer
    starts for speed -- documented in the sweep manifest).
    """
    r, wstd, F, freqs = _prep_pulsar(pulsar, nfreq)
    df = freqs[1] - freqs[0]
    Phi_c = np.zeros_like(freqs)  # no common term

    def nll(theta):
        val, _ = _noise_nll(r, wstd, F, Phi_c, df, freqs,
                            theta[0], theta[1], -30.0)
        return val

    starts = [(-15.0, 2.5), (-13.5, 4.5), (-16.0, 3.5), (-14.0, 5.0)]
    best = (np.inf, None)
    for la0, g0 in starts[:max(1, nstarts)]:
        res = optimize.minimize(
            nll, x0=np.array([la0, g0]),
            bounds=[(-18.0, -11.0), (0.5, 7.0)],
            method="L-BFGS-B", options={"maxiter": maxiter})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    logA_red, gamma_red = best[1]
    _, C = _noise_nll(r, wstd, F, Phi_c, df, freqs, logA_red, gamma_red,
                      -30.0)
    L = np.linalg.cholesky(C)
    return {"log10_A_red": float(logA_red), "gamma_red": float(gamma_red),
            "log10_A_common": -np.inf,
            "whitened_us": (np.linalg.solve(L, r)) * 1e6,
            "noise_cov": C, "fourier_F": F, "fourier_freqs": freqs}


def fit_array_noise_model(pulsars, nfreq=30, gamma_common=13.0 / 3.0,
                          maxiter=300):
    """Joint ML noise fit: one global common-process amplitude + per-pulsar
    intrinsic red noise (fixed white noise from TOA uncertainties).

    The common term (spectral index fixed to ``gamma_common``) absorbs the
    GWB's contribution to each pulsar's *auto*-covariance, so the intrinsic
    fits are not forced to soak up common power.  This is the poor-man's
    version of the Bayesian common-process noise model used in real
    analyses, and the recommended Stage-1 noise model.

    Returns a list of per-pulsar dicts with fitted parameters,
    ``whitened_us`` (whitened by the *intrinsic-only* model, so a common
    signal survives for the whitened-Pearson comparison),
    ``whitened_full_us`` (whitened by the full model, NANOGrav-style),
    ``noise_cov`` (weak-signal P_a = white + intrinsic, for the optimal
    statistic), ``noise_cov_full`` (white + intrinsic + common auto-term,
    for reference), ``fourier_F``, and ``fourier_freqs``.
    """
    prepped = [_prep_pulsar(p, nfreq) for p in pulsars]
    freqs = prepped[0][3]
    df = freqs[1] - freqs[0]
    Phi_c = powerlaw_psd(0.0, gamma_common, freqs) * df / 2.0
    n = len(pulsars)

    def total_nll(theta):
        logA_c = theta[0]
        tot = 0.0
        for i, (r, wstd, F, _) in enumerate(prepped):
            val, _ = _noise_nll(r, wstd, F, Phi_c, df, freqs,
                                theta[1 + 2 * i], theta[2 + 2 * i], logA_c)
            if val >= 1e30:
                return 1e30
            tot += val
        return tot

    x0 = np.array([-14.5] + [-14.0, 3.0] * n)
    bounds = [(-18.0, -11.0)] + [(-18.0, -11.0), (0.5, 7.0)] * n
    res = optimize.minimize(total_nll, x0=x0, bounds=bounds,
                            method="L-BFGS-B",
                            options={"maxiter": maxiter})
    theta = res.x
    logA_c = float(theta[0])
    models = []
    for i, (r, wstd, F, _) in enumerate(prepped):
        la, g = float(theta[1 + 2 * i]), float(theta[2 + 2 * i])
        _, C_full = _noise_nll(r, wstd, F, Phi_c, df, freqs, la, g, logA_c)
        _, C_intr = _noise_nll(r, wstd, F, Phi_c, df, freqs, la, g, -30.0)
        w_intr = np.linalg.solve(np.linalg.cholesky(C_intr), r) * 1e6
        w_full = np.linalg.solve(np.linalg.cholesky(C_full), r) * 1e6
        models.append({"log10_A_red": la, "gamma_red": g,
                       "log10_A_common": logA_c,
                       "whitened_us": w_intr,
                       "whitened_full_us": w_full,
                       # P_a for the optimal statistic: the weak-signal
                       # noise model (white + intrinsic red, NO common
                       # term).  Including the GWB's own auto-term in P_a
                       # suppresses exactly the modes carrying the signal
                       # (verified: SNR 9.1 -> 0.35 on the same
                       # realization); the weak-signal OS of
                       # Chamberlin et al. (2015) treats the common
                       # process as the signal, not the noise.
                       "noise_cov": C_intr,
                       "noise_cov_full": C_full,
                       "fourier_F": F, "fourier_freqs": freqs})
    return models


def red_subtracted_residuals_us(pulsar, nfreq=30, log10_A_red=None,
                                gamma_red=None, log10_A_common=None,
                                gamma_common=13.0 / 3.0):
    """Residuals with the posterior-mean red noise subtracted (Wiener filter).

    This is the simulation analog of NANOGrav's *published whitened
    residuals*: the best-estimate red-noise realization is subtracted and
    the result keeps real time units (microseconds) -- unlike the
    ``whitened_us`` fields elsewhere, which are L^{-1} r (dimensionless,
    unit RMS by construction) and therefore useless for RMS-reduction
    comparisons.  Used for the Project-1-style RMS-reduction selection
    groups: reduction = 1 - RMS(subtracted)/RMS(raw).

    With ``log10_A_common`` given, the common auto-term is included in the
    red-noise model (single-pulsar view: the common process's
    autocorrelation; cross-pulsar correlations cannot be subtracted
    single-pulsar and remain in the residuals).
    """
    r, wstd, F, freqs = _prep_pulsar(pulsar, nfreq)
    la = pulsar.log10_A_red if log10_A_red is None else log10_A_red
    g = pulsar.gamma_red if gamma_red is None else gamma_red
    df = freqs[1] - freqs[0]
    v = np.repeat(powerlaw_psd(la, g, freqs) * df / 2.0, 2)
    if log10_A_common is not None:
        v = v + (10.0 ** (2 * log10_A_common)) * np.repeat(
            powerlaw_psd(0.0, gamma_common, freqs) * df / 2.0, 2)
    Ninv = 1.0 / wstd ** 2
    # Posterior mean of Fourier coeffs via Woodbury:
    #   (V^{-1} + F^T N^{-1} F) mu = F^T N^{-1} r
    A = np.diag(1.0 / v) + (F * Ninv[:, None]).T @ F
    b = F.T @ (Ninv * r)
    mu = np.linalg.solve(A, b)
    return (r - F @ mu) * 1e6


def whitened_pearson(pulsars, tol_days=10.0, min_epochs=10, nfreq=30,
                     noise_models=None):
    """Pairwise Pearson r on epoch-matched *whitened* residuals."""
    if noise_models is None:
        noise_models = [fit_noise_model(p, nfreq) for p in pulsars]
    out = []
    for i, j, sep_deg in _pairwise_frame(pulsars):
        ia, ib = match_epochs(pulsars[i].mjds, pulsars[j].mjds, tol_days)
        r = np.nan
        if len(ia) >= min_epochs:
            a = noise_models[i]["whitened_us"][ia]
            b = noise_models[j]["whitened_us"][ib]
            if np.std(a) > 0 and np.std(b) > 0:
                r = float(np.corrcoef(a, b)[0, 1])
        out.append({"pair": (pulsars[i].name, pulsars[j].name),
                    "angsep_deg": sep_deg, "rho": r,
                    "n_matched": len(ia)})
    return out, noise_models


# ---------------------------------------------------------------------------
# Optimal statistic (Chamberlin et al. 2015)
# ---------------------------------------------------------------------------

def optimal_statistic(pulsars, nfreq=30, gamma_gwb=13.0 / 3.0,
                      noise_models=None):
    """Noise-weighted HD-template matched filter over pulsar pairs.

    The amplitude estimator (Chamberlin et al. 2015) is
        A^2_hat = sum_{a<b} Gamma_ab * M_ab / sum_{a<b} Gamma_ab^2 * T_ab,
    where M_ab = r_a^T P_a^-1 S_ab P_b^-1 r_b and
    T_ab = tr(P_a^-1 S_ab P_b^-1 S_ba), with S_ab the *unit-amplitude*
    HD-correlated GWB template (Fourier-basis, Gamma-free) and P the
    per-pulsar noise covariance.  Combining via the Gamma-free M_ab/T_ab
    avoids dividing by small HD coefficients.  Per-pair rho_ab values are
    returned for diagnostics/plotting only.

    Returns (pairwise list, summary dict with A_gwb^2 estimate and error).
    """
    if noise_models is None:
        noise_models = [fit_noise_model(p, nfreq) for p in pulsars]

    Z, Qb = [], []
    for m in noise_models:
        F, freqs = m["fourier_F"], m["fourier_freqs"]
        Cinv = np.linalg.inv(m["noise_cov"])
        # residuum back-projection through whitened inverse covariance:
        # r is recovered below from the pulsar list by index
        Z.append(F.T @ Cinv)
        Qb.append((F.T @ Cinv) @ F)
    Phi = powerlaw_psd(0.0, gamma_gwb, noise_models[0]["fourier_freqs"])
    freqs0 = noise_models[0]["fourier_freqs"]
    df = freqs0[1] - freqs0[0]
    # Unit-amplitude template weight per frequency: the GWB induces
    # Cov(r_a, r_b) = A_gwb^2 * Gamma_ab * sum_k [Phi_k * df/2 * F F^T],
    # so the matched-filter template must carry the same df/2 factor.
    W = Phi * df / 2.0
    nf = nfreq
    Qb = [Q.reshape(nf, 2, nf, 2) for Q in Qb]

    pair_rows = []
    num_A2 = 0.0  # sum_ab Gamma_ab * M_ab
    den_A2 = 0.0  # sum_ab Gamma_ab^2 * T_ab
    for i, j, sep_deg in _pairwise_frame(pulsars):
        gamma_ab = float(hellings_downs(np.radians(sep_deg)))
        ra = pulsars[i].residuals_us * 1e-6  # us -> s (match P, S units)
        rb = pulsars[j].residuals_us * 1e-6
        Za, Zb = Z[i] @ ra, Z[j] @ rb          # (2nf,) Fourier amplitudes
        Za = Za.reshape(nf, 2)
        Zb = Zb.reshape(nf, 2)
        # Gamma-free numerator and trace; the HD coefficient enters only
        # in the final combination (numerically stable -- no division by
        # small Gamma_ab).
        M_ab = float(np.sum(W * np.sum(Za * Zb, axis=1)))
        frob = np.einsum("klmn->km", Qb[i] * Qb[j])
        T_ab = float(np.sum(np.outer(W, W) * frob))
        # Per-pair correlation (diagnostic only; unstable when |Gamma|~0).
        rho_ab = M_ab / (gamma_ab * T_ab) if T_ab > 0 and gamma_ab != 0 \
            else np.nan
        sigma_ab = (1.0 / (abs(gamma_ab) * np.sqrt(T_ab))
                    if T_ab > 0 and gamma_ab != 0 else np.inf)
        pair_rows.append({"pair": (pulsars[i].name, pulsars[j].name),
                          "angsep_deg": sep_deg, "rho": rho_ab,
                          "sigma": sigma_ab, "gamma_ab": gamma_ab})
        num_A2 += gamma_ab * M_ab
        den_A2 += gamma_ab ** 2 * T_ab

    A2_hat = num_A2 / den_A2 if den_A2 > 0 else np.nan
    sigma_A2 = 1.0 / np.sqrt(den_A2) if den_A2 > 0 else np.inf
    summary = {"A2_hat": float(A2_hat), "sigma_A2": float(sigma_A2),
               "snr": float(A2_hat / sigma_A2)
               if np.isfinite(sigma_A2) and sigma_A2 > 0 else np.nan}
    return pair_rows, summary
