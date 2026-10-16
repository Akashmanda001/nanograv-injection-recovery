"""Pulsar timing array simulator for injection-recovery studies.

Stage-1 scope (see DESIGN.md):
  * Generates per-pulsar timing *residuals* directly (not TOAs through a
    timing model).  Injecting at the residual level is a known limitation:
    real analyses absorb some common covariance into the timing-model fit.
  * White noise: per-TOA measurement uncertainties with EFAC/EQUAD scaling.
  * Red noise: power-law Gaussian process realized through a Fourier basis.
  * Common signal: Hellings-Downs-correlated red noise injected through the
    Cholesky decomposition of the HD spatial covariance matrix.
  * Null scenarios: (a) no common signal, (b) loud uncorrelated red noise in
    a few pulsars, (c) common monopole- / dipole-correlated systematics
    (clock-like / ephemeris-like errors).

Epochs can come from real NANOGrav 15-yr par/tim files (via
``load_ng15_epochs``) or from a realistic synthetic schedule that mimics
NG15 observing properties (``synthetic_schedule``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 365.25 * SEC_PER_DAY
F_YR = 1.0 / SEC_PER_YEAR  # reference frequency, Hz

# Approximate sky positions (deg) of 8 well-known NANOGrav MSPs, used only to
# set realistic angular separations between pulsars in synthetic runs.
# Values are public knowledge (NANOGrav 15-yr paper); not precise astrometry.
KNOWN_PULSARS = {
    "J1909-3744": (287.447, -37.737),
    "J1713+0747": (258.456, 7.874),
    "J1744-1134": (266.122, -11.582),
    "J1600-3053": (240.088, -30.895),
    "J1918-0642": (289.700, -6.710),
    "J1012+5307": (153.139, 53.117),
    "J1640+2224": (250.070, 22.402),
    "J2317+1439": (349.288, 14.659),
}


# ---------------------------------------------------------------------------
# Angular-separation utilities
# ---------------------------------------------------------------------------

def angular_separation(ra1, dec1, ra2, dec2):
    """Angular separation in radians between sky positions (deg)."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cosang = np.clip(
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2),
        -1.0, 1.0,
    )
    return np.arccos(cosang)


def hellings_downs(xi):
    """Hellings-Downs overlap reduction function.

    xi : angular separation in radians.  Returns the GWB spatial correlation,
    normalized so that the autocorrelation (xi = 0) is 1.
    """
    x = (1.0 - np.cos(xi)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # Standard PTA form (e.g. Chamberlin et al. 2015); autocorrelation
        # (xi = 0, including the pulsar term) is normalized to 1.
        gamma = 0.5 + 1.5 * x * np.log(x) - 0.25 * x
    return np.where(xi == 0.0, 1.0, gamma)


def monopole(xi):
    """Clock-like systematic: fully correlated across all pulsars."""
    return np.ones_like(np.asarray(xi, dtype=float))


def dipole(xi):
    """Ephemeris-like systematic: cos(separation) correlation."""
    return np.cos(xi)


# ---------------------------------------------------------------------------
# Epoch loaders
# ---------------------------------------------------------------------------

def load_ng15_epochs(data_dir):
    """Load observing epochs + TOA errors from NANOGrav 15-yr par/tim files.

    Looks for ``*.tim`` files under ``data_dir`` and parses TEMPO2-format
    lines ``<file> <freq> <mjd> <toa_err_us> <residual> [flags...]``.
    Comment lines ('#', 'C', 'MODE') are skipped.

    Returns a list of dicts with keys: name, mjds (days), toa_errors_us.
    Raises FileNotFoundError if no .tim files are found -- callers should
    fall back to ``synthetic_schedule`` (never download data automatically).
    """
    import glob

    tim_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tim"),
                                 recursive=True))
    if not tim_files:
        raise FileNotFoundError(
            f"No .tim files found under {data_dir}. "
            "Point NG15_DATA_DIR at a NANOGrav 15-yr release, or use the "
            "synthetic schedule fallback."
        )
    pulsars = []
    for path in tim_files:
        name = os.path.splitext(os.path.basename(path))[0]
        mjds, errs = [], []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in ("#", "C", "M"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    mjds.append(float(parts[2]))
                    errs.append(float(parts[3]))
                except ValueError:
                    continue
        if mjds:
            idx = np.argsort(mjds)
            pulsars.append({
                "name": name,
                "mjds": np.asarray(mjds)[idx],
                "toa_errors_us": np.asarray(errs)[idx],
            })
    if not pulsars:
        raise FileNotFoundError(f"No usable TOAs parsed under {data_dir}.")
    return pulsars


def synthetic_schedule(n_psr=8, baseline_yr=15.0, seed=0, mjd_start=53000.0,
                       nominal_cadence_days=30.0, dropout_frac=0.15):
    """Generate a realistic synthetic observing schedule mimicking NG15.

    Mimic parameters (documented approximations of NANOGrav 15-yr practice):
      * 15-yr baseline; nominal ~monthly cadence with +/-7 d jitter.
      * ~15% random epoch dropouts (weather / maintenance / RFI losses).
      * 30% of pulsars get a high-cadence final 2 yr (14-day) campaign,
        as in NG15's high-cadence timing programs.
      * Per-TOA measurement errors: lognormal, median 0.3 us, 0.4 dex spread,
        clamped to [0.05, 3] us -- typical of NANOGrav's better-timed MSPs,
        so that intrinsic red noise is marginally detectable in single-
        pulsar fits (the regime where whitening choices matter).
      * Sky positions: 8 real NANOGrav MSPs when n_psr == 8, else uniform sky.

    Returns a list of dicts: name, ra_deg, dec_deg, mjds, toa_errors_us.
    """
    rng = np.random.default_rng(seed)
    names = list(KNOWN_PULSARS)
    pulsars = []
    for i in range(n_psr):
        if i < len(names):
            name = names[i]
            ra, dec = KNOWN_PULSARS[name]
        else:
            name = f"J{1000 + i:04d}+0000"
            ra = rng.uniform(0, 360)
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        t = mjd_start
        t_end = mjd_start + baseline_yr * 365.25
        high_cadence = i < int(0.3 * n_psr)
        mjds = []
        while t < t_end:
            cad = 14.0 if (high_cadence and t > t_end - 2 * 365.25) \
                else nominal_cadence_days
            t += cad + rng.uniform(-7.0, 7.0)
            if t < t_end and rng.random() > dropout_frac:
                mjds.append(t)
        mjds = np.asarray(sorted(mjds))
        errs = rng.lognormal(mean=np.log(0.3), sigma=0.4 * np.log(10),
                             size=len(mjds))
        errs = np.clip(errs, 0.05, 3.0)
        pulsars.append({"name": name, "ra_deg": ra, "dec_deg": dec,
                        "mjds": mjds, "toa_errors_us": errs})
    return pulsars


# ---------------------------------------------------------------------------
# Noise / signal realization
# ---------------------------------------------------------------------------

def fourier_basis(mjds, nfreq=30, tspan=None):
    """Fourier design matrix (cos,sin) at frequencies k/Tspan, k=1..nfreq.

    If ``tspan`` (seconds) is given, all pulsars share the same frequency
    grid -- required so that a common red process has one consistent
    Fourier representation across the array (as in enterprise).  Otherwise
    the pulsar's own data span is used.

    Returns (F, freqs_hz): F has shape (n_toa, 2*nfreq).
    """
    t = (np.asarray(mjds) - mjds[0]) * SEC_PER_DAY  # seconds
    tspan = tspan if tspan is not None else t[-1] - t[0]
    k = np.arange(1, nfreq + 1)
    freqs = k / tspan
    phase = 2.0 * np.pi * np.outer(t, freqs)
    F = np.empty((len(t), 2 * nfreq))
    F[:, 0::2] = np.cos(phase)
    F[:, 1::2] = np.sin(phase)
    return F, freqs


def powerlaw_psd(log10_A, gamma, freqs_hz):
    """One-sided power spectral density of timing residuals (s^2/Hz).

    P(f) = A^2 / (12 pi^2) * (f / f_yr)^(-gamma) * f_yr^-3
    (standard PTA convention; A is the characteristic strain amplitude).
    """
    A = 10.0 ** log10_A
    return (A ** 2 / (12.0 * np.pi ** 2)
            * (freqs_hz / F_YR) ** (-gamma) * F_YR ** -3)


@dataclass
class Pulsar:
    """One simulated pulsar: epochs, noise parameters, and residuals."""
    name: str
    ra_deg: float
    dec_deg: float
    mjds: np.ndarray
    toa_errors_us: np.ndarray
    efac: float = 1.0
    equad_us: float = 0.0
    log10_A_red: float = -14.0
    gamma_red: float = 3.0
    # Common Fourier-grid span (seconds) shared by the whole array; set by
    # simulate_array so injection, noise fits, and the optimal statistic
    # all use one consistent frequency grid (enterprise-style).
    fourier_tspan: float = None
    residuals_us: np.ndarray = field(default=None, repr=False)


def white_noise_std_us(toa_errors_us, efac, equad_us):
    """EFAC/EQUAD-style effective per-TOA white-noise standard deviation."""
    return np.sqrt((efac * toa_errors_us) ** 2 + equad_us ** 2)


def realize_red_noise(F, freqs_hz, log10_A, gamma, rng):
    """Draw one realization of power-law red noise through the Fourier basis."""
    psd = powerlaw_psd(log10_A, gamma, freqs_hz)
    df = freqs_hz[1] - freqs_hz[0] if len(freqs_hz) > 1 else freqs_hz[0]
    # Fourier-coefficient variance per (cos,sin) component: P(f)*df/2,
    # so that sum_k P(f_k)*df reproduces the process variance.
    coeff_var = np.repeat(psd * df / 2.0, 2)
    coeffs = rng.normal(0.0, np.sqrt(coeff_var))
    return (F @ coeffs) * 1e6  # seconds -> microseconds


def _gwb_design(pulsars, nfreq, tspan):
    """Per-pulsar Fourier design matrices on the shared frequency grid."""
    Fs, all_freqs = [], None
    for p in pulsars:
        F, freqs = fourier_basis(p.mjds, nfreq, tspan)
        Fs.append(F)
        all_freqs = freqs
    return Fs, all_freqs


def inject_common_red_noise(pulsars, log10_A_gwb, gamma_gwb=13.0 / 3.0,
                            orf=hellings_downs, nfreq=30, seed=1,
                            tspan=None):
    """Add a spatially-correlated red-noise process to each pulsar's residuals.

    Correlates Fourier coefficients across pulsars through the overlap
    reduction function (Hellings-Downs for a GWB; monopole/dipole for
    systematics) via the Cholesky factor of the spatial covariance matrix.
    All pulsars share one Fourier grid (``tspan``) so the common process
    has a single consistent representation.
    """
    rng = np.random.default_rng(seed)
    n = len(pulsars)
    xi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sep = angular_separation(pulsars[i].ra_deg, pulsars[i].dec_deg,
                                     pulsars[j].ra_deg, pulsars[j].dec_deg)
            xi[i, j] = xi[j, i] = sep
    corr = np.asarray(orf(xi))
    corr += 1e-10 * np.eye(n)  # numerical jitter
    L = np.linalg.cholesky(corr)

    Fs, freqs = _gwb_design(pulsars, nfreq, tspan)
    psd = powerlaw_psd(log10_A_gwb, gamma_gwb, freqs)
    df = freqs[1] - freqs[0]
    coeff_var = np.repeat(psd * df / 2.0, 2)

    for k in range(2 * nfreq):
        z = rng.normal(0.0, np.sqrt(coeff_var[k]), size=n)
        coeffs = L @ z  # spatially correlated Fourier coefficients
        for i, p in enumerate(pulsars):
            contrib = (Fs[i][:, k] * coeffs[i]) * 1e6  # s -> us
            p.residuals_us = (p.residuals_us if p.residuals_us is not None
                              else np.zeros_like(p.mjds)) + contrib


def simulate_array(schedule, scenario="gwb", seed=42, nfreq=30,
                   log10_A_gwb=-14.6, gamma_gwb=13.0 / 3.0,
                   red_noise_loud_pulsars=2,
                   log10_A_red_range=None, white_scale=1.0):
    """Build a full simulated pulsar array under a chosen scenario.

    schedule : list of dicts from ``synthetic_schedule`` / ``load_ng15_epochs``
    scenario : one of
        'gwb'      - HD-correlated common red noise + per-pulsar noise
        'null'     - per-pulsar white + red noise only, no common signal
        'loud_red' - like 'null' but ``red_noise_loud_pulsars`` pulsars get
                     very loud *uncorrelated* red noise (false-positive test)
        'monopole' - common monopole-correlated systematic (clock-like)
        'dipole'   - common dipole-correlated systematic (ephemeris-like)
    log10_A_red_range : (lo, hi) range for the intrinsic red-noise amplitude
        draw; defaults to (-17.0, -15.0) (Stage-1 behavior).  Stage-2 sweeps
        weak/fiducial/loud intrinsic noise through this parameter.
    white_scale : multiplicative factor on per-TOA measurement uncertainties.
    """
    rng = np.random.default_rng(seed)
    red_range = (-17.0, -15.0) if log10_A_red_range is None \
        else tuple(log10_A_red_range)
    pulsars = []
    for entry in schedule:
        mjds = np.asarray(entry["mjds"], dtype=float)
        p = Pulsar(
            name=entry["name"],
            ra_deg=entry.get("ra_deg", 0.0),
            dec_deg=entry.get("dec_deg", 0.0),
            mjds=mjds,
            toa_errors_us=(np.asarray(entry["toa_errors_us"], dtype=float)
                           * white_scale),
            efac=float(rng.uniform(1.0, 1.3)),
            equad_us=float(rng.uniform(0.0, 0.3)),
            # Intrinsic red noise kept modest (mostly below the GWB level),
            # as for the best PTA pulsars where the common process dominates
            # the low-frequency band.  Keeps the single-pulsar noise fits
            # identifiable: the joint fit recovers the common amplitude
            # without the intrinsic/common degeneracy swamping P_a.
            log10_A_red=float(rng.uniform(*red_range)),
            gamma_red=float(rng.uniform(1.5, 5.5)),
        )
        pulsars.append(p)
    # One shared Fourier grid for the whole array (enterprise-style).
    tspan = max((p.mjds[-1] - p.mjds[0]) for p in pulsars) * SEC_PER_DAY
    for idx, p in enumerate(pulsars):
        p.fourier_tspan = tspan
        wstd = white_noise_std_us(p.toa_errors_us, p.efac, p.equad_us)
        white = rng.normal(0.0, wstd)
        F, freqs = fourier_basis(p.mjds, nfreq, tspan)
        loud = scenario == "loud_red" and idx < red_noise_loud_pulsars
        if loud:
            # Record the loud parameters on the Pulsar so the *oracle*
            # noise model (true parameters) actually matches the truth.
            # Loud but plausible: -13.0 is at the upper end of NANOGrav
            # intrinsic red noise (not the absurd -12.5 that breaks
            # matrix conditioning).
            p.log10_A_red = -13.0
            p.gamma_red = 4.0
        red = realize_red_noise(
            F, freqs, log10_A=p.log10_A_red, gamma=p.gamma_red, rng=rng)
        p.residuals_us = white + red

    orf = {"gwb": hellings_downs,
           "monopole": monopole,
           "dipole": dipole}.get(scenario)
    if orf is not None:
        inject_common_red_noise(pulsars, log10_A_gwb, gamma_gwb,
                                orf=orf, nfreq=nfreq, seed=seed + 999,
                                tspan=tspan)
    return pulsars
