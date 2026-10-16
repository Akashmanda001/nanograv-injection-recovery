"""Stage-1 demo: one injected realization, 8 pulsars, all three estimators.

Runs end-to-end in well under two minutes on a laptop CPU.  Produces:
  * demo/figures/demo_hd_comparison.png -- measured pairwise correlations
    vs the Hellings-Downs curve, one panel per estimator.
  * printed metrics: HD-template-fit detection SNR per method, the
    optimal-statistic A_gwb^2 estimate vs the injected value, and the
    per-pulsar fitted noise parameters used for whitening.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulate import synthetic_schedule, simulate_array, hellings_downs  # noqa: E402
from estimators import (raw_pearson, whitened_pearson, optimal_statistic,
                        true_noise_model)  # noqa: E402
from metrics import hd_template_fit  # noqa: E402

N_PSR = 8
LOG10_A_GWB = -14.6          # injected GWB amplitude (NANOGrav-15yr-like)
GAMMA_GWB = 13.0 / 3.0
SEED = 7
FIG_PATH = os.path.join(os.path.dirname(__file__), "figures",
                        "demo_hd_comparison.png")


def main():
    t0 = time.time()
    print(f"Simulating {N_PSR} pulsars, scenario='gwb', "
          f"log10 A_gwb={LOG10_A_GWB} ...")
    schedule = synthetic_schedule(n_psr=N_PSR, seed=SEED)
    pulsars = simulate_array(schedule, scenario="gwb", seed=SEED,
                             log10_A_gwb=LOG10_A_GWB, gamma_gwb=GAMMA_GWB)
    ntoa = sum(len(p.mjds) for p in pulsars)
    print(f"  {ntoa} TOAs total "
          f"({time.time() - t0:.1f}s)")

    # Stage-1 uses ORACLE noise models (true simulation parameters) so the
    # estimator comparison is not confounded by noise-estimation error.
    # Per-pulsar ML noise estimation in the presence of a strong common
    # process is a separate, hard problem -- see DESIGN.md (Stage-2).
    print("Building oracle noise models (true parameters) ...")
    models = [true_noise_model(p) for p in pulsars]
    print(f"  {len(models)} pulsars, intrinsic red noise whitened; "
          f"common signal retained")

    print("Running estimators ...")
    raw = raw_pearson(pulsars)
    white, _ = whitened_pearson(pulsars, noise_models=models)
    os_pairs, os_sum = optimal_statistic(pulsars, noise_models=models)

    m_raw = hd_template_fit(raw, kind="pearson")
    m_white = hd_template_fit(white, kind="pearson")
    # For the optimal statistic, the proper amplitude/SNR comes from the
    # direct summary (Chamberlin et al.); the per-pair rho_ab are for
    # plotting only.
    m_os = {"amp": os_sum["A2_hat"], "sigma_amp": os_sum["sigma_A2"],
            "snr": os_sum["snr"]}

    A2_true = 10.0 ** (2 * LOG10_A_GWB)
    print("\n--- HD-template-fit detection SNR (single realization) ---")
    print(f"raw Pearson       : SNR = {m_raw['snr']:+.2f} "
          f"(amp={m_raw['amp']:+.3f} +- {m_raw['sigma_amp']:.3f})")
    print(f"whitened Pearson  : SNR = {m_white['snr']:+.2f} "
          f"(amp={m_white['amp']:+.3f} +- {m_white['sigma_amp']:.3f})")
    print(f"optimal statistic : SNR = {m_os['snr']:+.2f} "
          f"(amp={m_os['amp']:+.2e} +- {m_os['sigma_amp']:.2e})")
    print("\n--- Optimal-statistic amplitude recovery ---")
    print(f"A_gwb^2 recovered : {os_sum['A2_hat']:.2e} +- {os_sum['sigma_A2']:.2e}")
    print(f"A_gwb^2 injected  : {A2_true:.2e}")
    print(f"bias (1 real.)    : {os_sum['A2_hat'] - A2_true:+.2e}")

    # ---- figure ----
    xi = np.linspace(0.5, 179.5, 400)
    hd = hellings_downs(np.radians(xi))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    panels = [
        ("raw Pearson", raw, m_raw, "pearson"),
        ("whitened Pearson", white, m_white, "pearson"),
        ("optimal statistic", os_pairs, m_os, "os"),
    ]
    for ax, (title, rows, m, kind) in zip(axes, panels):
        xs = np.array([r["angsep_deg"] for r in rows])
        ys = np.array([r["rho"] for r in rows])
        ok = np.isfinite(ys)
        if kind == "os":
            sig = np.array([r["sigma"] for r in rows])
            ax.errorbar(xs[ok], ys[ok], yerr=sig[ok], fmt="o", ms=4,
                        alpha=0.7, capsize=2)
            ax.plot(xi, m["amp"] * hd, "r-", lw=2, label="fitted HD template")
            ax.set_ylabel(r"$\hat{\rho}_{ab}$  [$A_{\rm gwb}^2$ units]")
        else:
            ax.scatter(xs[ok], ys[ok], s=28, alpha=0.7)
            ax.plot(xi, m["amp"] * hd, "r-", lw=2, label="fitted HD template")
            ax.plot(xi, hd, "k--", lw=1, label="unit-amplitude HD")
            ax.set_ylabel("Pearson r")
        ax.set_xlabel("angular separation [deg]")
        ax.set_title(f"{title}\nHD-fit SNR = {m['snr']:+.2f}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Injection-recovery demo: 8 pulsars, "
                 f"log10 A_gwb = {LOG10_A_GWB} (1 realization)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    print(f"\nFigure saved to {FIG_PATH}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
