"""Stage-2 analysis: aggregate sweep results into figures + headline numbers.

Reads results/sweep_raw.npz + results/manifest.json; writes
results/figures/fig{1..5}_*.png and prints summary tables (captured for
RESULTS.md).
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
THRESH = 3.0


def load():
    d = np.load(os.path.join(RESULTS_DIR, "sweep_raw.npz"))
    with open(os.path.join(RESULTS_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)
    return d["rows"], d["extra"], manifest


def sel(rows, **kw):
    m = np.ones(len(rows), dtype=bool)
    for k, v in kw.items():
        col = rows[k]
        if isinstance(v, float) and np.isnan(v):
            m &= np.isnan(col.astype(float))
        else:
            m &= (col == v)
    return rows[m]


def snr_stats(sub):
    s = sub["snr"].astype(float)
    ok = s[np.isfinite(s)]
    if len(ok) == 0:
        return np.nan, np.nan, 0, np.nan
    return (float(np.mean(ok)), float(np.std(ok)), len(ok),
            float(np.mean(ok > THRESH)))


def os_bias_rmse(sub, true_A2):
    a = sub["A2_hat"].astype(float)
    ok = a[np.isfinite(a)]
    if len(ok) == 0:
        return np.nan, np.nan, 0
    bias = float(np.mean(ok - true_A2))
    rmse = float(np.sqrt(np.mean((ok - true_A2) ** 2)))
    return bias, rmse, len(ok)


def main():
    rows, extra, manifest = load()
    os.makedirs(FIG_DIR, exist_ok=True)
    amps = [-15.0, -14.6, -14.3]
    noises = ["weak", "fiducial", "loud"]
    nulls = ["null", "loud_red", "monopole", "dipole"]
    ests = ["raw_pearson", "whitened_pearson", "optimal_statistic"]
    est_labels = {"raw_pearson": "raw Pearson",
                  "whitened_pearson": "whitened Pearson",
                  "optimal_statistic": "optimal statistic"}
    colors = {"raw_pearson": "C0", "whitened_pearson": "C1",
              "optimal_statistic": "C2"}

    # ---------------- Fig 1: SNR vs injected amplitude ----------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for est in ests:
        # raw_pearson needs no noise model: stored as oracle only
        models = ("oracle",) if est == "raw_pearson" else ("oracle", "fitted")
        for model, ls in zip(models, ("-", "--")):
            means, stds = [], []
            for amp in amps:
                sub = sel(rows, scenario="gwb", log10_A_gwb=amp,
                          noise="fiducial", estimator=est, model=model,
                          selection="all")
                mu, sd, n, _ = snr_stats(sub)
                means.append(mu); stds.append(sd)
            if not any(np.isfinite(stds)):
                continue  # no usable realizations for this combo
            ax.errorbar(amps, means, yerr=stds, fmt="o", color=colors[est],
                        ls=ls, capsize=3,
                        label=f"{est_labels[est]} ({model})")
    ax.axhline(THRESH, color="k", ls=":", lw=1, label="detection threshold")
    ax.set_yscale("symlog", linthresh=3.0)
    ax.set_xlabel(r"injected $\log_{10} A_{\rm gwb}$")
    ax.set_ylabel("detection SNR (mean ± std over realizations)")
    ax.set_title("Fig 1: HD detection SNR vs injected GWB amplitude\n"
                 "12 pulsars, fiducial noise, all pulsars")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig1_snr_vs_amplitude.png",
                                    dpi=130)
    plt.close(fig)

    # ---------------- Fig 2: false-positive rates ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(nulls))
    width = 0.16
    order = [("raw_pearson", "oracle"), ("whitened_pearson", "oracle"),
             ("whitened_pearson", "fitted"), ("optimal_statistic", "oracle"),
             ("optimal_statistic", "fitted")]
    labels = ["raw", "wp oracle", "wp fitted", "OS oracle", "OS fitted"]
    for ax, nz in zip(axes, ["fiducial", "loud"]):
        for i, ((est, model), lab) in enumerate(zip(order, labels)):
            fprs = []
            for sc in nulls:
                sub = sel(rows, scenario=sc, noise=nz, estimator=est,
                          model=model, selection="all")
                _, _, _, fpr = snr_stats(sub)
                fprs.append(fpr)
            ax.bar(x + (i - 2) * width, fprs, width, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(nulls, rotation=15)
        ax.set_title(f"noise: {nz}")
        ax.set_ylabel("false-positive rate (SNR > 3)")
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 2: False-positive rate per null scenario and estimator\n"
                 "12 pulsars, all pulsars")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig2_false_positive_rates.png",
                                    dpi=130)
    plt.close(fig)

    # ---------------- Fig 3: OS amplitude bias vs noise ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    amp0 = -14.6
    true_A2 = 10.0 ** (2 * amp0)
    for ax, (model, title) in zip(axes, [("oracle", "OS, oracle noise model"),
                                         ("fitted", "OS, per-pulsar fitted "
                                          "noise model")]):
        rel_bias, rel_rmse = [], []
        for nz in noises:
            sub = sel(rows, scenario="gwb", log10_A_gwb=amp0, noise=nz,
                      estimator="optimal_statistic", model=model,
                      selection="all")
            b, r, _ = os_bias_rmse(sub, true_A2)
            rel_bias.append(b / true_A2); rel_rmse.append(r / true_A2)
        xx = np.arange(len(noises))
        ax.bar(xx - 0.2, rel_bias, 0.4, label="relative bias")
        ax.bar(xx + 0.2, rel_rmse, 0.4, label="relative RMSE")
        ax.set_xticks(xx); ax.set_xticklabels(noises)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title); ax.legend(fontsize=9)
    axes[0].set_ylabel(r"(estimate $-$ truth) / truth  [in $A^2_{\rm gwb}$]")
    fig.suptitle("Fig 3: Optimal-statistic amplitude recovery vs intrinsic "
                 "noise strength\n"
                 r"injected $\log_{10} A_{\rm gwb} = -14.6$, 12 pulsars")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig3_amplitude_bias.png",
                                    dpi=130)
    plt.close(fig)

    # ---------------- Fig 4: oracle vs fitted vs joint ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models4 = ["oracle", "fitted", "joint"]
    # detection probability, gwb -14.6 fiducial
    det = []
    for m in models4:
        sub = sel(rows, scenario="gwb", log10_A_gwb=-14.6, noise="fiducial",
                  estimator="optimal_statistic", model=m, selection="all")
        _, _, _, dp = snr_stats(sub)
        det.append(dp)
    axes[0].bar(models4, det, color=["C2", "C3", "C4"])
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("detection probability")
    axes[0].set_title("OS detection probability\n"
                      r"gwb $\log_{10}A=-14.6$, fiducial noise")
    # FPR, null fiducial
    fpr = []
    for m in models4:
        sub = sel(rows, scenario="null", noise="fiducial",
                  estimator="optimal_statistic", model=m, selection="all")
        _, _, _, fp = snr_stats(sub)
        fpr.append(fp)
    axes[1].bar(models4, fpr, color=["C2", "C3", "C4"])
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("false-positive rate")
    axes[1].set_title("OS false-positive rate\nnull, fiducial noise")
    # joint common-amplitude recovery annotation
    jx = extra[(extra["scenario"] == "gwb")]
    jx = jx[np.isfinite(jx["joint_log10_A_common"]
                        .astype(float))]["joint_log10_A_common"].astype(float)
    if len(jx):
        txt = (f"joint fit recovers common amplitude:\n"
               f"log10 A = {np.mean(jx):.2f} ± {np.std(jx):.2f} "
               f"(truth -14.60, n={len(jx)})")
        axes[0].text(0.02, 0.95, txt, transform=axes[0].transAxes, va="top",
                     fontsize=8, bbox=dict(fc="white", alpha=0.8))
    fig.suptitle("Fig 4: Noise-model handling for the optimal statistic "
                 "(12 pulsars)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig4_oracle_vs_fitted.png",
                                    dpi=130)
    plt.close(fig)

    # ---------------- Fig 5: selection groups ----------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sels = ["all", "strong", "moderate", "weak"]
    xx = np.arange(len(sels)); width = 0.25
    for i, est in enumerate(ests):
        means, stds, ns = [], [], []
        for s in sels:
            sub = sel(rows, scenario="gwb", log10_A_gwb=-14.6,
                      noise="fiducial", estimator=est, model="oracle",
                      selection=s)
            mu, sd, n, _ = snr_stats(sub)
            means.append(mu); stds.append(sd); ns.append(n)
        ax.bar(xx + (i - 1) * width, means, width, yerr=stds, capsize=3,
               color=colors[est], label=est_labels[est])
        for j, n in enumerate(ns):
            if np.isfinite(means[j]):
                ax.text(xx[j] + (i - 1) * width, means[j] + stds[j] + 0.4,
                        f"n={n}", ha="center", fontsize=7)
    ax.axhline(THRESH, color="k", ls=":", lw=1)
    ax.set_xticks(xx); ax.set_xticklabels(sels)
    ax.set_ylabel("detection SNR (mean ± std)")
    ax.set_title("Fig 5: Detection SNR by pulsar selection group\n"
                 r"gwb $\log_{10}A=-14.6$, fiducial noise, oracle models "
                 "(n = usable realizations)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig5_selection_groups.png",
                                    dpi=130)
    plt.close(fig)

    # ---------------- headline numbers ----------------
    print("=" * 70)
    print("HEADLINE NUMBERS (threshold SNR > 3.0)")
    print("=" * 70)
    print("\n[Detection probability] gwb, fiducial noise, all pulsars, oracle:")
    for amp in amps:
        line = f"  log10A={amp}: "
        for est in ests:
            sub = sel(rows, scenario="gwb", log10_A_gwb=amp, noise="fiducial",
                      estimator=est, model="oracle", selection="all")
            mu, sd, n, dp = snr_stats(sub)
            line += (f"{est_labels[est]} SNR={mu:.1f}±{sd:.1f} "
                     f"Pdet={dp:.2f}  ")
        print(line)
    print("\n[Degeneracy] gwb -14.6, fiducial, all pulsars:")
    for est in ("whitened_pearson", "optimal_statistic"):
        for model in ("oracle", "fitted"):
            sub = sel(rows, scenario="gwb", log10_A_gwb=-14.6,
                      noise="fiducial", estimator=est, model=model,
                      selection="all")
            mu, sd, n, dp = snr_stats(sub)
            print(f"  {est_labels[est]:17s} {model:6s}: "
                  f"SNR={mu:.2f}±{sd:.2f} Pdet={dp:.2f} (n={n})")
    print("\n[Degeneracy] OS amplitude recovery, gwb -14.6 (truth "
          f"A^2={true_A2:.2e}):")
    for nz in noises:
        for model in ("oracle", "fitted"):
            sub = sel(rows, scenario="gwb", log10_A_gwb=-14.6, noise=nz,
                      estimator="optimal_statistic", model=model,
                      selection="all")
            b, r, n = os_bias_rmse(sub, true_A2)
            print(f"  noise={nz:8s} {model:6s}: rel.bias={b/true_A2:+.2f} "
                  f"rel.RMSE={r/true_A2:.2f} (n={n})")
    print("\n[Joint fit] OS with joint array noise model:")
    for sc, amp in (("gwb", -14.6), ("null", np.nan)):
        for model in ("oracle", "fitted", "joint"):
            kw = dict(scenario=sc, noise="fiducial",
                      estimator="optimal_statistic", model=model,
                      selection="all")
            if sc == "gwb":
                kw["log10_A_gwb"] = amp
            sub = sel(rows, **kw)
            mu, sd, n, dp = snr_stats(sub)
            print(f"  {sc:4s} {model:6s}: SNR={mu:.2f}±{sd:.2f} "
                  f"Pdet/FPR={dp:.2f} (n={n})")
    print("\n[False positives] FPR by null scenario, fiducial noise, "
          "all pulsars:")
    for sc in nulls:
        line = f"  {sc:9s}: "
        for est, model in (("raw_pearson", "oracle"),
                           ("whitened_pearson", "oracle"),
                           ("optimal_statistic", "oracle"),
                           ("optimal_statistic", "fitted")):
            sub = sel(rows, scenario=sc, noise="fiducial", estimator=est,
                      model=model, selection="all")
            _, _, _, fpr = snr_stats(sub)
            line += f"{est_labels[est]}/{model}={fpr:.2f}  "
        print(line)
    print("\n[Selection] median group sizes (gwb -14.6, fiducial):")
    gx = extra[(extra["scenario"] == "gwb") & (extra["noise"] == "fiducial")]
    for k in ("n_strong", "n_moderate", "n_weak"):
        v = gx[k].astype(int)
        print(f"  {k}: median={np.median(v):.0f} "
              f"(range {v.min()}-{v.max()}, n={len(v)})")
    print("\nManifest: runtime %.0fs, %d realizations, fitted fallback "
          "fraction %.4f" % (manifest["runtime_s"],
                             manifest["n_realizations_total"],
                             manifest["fitted_fallback_fraction"]))


if __name__ == "__main__":
    main()
