"""Aggregate experiments/results/markov_tail_*.csv into tables and figures for docs/results.md.

    python -m experiments.analyze_markov_tail
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "figures"
TABLES = ROOT / "docs" / "tables"

# Reference categorical palette (dataviz skill, validated light-mode order) + secondary encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

U_ONLY = ["stop", "reassign_U", "majority", "frequency_soft", "markov", "markov_soft", "markov_a0.5",
          "markov_a0.5_soft", "markov_pooled", "markov_pooled_soft", "markov_w5", "markov_w20",
          "active_set_U", "hartigan_U", "fcm_soft", "gmm_plugin_soft"]


def load():
    frames = []
    for f in sorted(RESULTS.glob("markov_tail_*.csv")):
        if "skipped" in f.name:
            continue
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    return derive(df)


def derive(df):
    df = df.copy()
    nk = df["lloyd_dist_iter"]
    # base algorithm cost up to T0 (Hamerly or mini-batch) and full run
    df["total_base_dist"] = df["prefix_base_dist"] + df["extra_dist"]
    df["total_base_sec"] = df["prefix_base_sec"] + df["extra_seconds"]
    full = df["method"] == "full_convergence"
    df.loc[full, "total_base_dist"] = df.loc[full, "full_base_dist"]
    df.loc[full, "total_base_sec"] = df.loc[full, "full_base_sec"]
    df["save_dist_vs_base_full"] = 1 - df["total_base_dist"] / df["full_base_dist"]
    df["save_sec_vs_base_full"] = 1 - df["total_base_sec"] / df["full_base_sec"]
    # plain Lloyd as the base (exact regime): (T0+1) passes of n*k, BLAS timing per pass
    df["lloyd_full_dist"] = (df["T_conv"] + 1) * nk
    df["lloyd_total_dist"] = (df["T0"] + 1) * nk + df["extra_dist"]
    df.loc[full, "lloyd_total_dist"] = df.loc[full, "lloyd_full_dist"]
    df["save_dist_vs_lloyd_full"] = 1 - df["lloyd_total_dist"] / df["lloyd_full_dist"]
    g = df["lloyd_gemm_sec_iter"]
    df["lloyd_total_sec"] = (df["T0"] + 1) * g + df["extra_seconds"]
    df.loc[full, "lloyd_total_sec"] = (df.loc[full, "T_conv"] + 1) * g[full]
    df["save_sec_vs_lloyd_full"] = 1 - df["lloyd_total_sec"] / ((df["T_conv"] + 1) * g)
    # errors inside U: total disagreement minus the (shared) disagreement outside U
    stop = df[df.method == "stop"].set_index(["instance", "T0"])
    outside = (stop["n_disagree"] - stop["U_disagree_at_T0"]).rename("outside_err")
    df = df.join(outside, on=["instance", "T0"])
    df["U_err"] = np.where(df["method"].isin(U_ONLY), df["n_disagree"] - df["outside_err"], np.nan)
    df["U_err_rate"] = df["U_err"] / df["n_U"]
    return df


def summarize(df, cols, by=("regime", "T0", "method")):
    agg = df.groupby(list(by))[cols].agg(["mean", "min", "max"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


# ----------------------------------------------------------------------------- figures

def _style(ax, title, ylabel, xlabel="T0 (cut-off iteration)"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def lines(ax, frame, methods, col, labels=None, logy=False):
    for i, m in enumerate(methods):
        d = frame[frame.method == m].groupby("T0")[col].mean()
        if d.empty:
            continue
        ax.plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=(labels or {}).get(m, m),
                solid_capstyle="round")
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=3)


def figures(df):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    # Fig 1: how many points are unstable at T0, and how many of the later changes U catches
    st = df[df.method == "stop"].copy()
    st["recall_V"] = st["n_U_and_V"] / st["n_V"].replace(0, np.nan)
    st["share_U_wrong"] = st["U_disagree_at_T0"] / st["n_disagree"].replace(0, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title, ylab, logy in (
            (axes[0], "frac_U", "Unstable set U as a share of n", "|U| / n", True),
            (axes[1], "recall_V", "Later-changing points that are in U", "|U ∩ V| / |V|", False),
            (axes[2], "share_U_wrong", "Share of remaining errors that lie in U", "errors in U / all errors", False)):
        for i, reg in enumerate(["exact", "minibatch"]):
            d = st[st.regime == reg].groupby("T0")[col].mean()
            ax.plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=reg)
        if logy:
            ax.set_yscale("log")
        _style(ax, title, ylab)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_unstable_set.png", dpi=150)
    plt.close(fig)

    # Fig 2: compute saved by stopping at T0 + Markov assignment
    mk = df[df.method == "markov"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ex = mk[mk.regime == "exact"]
    for i, (col, lab) in enumerate((("save_dist_vs_lloyd_full", "distances vs Lloyd to convergence"),
                                    ("save_dist_vs_base_full", "distances vs Hamerly to convergence"),
                                    ("save_sec_vs_lloyd_full", "time vs BLAS Lloyd to convergence"),
                                    ("save_sec_vs_base_full", "time vs Hamerly to convergence"))):
        d = ex.groupby("T0")[col].mean() * 100
        axes[0].plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    _style(axes[0], "Exact regime: saving of Markov stop at T0", "saving (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    mb = mk[mk.regime == "minibatch"]
    for i, (col, lab) in enumerate((("save_dist_vs_base_full", "distances vs mini-batch to the horizon"),
                                    ("save_sec_vs_base_full", "time vs mini-batch to the horizon"))):
        d = mb.groupby("T0")[col].mean() * 100
        axes[1].plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    _style(axes[1], "Mini-batch regime: saving of Markov stop at T0", "saving (%)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_compute_saving.png", dpi=150)
    plt.close(fig)

    # Fig 3: errors inside U, per strategy
    names = {"stop": "keep current label", "majority": "majority vote", "markov": "Markov (proposed)",
             "active_set_U": "active-set Lloyd on U", "hartigan_U": "Hartigan on U",
             "reassign_U": "reassign U to nearest", "markov_a0.5": "Markov, smoothed (α=0.5)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, reg, methods in ((axes[0], "exact", ["stop", "majority", "markov", "markov_a0.5", "active_set_U", "hartigan_U"]),
                             (axes[1], "minibatch", ["stop", "majority", "markov", "markov_a0.5", "reassign_U"])):
        lines(ax, df[df.regime == reg], methods, "U_err_rate", names)
        _style(ax, f"{reg}: share of U left with a wrong label", "wrong labels in U / |U| (log)")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_error_in_U.png", dpi=150)
    plt.close(fig)

    # Fig 4: soft memberships — calibration against the converged label and error detection
    soft = ["markov_soft", "markov_a0.5_soft", "markov_pooled_soft", "frequency_soft", "gmm_plugin_soft", "stop"]
    snames = {"stop": "keep current label (0/1)", "markov_soft": "Markov (proposed)", "markov_a0.5_soft": "Markov, smoothed α=0.5",
              "markov_pooled_soft": "Markov, pooled prior", "frequency_soft": "label frequency",
              "gmm_plugin_soft": "plug-in GMM posterior"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col_i, reg in enumerate(["exact", "minibatch"]):
        d = df[df.regime == reg]
        lines(axes[0, col_i], d, soft, "brier_ref", snames)
        _style(axes[0, col_i], f"{reg}: Brier score vs converged label (lower is better)", "Brier")
        lines(axes[1, col_i], d, soft, "auroc_err_ref", snames)
        _style(axes[1, col_i], f"{reg}: AUROC, uncertainty flags wrong labels", "AUROC")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_soft_quality.png", dpi=150)
    plt.close(fig)


def tables(df):
    TABLES.mkdir(parents=True, exist_ok=True)
    state = summarize(df[df.method == "stop"], ["frac_U", "n_U", "n_V", "n_U_and_V", "U_disagree_at_T0",
                                                "n_disagree", "changes_at_T0"], by=("regime", "T0"))
    state.to_csv(TABLES / "state_per_T0.csv", index=False)
    compute = summarize(df, ["save_dist_vs_lloyd_full", "save_dist_vs_base_full", "save_sec_vs_lloyd_full",
                             "save_sec_vs_base_full", "extra_dist", "extra_seconds"])
    compute.to_csv(TABLES / "compute_per_T0.csv", index=False)
    err = summarize(df, ["n_disagree", "disagree_rate", "U_err", "U_err_rate", "rel_sse", "center_err",
                         "acc_truth", "ari_ref"])
    err.to_csv(TABLES / "error_per_T0.csv", index=False)
    softcols = [c for c in ["brier_ref", "p_ref_label", "ece_ref", "auroc_err_ref", "entropy", "frac_fractional",
                            "tv_model_post", "tv_ref_soft", "n_U_matched", "tv_true_post", "brier_truth",
                            "ece_truth", "auroc_err_truth"] if c in df]
    soft = summarize(df, softcols)
    soft.to_csv(TABLES / "soft_per_T0.csv", index=False)
    inst = df.groupby("instance")[["regime", "n", "d", "k", "sep", "seed", "T_conv", "converged",
                                   "ref_acc_truth", "n_matched_clusters", "lloyd_gemm_sec_iter",
                                   "full_base_sec"]].first()
    inst.to_csv(TABLES / "instances.csv")
    return state, compute, err, soft, inst


def oscillator_report():
    """Reliability of fractional π: does π(a) match how often the point really ends in a?"""
    rows = []
    for f in sorted(RESULTS.glob("oscillators_*.npz")):
        z = np.load(f)
        if len(z["idx"]) == 0:
            continue
        regime = f.name.split("_")[1].split("-")[0]
        out = (z["occ_a"] if regime == "minibatch" else (z["ref"] == z["a"]).astype(float))
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
            for name, pred in (("markov_pi", z["pi_a"]), ("frequency", z["freq_a"]),
                               ("model_posterior", z["model_a"] / np.maximum(z["model_a"] + z["model_b"], 1e-300))):
                m = (pred >= lo) & (pred < hi) if hi < 1 else (pred >= lo)
                if m.any():
                    rows.append(dict(file=f.name, regime=regime, predictor=name, bin=f"[{lo:.1f},{hi:.1f})",
                                     n=int(m.sum()), mean_pred=float(pred[m].mean()),
                                     observed=float(out[m].mean())))
        for name, pred in (("markov_pi", z["pi_a"]), ("frequency", z["freq_a"]),
                           ("model_posterior", z["model_a"] / np.maximum(z["model_a"] + z["model_b"], 1e-300))):
            rows.append(dict(file=f.name, regime=regime, predictor=name, bin="all", n=len(pred),
                             mean_pred=float(pred.mean()), observed=float(out.mean()),
                             brier=float(((pred - out) ** 2).mean()),
                             corr=float(np.corrcoef(pred, out)[0, 1]) if pred.std() > 0 and out.std() > 0 else float("nan")))
    rep = pd.DataFrame(rows)
    if len(rep):
        TABLES.mkdir(parents=True, exist_ok=True)
        rep.to_csv(TABLES / "oscillator_reliability.csv", index=False)
    return rep


def main():
    df = load()
    figures(df)
    t = tables(df)
    osc = oscillator_report()
    if len(osc):
        print(osc.to_string())
    print(t[4].to_string())
    print(f"rows={len(df)} instances={df.instance.nunique()}")


if __name__ == "__main__":
    main()
