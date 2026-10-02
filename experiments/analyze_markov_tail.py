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
REGIMES = ["exact", "minibatch", "minibatch_const"]
RNAME = {"exact": "exact Lloyd", "minibatch": "mini-batch (1/count)", "minibatch_const": "mini-batch (constant step)"}
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
        for i, reg in enumerate(REGIMES):
            d = st[st.regime == reg].groupby("T0")[col].mean()
            if d.empty:
                continue
            ax.plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=RNAME[reg])
        if logy:
            ax.set_yscale("log")
        _style(ax, title, ylab)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_unstable_set.png", dpi=150)
    plt.close(fig)

    # Fig 2: compute saved by stopping at T0 + Markov assignment
    mk_ = df[df.method == "markov"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ex = mk_[mk_.regime == "exact"]
    for i, (col, lab) in enumerate((("save_dist_vs_lloyd_full", "distances vs Lloyd to convergence"),
                                    ("save_dist_vs_base_full", "distances vs Hamerly to convergence"),
                                    ("save_sec_vs_lloyd_full", "time vs BLAS Lloyd to convergence"),
                                    ("save_sec_vs_base_full", "time vs Hamerly to convergence"))):
        d = ex.groupby("T0")[col].mean() * 100
        axes[0].plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    _style(axes[0], "exact Lloyd: saving of stopping at T0", "saving (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for ax, reg in ((axes[1], "minibatch"), (axes[2], "minibatch_const")):
        mb = mk_[mk_.regime == reg]
        for i, (col, lab) in enumerate((("save_dist_vs_base_full", "distances vs 200-epoch run"),
                                        ("save_sec_vs_base_full", "time vs 200-epoch run"))):
            d = mb.groupby("T0")[col].mean() * 100
            if d.empty:
                continue
            ax.plot(d.index, d.values, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        _style(ax, f"{RNAME[reg]}: saving of stopping at T0", "saving (%)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_compute_saving.png", dpi=150)
    plt.close(fig)

    # Fig 3: errors inside U, per strategy
    names = {"stop": "keep current label", "majority": "majority vote", "markov": "Markov (proposed)",
             "active_set_U": "active-set Lloyd on U", "hartigan_U": "Hartigan on U",
             "reassign_U": "reassign U to nearest", "markov_a0.5": "Markov, smoothed (α=0.5)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    mb_methods = ["stop", "majority", "markov", "markov_a0.5", "reassign_U"]
    for ax, reg, methods in ((axes[0], "exact", ["stop", "majority", "markov", "markov_a0.5", "active_set_U", "hartigan_U"]),
                             (axes[1], "minibatch", mb_methods), (axes[2], "minibatch_const", mb_methods)):
        lines(ax, df[df.regime == reg], methods, "U_err_rate", names)
        _style(ax, f"{RNAME[reg]}: wrong labels left in U", "wrong labels in U / |U| (log)")
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
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for col_i, reg in enumerate(REGIMES):
        d = df[df.regime == reg]
        lines(axes[0, col_i], d, soft, "brier_ref", snames)
        _style(axes[0, col_i], f"{RNAME[reg]}: Brier vs final label (lower = better)", "Brier")
        lines(axes[1, col_i], d, soft, "auroc_err_ref", snames)
        _style(axes[1, col_i], f"{RNAME[reg]}: AUROC, uncertainty flags wrong labels", "AUROC")
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


def predictors(z):
    """Two-state probability that the point ends in state a, from each competing source."""
    two = lambda pa, pb: pa / np.maximum(pa + pb, 1e-300)
    out = [("markov_pi", z["pi_a"]), ("frequency", z["freq_a"])]
    if "plug_a" in z.files:
        out.append(("plugin_posterior_T0", two(z["plug_a"], z["plug_b"])))
    out.append(("model_posterior_oracle", two(z["model_a"], z["model_b"])))
    return out


BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001))


def oscillator_report():
    """Reliability of fractional π: does π(a) match how often the point really ends in a?

    Outcome: long-run occupancy of state a (mini-batch regimes) or 1[converged label == a]
    (exact).  Reported separately for T0 = 10 and for the few-unstable-points regime T0 >= 20.
    """
    rows = []
    for f in sorted(RESULTS.glob("oscillators_*.npz")):
        z = np.load(f)
        if len(z["idx"]) == 0:
            continue
        regime = f.name[len("oscillators_"):].split("-")[0]
        out = (z["occ_a"] if regime.startswith("minibatch") else (z["ref"] == z["a"]).astype(float))
        for subset, sel in (("T0=10", z["T0"] == 10), ("T0>=20", z["T0"] >= 20)):
            if not sel.any():
                continue
            o = out[sel]
            for name, pred in predictors(z):
                pr = pred[sel]
                for lo, hi in BINS:
                    m = (pr >= lo) & (pr < hi)
                    if m.any():
                        rows.append(dict(regime=regime, subset=subset, predictor=name, bin=f"[{lo:.1f},{min(hi, 1):.1f})",
                                         n=int(m.sum()), mean_pred=float(pr[m].mean()), observed=float(o[m].mean())))
                rows.append(dict(regime=regime, subset=subset, predictor=name, bin="all", n=int(sel.sum()),
                                 mean_pred=float(pr.mean()), observed=float(o.mean()),
                                 brier=float(((pr - o) ** 2).mean()),
                                 corr=float(np.corrcoef(pr, o)[0, 1]) if pr.std() > 0 and o.std() > 0 else float("nan"),
                                 frac_outcome_mixed=float(((o > 0.05) & (o < 0.95)).mean()),
                                 ends_in_label_T0=float((z["ref"][sel] == z["label_T0"][sel]).mean())))
    rep = pd.DataFrame(rows)
    if len(rep):
        TABLES.mkdir(parents=True, exist_ok=True)
        rep.to_csv(TABLES / "oscillator_reliability.csv", index=False)
        fig5(rep)
    return rep


def fig5(rep):
    regs = [r for r in REGIMES if r in set(rep.regime)]
    fig, axes = plt.subplots(1, len(regs), figsize=(5.4 * len(regs), 4.8), squeeze=False)
    names = {"markov_pi": "Markov π (proposed)", "frequency": "label frequency",
             "plugin_posterior_T0": "plug-in GMM posterior at T0", "model_posterior_oracle": "converged-model posterior (oracle)"}
    for ax, reg in zip(axes[0], regs):
        d = rep[(rep.regime == reg) & (rep.subset == "T0>=20") & (rep.bin != "all")]
        ax.plot([0.5, 1], [0.5, 1], color=GRID, lw=1.5, zorder=0)
        for i, pname in enumerate(["markov_pi", "frequency", "plugin_posterior_T0", "model_posterior_oracle"]):
            q = d[d.predictor == pname]
            if q.empty:
                continue
            size = 30 + 400 * np.sqrt(q.n / q.n.max())   # marker area grows with the bin's point count
            ax.scatter(q.mean_pred, q.observed, s=size, color=SERIES[i], marker=MARKERS[i],
                       edgecolor=SURFACE, linewidth=1.5, label=names[pname], alpha=0.9)
        n = int(rep[(rep.regime == reg) & (rep.subset == "T0>=20") & (rep.bin == "all")].n.max() or 0)
        _style(ax, f"{RNAME[reg]} (T0 ≥ 20, {n} points)", "observed share ending in a", "predicted probability of state a")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.47, 1.0)
        leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                        ncol=2, markerscale=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_fractional_reliability.png", dpi=150)
    plt.close(fig)


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
