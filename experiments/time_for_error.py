"""Time needed for a given error, per procedure: each procedure's own setting swept.

A user picks one setting without knowing the outcome, so every setting is applied to all
instances of a regime (the 10 trajectories of experiments/pareto.py, n = 10^6) and gives one
point: x = mean share of labels that differ from the reference, y = mean time relative to the
full run.  Sweeping the setting traces the procedure's curve; a curve further down and left
reaches the same error in less time.

    markov-kmeans (auto)      unstable_tol = change_tol swept (other thresholds at their defaults)
    markov-kmeans (t0)        fixed number of iterations, unstable points settled
    sklearn tol               tolerance on the squared center shift swept
    Perez-Ortega              threshold on the share of changed labels swept
    fixed iterations          fixed number of iterations, labels kept

    python -m experiments.time_for_error   # experiments/results/time_for_error.csv, docs/figures/fig9_time_for_error.png
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.analyze_markov_tail import FIG, GRID, INK2, SERIES, SURFACE, _style, fill_report, md, plt
from experiments.auto_rule import reference, replay_stop
from experiments.pareto import INSTANCES, K, RESULTS, first_t, name_of, settle_labels, trajectory, warm_up
from kmeans_accel import datasets
from markov_kmeans import _rules

ITERS = (2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 230)
TOLS = tuple(sorted(set(np.geomspace(3e-2, 1e-5, 15)) | {1e-3}, reverse=True))
SK_TOLS = tuple(np.geomspace(1e-1, 1e-9, 17))
PEREZ = tuple(sorted(set(np.geomspace(1e-1, 1e-6, 16)) | {0.03}, reverse=True))
FAMILIES = ["markov-kmeans (auto)", "markov-kmeans (t0)", "sklearn tol", "Perez-Ortega", "fixed iterations"]
DEFAULTS = {"markov-kmeans (auto)": 1e-3, "sklearn tol": 1e-4, "Perez-Ortega": 0.03, "fixed iterations": 25}


def sweep(regime, sep, seed, eta):
    traj, X = trajectory(regime, sep, seed, eta)
    H, C, counts, secs = traj["H"], traj["C"], traj["counts"], traj["secs"]
    T = len(H)
    n = H.shape[1]
    exact = regime == "exact"
    ref, _ = reference(regime, traj, K)
    full = secs[T - 1]
    err = lambda lab: float((lab != ref).mean())
    rows = []
    add = lambda fam, par, t, extra, e: rows.append(dict(family=fam, param=par, T=t, cost=(secs[t] + extra) / full, error=e))

    # monitor bookkeeping paid by markov-kmeans with t0 (push only, no judgement)
    mon = _rules.StateMonitor(n, K, _rules.StopParams(), exact=exact)
    mon.push(H[0], C[0])
    push = np.zeros(T)
    for t in range(1, T):
        t0 = time.perf_counter()
        mon.push(H[t], C[t], int(counts[t - 1]))
        push[t] = push[t - 1] + time.perf_counter() - t0

    for it in ITERS:
        t = min(it, T - 1)
        add("fixed iterations", it, t, 0.0, err(H[t]))
        lab, s = settle_labels(H, t, None, "auto")
        add("markov-kmeans (t0)", it, t, push[t] + s, err(lab))
    for tol in TOLS:
        params = _rules.StopParams(unstable_tol=tol, change_tol=tol)
        t, reason, _, st = replay_stop(traj, params, exact, "auto", K)
        regime_name = None if reason == "max_iter" else ("oscillation" if reason == "oscillation" else "drift")
        lab, s = settle_labels(H, t, regime_name, "keep" if reason == "converged" else "auto")
        add("markov-kmeans (auto)", tol, t, st["monitor_seconds"] + s, err(lab))
    if X is None:
        X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
    var = float(np.var(X, axis=0).mean())
    shift = np.r_[np.inf, ((C[1:] - C[:-1]) ** 2).sum((1, 2))]
    for tol in SK_TOLS:
        t = first_t(lambda t_: shift[t_] <= tol * var, T)
        add("sklearn tol", tol, t, 0.0, err(H[t]))
    for f in PEREZ:
        t = first_t(lambda t_: counts[t_ - 1] < f * n, T)
        add("Perez-Ortega", f, t, 0.0, err(H[t]))
    add("full run", 0, T - 1, 0.0, err(H[T - 1]))
    df = pd.DataFrame(rows)
    df["regime"], df["instance"] = regime, name_of(regime, sep, seed, eta)
    return df


def curves(df):
    """Mean error and mean time of every setting over the instances of a regime."""
    g = df.groupby(["regime", "family", "param"])
    return g.agg(error=("error", "mean"), error_max=("error", "max"), cost=("cost", "mean"),
                 n=("instance", "nunique")).reset_index()


def time_for(cv, targets=(0.01, 0.005, 0.003, 0.002, 0.001)):
    """Least mean time among a procedure's settings whose mean error is at most each target."""
    rows = []
    for (reg, fam), g in cv.groupby(["regime", "family"]):
        row = {"regime": reg, "family": fam}
        for x in targets:
            ok = g[g.error <= x + 1e-12]
            row[f"{x * 100:g}%"] = float(ok.cost.min()) if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


TARGETS = {"exact": (0.02, 0.01, 0.005, 0.002, 0.001), "minibatch": (0.002, 0.001, 0.0005, 0.0002, 0.0001),
           "minibatch_const": (0.01, 0.005, 0.003, 0.002, 0.001)}
NAMES = {"markov-kmeans (auto)": "markov-kmeans 자동 규칙", "markov-kmeans (t0)": "markov-kmeans t0 지정 + 편입",
         "sklearn tol": "sklearn tol", "Perez-Ortega": "Pérez-Ortega", "fixed iterations": "고정 반복 수",
         "full run": "끝까지"}


def report_tables(cv):
    """Per regime: least mean time (share of the full run) to reach each mean-error target."""
    T = {}
    for reg, targets in TARGETS.items():
        tf = time_for(cv[cv.regime == reg], targets).set_index("family")
        tf = tf.drop(columns="regime").reindex([f for f in FAMILIES + ["full run"] if f in tf.index])
        tf.index = [NAMES[f] for f in tf.index]
        tf.columns = [f"오류 ≤ {c}" for c in tf.columns]
        T[f"time_for_error_{reg}"] = md(tf, "{:.0%}", index_name="방법 (설정을 바꿔 가며 가장 빠른 것)")
    return T


TITLE = {"exact": "exact Lloyd (4 instances)", "minibatch": "mini-batch 1/count (2 instances)",
         "minibatch_const": "mini-batch constant step (4 instances)"}


def figure(cv):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    style = {"markov-kmeans (auto)": (SERIES[0], "o", "-", 2.6), "markov-kmeans (t0)": (SERIES[0], "o", "--", 1.4),
             "sklearn tol": (SERIES[1], "s", "-", 1.8), "Perez-Ortega": (SERIES[2], "^", "-", 1.8),
             "fixed iterations": (SERIES[3], "D", "-", 1.8)}
    for ax, reg in zip(axes, ("exact", "minibatch", "minibatch_const")):
        d = cv[cv.regime == reg]
        for fam in FAMILIES:
            q = d[d.family == fam].sort_values("error")
            col, mk, ls, lw = style[fam]
            # keep only the lower envelope: a setting that costs more for a larger error is never chosen
            env = q[q.cost <= q.cost.cummin()]
            ax.plot(env.error * 100, env.cost, color=col, ls=ls, lw=lw, marker=mk, ms=5 if lw < 2 else 6,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=fam, zorder=4 if "auto" in fam else 3)
            if fam in DEFAULTS:
                dq = q[np.isclose(q.param, DEFAULTS[fam])]
                ax.scatter(dq.error * 100, dq.cost, s=170, facecolor="none", edgecolor=col, linewidth=2, zorder=5)
        fr = d[d.family == "full run"]
        if (fr.error > 0).all():
            ax.scatter(fr.error * 100, fr.cost, marker="*", s=220, color=INK2, edgecolor=SURFACE, zorder=5, label="full run")
        else:
            ax.axhline(1.0, color=GRID, lw=1.5, ls=":", zorder=1)
            ax.text(0.98, 1.0, "full run: error 0 by definition", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=8, color=INK2)
        ax.scatter([], [], s=170, facecolor="none", edgecolor=INK2, linewidth=2, label="default setting")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.08)
        _style(ax, TITLE[reg], "time / full run (mean)", "labels differing from the reference (%, mean, log)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig9_time_for_error.png", dpi=150)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    warm_up()
    df = pd.concat([sweep(*inst) for inst in INSTANCES], ignore_index=True)
    df.to_csv(RESULTS / "time_for_error.csv", index=False)
    cv = curves(df)
    figure(cv)
    T = report_tables(cv)
    fill_report(T)
    for k, v in T.items():
        print(f"== {k}\n{v}")
    return df, cv


if __name__ == "__main__":
    main()
