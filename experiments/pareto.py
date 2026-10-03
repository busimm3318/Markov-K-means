"""Joint cost-error comparison of complete clustering procedures on the same trajectories.

Every procedure is evaluated as a user would run it, on one recorded trajectory per
instance (the 10 instances of experiments/auto_rule.py, n = 10^6):

    markov-kmeans 0.2.0   automatic stop + regime-aware settlement (package defaults)
    markov-kmeans 0.1.0   stop when |U|/n <= 1e-3, Markov settlement
    sklearn tol           stop when the squared center shift <= 1e-4 * mean variance, keep labels
    Perez-Ortega          stop when fewer than 0.03 n labels change, keep labels
    fixed 20 / 25 / 50    stop after a fixed number of iterations (25: faiss default), keep labels
    full run              exact: to convergence; mini-batch: 200 epochs

Cost is the time spent relative to the full run, including the monitor's time for the
markov-kmeans rules.  Error is the share of labels that differ from the reference (exact:
converged labels; mini-batch: modal label of the last 50 of 200 epochs).

Two hindsight frontiers give a scale: stopping at the best fixed iteration t chosen with
knowledge of the outcome and keeping the labels ("oracle stop + keep"), or settling the
unstable points as markov-kmeans does ("oracle stop + settle").  For every procedure we
report whether another procedure is at least as cheap and as accurate (dominated), the
error excess over the oracle frontier at the same cost, and the cost ratio to the oracle
frontier at the same error.

    python -m experiments.pareto          # writes experiments/results/pareto.csv, docs/figures/fig8_cost_error.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus

from experiments.analyze_markov_tail import FIG, GRID, INK2, MARKERS, SERIES, SURFACE, _style, plt
from experiments.auto_rule import record, reference, replay_stop
from kmeans_accel import datasets
from markov_kmeans import _rules

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "auto_traj"
RESULTS = ROOT / "experiments" / "results"
K = 50
INSTANCES = [("exact", 1.5, 0, 0.0), ("exact", 1.5, 1, 0.0), ("exact", 2.0, 0, 0.0), ("exact", 2.0, 1, 0.0),
             ("minibatch", 1.5, 0, 0.0), ("minibatch", 2.0, 0, 0.0),
             ("minibatch_const", 1.5, 0, 2e-4), ("minibatch_const", 2.0, 0, 2e-4),
             ("minibatch_const", 1.5, 1, 2e-4), ("minibatch_const", 1.5, 0, 1e-3)]
PROCS = ["markov-kmeans 0.2.0", "markov-kmeans 0.1.0", "sklearn tol", "Perez-Ortega", "fixed 20", "fixed 25 (faiss)",
         "fixed 50", "full run"]


def name_of(regime, sep, seed, eta):
    return f"{regime}-sep{sep}-s{seed}" + (f"-eta{eta:g}" if regime == "minibatch_const" else "")


def trajectory(regime, sep, seed, eta):
    f = CACHE / f"{name_of(regime, sep, seed, eta)}.npz"
    if f.exists():
        return {k: v for k, v in np.load(f).items()}, None
    X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
    C0, _ = kmeans_plusplus(X, K, random_state=seed)
    traj = record(regime, X, C0, seed, eta)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **traj)
    return traj, X


def settle_labels(H, t, regime_name, rule):
    """Labels at t with the points that moved in the last W iterations settled by `rule`."""
    W = _rules.StopParams().window
    L = H[t].astype(np.int64)
    lo = max(0, t - W)
    moved = (H[lo + 1:t + 1] != H[lo:t]).any(0)
    U = np.flatnonzero(moved)
    if len(U) == 0 or rule == "keep":
        return L, 0.0
    t0 = time.perf_counter()
    Hw = np.ascontiguousarray(H[lo:t + 1][:, U].astype(np.int64))
    regime = regime_name
    if regime is None:
        regime = "oscillation" if (_rules.switches(Hw) >= 2).mean() >= _rules.StopParams().osc_share else "drift"
    lab, _, _, _ = _rules.settle(Hw, L[U], rule=rule, n_clusters=K, regime=regime)
    out = L.copy()
    out[U] = lab
    return out, time.perf_counter() - t0


def first_t(cond, T):
    for t in range(1, T):
        if cond(t):
            return t
    return T - 1


def analyse(regime, sep, seed, eta):
    traj, X = trajectory(regime, sep, seed, eta)
    H, C, counts, secs = traj["H"], traj["C"], traj["counts"], traj["secs"]
    T = len(H)
    exact = regime == "exact"
    ref, _ = reference(regime, traj, K)
    full = secs[T - 1]
    cost = lambda t, extra=0.0: (secs[t] + extra) / full
    err = lambda lab: float((lab != ref).mean())
    rows = []

    # hindsight frontiers: every stopping iteration, labels kept or settled
    keep_err = np.array([err(H[t]) for t in range(T)])
    settle_err = np.full(T, np.nan)
    for t in range(1, T):
        settle_err[t] = err(settle_labels(H, t, None, "auto")[0])
    frontier = pd.DataFrame({"t": np.arange(T), "cost": secs / full, "err_keep": keep_err, "err_settle": settle_err})

    params = _rules.StopParams()
    t, reason, _, st = replay_stop(traj, params, exact, "auto", K)
    regime_name = None if reason == "max_iter" else ("oscillation" if reason == "oscillation" else "drift")
    lab, s_sec = settle_labels(H, t, regime_name, "keep" if reason == "converged" else "auto")
    rows.append(("markov-kmeans 0.2.0", t, cost(t, st["monitor_seconds"] + s_sec), err(lab)))
    t, reason, _, st = replay_stop(traj, params, exact, "unstable", K)
    lab, s_sec = settle_labels(H, t, None, "keep" if reason == "converged" else "markov")
    rows.append(("markov-kmeans 0.1.0", t, cost(t, st["monitor_seconds"] + s_sec), err(lab)))
    if X is None:
        X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
    var = float(np.var(X, axis=0).mean())
    t = first_t(lambda t_: float(((C[t_] - C[t_ - 1]) ** 2).sum()) <= 1e-4 * var, T)
    rows.append(("sklearn tol", t, cost(t), err(H[t])))
    t = first_t(lambda t_: counts[t_ - 1] < 0.03 * H.shape[1], T)
    rows.append(("Perez-Ortega", t, cost(t), err(H[t])))
    for T0, nm in ((20, "fixed 20"), (25, "fixed 25 (faiss)"), (50, "fixed 50")):
        t = min(T0, T - 1)
        rows.append((nm, t, cost(t), err(H[t])))
    rows.append(("full run", T - 1, 1.0, err(H[T - 1])))

    df = pd.DataFrame(rows, columns=["procedure", "T", "cost", "error"])
    # dominance among the real procedures
    dom = []
    for i, r in df.iterrows():
        by = [q.procedure for j, q in df.iterrows() if j != i and q.cost <= r.cost and q.error <= r.error
              and (q.cost < r.cost or q.error < r.error)]
        dom.append(", ".join(by))
    df["dominated_by"] = dom
    # scale against the hindsight frontier (labels kept)
    fr = frontier.iloc[1:]
    best_err_at = lambda c: float(fr.loc[fr.cost <= c + 1e-12, "err_keep"].min()) if (fr.cost <= c + 1e-12).any() else np.nan
    best_cost_for = lambda e: float(fr.loc[fr.err_keep <= e + 1e-12, "cost"].min()) if (fr.err_keep <= e + 1e-12).any() else np.nan
    df["oracle_err_at_cost"] = [best_err_at(c) for c in df.cost]
    df["oracle_cost_for_err"] = [best_cost_for(e) for e in df.error]
    df["excess_error"] = df.error - df.oracle_err_at_cost
    df["cost_ratio"] = df.cost / df.oracle_cost_for_err
    for k, v in dict(regime=regime, sep=sep, seed=seed, eta=eta, instance=name_of(regime, sep, seed, eta)).items():
        df[k] = v
    frontier["instance"] = name_of(regime, sep, seed, eta)
    return df, frontier


def figure(res, fronts):
    show = [name_of(*INSTANCES[2]), name_of(*INSTANCES[4]), name_of(*INSTANCES[6])]
    titles = ["exact Lloyd (sep 2.0, seed 0)", "mini-batch 1/count (sep 1.5)", "mini-batch constant step (sep 1.5, η=2e-4)"]
    others = ["Perez-Ortega", "fixed 20", "fixed 25 (faiss)", "fixed 50", "sklearn tol", "full run"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, inst, title in zip(axes, show, titles):
        fr = fronts[fronts.instance == inst].iloc[1:]
        ax.plot(fr.cost, np.maximum(fr.err_keep, 1e-5) * 100, color=GRID, lw=3, label="hindsight: best fixed stop + keep labels")
        ax.plot(fr.cost, np.maximum(fr.err_settle, 1e-5) * 100, color=INK2, lw=1.2, ls="--",
                label="hindsight: best fixed stop + settle (markov-kmeans)")
        d = res[res.instance == inst]
        for i, p in enumerate(others):
            q = d[d.procedure == p]
            ax.scatter(q.cost, np.maximum(q.error, 1e-5) * 100, s=170 if p == "full run" else 60,
                       color=SERIES[1 + i % (len(SERIES) - 1)],
                       marker=MARKERS[1 + i % (len(MARKERS) - 1)] if p != "full run" else "*",
                       edgecolor=SURFACE, linewidth=1.2, zorder=3, label=p)
        q = d[d.procedure == "markov-kmeans 0.2.0"]
        ax.scatter(q.cost, np.maximum(q.error, 1e-5) * 100, s=150, color=SERIES[0], marker="o", edgecolor=SURFACE,
                   linewidth=2, zorder=5, label="markov-kmeans 0.2.0")
        ax.set_yscale("log")
        _style(ax, title, "labels differing from the reference (%, log)", "time / full run")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig8_cost_error.png", dpi=150)
    plt.close(fig)


EXTERNAL = ["markov-kmeans 0.2.0", "sklearn tol", "Perez-Ortega", "fixed 20", "fixed 25 (faiss)", "fixed 50", "full run"]
RNAME = {"exact": "정확 Lloyd", "minibatch": "mini-batch 1/count", "minibatch_const": "mini-batch 상수 스텝"}
EPS_COST, EPS_ERR = 0.01, 0.02     # differences within 1 %p of time or 2 % of the error count as ties


def dominates(a, b):
    no_worse = a.cost <= b.cost + EPS_COST and a.error <= b.error * (1 + EPS_ERR) + 1e-6
    better = a.cost < b.cost - EPS_COST or a.error < b.error * (1 - EPS_ERR) - 1e-6
    return no_worse and better


def report_tables(res):
    """Markdown tables for docs/results.md (section 6.3)."""
    from experiments.analyze_markov_tail import md

    T = {}
    d = res[res.procedure.isin(EXTERNAL)].copy()
    d["regime"] = d.instance.str.split("-").str[0]
    dom = []
    for _, g in d.groupby("instance", sort=False):
        g = g.set_index("procedure")
        for p in EXTERNAL:
            dom.append(sum(dominates(g.loc[q], g.loc[p]) for q in EXTERNAL if q != p) > 0)
    d["dominated"] = dom
    sse = pd.read_csv(RESULTS / "auto_rule.csv")
    sse = sse[sse.n == 1_000_000]
    names = {"auto_stop+auto": "markov-kmeans 0.2.0", "sklearn_tol(center_shift<=1e-4 var)": "sklearn tol",
             "perez_ortega(changes<0.03n)": "Perez-Ortega", "fixed_T20": "fixed 20", "fixed_T50": "fixed 50",
             "full_run": "full run"}
    sse = sse[sse.policy.isin(names)].assign(procedure=lambda x: x.policy.map(names))
    sse_max = sse.groupby(["regime", "procedure"]).rel_sse_vs_full.max()
    rows = []
    for reg in ("exact", "minibatch", "minibatch_const"):
        g = d[d.regime == reg]
        for p in EXTERNAL:
            q = g[g.procedure == p]
            rows.append({"사례": f"{RNAME[reg]} / {p}", "시간(중앙값)": q.cost.median(),
                         "시간 범위": f"{q.cost.min():.0%}–{q.cost.max():.0%}", "오류율(중앙값)": q.error.median(),
                         "오류율(최대)": q.error.max(), "목적함수 증가(최대)": sse_max.get((reg, p), np.nan),
                         "지배당한 사례": f"{int(q.dominated.sum())}/{len(q)}"})
    t = pd.DataFrame(rows).set_index("사례")
    T["pareto_summary"] = md(t, {"시간(중앙값)": "{:.0%}", "시간 범위": "{}", "오류율(중앙값)": "{:.2%}",
                                 "오류율(최대)": "{:.2%}", "목적함수 증가(최대)": "{:+.1e}", "지배당한 사례": "{}"},
                             index_name="체제 / 방법")
    lams = np.unique(np.r_[0, np.geomspace(0.01, 1000, 4001)])
    rows = []
    for reg in ("exact", "minibatch", "minibatch_const"):
        m = d[d.regime == reg].groupby("procedure").agg(cost=("cost", "mean"), err=("error", "mean"))
        best = [m.index[np.argmin(m.cost + lam * m.err * 100)] for lam in lams]
        start = 0
        for i in range(1, len(lams) + 1):
            if i == len(lams) or best[i] != best[i - 1]:
                hi = "∞" if i == len(lams) else f"{lams[i]:.2g}"
                rows.append({"체제": RNAME[reg], "가장 나은 방법": best[i - 1], "오류 1%p의 가치 (끝까지 돌린 시간 대비)": f"{lams[start]:.2g} ~ {hi}"})
                start = i
    t = pd.DataFrame(rows)
    t.index = range(1, len(t) + 1)
    T["pareto_lambda"] = md(t, "{}", index_name="#")
    return T


def warm_up():
    """Compile the numba kernels once so that no procedure is charged for JIT compilation."""
    H = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [2, 2, 2, 2]] * 4, dtype=np.uint8)[:11]
    for rule, regime in (("auto", "drift"), ("auto", "oscillation"), ("markov", None)):
        settle_labels(H, 10, regime, rule)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    warm_up()
    res, fronts = [], []
    for inst in INSTANCES:
        df, fr = analyse(*inst)
        res.append(df)
        fronts.append(fr)
        print(f"[{time.strftime('%H:%M:%S')}] {name_of(*inst)}", flush=True)
        print(df[["procedure", "T", "cost", "error", "excess_error", "cost_ratio", "dominated_by"]].round(4).to_string(index=False))
    res, fronts = pd.concat(res, ignore_index=True), pd.concat(fronts, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "pareto.csv", index=False)
    fronts.to_csv(RESULTS / "pareto_frontier.csv", index=False)
    figure(res, fronts)
    from experiments.analyze_markov_tail import fill_report
    fill_report(report_tables(res))


if __name__ == "__main__":
    main()
