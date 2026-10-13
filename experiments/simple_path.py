"""The 0.6 default path against the 0.5 default, on the 10 trajectories of experiments/pareto.py.

    0.5 default   stop_rule="auto" (the full monitor), assign_rule="auto" (history settlement of
                  oscillating points also in a drift), pseudo representatives
    0.6 default   exact engines: stop_rule="unstable" (the windowed changed-share threshold) with a
                  history-free monitor; mini-batch: the full monitor as before; assign_rule=
                  "oscillation" (label histories only in an oscillating run); pseudo representatives

Both monitors are fed the recorded labels with the list of movers, as MarkovKMeans.fit feeds
them from the engines, and their own time (push + assess) is counted. The pseudo step runs on
the base algorithm's own state (Hamerly replayed to the stop for exact runs).

    python -m experiments.simple_path      # -> experiments/results/simple_path.csv
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.auto_rule import reference
from experiments.pareto import INSTANCES, K, RESULTS, name_of, trajectory
from experiments.time_for_error import _Brute
from kmeans_accel import datasets
from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import compute_centers, inertia
from markov_kmeans.estimator import pseudo_finish

REASON = {"converged": "converged", "oscillation": "oscillation"}


def replay(traj, exact, rule, history):
    """(stop t, reason, monitor, seconds spent in the monitor) for one stop rule on a trajectory."""
    H, C, counts = traj["H"], traj["C"], traj["counts"]
    p = _rules.StopParams()
    mon = _rules.StateMonitor(H.shape[1], K, p, exact=exact, rule=rule, history=history)
    mon.push(H[0], C[0])
    spent = 0.0
    for t in range(1, len(H)):
        movers = np.flatnonzero(H[t] != H[t - 1])           # what the engine reports (not timed)
        t0 = time.perf_counter()
        mon.push(H[t], C[t], int(counts[t - 1]), movers=movers)
        stop, regime, state = mon.assess()
        spent += time.perf_counter() - t0
        if stop:
            reason = REASON.get(regime) or ("few_unstable" if state.get("unstable_frac", 1) <= p.unstable_tol
                                            else "stable_drift")
            return t, reason, mon, spent
    return len(H) - 1, "max_iter", mon, spent


def finish(X, traj, t, reason, mon, assign, exact):
    """Settlement and pseudo step at stop t: (labels, seconds)."""
    H, C = traj["H"], traj["C"]
    W = mon.params.window
    L = H[t].astype(np.int64)
    if reason == "converged":
        return L, 0.0
    if exact:
        eng = HamerlyEngine(X, C[0])
        for _ in range(t):
            eng.step()
        assert np.array_equal(eng.labels, H[t])
    else:
        eng = _Brute(X, C[t])
    t0 = time.perf_counter()
    U = mon.unstable().astype(np.int64)
    regime = "oscillation" if reason == "oscillation" else "drift"
    if mon.history:
        lab_U, S, P, osc = _rules.settle(mon.window(U), L[U], rule=assign, n_clusters=K, regime=regime)
    else:
        lab_U, S, P, osc = _rules.settle(None, L[U], rule=assign, n_clusters=K, regime=regime,
                                         n_switches=mon.switch_counts(U), width=W + 1)
    lab = L.copy()
    lab[U] = lab_U
    lo = max(0, t - W)
    out = pseudo_finish(X, eng, SimpleNamespace(centers_window=lambda: C[lo:t + 1]), L, lab, U, S, P, osc,
                        inplace=True)
    return out[0], time.perf_counter() - t0


def run():
    rows = []
    for inst in INSTANCES:
        regime, sep, seed, eta = inst
        traj, X = trajectory(*inst)
        if X is None:
            X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
        exact = regime == "exact"
        ref, _ = reference(regime, traj, K)
        sse_ref = float(inertia(X, compute_centers(X, ref, traj["C"][-1])[0], ref))
        full = traj["secs"][len(traj["H"]) - 1]
        for name, rule, history, assign in (
                ("0.5 default", "auto", True, "history"),
                ("0.6 default", "unstable" if exact else "auto", not exact, "oscillation")):
            t, reason, mon, mon_s = replay(traj, exact, rule, history)
            lab, fin_s = finish(X, traj, t, reason, mon, assign, exact)
            sse = float(inertia(X, compute_centers(X, lab, traj["C"][t])[0], lab))
            rows.append(dict(instance=name_of(*inst), regime=regime, path=name, t=t, reason=reason,
                             error=float((lab != ref).mean()), sse_excess=sse / sse_ref - 1,
                             cost=(traj["secs"][t] + mon_s + fin_s) / full, monitor_share=mon_s / full,
                             history=history))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/simple_path.csv", index=False)
    return df


if __name__ == "__main__":
    print(run().round(5).to_string(index=False))
