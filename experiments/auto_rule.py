"""Evaluate the automatic stop / assignment rules of markov_kmeans against alternatives.

One trajectory per instance is recorded with the package's own engines (labels of every
point at every iteration, change counts, cost, centers).  Stopping policies are then
replayed on that trajectory with the package's own decision functions
(``markov_kmeans._rules``), so each policy is evaluated on the identical run.  The first
small instance also checks that the replay reproduces ``MarkovKMeans.fit`` exactly.

Reference: exact Lloyd -> converged labels; mini-batch -> modal label over the last 50
of 200 epochs (soft reference: the occupancy over those epochs).

    python -m experiments.auto_rule [--n 1000000] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import kmeans_plusplus

from kmeans_accel import datasets
from markov_kmeans import MarkovKMeans, _rules
from markov_kmeans._engines import HamerlyEngine, MiniBatchEngine
from markov_kmeans._kernels import compute_centers, inertia
from markov_kmeans._markov import to_dense

RESULTS = Path(__file__).resolve().parent / "results"
HORIZON = 200
REF_WINDOW = 50


def record(regime, X, C0, seed, eta):
    if regime == "exact":
        eng = HamerlyEngine(X, C0)
    else:
        eng = MiniBatchEngine(X, C0, 4096, "constant" if regime == "minibatch_const" else "count", eta, seed)
    labels = [eng.labels.astype(np.uint8)]
    centers = [eng.centers.copy()]
    counts, ndist, secs = [], [eng.n_dist], [0.0]
    t0 = time.perf_counter()
    cap = 3000 if regime == "exact" else HORIZON
    for _ in range(cap):
        t = time.perf_counter()
        ch = eng.step()
        secs.append(secs[-1] + time.perf_counter() - t)
        counts.append(ch)
        ndist.append(eng.n_dist)
        labels.append(eng.labels.astype(np.uint8))
        centers.append(eng.centers.copy())
        if regime == "exact" and ch == 0:
            break
    return dict(H=np.array(labels), C=np.array(centers), counts=np.array(counts),
                ndist=np.array(ndist, dtype=float), secs=np.array(secs), wall=time.perf_counter() - t0)


def reference(regime, traj, k):
    H = traj["H"]
    if regime == "exact":
        return H[-1].astype(np.int64), None
    tail = H[-REF_WINDOW:].astype(np.int64)
    n = H.shape[1]
    occ = np.zeros((n, k), dtype=np.float32)
    for row in tail:
        occ[np.arange(n), row] += 1.0 / REF_WINDOW
    return occ.argmax(1), occ


REASON = lambda regime, state, p: {"converged": "converged", "oscillation": "oscillation",
                                   "drift": "few_unstable" if state.get("unstable_frac", 1) <= p.unstable_tol
                                   else "stable_drift"}[regime]
STATE_KEYS = ("unstable_frac", "center_drift", "center_travel", "cluster_unstable", "size_change", "net_flow",
              "oscillating_share", "projected_changes")


def replay_stop(traj, params, exact, rule, k):
    """Same StateMonitor as MarkovKMeans.fit, fed with the recorded trajectory.

    The monitor's own time (push + assess, as paid inside fit) is returned in
    state["monitor_seconds"] so that the cost of a stopping rule includes it.
    """
    H, C, counts = traj["H"], traj["C"], traj["counts"]
    mon = _rules.StateMonitor(H.shape[1], k, params, exact=exact, rule=rule)
    mon.push(H[0], C[0])
    state, spent = {}, 0.0
    for t in range(1, len(H)):
        t0 = time.perf_counter()
        mon.push(H[t], C[t], int(counts[t - 1]))
        stop, regime, state = mon.assess()
        spent += time.perf_counter() - t0
        if stop:
            return t, REASON(regime, state, params), mon.last_change.copy(), {**state, "monitor_seconds": spent}
    return len(H) - 1, "max_iter", mon.last_change.copy(), {**state, "monitor_seconds": spent}


def first_t(cond, T):
    for t in range(1, T):
        if cond(t):
            return t
    return T - 1


def evaluate(name, regime, traj, X, k, t, reason, last_change, assign, ref, occ, W, extra=None):
    H = traj["H"]
    L = H[t].astype(np.int64)
    U = np.flatnonzero(last_change > t - W) if reason != "converged" else np.empty(0, dtype=np.int64)
    labels = L.copy()
    soft_tv = float("nan")
    if len(U) and assign != "keep_all":
        Hw = H[max(0, t - W):t + 1][:, U].astype(np.int64)
        regime_ = "oscillation" if reason == "oscillation" else "drift"
        lab_U, S, P, _ = _rules.settle(Hw, L[U], rule=assign, regime=regime_)
        labels[U] = lab_U
        if occ is not None:
            soft_tv = float(0.5 * np.abs(to_dense(S, P, k) - occ[U]).sum(1).mean())
    if regime == "exact":
        C = compute_centers(X, labels, traj["C"][t])[0]
    else:
        C = traj["C"][t]
    full = len(H) - 1
    monitor = (extra or {}).get("monitor_seconds", 0.0)
    row = dict(policy=name, T_stop=t, stop_reason=reason, n_U=len(U),
               dist_frac=traj["ndist"][t] / traj["ndist"][full],
               time_frac=(traj["secs"][t] + monitor) / traj["secs"][full], monitor_seconds=monitor,
               disagree_rate=float((labels != ref).mean()),
               U_err=float((labels[U] != ref[U]).mean()) if len(U) else 0.0,
               sse=float(inertia(X, C, labels)), soft_tv_U=soft_tv)
    row.update(extra or {})
    return row


def write(path, row, fields):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def run_instance(regime, n, d, k, sep, seed, eta, out, check=False):
    X, z, means = datasets.gmm(n, d, k, sep=sep, seed=seed)
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    traj = record(regime, X, C0, seed, eta)
    exact = regime == "exact"
    ref, occ = reference(regime, traj, k)
    params = _rules.StopParams()
    W = params.window
    T = len(traj["H"])
    inst = dict(regime=regime, n=n, d=d, k=k, sep=sep, seed=seed, eta=eta if regime == "minibatch_const" else "",
                T_full=T - 1, record_seconds=round(traj["wall"], 1))
    print(f"[{time.strftime('%H:%M:%S')}] {inst}", flush=True)
    full_sse = None
    rows = []
    # 1) the package's automatic rules (and variants of the assignment at the same stop)
    t, reason, lc, state = replay_stop(traj, params, exact, "auto", k)
    at_stop = {f"state_{key}": state.get(key, "") for key in STATE_KEYS}
    at_stop["monitor_seconds"] = state["monitor_seconds"]
    if check:
        mk = MarkovKMeans(k, init=C0, algorithm="hamerly" if exact else "minibatch",
                          learning_rate="constant" if regime == "minibatch_const" else "count",
                          eta=eta, max_iter=len(traj["H"]) - 1, random_state=seed).fit(X)
        assert mk.n_iter_ == t and mk.stop_reason_ == reason, (mk.n_iter_, t, mk.stop_reason_, reason)
        print(f"  replay matches MarkovKMeans.fit: stop at {t} ({reason})", flush=True)
    for assign in ("auto", "markov", "frequency", "keep"):
        rows.append(evaluate(f"auto_stop+{assign}", regime, traj, X, k, t, reason, lc, assign, ref, occ, W, at_stop))
    # 2) the 0.1.0 default: unstable fraction <= 1e-3 only, Markov assignment
    t, reason, lc, st = replay_stop(traj, params, exact, "unstable", k)
    rows.append(evaluate("v0.1.0(unstable_tol)+markov", regime, traj, X, k, t, reason, lc, "markov", ref, occ, W,
                         {"monitor_seconds": st["monitor_seconds"]}))
    # 3) common stopping rules of other implementations, labels kept as they are
    var = float(np.var(X, axis=0).mean())
    Cs = traj["C"]
    shift = lambda t_: float(((Cs[t_] - Cs[t_ - 1]) ** 2).sum())
    t = first_t(lambda t_: shift(t_) <= 1e-4 * var, T)
    rows.append(evaluate("sklearn_tol(center_shift<=1e-4 var)", regime, traj, X, k, t, "rule", lc_at(traj, t), "keep_all", ref, occ, W))
    t = first_t(lambda t_: traj["counts"][t_ - 1] < 0.03 * n, T)
    rows.append(evaluate("perez_ortega(changes<0.03n)", regime, traj, X, k, t, "rule", lc_at(traj, t), "keep_all", ref, occ, W))
    for T0 in (20, 50):
        t = min(T0, T - 1)
        rows.append(evaluate(f"fixed_T{T0}", regime, traj, X, k, t, "t0", lc_at(traj, t), "keep_all", ref, occ, W))
    # 4) full run
    rows.append(evaluate("full_run", regime, traj, X, k, T - 1, "full", lc_at(traj, T - 1), "keep_all", ref, occ, W))
    full_sse = rows[-1]["sse"]
    for r in rows:
        r["rel_sse_vs_full"] = r["sse"] / full_sse - 1.0
    fields = list(dict.fromkeys(k for r in rows for k in {**inst, **r}))
    for r in rows:
        write(out, {**inst, **r}, fields)
        print(f"  {r['policy']:38s} stop={r['T_stop']:4d} ({r['stop_reason']:12s}) cost={r['dist_frac']:.3f}/{r['time_frac']:.3f} "
              f"err={r['disagree_rate']:.5f} U_err={r['U_err']:.3f} |U|={r['n_U']} tv={r['soft_tv_U']:.3f}", flush=True)


def lc_at(traj, t):
    H = traj["H"]
    lc = np.full(H.shape[1], -1, dtype=np.int32)
    for s in range(1, t + 1):
        lc[H[s] != H[s - 1]] = s
    return lc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="auto_rule.csv")
    a = p.parse_args(argv)
    out = RESULTS / a.out
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.quick:
        run_instance("exact", 50_000, 8, 20, 1.2, 0, 0.0, out, check=True)
        run_instance("minibatch_const", 50_000, 8, 20, 1.2, 0, 1e-3, out, check=True)
        return
    run_instance("exact", 100_000, 16, 50, 1.5, 0, 0.0, out, check=True)   # equivalence check
    for sep in (1.5, 2.0):
        for seed in (0, 1):
            run_instance("exact", a.n, 16, 50, sep, seed, 0.0, out)
        run_instance("minibatch", a.n, 16, 50, sep, 0, 0.0, out)
        run_instance("minibatch_const", a.n, 16, 50, sep, 0, 2e-4, out)
    run_instance("minibatch_const", a.n, 16, 50, 1.5, 1, 2e-4, out)    # second seed
    run_instance("minibatch_const", a.n, 16, 50, 1.5, 0, 1e-3, out)    # larger step: stronger jitter
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    main()
