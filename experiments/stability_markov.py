"""Label stabilization: does the Markov layer keep mini-batch labels steady without losing quality?

Purpose under test (set by the author after experiments/floor_markov.py): the Markov layer
stabilizes labels. Two meanings, both measured:

    churn            a running mini-batch reports labels once per epoch (a downstream system
                     reads them while the model keeps training); churn is the share of points
                     whose reported label changed since the previous report, averaged over
                     epochs 11..100
    reproducibility  two runs from the same k-means++ centers that differ only in the order of
                     the minibatches; disagreement is the share of points labelled differently
                     at epoch 100 (clusters matched one to one, Hungarian)

Both are trivially zero for a method that never updates, so every comparison carries a quality
condition: the SSE excess over the exact K-means fixed point from the same start (the
reference of experiments/floor_markov.py) may not be worse than the base's by more than 10 %
of its size plus 1e-5 (mean over epochs 11, 15, 20, ..., 100 for churn; epoch 100 for
reproducibility).

Bases (markov-kmeans' epoch mini-batch engine, batch 4096, 100 epochs): constant step
eta = 2e-4 and 1e-3 (points keep switching) and the 1/count step. Reported labels per epoch:

    base     the current centers assigned once (MiniBatchKMeans' labels_)
    markov   MarkovKMeans' answer if stopped now (t0 = epoch): settlement (assign_rule=
             "oscillation") and pseudo representatives; the base answer at epoch 1
    avg10    the mean of the last 10 epoch-end centers, assigned once
    ema      an exponential average of the epoch-end centers (decay 0.9), assigned once
    hyst3    the base labels with hysteresis: a point's reported label changes only after the
             base gave it the same new label at 3 consecutive reports
    polish1  one exact Lloyd iteration (Hamerly) from the current centers, at epoch 100 only
             (reproducibility; it costs about two passes over the data per report)

Pre-registered criterion (fixed before the runs; the Markov layer runs with its defaults). In a
configuration the Markov layer stabilizes labels when

    S1  churn: its churn <= 0.5 x the base's, at no worse quality;
    S2  reproducibility: its disagreement <= 0.5 x the base's, at no worse quality; and
    U   for S1 and for S2, no simple stabilizer (avg10, ema, hyst3) that meets the same quality
        condition is more stable than it.

The purpose is supported for a base when S1, S2 and U hold in at least 2/3 of the
configurations. Reported alongside: polish1, the costs, the monitor's own stop epoch.

Configurations: covtype and MNIST with k = 16, 64 and k-means++ seeds 0, 1; USCensus1990 with
k = 16, 64, seed 0; the two constant-step mixtures of results.md section 6 (n = 10^6, d = 16,
k = 50, separation 1.5, 2, seed 0). Minibatch orders: seeds 1000 and 1001. Everything runs with
BLAS on one thread (see experiments/floor_markov.py). A pilot on a small mixture is not counted.

    python -m experiments.stability_markov pilot
    python -m experiments.stability_markov run      # -> results/stability_markov.csv
    python -m experiments.stability_markov report
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

from experiments.accel_engines import Hamerly, clone
from experiments.accelerators_markov import RESULTS, markov_finish, warmup
from experiments.floor_markov import BASES, _nearest_all, load, polish
from experiments.minibatch_markov import Scorer
from markov_kmeans import _rules
from markov_kmeans._engines import MiniBatchEngine

OUT = os.path.join(RESULTS, "stability_markov.csv")
EPOCHS, BATCH, FROM = 100, 4096, 11
SSE_EPOCHS = (11,) + tuple(range(15, EPOCHS + 1, 5))
ORDERS = (1000, 1001)
METHODS = ("base", "markov", "avg10", "ema", "hyst3")
SIMPLE = ("avg10", "ema", "hyst3")
GRID = ([(d, None, k, s) for d in ("covtype", "mnist") for k in (16, 64) for s in (0, 1)]
        + [("census", None, k, 0) for k in (16, 64)]
        + [("gmm", sep, 50, 0) for sep in (1.5, 2.0)])
HALF, Q_REL, Q_ABS = 0.5, 0.10, 1e-5


def key_of(name, sep, k, seed):
    return f"{name}" + (f"-sep{sep:g}" if sep is not None else "") + f"-k{k}-s{seed}"


def matched_disagreement(a, b, k):
    """Share of points labelled differently by a and b after the best one-to-one cluster matching."""
    from scipy.optimize import linear_sum_assignment

    conf = np.bincount(a.astype(np.int64) * k + b.astype(np.int64), minlength=k * k).reshape(k, k)
    r, c = linear_sum_assignment(-conf)
    return 1.0 - conf[r, c].sum() / len(a)


class Hysteresis:
    """Reported labels that follow the base only after `m` consecutive reports of the same new label."""

    def __init__(self, labels, m=3):
        self.rep = labels.astype(np.int64).copy()
        self.cand = self.rep.copy()
        self.cnt = np.zeros(len(labels), dtype=np.int16)
        self.m = m

    def update(self, labels):
        L = labels.astype(np.int64)
        same = L == self.rep
        follow = ~same & (L == self.cand)
        new = ~same & ~follow
        self.cnt[same] = 0
        self.cnt[follow] += 1
        self.cand[new] = L[new]
        self.cnt[new] = 1
        flip = self.cnt >= self.m
        self.rep[flip] = L[flip]
        self.cnt[flip] = 0
        return self.rep


def one_run(X, C0, order, lr, eta, score, nt):
    n, k = X.shape[0], C0.shape[0]
    t0 = time.perf_counter()
    eng = MiniBatchEngine(X, C0, batch_size=BATCH, learning_rate=lr, eta=eta, seed=order)
    base_s = time.perf_counter() - t0
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True)
    t0 = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    cost = {m: 0.0 for m in METHODS}
    cost["markov"] += time.perf_counter() - t0
    hist, ema, hyst = deque(maxlen=10), None, None
    prev = {m: None for m in METHODS}
    churn = {m: [] for m in METHODS}
    sse = {m: [] for m in METHODS}
    stop_epoch = None
    for e in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        ch = eng.step()
        base_s += time.perf_counter() - t0
        rep = {}
        t0 = time.perf_counter()
        rep["base"] = _nearest_all(X, eng.centers, nt)
        cost["base"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        hist.append(eng.centers.copy())
        ema = eng.centers.copy() if ema is None else 0.9 * ema + 0.1 * eng.centers
        rep["avg10"] = _nearest_all(X, np.mean(hist, axis=0), nt)
        cost["avg10"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        rep["ema"] = _nearest_all(X, ema, nt)
        cost["ema"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        hyst = Hysteresis(rep["base"]) if hyst is None else hyst
        rep["hyst3"] = hyst.update(rep["base"]).copy()
        cost["hyst3"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers, ch, movers=eng.movers)
        stop, regime, _ = mon.assess()
        cost["markov"] += time.perf_counter() - t0
        if mon.t >= 2:
            c = clone(eng)
            t0 = time.perf_counter()
            lab, _ = markov_finish(X, c, mon, eng.labels.copy(), k, regime if regime != "running" else "drift")
            cost["markov"] += time.perf_counter() - t0
            del c
            rep["markov"] = np.asarray(lab)
        else:
            rep["markov"] = rep["base"]
        if stop and stop_epoch is None and mon.t >= 2:
            stop_epoch = e
        for m in METHODS:
            r = np.asarray(rep[m])
            if e >= FROM and prev[m] is not None:
                churn[m].append(float((r != prev[m]).mean()))
            if e in SSE_EPOCHS:
                sse[m].append(score.sse(r))
            prev[m] = r.astype(np.int64) if r.dtype != np.int64 else r.copy()
    final = {m: prev[m] for m in METHODS}
    t0 = time.perf_counter()
    final["polish1"] = np.asarray(polish(X, eng.centers, 1, nt)).astype(np.int64)
    p_cost = time.perf_counter() - t0
    rows = []
    for m in METHODS:
        rows.append(dict(method=m, churn=float(np.mean(churn[m])), sse_mean=float(np.mean(sse[m])), sse_100=sse[m][-1],
                         cost=cost[m], cost_share=cost[m] / base_s))
    rows.append(dict(method="polish1", churn=np.nan, sse_mean=np.nan, sse_100=score.sse(final["polish1"]),
                     cost=p_cost, cost_share=p_cost / base_s))
    for r in rows:
        r.update(order=order, base_seconds=base_s, stop_epoch=stop_epoch)
    return rows, final


def run_config(name, sep, k, seed):
    from sklearn.cluster import kmeans_plusplus
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from threadpoolctl import threadpool_limits

    key = key_of(name, sep, k, seed)
    X = load(name, sep)
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    score = Scorer(X, k)
    nt = _openmp_effective_n_threads()
    t_wall = time.perf_counter()
    out = []
    with threadpool_limits(limits=1, user_api="blas"):
        ref = Hamerly(X, C0)
        while ref.step():
            pass
        exact = score.sse(ref.labels)
        del ref
        for base, lr, eta in BASES:
            finals = []
            for order in ORDERS:
                rows, final = one_run(X, C0, order, lr, eta, score, nt)
                for r in rows:
                    r["base"] = base
                out += rows
                finals.append(final)
                gc.collect()
            for m in METHODS + ("polish1",):
                out.append(dict(base=base, method=m, order="pair",
                                disagreement=matched_disagreement(finals[0][m], finals[1][m], k)))
            del finals
    df = pd.DataFrame(out)
    df.insert(0, "config", key)
    df["dataset"], df["k"], df["seed"], df["n"], df["exact_sse"] = name, k, seed, X.shape[0], exact
    for c in ("sse_mean", "sse_100"):
        df[c.replace("sse", "excess")] = df[c] / exact - 1
    print(key, f"{time.perf_counter() - t_wall:.0f}s", flush=True)
    return df


def _ok(q, q_base):
    return q <= q_base + Q_REL * abs(q_base) + Q_ABS


def judge(df):
    out = []
    for (key, base), g in df.groupby(["config", "base"], sort=False):
        runs = g[g.order != "pair"].groupby("method")[["churn", "excess_mean", "excess_100", "cost_share"]].mean()
        pair = g[g.order == "pair"].set_index("method").disagreement
        row = dict(config=key, base=base, stop_epoch=g[g.order != "pair"].stop_epoch.dropna().median()
                   if g[g.order != "pair"].stop_epoch.notna().any() else np.nan)
        for m in METHODS:
            row[f"churn_{m}"] = runs.loc[m, "churn"]
            row[f"q_{m}"] = runs.loc[m, "excess_mean"]
            row[f"dis_{m}"] = pair[m]
            row[f"q100_{m}"] = runs.loc[m, "excess_100"]
            row[f"cost_{m}"] = runs.loc[m, "cost_share"]
        row["dis_polish1"], row["q100_polish1"] = pair["polish1"], runs.loc["polish1", "excess_100"]
        ok1 = {m: _ok(row[f"q_{m}"], row["q_base"]) for m in METHODS}
        ok2 = {m: _ok(row[f"q100_{m}"], row["q100_base"]) for m in METHODS}
        row["S1"] = ok1["markov"] and row["churn_markov"] <= HALF * row["churn_base"]
        row["S2"] = ok2["markov"] and row["dis_markov"] <= HALF * row["dis_base"]
        u1 = all(row["churn_markov"] <= row[f"churn_{m}"] for m in SIMPLE if ok1[m])
        u2 = all(row["dis_markov"] <= row[f"dis_{m}"] for m in SIMPLE if ok2[m])
        row["U"] = u1 and u2
        row["stabilizes"] = row["S1"] and row["S2"] and row["U"]
        out.append(row)
    return pd.DataFrame(out)


def report(path=OUT):
    j = judge(pd.read_csv(path))
    j = j[~j.config.str.startswith("pilot")]
    pd.set_option("display.width", 300)
    cols = ["config", "base", "stop_epoch"] + [f"churn_{m}" for m in METHODS] + [f"dis_{m}" for m in METHODS] + \
           ["dis_polish1", "q_base", "q_markov", "q100_base", "q100_markov", "q100_polish1", "cost_markov", "S1", "S2", "U", "stabilizes"]
    print(j[cols].round(5).to_string(index=False))
    for base, g in j.groupby("base", sort=False):
        print(f"{base:10s} stabilizes {int(g.stabilizes.sum())}/{len(g)} (supported: {g.stabilizes.mean() >= 2 / 3}) "
              f"| S1 {int(g.S1.sum())} S2 {int(g.S2.sum())} U {int(g.U.sum())}")
    j.to_csv(os.path.join(RESULTS, "stability_markov_judgement.csv"), index=False)
    return j


def run(grid=GRID, out=OUT):
    old = pd.read_csv(out) if os.path.exists(out) else None
    done = set(old.config) if old is not None else set()
    warmup()
    for cfg in grid:
        key = key_of(*cfg)
        if key in done:
            continue
        df = run_config(*cfg)
        old = df if old is None else pd.concat([old, df], ignore_index=True)
        old.to_csv(out, index=False)
        del df
        gc.collect()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "pilot":
        p = os.path.join(RESULTS, "stability_markov_pilot.csv")
        run(grid=[("pilot", None, 20, 0)], out=p)
        print(judge(pd.read_csv(p)).round(5).T.to_string())
    elif cmd == "run":
        run()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else OUT)
