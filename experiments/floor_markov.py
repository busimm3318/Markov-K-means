"""Accuracy the accelerator alone cannot reach: does the Markov layer buy it cheaply?

Purpose under test (set by the author after experiments/accelerators_markov.py and
experiments/minibatch_markov.py): the Markov layer is not meant to win time and accuracy at
once, but to reach an accuracy that the accelerator by itself never reaches, at a small cost.
Exact accelerators reach the exact fixed point when run on, so only accelerators with an
accuracy floor qualify. Bases (markov-kmeans' epoch mini-batch engine, batch 4096, 150 epochs,
the same k-means++ start as the reference):

    const2e-4, const1e-3   constant per-point step eta: never converges, points keep switching
    count                  Sculley's 1/count step (control: it converges to its own fixed point)

Primary reference R1: the exact K-means fixed point reached from the same start (incremental
Hamerly to convergence). Accuracy is the SSE excess of a partition (clusters at their
centroids) over R1's; the label agreement with R1 is reported too.

    floor          the lowest SSE excess of the base's own answers over epochs 1..100; its answer
                   at an epoch is its current centers assigned once (as MiniBatchKMeans' labels_)
    markov         MarkovKMeans' answer at the monitor's stop (the full monitor, epoch 100 when it
                   never stops): settlement (assign_rule="oscillation") and pseudo representatives
    simple remedies at that same epoch, each assigned once:
        avg10      the mean of the last 10 epoch-end centers (Polyak-Ruppert tail average)
        ema        an exponential average of the epoch-end centers (decay 0.9)
        polish1/3  1 / 3 exact Lloyd iterations (Hamerly) from the current centers

Pre-registered criterion (fixed before the runs; the Markov layer runs with its defaults).
In a configuration the Markov layer fulfils the purpose when

    (i)   it goes below the floor: SSE excess <= 0.75 x floor (a quarter of the gap the base cannot
          close is removed),
    (ii)  cheaply: its monitor and finish cost <= 10 % of the base's time up to the stop, and
    (iii) no simple remedy does better at that epoch: SSE excess <= that of avg10, ema and polish1
          (polish3 is reported).

Configurations whose floor is <= 1e-4 (no gap to close) are reported and left out of the count.
The purpose is supported for a base when (i)-(iii) hold in at least 2/3 of its counted
configurations. Secondary reference R2: each point's most frequent label over epochs 101..150 of
the same run (the base's own long-run state, the reference of results.md section 6); the same
tests with the label disagreement in place of the SSE excess.

Threads: everything runs with BLAS on one thread (numba kernels on all cores). Once OpenBLAS has
run multithreaded, its threads slow the compiled parallel kernels that follow (the monitor 11x on
a small mixture, even after the BLAS limit is restored), so one BLAS thread is the consistent
setting. It costs the base's epochs 0-23 % on these data (covtype 0, USCensus 7, MNIST 23 %);
the report gives test (ii) also with the base that much faster.

Configurations: covtype, MNIST and USCensus1990 with k = 16, 64 and k-means++ seeds 0, 1; the two
constant-step mixtures of results.md section 6 (n = 10^6, d = 16, k = 50, separation 1.5 and 2,
seed 0). A pilot on a small mixture is not counted.

    python -m experiments.floor_markov pilot
    python -m experiments.floor_markov run        # -> results/floor_markov.csv
    python -m experiments.floor_markov report
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
from experiments.accelerators_markov import RESULTS, dataset, markov_finish, warmup
from experiments.minibatch_markov import Scorer
from experiments.minibatch_markov import _nearest_all as _sk_nearest
from markov_kmeans import _rules
from markov_kmeans._engines import MiniBatchEngine

OUT = os.path.join(RESULTS, "floor_markov.csv")
EPOCHS, REF_EPOCHS, BATCH = 100, 50, 4096
BASES = (("const2e-4", "constant", 2e-4), ("const1e-3", "constant", 1e-3), ("count", "count", 0.0))
CHECKPOINTS = (2, 5, 10, 20, 30, 50, 75, 100)
GRID = ([(d, None, k, s) for d in ("covtype", "mnist", "census") for k in (16, 64) for s in (0, 1)]
        + [("gmm", sep, 50, 0) for sep in (1.5, 2.0)])
FLOOR_MIN, CROSS, CHEAP = 1e-4, 0.75, 0.10


def key_of(name, sep, k, seed):
    return f"{name}" + (f"-sep{sep:g}" if sep is not None else "") + f"-k{k}-s{seed}"


def load(name, sep):
    from kmeans_accel import datasets

    if name == "gmm":
        X, _, _ = datasets.gmm(1_000_000, 16, 50, sep=sep, seed=0)
        return np.ascontiguousarray(X)
    if name == "pilot":
        X, _, _ = datasets.gmm(120_000, 8, 20, sep=1.0, seed=5)
        return np.ascontiguousarray(X)
    return dataset(name)


def _nearest_all(X, C, nt):
    """scikit-learn's assignment (OpenMP over chunks) with BLAS on one thread, as scikit-learn runs it;
    everything else (the base engine's matrix products included) runs as in normal use."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        return _sk_nearest(X, C, nt)


def small(lab, k):
    return np.asarray(lab).astype(np.uint8 if k <= 256 else np.uint16)


def polish(X, C, m, nt):
    """m exact Lloyd iterations (Hamerly) from the centers C; the final assignment."""
    eng = Hamerly(X, C)
    for _ in range(m):
        if eng.step() == 0:
            break
    m_ = eng.cnt > 0
    Cn = eng.centers.copy()
    Cn[m_] = eng.S[m_] / eng.cnt[m_, None]
    return _nearest_all(X, Cn, nt)


def base_run(X, C0, seed, lr, eta, score, nt):
    """One base run of 150 epochs: per-epoch answers of the base and of the Markov layer, remedies at
    the checkpoints and at the monitor's stop; labels kept for the secondary reference."""
    n, k = X.shape[0], C0.shape[0]
    t0 = time.perf_counter()
    eng = MiniBatchEngine(X, C0, batch_size=BATCH, learning_rate=lr, eta=eta, seed=seed)
    secs = time.perf_counter() - t0
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True)
    t0 = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    mon_s = time.perf_counter() - t0
    hist, ema = deque(maxlen=10), None
    rows, keep, stop_epoch, regime_at_stop = [], {}, None, None
    ref_hist = np.empty((REF_EPOCHS, n), dtype=np.uint8 if k <= 256 else np.uint16)

    def timed(f):
        t = time.perf_counter()
        out = f()
        return out, time.perf_counter() - t

    for e in range(1, EPOCHS + REF_EPOCHS + 1):
        t0 = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t0
        if e > EPOCHS:
            ref_hist[e - EPOCHS - 1] = eng.labels
            continue
        hist.append(eng.centers.copy())
        ema = eng.centers.copy() if ema is None else 0.9 * ema + 0.1 * eng.centers
        lab, es = timed(lambda: _nearest_all(X, eng.centers, nt))
        s, a = score(lab)
        rows.append(dict(method="base", epoch=e, tau=secs + es, extra=es, sse=s, agree=a))
        keep[("base", e)] = small(lab, k)
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers, ch, movers=eng.movers)
        stop, regime, _ = mon.assess()
        mon_s += time.perf_counter() - t0
        if mon.t >= 2:
            c = clone(eng)
            (lab, _), fin = timed(lambda: markov_finish(X, c, mon, eng.labels.copy(), k,
                                                        regime if regime != "running" else "drift"))
            del c
            s, a = score(lab)
            rows.append(dict(method="markov", epoch=e, tau=secs + mon_s + fin, extra=mon_s + fin, sse=s, agree=a,
                             base_tau=secs, regime=regime))
            keep[("markov", e)] = small(lab, k)
        if stop and stop_epoch is None and mon.t >= 2:
            stop_epoch, regime_at_stop = e, regime
        if e in CHECKPOINTS or (stop_epoch == e) or (e == EPOCHS and stop_epoch is None):
            C_avg = np.mean(hist, axis=0)
            for name, f in (("avg10", lambda: _nearest_all(X, C_avg, nt)), ("ema", lambda: _nearest_all(X, ema, nt)),
                            ("polish1", lambda: polish(X, eng.centers, 1, nt)),
                            ("polish3", lambda: polish(X, eng.centers, 3, nt))):
                lab, x = timed(f)
                s, a = score(lab)
                rows.append(dict(method=name, epoch=e, tau=secs + x, extra=x, sse=s, agree=a))
                keep[(name, e)] = small(lab, k)
    # secondary reference: most frequent label over epochs 101..150
    counts = np.zeros((k, n), dtype=np.uint16)
    for j in range(k):
        counts[j] = (ref_hist == j).sum(0)
    modal = counts.argmax(0)
    del counts, ref_hist
    for r in rows:
        r["agree_r2"] = float((keep[(r["method"], r["epoch"])] == modal).mean())
    return rows, stop_epoch, regime_at_stop


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
        score.ref = ref.labels.copy()
        exact = score.sse(ref.labels)
        del ref
        for base, lr, eta in BASES:
            rows, stop_epoch, regime = base_run(X, C0, seed, lr, eta, score, nt)
            df = pd.DataFrame(rows)
            df["base"], df["stop_epoch"], df["regime_at_stop"] = base, stop_epoch, regime
            out.append(df)
            gc.collect()
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "config", key)
    df["dataset"], df["k"], df["seed"], df["n"] = name, k, seed, X.shape[0]
    df["sse_excess"] = df.sse / exact - 1
    print(key, f"{time.perf_counter() - t_wall:.0f}s", flush=True)
    return df


def judge(curves):
    """One row per configuration and base: floor, the Markov answer at its stop, remedies, verdict."""
    out = []
    for (key, base), g in curves.groupby(["config", "base"], sort=False):
        se = g.stop_epoch.iloc[0]
        never = pd.isna(se)
        e = EPOCHS if never else int(se)
        b = g[g.method == "base"]
        floor = float(b.sse_excess.min())
        floor_r2 = float(1 - b.agree_r2.max())
        at = g[g.epoch == e].set_index("method")
        mk = at.loc["markov"]
        base_tau = float(mk.base_tau)
        row = dict(config=key, base=base, stop_epoch=e, never_stopped=bool(never), regime=g.regime_at_stop.iloc[0],
                   floor=floor, floor_epoch=int(b.loc[b.sse_excess.idxmin(), "epoch"]), markov=float(mk.sse_excess),
                   markov_cost_share=float(mk.extra) / base_tau, base_at_stop=float(at.loc["base", "sse_excess"]),
                   floor_r2=floor_r2, markov_r2=float(1 - mk.agree_r2), markov_err_r1=float(1 - mk.agree),
                   floor_err_r1=float(1 - b.agree.max()))
        for m in ("avg10", "ema", "polish1", "polish3"):
            row[m] = float(at.loc[m, "sse_excess"])
            row[f"{m}_r2"] = float(1 - at.loc[m, "agree_r2"])
            row[f"{m}_cost_share"] = float(at.loc[m, "extra"]) / base_tau
        row["counted"] = floor > FLOOR_MIN
        row["i"] = row["markov"] <= CROSS * floor
        row["ii"] = row["markov_cost_share"] <= CHEAP
        row["ii_fast_base"] = row["markov_cost_share"] / 0.77 <= CHEAP   # base 23 % faster (MNIST, multithreaded BLAS)
        row["iii"] = all(row["markov"] <= row[m] for m in ("avg10", "ema", "polish1"))
        row["fulfils"] = row["i"] and row["ii"] and row["iii"]
        # secondary reference R2 (label disagreement with the base's own long-run modal labels)
        row["counted_r2"] = floor_r2 > FLOOR_MIN
        row["i_r2"] = row["markov_r2"] <= CROSS * floor_r2
        row["iii_r2"] = all(row["markov_r2"] <= row[f"{m}_r2"] for m in ("avg10", "ema", "polish1"))
        row["fulfils_r2"] = row["i_r2"] and row["ii"] and row["iii_r2"]
        out.append(row)
    return pd.DataFrame(out)


def report(path=OUT):
    j = judge(pd.read_csv(path))
    j = j[~j.config.str.startswith("pilot")]
    pd.set_option("display.width", 250)
    cols = ["config", "base", "stop_epoch", "never_stopped", "floor", "markov", "avg10", "ema", "polish1", "polish3",
            "markov_cost_share", "i", "ii", "iii", "fulfils", "floor_r2", "markov_r2", "avg10_r2", "fulfils_r2"]
    print(j[cols].round(5).to_string(index=False))
    for base, g in j.groupby("base", sort=False):
        c = g[g.counted]
        c2 = g[g.counted_r2]
        print(f"{base:10s} R1: {int(c.fulfils.sum())}/{len(c)} counted (supported: {len(c) > 0 and c.fulfils.mean() >= 2 / 3})"
              f" | i {int(c.i.sum())} ii {int(c.ii.sum())} (fast base {int(c.ii_fast_base.sum())}) iii {int(c.iii.sum())}"
              f" || R2: {int(c2.fulfils_r2.sum())}/{len(c2)} (supported: {len(c2) > 0 and c2.fulfils_r2.mean() >= 2 / 3})")
    j.to_csv(os.path.join(RESULTS, "floor_markov_judgement.csv"), index=False)
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
        p = os.path.join(RESULTS, "floor_markov_pilot.csv")
        run(grid=[("pilot", None, 20, 0)], out=p)
        print(judge(pd.read_csv(p)).round(5).T.to_string())
    elif cmd == "run":
        run()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else OUT)
