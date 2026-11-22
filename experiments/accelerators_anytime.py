"""Accuracy as a function of the time spent: each accelerator alone and with the Markov layer.

For one configuration of experiments/accelerators_markov.py, every accelerator runs on its
own (no other run sharing the CPU) to its end, and after every iteration two answers are
formed outside the timing:

    plain     what the accelerator returns if stopped now: its current labels (mini-batch: a
              final assignment to the current centers, as MiniBatchKMeans does, timed)
    markov    what MarkovKMeans returns if stopped now (t0 = this iteration): the history-free
              settlement and the pseudo representative step on a copy of the engine, timed

The time of a plain answer is the engine's time so far (plus the final assignment for
mini-batch); the time of a markov answer adds the monitor (push and assess every iteration)
and the finishing step. Accuracy is the share of labels equal to the accelerator's own end
(convergence; mini-batch: 100 epochs). The curves sample, at every whole second from 0 to
the end, the most recent answer available within that budget.

Task difficulty: Gaussian mixtures of n = 2 x 10^6 points in d = 16 with 50 components whose
means are drawn from N(0, sep^2 I) and unit noise, clustered with k = 50; only the separation
changes: sep = 4 (very easy: converged in 3 iterations, ARI 1.00 against the components),
2 (easy: 66, 0.95), 1 (hard: 154, 0.75), 0.5 (very hard: 404, 0.13; iterations and ARI from a
4 x 10^5-point check).

    python -m experiments.accelerators_anytime covtype 64 0   # -> results/accelerators_anytime_<config>.csv
    python -m experiments.accelerators_anytime figure covtype-k64-s0
    python -m experiments.accelerators_anytime difficulty     # the four separations, one figure each
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.accel_engines import clone
from experiments.accelerators_markov import (ENGINE_KO, ENGINES, EXACT, MAX_ITER, MB_EPOCHS, RESULTS, SkMiniBatch,
                                             _korean_font, config_key, dataset, make, markov_finish, warmup)
from markov_kmeans import _rules


def _small(labels, k):
    return np.asarray(labels).astype(np.uint8 if k <= 256 else np.uint16)


def trace_exact(name, X, C0):
    n, k = X.shape[0], C0.shape[0]
    t0 = time.perf_counter()
    eng = make(name, X, C0)
    secs = time.perf_counter() - t0
    external = name in ("squarem", "anderson")
    prev = eng.labels.copy()
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=True, rule="unstable", history=False)
    t0 = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    mon_s = time.perf_counter() - t0
    plain, mk, rows = [_small(eng.labels, k)], [None], [dict(t=0, engine_s=secs, monitor_s=mon_s, finish_s=np.nan)]
    auto = None
    for t in range(1, MAX_ITER + 1):
        t0 = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t0
        movers = np.flatnonzero(eng.labels != prev) if external else eng.movers
        if external:
            prev[movers] = eng.labels[movers]
            ch = len(movers)
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers, ch, movers=movers)
        stop, regime, _ = mon.assess()
        mon_s += time.perf_counter() - t0
        if stop and auto is None:
            auto = t
        plain.append(_small(eng.labels, k))
        fin = np.nan
        lab = None
        if ch == 0:                                            # converged: MarkovKMeans returns these labels
            fin, lab = 0.0, _small(eng.labels, k)
        elif mon.t >= 2:
            c = clone(eng)                                     # not timed
            L = eng.labels.copy()
            t0 = time.perf_counter()
            lab, _ = markov_finish(X, c, mon, L, k, "drift")
            fin = time.perf_counter() - t0
            del c
            lab = _small(lab, k)
        mk.append(lab)
        rows.append(dict(t=t, engine_s=secs, monitor_s=mon_s, finish_s=fin))
        if ch == 0:
            break
    ref = plain[-1]
    df = pd.DataFrame(rows)
    df["acc_plain"] = [float((p == ref).mean()) for p in plain]
    df["acc_markov"] = [float((m == ref).mean()) if m is not None else np.nan for m in mk]
    df["time_plain"] = df.engine_s
    df["time_markov"] = df.engine_s + df.monitor_s + df.finish_s
    df["auto_stop"] = df.t == auto
    return df


def trace_minibatch(X, C0, seed):
    from threadpoolctl import threadpool_limits

    n, k = X.shape[0], C0.shape[0]
    with threadpool_limits(limits=1, user_api="blas"):
        t0 = time.perf_counter()
        eng = SkMiniBatch(X, C0, seed)
        secs = time.perf_counter() - t0
        mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True)
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers)
        mon_s = time.perf_counter() - t0
        spe = n // eng.batch
        early_samples = {2 ** j for j in range(20) if 2 ** j < spe}
        prev = eng.labels.copy()
        rows, plain, mk = [], [], []
        auto = default = None
        epoch = 0

        def sample(markov_lab, fin):
            t0 = time.perf_counter()
            lab, _ = eng.nearest(eng.centers)                  # MiniBatchKMeans' final labels_
            es = time.perf_counter() - t0
            plain.append(_small(lab, k))
            mk.append(markov_lab)
            rows.append(dict(t=eng.step_i / spe, engine_s=secs, estep_s=es, monitor_s=mon_s, finish_s=fin))

        sample(None, np.nan)
        for i in range((MB_EPOCHS * n) // eng.batch):
            t0 = time.perf_counter()
            early = eng.minibatch()
            secs += time.perf_counter() - t0
            if early and default is None:
                default = eng.step_i / spe
            if eng.step_i % spe:
                if eng.step_i in early_samples:
                    sample(None, np.nan)
                continue
            epoch += 1
            movers = np.flatnonzero(eng.labels != prev)
            prev[movers] = eng.labels[movers]
            t0 = time.perf_counter()
            mon.push(eng.labels, eng.centers, len(movers), movers=movers)
            stop, regime, _ = mon.assess()
            mon_s += time.perf_counter() - t0
            if stop and auto is None:
                auto = epoch
            lab, fin = None, np.nan
            if mon.t >= 2:
                c = clone(eng)
                L = eng.labels.astype(np.int64)
                t0 = time.perf_counter()
                lab, _ = markov_finish(X, c, mon, L, k, regime if regime != "running" else "drift")
                fin = time.perf_counter() - t0
                lab = _small(lab, k)
            sample(lab, fin)
    ref = plain[-1]
    df = pd.DataFrame(rows)
    df["acc_plain"] = [float((p == ref).mean()) for p in plain]
    df["acc_markov"] = [float((m == ref).mean()) if m is not None else np.nan for m in mk]
    df["time_plain"] = df.engine_s + df.estep_s
    df["time_markov"] = df.engine_s + df.monitor_s + df.finish_s
    df["auto_stop"] = np.isclose(df.t, auto) if auto else False
    df.attrs["default"] = default
    return df


DIFFICULTY = ((4.0, "매우 쉬움"), (2.0, "쉬움"), (1.0, "어려움"), (0.5, "매우 어려움"))
DIFF_N, DIFF_K = 2_000_000, 50


def difficulty_key(sep):
    return f"gmm2e6-sep{sep:g}-k{DIFF_K}-s0"


def run_difficulty(seps=None):
    """Each accelerator in its own process (its label snapshots are freed with it); finished ones are kept."""
    import subprocess

    for sep, _ in DIFFICULTY:
        if seps and sep not in seps:
            continue
        for e in ENGINES:
            subprocess.run([sys.executable, "-u", "-m", "experiments.accelerators_anytime", "one", str(sep), e], check=True)
        print(figure(difficulty_key(sep), title=f"과제 난이도 {dict(DIFFICULTY)[sep]} (GMM 분리도 {sep:g}): "
                     f"n = 2,000,000, d = 16, k = {DIFF_K}"), flush=True)


def run_one(sep, engine):
    from kmeans_accel import datasets

    key = difficulty_key(sep)
    path = os.path.join(RESULTS, f"accelerators_anytime_{key}.csv")
    if os.path.exists(path) and engine in set(pd.read_csv(path).engine):
        return
    X, _, _ = datasets.gmm(DIFF_N, 16, DIFF_K, sep=sep, seed=0)
    run(None, DIFF_K, 0, engines=(engine,), X=np.ascontiguousarray(X), key=key)


def run(name, k, seed, engines=ENGINES, X=None, key=None, warm=True):
    from sklearn.cluster import kmeans_plusplus
    from threadpoolctl import threadpool_limits

    key = key or config_key(name, None, None, k, seed)
    X = dataset(name) if X is None else X
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    if warm:
        warmup()
    path = os.path.join(RESULTS, f"accelerators_anytime_{key}.csv")
    out = [pd.read_csv(path)] if os.path.exists(path) else []
    for e in engines:
        t0 = time.perf_counter()
        if e in EXACT:
            with threadpool_limits(limits=1, user_api="blas"):
                df = trace_exact(e, X, C0)
        else:
            df = trace_minibatch(X, C0, seed)
        df.insert(0, "engine", e)
        out.append(df)
        last = df.iloc[-1]
        print(e, f"{len(df) - 1} iterations, end {last.time_plain:.1f}s, wall {time.perf_counter() - t0:.0f}s", flush=True)
        pd.concat(out).to_csv(path, index=False)
    return pd.concat(out)


def per_second(g, which):
    """Accuracy of the most recent answer available at every whole second up to the end."""
    T = int(np.ceil(g.time_plain.max()))
    tm, acc = g[f"time_{which}"].to_numpy(), g[f"acc_{which}"].to_numpy()
    ok = ~np.isnan(tm) & ~np.isnan(acc)
    tm, acc = tm[ok], acc[ok]
    order = np.argsort(tm, kind="stable")
    tm, acc = tm[order], acc[order]
    s = np.arange(0, T + 1)
    idx = np.searchsorted(tm, s, side="right") - 1
    return s, np.where(idx >= 0, acc[np.maximum(idx, 0)], np.nan)


def figure(key="covtype-k64-s0", out=None, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from experiments.analyze_markov_tail import FIG, GRID as GRIDLINE, INK, INK2, SERIES, SURFACE

    plt.rcParams.update({"font.family": _korean_font(), "axes.unicode_minus": False, "figure.facecolor": SURFACE})
    df = pd.read_csv(os.path.join(RESULTS, f"accelerators_anytime_{key}.csv"))
    PLAIN, MK = SERIES[1], SERIES[0]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.8))
    for ax, eng in zip(axes.ravel(), ENGINES):
        g = df[df.engine == eng]
        ax.set_facecolor(SURFACE)
        lo = 1.0
        for which, col, mk in (("plain", PLAIN, "o"), ("markov", MK, "o")):
            s, a = per_second(g, which)
            ax.plot(s, 100 * a, color=col, lw=2, marker=mk, ms=4.5, mfc=col, mec=SURFACE, mew=0.8,
                    zorder=3 if which == "markov" else 2)
            lo = min(lo, np.nanmin(a))
        a_stop = g[g.auto_stop]
        if len(a_stop):
            r = a_stop.iloc[0]
            ax.scatter([r.time_markov], [100 * r.acc_markov], s=110, marker="D", facecolor=SURFACE, edgecolor=MK,
                       lw=1.8, zorder=4)
        T = int(np.ceil(g.time_plain.max()))
        ax.set_xlim(0, T + 0.5)
        ax.set_ylim(max(0, 100 * lo - 1.5), 100.4)
        ax.set_title(f"{ENGINE_KO[eng]}  (수렴 {g.time_plain.max():.1f}초)", loc="left", fontsize=11, color=INK,
                     fontweight="bold")
        ax.grid(True, color=GRIDLINE, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("걸린 시간 (초)", color=INK2, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("정확도 (수렴 결과와 같은 라벨 비율)", color=INK2, fontsize=10)
    handles = [Line2D([], [], color=PLAIN, lw=2, marker="o", ms=5, label="가속법 그대로: 그 시각에 멈췄을 때의 결과"),
               Line2D([], [], color=MK, lw=2, marker="o", ms=5,
                      label="Markov 적용: 그 시각에 멈추고 이력 정리·의사 대표점 적용 (감시·마무리 시간 포함)"),
               Line2D([], [], ls="", marker="D", ms=8, mfc=SURFACE, mec=MK, mew=1.8, label="Markov 자동 정지 지점")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2)
    ds, rest = key.rsplit("-k", 1)
    title = title or f"{ds}, k = {rest.split('-')[0]}"
    fig.suptitle(f"시간–정확도 교환 곡선: {title} (1초 간격, 각 초에 그때까지 얻을 수 있는 가장 최근 결과)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.935, "정확도 기준: 각 가속법 자신의 수렴 결과(MiniBatchKMeans는 100 epoch 뒤 결과). 0초에는 아직 결과가 없어 1초부터 찍힌다.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = out or os.path.join(FIG, f"fig15_anytime_{key}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "one":
        run_one(float(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "difficulty":
        run_difficulty([float(v) for v in sys.argv[2:]] or None)
    elif sys.argv[1] == "figure":
        print(figure(*sys.argv[2:3]))
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
