"""Mini-batch K-means with the Markov layer against the existing fast K-means methods.

Question: for the same time, does mini-batch K-means with the Markov layer give a better
clustering than the other ways people make K-means fast? Every method starts from the same
k-means++ centers of a configuration, and every answer is scored on one scale: the SSE of
its partition (each cluster at its centroid) over that of the exact K-means fixed point
reached from the same start (incremental Hamerly run to convergence), i.e. the SSE excess.

Methods (an "answer" is what the method returns if stopped at that moment; its time counts
everything the method itself does up to and including that answer, nothing else):

    A  markov       scikit-learn's MiniBatchKMeans loop (batch 1024, experiments/accelerators_markov.py)
                    with the Markov layer: the full monitor once per epoch, and at each epoch the
                    0.6 finish (settlement and pseudo representatives) as MarkovKMeans(t0=epoch)
                    returns it; the initial labelling pass the monitor needs is charged to it
       markov_auto  the same with the monitor's own stop (one answer)
    A' pkg_markov   markov-kmeans' epoch mini-batch engine (Sculley 1/count, batch 4096) with the
                    Markov layer; pkg_plain its centers assigned once (reported, not a competitor)
    B  mb_plain     MiniBatchKMeans stopped at that moment: its final assignment to the current
                    centers (= labels_), at every epoch and after 1, 2, 4, ... minibatches
       sk_default   MiniBatchKMeans with its early stopping (no EWA-inertia improvement in 10 batches)
    C  mb_avg10     the same run, the mean of the last 10 epoch-end centers assigned once
       mb_ema       the same run, an exponential average of the epoch-end centers (decay 0.9) assigned once
    D  mb_b4096, mb_b16384   MiniBatchKMeans with batch 4096 / 16384, stopped at that moment
    E  sub_<f>      exact K-means (incremental Hamerly) on a uniform random f = 1, 3, 10, 30 % of the
                    points (at least 20 k points) to convergence, then one assignment of all points
    F  hamerly      exact incremental Hamerly stopped after any iteration; hamerly_tol its stop at
                    scikit-learn's tol = 1e-4; lloyd, lloyd_tol the same for scikit-learn's Lloyd
                    kernel (run at most 1.5 x the 100-epoch time of B)

Pre-registered criterion (fixed before the runs; nothing in the Markov layer is tuned):
for every method and target eps in (3 %, 1 %, 0.3 %) SSE excess, the time to target is the first
time from which the method's answers stay at or below eps until its last answer (a one-answer
method: its time if it is at or below eps, else never). A wins a target in a configuration when
it reaches it and its time is at most 0.8 x the fastest of all other methods (A' and markov_auto
excluded); a target no method reaches is void. A configuration favours A when A wins at least
two of its non-void targets (all of them if fewer than two). Mini-batch with the Markov layer
holds an advantage over the existing methods when at least 2/3 of the configurations favour A.

Reported alongside: A against C alone (the Markov finish against plain center averaging on the
same trajectory), the automatic stops (markov_auto, sk_default, hamerly_tol, lloyd_tol), and
how often the monitor never stops within 100 epochs.

Configurations: covtype and MNIST with k = 16, 64, 256, USCensus1990 with k = 16, 64, k-means++
seeds 0, 1, 2; the four task difficulties of experiments/accelerators_anytime.py (n = 2 x 10^6,
d = 16, k = 50, separation 4, 2, 1, 0.5, seed 0). A pilot on a small mixture is not counted.

    python -m experiments.minibatch_markov pilot
    python -m experiments.minibatch_markov run          # -> results/minibatch_markov_runs.csv
    python -m experiments.minibatch_markov report
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

from experiments.accel_engines import Hamerly, SkLloyd, clone
from experiments.accelerators_markov import RESULTS, SkMiniBatch, dataset, markov_finish, warmup
from markov_kmeans import _rules
from markov_kmeans._engines import MiniBatchEngine
from markov_kmeans._kernels import compute_centers

OUT = os.path.join(RESULTS, "minibatch_markov_runs.csv")
EPOCHS = 100
TARGETS = (0.03, 0.01, 0.003)
SUB_FRACS = (0.01, 0.03, 0.1, 0.3)
SEEDS = (0, 1, 2)
GRID = ([("covtype", None, k, s) for k in (16, 64, 256) for s in SEEDS]
        + [("mnist", None, k, s) for k in (16, 64, 256) for s in SEEDS]
        + [("census", None, k, s) for k in (16, 64) for s in SEEDS]
        + [("gmm2e6", sep, 50, 0) for sep in (4.0, 2.0, 1.0, 0.5)])
SELF = ("markov", "markov_auto", "pkg_markov", "pkg_plain")


def key_of(name, sep, k, seed):
    return f"{name}" + (f"-sep{sep:g}" if sep is not None else "") + f"-k{k}-s{seed}"


def load(name, sep):
    if name == "gmm2e6":
        from kmeans_accel import datasets

        X, _, _ = datasets.gmm(2_000_000, 16, 50, sep=sep, seed=0)
        return np.ascontiguousarray(X)
    if name == "pilot":
        from kmeans_accel import datasets

        X, _, _ = datasets.gmm(150_000, 8, 20, sep=1.0, seed=11)
        return np.ascontiguousarray(X)
    return dataset(name)


class Scorer:
    """Partition SSE (clusters at their centroids) and agreement with the exact fixed point."""

    def __init__(self, X, k):
        self.X, self.k = X, k
        self.xsq = float(np.einsum("ij,ij->", X, X))
        self.ref = None

    def sse(self, labels):
        lab = np.asarray(labels, dtype=np.int64)
        C, cnt = compute_centers(self.X, lab, np.zeros((self.k, self.X.shape[1])))
        return float(self.xsq - (cnt[:, None] * C * C).sum())

    def __call__(self, labels):
        lab = np.asarray(labels)
        agree = float((lab == self.ref).mean()) if self.ref is not None else np.nan
        return self.sse(lab), agree


def _nearest_all(X, C, n_threads):
    from sklearn.cluster._kmeans import _labels_inertia

    return _labels_inertia(X, np.ones(X.shape[0]), np.ascontiguousarray(C), n_threads=n_threads, return_inertia=False)


def exact_hamerly(X, C0, score, tol_abs, max_iter=10_000):
    """Incremental Hamerly to convergence; partition SSE of every iterate from the running sums."""
    rows = []
    t0 = time.perf_counter()
    eng = Hamerly(X, C0)
    secs = time.perf_counter() - t0

    def sse_sums():
        m = eng.cnt > 0
        return float(score.xsq - ((eng.S[m] ** 2).sum(1) / eng.cnt[m]).sum())

    rows.append(dict(method="hamerly", tau=secs, sse=sse_sums()))
    prev, tol_done = eng.centers.copy(), False
    for _ in range(max_iter):
        t0 = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t0
        rows.append(dict(method="hamerly", tau=secs, sse=sse_sums()))
        if not tol_done and ch and float(((eng.centers - prev) ** 2).sum()) <= tol_abs:
            rows.append(dict(method="hamerly_tol", tau=secs, sse=rows[-1]["sse"]))
            tol_done = True
        prev = eng.centers.copy()
        if ch == 0:
            break
    if not tol_done:
        rows.append(dict(method="hamerly_tol", tau=secs, sse=rows[-1]["sse"]))
    score.ref = eng.labels.copy()
    exact = score.sse(eng.labels)
    for r in rows:
        r["agree"] = np.nan
    return rows, exact, secs


def exact_lloyd(X, C0, score, tol_abs, budget):
    rows = []
    t0 = time.perf_counter()
    eng = SkLloyd(X, C0)
    secs = time.perf_counter() - t0
    rows.append(dict(method="lloyd", tau=secs, **dict(zip(("sse", "agree"), score(eng.labels)))))
    prev, tol_done = eng.centers.copy(), False
    while secs < budget:
        t0 = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t0
        s, a = score(eng.labels)
        rows.append(dict(method="lloyd", tau=secs, sse=s, agree=a))
        if not tol_done and ch and float(((eng.centers - prev) ** 2).sum()) <= tol_abs:
            rows.append(dict(method="lloyd_tol", tau=secs, sse=s, agree=a))
            tol_done = True
        prev = eng.centers.copy()
        if ch == 0:
            if not tol_done:
                rows.append(dict(method="lloyd_tol", tau=secs, sse=s, agree=a))
            break
    return rows, secs


def sk_minibatch(X, C0, seed, score, batch=1024, post=True):
    """One MiniBatchKMeans run of 100 epochs; answers of B (and C and A when ``post``)."""
    n, k = X.shape[0], C0.shape[0]
    rows = []
    name = "mb_plain" if batch == 1024 else f"mb_b{batch}"
    t0 = time.perf_counter()
    eng = SkMiniBatch(X, C0, seed, batch_size=batch)
    init = time.perf_counter() - t0
    secs = init - eng.init_label_secs                     # MiniBatchKMeans itself does no initial labelling
    label0 = eng.init_label_secs
    spe = n // eng.batch
    early = {2 ** j for j in range(30) if 2 ** j < spe}
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True) if post else None
    mon_s = 0.0
    if post:
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers)
        mon_s += time.perf_counter() - t0
    hist, ema = deque(maxlen=10), None
    prev = eng.labels.copy()
    auto = default = False

    def answer(method, C, extra, base):
        t0 = time.perf_counter()
        lab = _nearest_all(X, C, eng.n_threads)
        s, a = score(lab)
        rows.append(dict(method=method, tau=base + extra + time.perf_counter() - t0, sse=s, agree=a,
                         epoch=eng.step_i / spe))

    for _ in range((EPOCHS * n) // eng.batch):
        t0 = time.perf_counter()
        fired = eng.minibatch()
        secs += time.perf_counter() - t0
        if fired and not default:
            default = True
            if batch == 1024:
                answer("sk_default", eng.centers, 0.0, secs)
        if eng.step_i % spe:
            if eng.step_i in early:
                answer(name, eng.centers, 0.0, secs)
            continue
        answer(name, eng.centers, 0.0, secs)
        if not post:
            continue
        t0 = time.perf_counter()
        hist.append(eng.centers.copy())
        avg = np.mean(hist, axis=0)
        ema = eng.centers.copy() if ema is None else 0.9 * ema + 0.1 * eng.centers
        avg_s = time.perf_counter() - t0
        answer("mb_avg10", avg, avg_s, secs)
        answer("mb_ema", ema, avg_s, secs)
        movers = np.flatnonzero(eng.labels != prev)          # an epoch engine reports these (not timed)
        prev[movers] = eng.labels[movers]
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers, len(movers), movers=movers)
        stop, regime, _ = mon.assess()
        mon_s += time.perf_counter() - t0
        if mon.t >= 2:
            c = clone(eng)
            L = eng.labels.astype(np.int64)
            t0 = time.perf_counter()
            lab, _ = markov_finish(X, c, mon, L, k, regime if regime != "running" else "drift")
            fin = time.perf_counter() - t0
            del c
            s, a = score(lab)
            r = dict(method="markov", tau=secs + label0 + mon_s + fin, sse=s, agree=a, epoch=eng.step_i / spe)
            rows.append(r)
            if stop and not auto:
                auto = True
                rows.append(dict(r, method="markov_auto"))
    if post and not auto:
        rows.append(dict(rows[-1], method="markov_auto", never_stopped=True))
    return rows, secs


def pkg_minibatch(X, C0, seed, score, batch=4096):
    n, k = X.shape[0], C0.shape[0]
    rows = []
    t0 = time.perf_counter()
    eng = MiniBatchEngine(X, C0, batch_size=batch, learning_rate="count", seed=seed)
    secs = time.perf_counter() - t0
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True)
    t0 = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    mon_s = time.perf_counter() - t0
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    nt = _openmp_effective_n_threads()
    for e in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t0
        t0 = time.perf_counter()
        lab = _nearest_all(X, eng.centers, nt)
        es = time.perf_counter() - t0
        s, a = score(lab)
        rows.append(dict(method="pkg_plain", tau=secs + es, sse=s, agree=a, epoch=e))
        t0 = time.perf_counter()
        mon.push(eng.labels, eng.centers, ch, movers=eng.movers)
        stop, regime, _ = mon.assess()
        mon_s += time.perf_counter() - t0
        if mon.t >= 2:
            c = clone(eng)
            t0 = time.perf_counter()
            lab, _ = markov_finish(X, c, mon, eng.labels.copy(), k, regime if regime != "running" else "drift")
            fin = time.perf_counter() - t0
            del c
            s, a = score(lab)
            rows.append(dict(method="pkg_markov", tau=secs + mon_s + fin, sse=s, agree=a, epoch=e))
    return rows


def subsample(X, C0, seed, score, frac):
    n, k = X.shape[0], C0.shape[0]
    m = max(int(frac * n), 20_000, 20 * k)
    if m >= n:
        return []
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    t0 = time.perf_counter()
    idx = np.sort(np.random.default_rng(seed).choice(n, m, replace=False))
    Xs = np.ascontiguousarray(X[idx])
    eng = Hamerly(Xs, C0)
    for _ in range(10_000):
        if eng.step() == 0:
            break
    m_ = eng.cnt > 0
    C = eng.centers.copy()
    C[m_] = eng.S[m_] / eng.cnt[m_, None]
    lab = _nearest_all(X, C, _openmp_effective_n_threads())
    secs = time.perf_counter() - t0
    s, a = score(lab)
    return [dict(method=f"sub_{frac:g}", tau=secs, sse=s, agree=a)]


def run_config(name, sep, k, seed):
    from sklearn.cluster import kmeans_plusplus
    from threadpoolctl import threadpool_limits

    key = key_of(name, sep, k, seed)
    X = load(name, sep)
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    score = Scorer(X, k)
    tol_abs = 1e-4 * float(np.var(X, axis=0).mean())
    t_wall = time.perf_counter()
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        r, exact_sse, ham_s = exact_hamerly(X, C0, score, tol_abs)      # also sets the reference partition
        for x in r:
            x["agree"] = np.nan
        rows += r
        r, mb_s = sk_minibatch(X, C0, seed, score)
        rows += r
        for b in (4096, 16384):
            rows += sk_minibatch(X, C0, seed, score, batch=b, post=False)[0]
        rows += pkg_minibatch(X, C0, seed, score)
        for f in SUB_FRACS:
            rows += subsample(X, C0, seed, score, f)
        rows += exact_lloyd(X, C0, score, tol_abs, budget=1.5 * mb_s)[0]
    df = pd.DataFrame(rows)
    df.insert(0, "config", key)
    df["dataset"], df["k"], df["seed"], df["n"] = name, k, seed, X.shape[0]
    df["sse_excess"] = df.sse / exact_sse - 1
    df["exact_sse"], df["hamerly_converged_s"], df["mb_100_epochs_s"] = exact_sse, ham_s, mb_s
    print(key, f"{time.perf_counter() - t_wall:.0f}s", flush=True)
    return df


def time_to_target(g, eps):
    """First time from which the answers stay <= eps (one-answer methods: their time if <= eps)."""
    g = g.sort_values("tau")
    e = g.sse_excess.to_numpy()
    ok = e <= eps
    if not ok[-1]:
        return np.inf
    bad = np.flatnonzero(~ok)
    i = bad[-1] + 1 if len(bad) else 0
    return float(g.tau.to_numpy()[i])


def judge(df):
    out = []
    for key, g in df.groupby("config", sort=False):
        t = {(m, eps): time_to_target(h, eps) for m, h in g.groupby("method") for eps in TARGETS}
        methods = g.method.unique()
        rivals = [m for m in methods if m not in SELF]
        wins, void = 0, 0
        row = dict(config=key)
        for eps in TARGETS:
            best = min(t[(m, eps)] for m in rivals)
            best_m = min(rivals, key=lambda m: t[(m, eps)])
            a = t[("markov", eps)]
            reached = np.isfinite(best) or np.isfinite(a)
            win = bool(np.isfinite(a) and a <= 0.8 * best)
            void += not reached
            wins += win
            row.update({f"markov_{eps:g}": a, f"best_{eps:g}": best, f"best_method_{eps:g}": best_m, f"win_{eps:g}": win,
                        f"avg10_{eps:g}": t[("mb_avg10", eps)], f"ema_{eps:g}": t[("mb_ema", eps)]})
        live = len(TARGETS) - void
        row["favours_markov"] = wins >= min(2, live) and live > 0
        auto = g[g.method == "markov_auto"].iloc[0]
        row.update(markov_auto_tau=auto.tau, markov_auto_excess=auto.sse_excess,
                   markov_never_stopped=bool(auto.get("never_stopped") == True))  # noqa: E712
        for m in ("sk_default", "hamerly_tol", "lloyd_tol"):
            h = g[g.method == m]
            if len(h):
                row[f"{m}_tau"], row[f"{m}_excess"] = h.tau.iloc[0], h.sse_excess.iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def report(path=OUT):
    df = pd.read_csv(path)
    df = df[~df.config.str.startswith("pilot")]
    j = judge(df)
    pd.set_option("display.width", 250)
    print(j.round(4).to_string(index=False))
    share = j.favours_markov.mean()
    print(f"configurations favouring A: {j.favours_markov.sum()}/{len(j)} ({share:.0%}); "
          f"advantage (>= 2/3): {share >= 2 / 3}")
    j.to_csv(os.path.join(RESULTS, "minibatch_markov_judgement.csv"), index=False)
    return j


METHOD_KO = (("markov", "Markov 적용 미니배치 (A)"), ("mb_plain", "미니배치 그대로 (배치 1024)"),
             ("mb_avg10", "미니배치 + 최근 10 epoch 중심 평균"), ("mb_ema", "미니배치 + 중심 지수 평균"),
             ("mb_b4096", "미니배치, 배치 4096"), ("mb_b16384", "미니배치, 배치 16384"),
             ("sub_0.01", "표본 1% → 정확 K-means → 전체 할당"), ("sub_0.03", "표본 3% → 정확 K-means"),
             ("sub_0.1", "표본 10% → 정확 K-means"), ("sub_0.3", "표본 30% → 정확 K-means"),
             ("hamerly", "정확 Hamerly (증분), 아무 반복에서 멈춤"), ("lloyd", "정확 scikit-learn Lloyd, 아무 반복에서 멈춤"))


def figure(path=OUT, out=None):
    """Per target: each method's time to the target over the fastest method's, one dot per configuration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.accelerators_markov import _korean_font
    from experiments.analyze_markov_tail import FIG, GRID as GRIDLINE, INK, INK2, SERIES, SURFACE

    plt.rcParams.update({"font.family": _korean_font(), "axes.unicode_minus": False, "figure.facecolor": SURFACE})
    df = pd.read_csv(path)
    df = df[~df.config.str.startswith("pilot")]
    names = [m for m, _ in METHOD_KO]
    T = {(c, m, e): time_to_target(h, e) for (c, m), h in df.groupby(["config", "method"]) for e in TARGETS}
    configs = df.config.unique()
    colour = {m: SERIES[0] if m == "markov" else SERIES[1] if m.startswith("mb_") else SERIES[2] if m.startswith("sub_")
              else INK2 for m in names}
    cap = 300.0
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, e in zip(axes, TARGETS):
        ax.set_facecolor(SURFACE)
        for i, m in enumerate(names):
            r = []
            for c in configs:
                best = min(T.get((c, x, e), np.inf) for x in names)
                t = T.get((c, m, e), np.inf)
                if np.isfinite(best):
                    r.append(min(t / best, cap))
            r = np.array(r)
            y = len(names) - 1 - i + rng.uniform(-0.18, 0.18, len(r))
            ax.scatter(r, y, s=16, color=colour[m], alpha=0.75, edgecolor="none", zorder=3)
            ok = r < cap
            if ok.any():
                med = np.median(r)
                ax.plot([med, med], [len(names) - 1 - i - 0.32, len(names) - 1 - i + 0.32], color=INK, lw=2, zorder=4)
            ax.text(cap * 1.12, len(names) - 1 - i, f"{ok.sum()}/{len(r)}", va="center", fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.8, cap * 1.6)
        ax.axvline(cap, color=GRIDLINE, lw=6, zorder=1)
        ax.set_xticks([1, 3, 10, 30, 100, cap])
        ax.set_xticklabels(["1배\n(가장 빠름)", "3", "10", "30", "100", "도달\n못 함"])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([k for _, k in METHOD_KO][::-1], fontsize=9.5)
        ax.grid(True, axis="x", color=GRIDLINE, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"목표: SSE 초과 ≤ {e:.1%}".replace(".0%", "%"), loc="left", fontsize=11.5, color=INK, fontweight="bold")
        ax.set_xlabel("목표에 도달한 시간 ÷ 그 구성에서 가장 빠른 방법의 시간 (로그)", color=INK2, fontsize=9.5)
    fig.suptitle("미니배치 + Markov 층과 기존의 빠른 K-means: 목표 품질에 도달하는 시간 (점 하나 = 구성 하나, 세로 막대 = 중앙값, "
                 "오른쪽 숫자 = 도달한 구성 수)", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out or os.path.join(FIG, "fig16_minibatch_markov.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


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
        p = os.path.join(RESULTS, "minibatch_markov_pilot.csv")
        run(grid=[("pilot", None, 20, 0)], out=p)
        print(judge(pd.read_csv(p)).round(4).T.to_string())
    elif cmd == "run":
        run()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else OUT)
    elif cmd == "figure":
        print(figure())
