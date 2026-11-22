"""Markov-K-means on top of existing K-means accelerators: does the Markov layer still pay?

The Markov layer of markov-kmeans watches the label and center sequence of any iterative
K-means; it does not care how an iteration is computed. This study puts it on top of the
accelerators people use (experiments/accel_engines.py):

    exact, cheaper iterations   sk_lloyd (scikit-learn's GEMM Lloyd), sk_elkan (scikit-learn's Elkan),
                                hamerly, yinyang (incremental center sums), hamerly_pkg (the
                                markov-kmeans 0.5 Hamerly engine, base of the next two)
    exact, fewer iterations     squarem, anderson (experiments/accelerate.py: SQUAREM / Anderson
                                extrapolation of the center chain, kept only if the SSE falls)
    approximate                 sk_minibatch (scikit-learn's MiniBatchKMeans loop and kernels,
                                batch 1024, its learning rates and reassignment; uniform draws)

and compares, on each accelerator's own run, the stopping rules

    full          run to its end: no label changes (exact; cap 5000 iterations), or scikit-learn's
                  max_iter = 100 epochs (mini-batch), then a final assignment (mini-batch)
    default       the accelerator's own default stop: scikit-learn's tol = 1e-4 on the squared center
                  shift (relative to the mean feature variance; exact engines, judged one iteration
                  later so that the labels are those of scikit-learn's final E-step), or
                  MiniBatchKMeans' early stopping (no EWA-inertia improvement in 10 minibatches),
                  then a final assignment
    threshold     changed share of one iteration < 1e-4 (Perez-Ortega's rule, no window)
    markov_stop   the 0.6 default stop alone: exact engines stop_rule="unstable" with a history-free
                  monitor (points that changed in the last W = 10 iterations <= 0.1 %); mini-batch the
                  full monitor, fed once per epoch with the labels last given to each point; then the
                  labels as they are (mini-batch: a final assignment)
    markov        the 0.6 default path: the same stop, settlement (assign_rule="oscillation") and the
                  pseudo representative step (AR(1) long-run centers, one pruned assignment pass,
                  kept only if the SSE does not rise), as MarkovKMeans.fit does it

Times count the engine (its initial assignment and every iteration), the monitor (push and
assess) for the Markov rules, and the finishing pass. A finish runs on a copy of the engine
made at the stop (the copy is not timed); seeding (k-means++, the same centers for every
accelerator of a configuration) is not timed. Quality is the SSE of the returned partition
(each cluster at its centroid) and the share of labels that differ, both against the
accelerator's own end and against the exact converged partition (hamerly_pkg, bit-identical
to Lloyd).

Pre-registered criterion (fixed before the evaluation runs; the markov-kmeans parameters are
the package defaults and nothing is tuned). The Markov layer is a meaningful gain on top of
accelerator A when, over the evaluation configurations,

    (a) the median time saving of A + markov against A's full run is at least 20 %,
    (b) its SSE excess over A's full run is at most 1e-3 in at least 80 % of the configurations, and
    (c) A + markov is dominated by A + default in at most half of the configurations
        (dominated: the other rule is no slower, within 1 % of A's full time, and has no larger
        SSE excess, within 1e-6, and is better beyond one of those margins).

Reported alongside, without pass/fail: the saving against A's default stop, the pseudo step's
net effect (markov against markov_stop), the windowed stop against the plain threshold, the
fastest accelerator-and-rule route to an SSE excess <= 1e-3 over the exact optimum, and whether
the exact engines followed the same trajectory.

Evaluation configurations: the four exact synthetic instances of experiments/pareto.py
(n = 10^6, d = 16, k = 50), covtype (581012 x 54) and MNIST (70000 x 784) with k = 16, 64, 256,
USCensus1990 (2458285 x 68) with k = 16, 64, k-means++ seeds 0, 1, 2 (the pareto instances use
their own seed). A pilot (gmm 200000 x 16, k = 50) only checks the harness and is not counted.

    python -m experiments.accelerators_markov pilot
    python -m experiments.accelerators_markov run [dataset ...]    # -> results/accelerators_markov.csv
    python -m experiments.accelerators_markov report
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.accel_engines import Hamerly, SkElkan, SkLloyd, Yinyang, clone
from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import compute_centers, inertia
from markov_kmeans.estimator import pseudo_finish

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT = os.path.join(RESULTS, "accelerators_markov.csv")
CURVES = os.path.join(RESULTS, "accelerators_markov")         # per-iteration times (not committed)
EXACT = ("sk_lloyd", "sk_elkan", "hamerly", "yinyang", "hamerly_pkg", "squarem", "anderson")
ENGINES = EXACT + ("sk_minibatch",)
RULES = ("full", "default", "threshold", "markov_stop", "markov")
REF = "hamerly_pkg"
MAX_ITER = 5000
MB_EPOCHS, MB_BATCH = 100, 1024
TOL, THRESHOLD = 1e-4, 1e-4
SEEDS = (0, 1, 2)
GRID = ([("gmm", sep, s, 50, s) for sep in (1.5, 2.0) for s in (0, 1)]
        + [("covtype", None, None, k, s) for k in (16, 64, 256) for s in SEEDS]
        + [("mnist", None, None, k, s) for k in (16, 64, 256) for s in SEEDS]
        + [("census", None, None, k, s) for k in (16, 64) for s in SEEDS])


def dataset(name, sep=None, data_seed=None):
    if name == "gmm":
        from kmeans_accel import datasets

        X, _, _ = datasets.gmm(1_000_000, 16, 50, sep=sep, seed=data_seed)
    elif name == "pilot":
        from kmeans_accel import datasets

        X, _, _ = datasets.gmm(200_000, 16, 50, sep=1.5, seed=7)
    else:
        X = np.load(os.path.join(DATA, f"{name}.npy"))
    return np.ascontiguousarray(X, dtype=np.float64)


def config_key(name, sep, data_seed, k, seed):
    return f"{name}" + (f"-sep{sep}-d{data_seed}" if sep is not None else "") + f"-k{k}-s{seed}"


def make(name, X, C0):
    from experiments.accelerate import AcceleratedHamerly

    return {"sk_lloyd": lambda: SkLloyd(X, C0), "sk_elkan": lambda: SkElkan(X, C0),
            "hamerly": lambda: Hamerly(X, C0), "yinyang": lambda: Yinyang(X, C0),
            "hamerly_pkg": lambda: HamerlyEngine(X, C0),
            "squarem": lambda: AcceleratedHamerly(X, C0, method="squarem"),
            "anderson": lambda: AcceleratedHamerly(X, C0, method="anderson")}[name]()


def small(labels):
    return np.asarray(labels).astype(np.uint16)


def markov_finish(X, eng, mon, L, k, regime="drift"):
    """Settlement and pseudo step as MarkovKMeans.fit (0.6 path); returns (labels, accepted)."""
    W = mon.params.window
    L = np.asarray(L, dtype=np.int64)
    U = mon.unstable().astype(np.int64)
    if mon.history:
        Hw = mon.window(U)
        n_sw = _rules.switches(Hw)
    else:
        Hw, n_sw = None, mon.switch_counts(U)
    rule = "oscillation" if (Hw is not None or regime != "oscillation") else "keep"
    labels_U, S, P, osc = _rules.settle(Hw, L[U], rule=rule, n_clusters=k, regime=regime, n_switches=n_sw,
                                        width=W + 1)
    if mon.t < 2:
        out = L.copy()
        out[U] = labels_U
        return out, False
    res = pseudo_finish(X, eng, mon, L, None, U, S, P, osc, inplace=True, labels_U=labels_U)
    return res[0], bool(res[8])


class Track:
    """One accelerator's run with every stopping rule judged on it."""

    def __init__(self, name, X, C0, tol_abs):
        self.name = name
        self.n, self.k = X.shape[0], C0.shape[0]
        t0 = time.perf_counter()
        self.eng = make(name, X, C0)
        self.secs = time.perf_counter() - t0
        self.init_secs = self.secs
        self.external = name in ("squarem", "anderson")      # these engines do not report movers
        self.prev = self.eng.labels.copy() if self.external else None
        self.mon = _rules.StateMonitor(self.n, self.k, _rules.StopParams(), exact=True, rule="unstable", history=False)
        t0 = time.perf_counter()
        self.mon.push(self.eng.labels, self.eng.centers)
        self.mon_secs = time.perf_counter() - t0
        self.prevC = self.eng.centers.copy()
        self.tol_abs = tol_abs
        self.t = 0
        self.stops = {}
        self.changes = []
        self.cum = []                                         # engine seconds after each iteration
        self.done = False

    def record(self, rule, labels, seconds, **extra):
        self.stops[rule] = dict(t=self.t, seconds=seconds, labels=small(labels), **extra)

    def step(self, X):
        eng = self.eng
        t0 = time.perf_counter()
        ch = eng.step()
        self.secs += time.perf_counter() - t0
        self.t += 1
        if self.external:                                    # net movers of the step (not timed)
            movers = np.flatnonzero(eng.labels != self.prev)
            self.prev[movers] = eng.labels[movers]
            ch = len(movers)
        else:
            movers = eng.movers
        self.changes.append(ch)
        self.cum.append(self.secs)
        C = eng.centers
        if ch == 0 or self.t >= MAX_ITER:
            conv = ch == 0
            self.record("full", eng.labels, self.secs, converged=conv)
            for r in ("default", "threshold"):
                self.stops.setdefault(r, dict(self.stops["full"]))
            if "markov" not in self.stops:
                s = self.secs + self.mon_secs
                self.stops["markov_stop"] = dict(self.stops["full"], seconds=s, monitor_secs=self.mon_secs)
                self.stops["markov"] = dict(self.stops["full"], seconds=s, monitor_secs=self.mon_secs, finish_secs=0.0,
                                            accepted=False)
            self.done = True
            return
        if "default" not in self.stops and float(((C - self.prevC) ** 2).sum()) <= self.tol_abs:
            self.record("default", eng.labels, self.secs)
        if "threshold" not in self.stops and ch < THRESHOLD * self.n:
            self.record("threshold", eng.labels, self.secs)
        if "markov" not in self.stops:
            t0 = time.perf_counter()
            self.mon.push(eng.labels, C, ch, movers=movers)
            stop, regime, _ = self.mon.assess()
            self.mon_secs += time.perf_counter() - t0
            if stop:
                base = self.secs + self.mon_secs
                self.record("markov_stop", eng.labels, base, monitor_secs=self.mon_secs)
                copy_ = clone(eng)                            # not timed: the package finishes in place
                L = eng.labels.copy()
                t0 = time.perf_counter()
                lab, acc = markov_finish(X, copy_, self.mon, L, self.k, regime)
                fin = time.perf_counter() - t0
                del copy_
                self.record("markov", lab, base + fin, monitor_secs=self.mon_secs, finish_secs=fin, accepted=acc)
        self.prevC = C.copy()


def run_exact(X, C0, engines=EXACT):
    tol_abs = TOL * float(np.var(X, axis=0).mean())
    tracks = []
    for name in engines:
        tracks.append(Track(name, X, C0, tol_abs))
    active = list(tracks)
    while active:
        for tr in active:
            tr.step(X)
            if tr.done:
                tr.eng = None                                  # free bounds as soon as a run ends
        active = [tr for tr in active if not tr.done]
        gc.collect() if any(tr.eng is None for tr in tracks) and len(active) < len(tracks) else None
    return tracks


# ---------------------------------------------------------------------------------------------
# scikit-learn MiniBatchKMeans, step by step


class SkMiniBatch:
    """The loop of MiniBatchKMeans.fit with its kernels; ``labels`` holds the label each point last got."""

    exact = False

    def __init__(self, X, C0, seed, batch_size=MB_BATCH, max_no_improvement=10, reassignment_ratio=0.01):
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        self.X = X
        self.n, self.k = X.shape[0], C0.shape[0]
        self.batch = min(batch_size, self.n)
        self.n_threads = _openmp_effective_n_threads()
        self.rng = np.random.RandomState(seed)
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        self.new = np.empty_like(self.centers)
        self.counts = np.zeros(self.k)
        self.unit = np.ones(self.batch)
        self.w_all = np.ones(self.n)
        self.ratio = reassignment_ratio
        self.max_no_improvement = max_no_improvement
        self.ewa = self.ewa_min = None
        self.no_improvement = 0
        self.since_reassign = 0
        self.step_i = 0
        self.labels = np.full(self.n, -1, dtype=np.int32)
        lab, _ = self.nearest(self.centers)                  # every point starts with a label
        self.labels[:] = lab

    def nearest(self, C, inplace=False):
        from sklearn.cluster._kmeans import _labels_inertia

        lab = _labels_inertia(self.X, self.w_all, np.ascontiguousarray(C), n_threads=self.n_threads,
                              return_inertia=False)
        if inplace:
            self.labels[:] = lab
        return np.asarray(lab, dtype=np.int64), self.n * C.shape[0]

    def minibatch(self):
        """One scikit-learn minibatch step; returns True when its early stopping fires."""
        from sklearn.cluster._k_means_minibatch import _minibatch_update_dense
        from sklearn.cluster._kmeans import _labels_inertia

        idx = self.rng.randint(0, self.n, self.batch)          # = choice(n, batch, p=uniform), O(batch)
        Xb = self.X[idx]
        lab, inertia_b = _labels_inertia(Xb, self.unit, self.centers, n_threads=self.n_threads)
        self.labels[idx] = lab
        _minibatch_update_dense(Xb, self.unit, self.centers, self.new, self.counts, lab, self.n_threads)
        self.since_reassign += self.batch
        if (self.counts == 0).any() or self.since_reassign >= 10 * self.k:
            self.since_reassign = 0
            to_re = self.counts < self.ratio * self.counts.max()
            if to_re.sum() > 0.5 * self.batch:
                to_re[np.argsort(self.counts)[int(0.5 * self.batch):]] = False
            m = int(to_re.sum())
            if m:
                self.new[to_re] = Xb[self.rng.choice(self.batch, replace=False, size=m)]
                self.counts[to_re] = np.min(self.counts[~to_re])
        self.centers, self.new = self.new, self.centers
        # MiniBatchKMeans._mini_batch_convergence with tol = 0
        self.step_i += 1
        b = inertia_b / self.batch
        if self.step_i == 1:
            return False
        if self.ewa is None:
            self.ewa = b
        else:
            a = min(self.batch * 2.0 / (self.n + 1), 1.0)
            self.ewa = self.ewa * (1 - a) + b * a
        if self.ewa_min is None or self.ewa < self.ewa_min:
            self.no_improvement = 0
            self.ewa_min = self.ewa
        else:
            self.no_improvement += 1
        return self.no_improvement >= self.max_no_improvement


def run_minibatch(X, C0, seed):
    """One MiniBatchKMeans run of 100 epochs with every rule judged on it; returns a Track-like object."""
    from threadpoolctl import threadpool_limits

    n, k = X.shape[0], C0.shape[0]
    tr = Track.__new__(Track)
    tr.name, tr.n, tr.k, tr.stops, tr.changes, tr.cum = "sk_minibatch", n, k, {}, [], []
    t0 = time.perf_counter()
    eng = SkMiniBatch(X, C0, seed)
    tr.secs = tr.init_secs = time.perf_counter() - t0
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=False, rule="monitor", history=True)
    t0 = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    tr.mon_secs = time.perf_counter() - t0
    steps_per_epoch = n // eng.batch
    n_steps = (MB_EPOCHS * n) // eng.batch
    prev = eng.labels.copy()
    tr.t = 0
    with threadpool_limits(limits=1, user_api="blas"):
        def final(rule, extra_secs=0.0, **extra):
            t0 = time.perf_counter()
            lab, _ = eng.nearest(eng.centers)                   # MiniBatchKMeans' final labels_
            tr.stops[rule] = dict(t=tr.t, seconds=tr.secs + extra_secs + time.perf_counter() - t0,
                                  labels=small(lab), steps=eng.step_i, **extra)
        for i in range(n_steps):
            t0 = time.perf_counter()
            early = eng.minibatch()
            tr.secs += time.perf_counter() - t0
            if early and "default" not in tr.stops:
                final("default")
            if eng.step_i % steps_per_epoch == 0:
                tr.t += 1
                movers = np.flatnonzero(eng.labels != prev)        # not timed: an epoch engine reports them
                prev[movers] = eng.labels[movers]
                tr.changes.append(len(movers))
                tr.cum.append(tr.secs)
                if "markov" not in tr.stops:
                    t0 = time.perf_counter()
                    mon.push(eng.labels, eng.centers, len(movers), movers=movers)
                    stop, regime, _ = mon.assess()
                    tr.mon_secs += time.perf_counter() - t0
                    if stop:
                        final("markov_stop", tr.mon_secs, monitor_secs=tr.mon_secs)
                        e2 = clone(eng)
                        L = eng.labels.astype(np.int64)
                        t0 = time.perf_counter()
                        lab, acc = markov_finish(X, e2, mon, L, k, regime)
                        fin = time.perf_counter() - t0
                        del e2
                        tr.stops["markov"] = dict(t=tr.t, seconds=tr.secs + tr.mon_secs + fin, labels=small(lab),
                                                  steps=eng.step_i, monitor_secs=tr.mon_secs, finish_secs=fin,
                                                  accepted=acc, regime=regime)
        final("full", converged=False)
    for r in ("default", "threshold", "markov_stop", "markov"):
        if r not in tr.stops:
            tr.stops[r] = dict(tr.stops["full"], missing=True) if r != "threshold" else dict(tr.stops["full"])
    tr.stops["threshold"] = dict(tr.stops["full"], not_applicable=True)
    return tr


# ---------------------------------------------------------------------------------------------


def sse(X, labels, k):
    lab = labels.astype(np.int64)
    C = compute_centers(X, lab, np.zeros((k, X.shape[1])))[0]
    return float(inertia(X, C, lab))


def rows_for(key, meta, X, tracks, ref_labels, sse_ref):
    rows = []
    k = meta["k"]
    for tr in tracks:
        own = tr.stops["full"]["labels"]
        sse_own = sse(X, own, k)
        for r in RULES:
            s = tr.stops[r]
            lab = s["labels"]
            v = sse(X, lab, k) if r != "full" else sse_own
            rows.append(dict(config=key, **meta, engine=tr.name, rule=r, iterations=s["t"], seconds=s["seconds"],
                             init_secs=tr.init_secs, monitor_secs=s.get("monitor_secs", 0.0),
                             finish_secs=s.get("finish_secs", 0.0), accepted=s.get("accepted", np.nan),
                             converged=s.get("converged", np.nan), steps=s.get("steps", np.nan),
                             missing=bool(s.get("missing", False)), not_applicable=bool(s.get("not_applicable", False)),
                             sse=v, sse_excess=v / sse_own - 1, sse_excess_exact=v / sse_ref - 1,
                             label_err=float((lab != own).mean()), label_err_exact=float((lab != ref_labels).mean()),
                             same_as_exact=bool(np.array_equal(own, ref_labels)) if r == "full" else np.nan,
                             full_seconds=tr.stops["full"]["seconds"]))
    return rows


def run_config(name, sep, data_seed, k, seed, engines=ENGINES):
    from sklearn.cluster import kmeans_plusplus
    from threadpoolctl import threadpool_limits

    key = config_key(name, sep, data_seed, k, seed)
    X = dataset(name, sep, data_seed)
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    meta = dict(dataset=name, n=X.shape[0], d=X.shape[1], k=k, seed=seed)
    t0 = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):         # as scikit-learn runs its own kernels
        tracks = run_exact(X, C0, [e for e in engines if e in EXACT])
    ref_tr = next(tr for tr in tracks if tr.name == REF)
    ref_labels = ref_tr.stops["full"]["labels"]
    sse_ref = sse(X, ref_labels, k)
    if "sk_minibatch" in engines:
        tracks.append(run_minibatch(X, C0, seed))
    rows = rows_for(key, meta, X, tracks, ref_labels, sse_ref)
    os.makedirs(CURVES, exist_ok=True)
    np.savez_compressed(os.path.join(CURVES, f"{key}.npz"), init=np.array([tr.init_secs for tr in tracks]),
                        names=np.array([tr.name for tr in tracks]),
                        **{f"cum_{tr.name}": np.array(tr.cum) for tr in tracks},
                        **{f"changes_{tr.name}": np.array(tr.changes) for tr in tracks})
    print(key, f"{time.perf_counter() - t0:.0f}s", flush=True)
    for r in rows:
        if r["rule"] in ("full", "default", "markov"):
            print(f"  {r['engine']:12s} {r['rule']:11s} it {r['iterations']:5d} {r['seconds']:8.2f}s "
                  f"({r['seconds'] / r['full_seconds']:6.1%})  sse+ {r['sse_excess']:+.2e}  err {r['label_err']:.4%}",
                  flush=True)
    return rows


def warmup():
    """Compile every kernel (and the dtype variants) before anything is timed."""
    from kmeans_accel import datasets
    from sklearn.cluster import kmeans_plusplus

    X, _, _ = datasets.gmm(20_000, 4, 8, sep=1.0, seed=3)
    X = np.ascontiguousarray(X, dtype=np.float64)
    for k in (8, 24):
        C0, _ = kmeans_plusplus(X, k, random_state=0)
        tracks = run_exact(X, C0)
        tracks.append(run_minibatch(X, C0, 0))
        rows_for("warmup", dict(k=k), X, tracks, tracks[0].stops["full"]["labels"], 1.0)


def run(datasets_=None, grid=GRID, out=OUT):
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["config"] for r in rows}
    warmup()
    for cfg in grid:
        if datasets_ and cfg[0] not in datasets_:
            continue
        key = config_key(*cfg)
        if key in done:
            continue
        rows += run_config(*cfg)
        pd.DataFrame(rows).to_csv(out, index=False)
        gc.collect()
    return pd.DataFrame(rows)


ENGINE_KO = {"sk_lloyd": "scikit-learn Lloyd", "sk_elkan": "scikit-learn Elkan", "hamerly": "Hamerly (증분 중심)",
             "yinyang": "Yinyang (증분 중심)", "hamerly_pkg": "Hamerly (markov-kmeans 엔진)",
             "squarem": "SQUAREM + Hamerly", "anderson": "Anderson + Hamerly",
             "sk_minibatch": "scikit-learn MiniBatchKMeans"}
DATASET_KO = {"gmm": "합성 GMM", "covtype": "covtype", "mnist": "MNIST", "census": "USCensus1990"}


def dominated(t_m, e_m, t_o, e_o, t_tol=0.01, e_tol=1e-6):
    """Is (t_m, e_m) dominated by (t_o, e_o)? Times as fractions of the full run, SSE excesses."""
    return (t_o <= t_m + t_tol and e_o <= e_m + e_tol) and (t_o < t_m - t_tol or e_o < e_m - e_tol)


def summary(df):
    """Per engine: savings, quality, the pre-registered verdict."""
    df = df.copy()
    df["frac"] = df.seconds / df.full_seconds
    w = df.pivot_table(index=["config", "engine"], columns="rule", values=["frac", "sse_excess", "label_err",
                                                                         "sse_excess_exact"], aggfunc="first")
    out = []
    for eng in ENGINES:
        if eng not in w.index.get_level_values(1):
            continue
        g = w.xs(eng, level=1)
        fm, em = g["frac"]["markov"], g["sse_excess"]["markov"]
        fd, ed = g["frac"]["default"], g["sse_excess"]["default"]
        dom = np.array([dominated(a, b, c, d) for a, b, c, d in zip(fm, em, fd, ed)])
        sav = 1 - fm
        a, b, c = float(np.median(sav)) >= 0.20, float((em <= 1e-3).mean()) >= 0.80, float(dom.mean()) <= 0.5
        out.append(dict(engine=eng, configs=len(g), saving_markov=float(np.median(sav)),
                        saving_min=float(sav.min()), saving_max=float(sav.max()),
                        saving_markov_stop=float(np.median(1 - g["frac"]["markov_stop"])),
                        saving_default=float(np.median(1 - fd)), saving_threshold=float(np.median(1 - g["frac"]["threshold"])),
                        sse_markov_median=float(np.median(em)), sse_markov_max=float(em.max()),
                        sse_ok_share=float((em <= 1e-3).mean()),
                        sse_default_median=float(np.median(ed)), sse_default_max=float(ed.max()),
                        err_markov_median=float(np.median(g["label_err"]["markov"])),
                        err_markov_stop_median=float(np.median(g["label_err"]["markov_stop"])),
                        err_default_median=float(np.median(g["label_err"]["default"])),
                        dominated_by_default=float(dom.mean()),
                        markov_vs_default_time=float(np.median(fm - fd)),
                        a=a, b=b, c=c, meaningful=a and b and c))
    return pd.DataFrame(out)


def report(path=OUT):
    df = pd.read_csv(path)
    df = df[~df.config.str.startswith("pilot")]
    s = summary(df)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string(index=False))
    df["frac"] = df.seconds / df.full_seconds
    m = df[df.rule == "markov"].assign(saving=lambda d: 1 - d.frac)
    print(m.pivot_table(index="engine", columns=["dataset", "k"], values="saving", aggfunc="median").round(3).to_string())
    # identical trajectories of the exact engines
    full = df[(df.rule == "full") & df.engine.isin(EXACT[:5])]
    print("exact engines with the exact converged partition:", full.groupby("engine").same_as_exact.mean().round(3).to_dict())
    # fastest route to SSE excess <= 1e-3 over the exact optimum
    best = []
    for key, g in df.groupby("config"):
        ok = g[g.sse_excess_exact <= 1e-3]
        fastest_full = g[(g.rule == "full") & g.engine.isin(EXACT)].seconds.min()
        r = ok.loc[ok.seconds.idxmin()]
        best.append(dict(config=key, engine=r.engine, rule=r.rule, seconds=r.seconds, vs_fastest_full=r.seconds / fastest_full))
    best = pd.DataFrame(best)
    print(best.to_string(index=False))
    s.to_csv(os.path.join(RESULTS, "accelerators_markov_summary.csv"), index=False)
    best.to_csv(os.path.join(RESULTS, "accelerators_markov_fastest.csv"), index=False)
    return s, best


FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-{}.ttf"


def _korean_font():
    """NanumGothic (OFL) for the Korean labels, fetched into data/fonts on first use."""
    import urllib.request

    from matplotlib import font_manager

    os.makedirs(os.path.join(DATA, "fonts"), exist_ok=True)
    for w in ("Regular", "Bold"):
        f = os.path.join(DATA, "fonts", f"NanumGothic-{w}.ttf")
        if not os.path.exists(f):
            urllib.request.urlretrieve(FONT_URL.format(w), f)
        font_manager.fontManager.addfont(f)
    return "NanumGothic"


def figure(path=OUT, out=None):
    """Time against accuracy per accelerator: its own default stop -> with the Markov layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from experiments.analyze_markov_tail import FIG, GRID as GRIDLINE, INK, INK2, SERIES, SURFACE

    plt.rcParams.update({"font.family": _korean_font(), "axes.unicode_minus": False, "figure.facecolor": SURFACE})
    df = pd.read_csv(path)
    df = df[~df.config.str.startswith("pilot")]
    df["time"] = 100 * df.seconds / df.full_seconds
    top = 1 - 2e-6
    df["acc"] = (1 - df.label_err).clip(0.5, top)
    n_cfg = df.config.nunique()
    DEF, MK = SERIES[1], SERIES[0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, eng in zip(axes.ravel(), ENGINES):
        g = df[df.engine == eng].pivot_table(index="config", columns="rule", values=["time", "acc"], aggfunc="first")
        ax.set_facecolor(SURFACE)
        for c in g.index:
            ax.plot([g.time.default[c], g.time.markov[c]], [g.acc.default[c], g.acc.markov[c]], color=GRIDLINE,
                    lw=1.2, zorder=1, solid_capstyle="round")
        ax.scatter(g.time.default, g.acc.default, s=46, facecolor=SURFACE, edgecolor=DEF, lw=1.8, zorder=3)
        ax.scatter(g.time.markov, g.acc.markov, s=46, color=MK, edgecolor=SURFACE, lw=1.0, zorder=4)
        ax.scatter([100], [top], s=90, marker="*", color=INK, zorder=5)
        md, mm = g.time.default.median(), g.time.markov.median()
        ed, em = 1 - g.acc.default.median(), 1 - g.acc.markov.median()
        ax.text(0.03, 0.97, f"중앙값\n기본 정지  시간 {md:.0f}% · 오차 {ed:.2%}\nMarkov     시간 {mm:.0f}% · 오차 {em:.2%}",
                transform=ax.transAxes, fontsize=8.6, color=INK2, va="top", linespacing=1.45)
        ax.set_title(ENGINE_KO[eng], loc="left", fontsize=11.5, color=INK, fontweight="bold")
        ax.set_yscale("logit")
        ax.set_ylim(0.6, 1 - 1e-6)
        ax.set_yticks([0.7, 0.9, 0.99, 0.999, 0.9999, 0.99999, top])
        ax.set_yticklabels(["70%", "90%", "99%", "99.9%", "99.99%", "99.999%", "100%"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(-3, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, color=GRIDLINE, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("걸린 시간 (그 가속법을 끝까지 돌린 시간 대비, %)", color=INK2, fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("정확도 (끝까지 돌린 결과와 같은 라벨 비율)", color=INK2, fontsize=9.5)
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=DEF, mew=1.8,
                      label="기존: 가속법 자체의 기본 정지 (scikit-learn tol = 1e-4, MiniBatch 조기 정지)"),
               Line2D([], [], ls="", marker="o", ms=7, color=MK, label="Markov 적용 (0.6 기본 경로: 정지 + 의사 대표점)"),
               Line2D([], [], ls="", marker="*", ms=10, color=INK, label="끝까지 (정확도 100% 기준)"),
               Line2D([], [], color=GRIDLINE, lw=1.2, label="같은 구성끼리 연결")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.suptitle(f"기존 가속법과 Markov 적용 가속법: 걸린 시간 대 정확도 (점 하나 = 구성 하나, {n_cfg}개 구성)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = out or os.path.join(FIG, "fig14_accelerators_markov.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def pilot():
    return run(grid=[("pilot", None, None, 50, 0)], out=os.path.join(RESULTS, "accelerators_markov_pilot.csv"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "pilot":
        pilot()
    elif cmd == "run":
        run(sys.argv[2:] or None)
    elif cmd == "figure":
        print(figure())
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else OUT)
