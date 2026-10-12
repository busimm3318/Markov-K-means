"""Cost of the Markov part of Markov-K-means: the label-history bookkeeping and the long-run laws.

The reference implementation is the one released as v0.4.0 (read from the git tag); the
current one is the working tree. Both are fed the same iterates of one Sentinel-2 scene
(Seoul, 20 m, about thirty million pixels, the scene of experiments/sentinel2_tail.py), so
their decisions can be compared one to one:

    exact      Hamerly, k = 12, until the automatic stop: per-iteration cost of the monitor
               (recording the label chains and judging the run), identical decisions required
    oscillate  constant-step mini-batch (an EMA codebook), k = 12, a fixed number of epochs, every
               epoch recorded and judged: the unstable set is large, so the cost of the label
               chains' long-run laws shows

    python -m experiments.markov_speed     # -> experiments/results/markov_speed.csv
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_TAG = "v0.4.0"
SEOUL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/52/S/CG/2024/11/S2A_52SCG_20241112_0_L2A/"


def reference_modules(tag=REF_TAG):
    """The released _rules and _markov modules, loaded under their own package name."""
    d = tempfile.mkdtemp(prefix="mk_ref_")
    pkg = os.path.join(d, "mk_ref")
    os.makedirs(pkg)
    for name in ("_rules", "_markov"):
        src = subprocess.run(["git", "show", f"{tag}:markov_kmeans/{name}.py"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
        with open(os.path.join(pkg, f"{name}.py"), "w") as f:
            f.write(src)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, d)
    import mk_ref._markov as ref_markov  # noqa: E402
    import mk_ref._rules as ref_rules  # noqa: E402
    return ref_rules, ref_markov


def _timed(f, *a, **kw):
    t0 = time.perf_counter()
    out = f(*a, **kw)
    return out, time.perf_counter() - t0


def lockstep(engine, monitors, max_iter, rule="auto", halt=True):
    """Feed the same iterates to several monitors; per monitor: seconds, trace, stop iteration.

    ``halt=False`` keeps feeding (and judging) a monitor after it decided to stop.
    """
    for m in monitors:
        m.push(engine.labels, engine.centers)
    secs = [0.0] * len(monitors)
    stopped = [None] * len(monitors)
    step_s, t = 0.0, 0
    while t < max_iter and not (halt and all(s is not None for s in stopped)):
        (changed), dt = _timed(engine.step)
        step_s += dt
        t += 1
        for j, m in enumerate(monitors):
            if stopped[j] is not None:
                continue
            t0 = time.perf_counter()
            if "movers" in inspect.signature(m.push).parameters:      # the engine tells who moved
                m.push(engine.labels, engine.centers, changed, movers=engine.movers)
            else:
                m.push(engine.labels, engine.centers, changed)
            stop, _, _ = m.assess() if rule == "auto" else (False, None, None)
            secs[j] += time.perf_counter() - t0
            if stop and halt:
                stopped[j] = t
        if changed == 0 and engine.exact:
            break
    return secs, step_s, t, stopped


def run():
    from sklearn.cluster import kmeans_plusplus

    from experiments.sentinel2_tail import load
    from markov_kmeans import _markov, _rules
    from markov_kmeans._engines import HamerlyEngine, MiniBatchEngine

    ref_rules, ref_markov = reference_modules()
    cube, valid = load(SEOUL)
    X = np.ascontiguousarray(cube[valid], dtype=np.float64)
    del cube, valid
    n, k = len(X), 12
    rng = np.random.default_rng(0)
    C0, _ = kmeans_plusplus(X[rng.choice(n, 1_000_000, replace=False)], k, random_state=0)
    rows = []

    # exact: the monitor until the automatic stop
    mons = [ref_rules.StateMonitor(n, k, ref_rules.StopParams(), exact=True),
            _rules.StateMonitor(n, k, _rules.StopParams(), exact=True)]
    secs, step_s, t, stopped = lockstep(HamerlyEngine(X, C0), mons, 1500)
    same = mons[0].trace == mons[1].trace and np.array_equal(mons[0].unstable(), mons[1].unstable())
    U = mons[1].unstable()
    for name, mon, s in (("v0.4.0", mons[0], secs[0]), ("current", mons[1], secs[1])):
        rows.append(dict(case="exact k=12", impl=name, n=n, iterations=stopped[1], monitor_s=s,
                         monitor_ms_per_iter=s / stopped[1] * 1e3, engine_s=step_s, monitor_share=s / (s + step_s),
                         unstable=len(U), identical_decisions=same))
    print(rows[-2:], flush=True)
    for mon in mons:
        del mon
    del mons

    # constant step: large unstable set, long-run laws of the label chains
    eng = MiniBatchEngine(X, C0, batch_size=4096, learning_rate="constant", eta=2e-4, seed=0)
    epochs = 15
    mons = [ref_rules.StateMonitor(n, k, ref_rules.StopParams(), exact=False),
            _rules.StateMonitor(n, k, _rules.StopParams(), exact=False)]
    secs, step_s, t, _ = lockstep(eng, mons, epochs, halt=False)
    same_trace = mons[0].trace == mons[1].trace
    U0, U1 = mons[0].unstable(), mons[1].unstable()
    Hw0, Hw1 = mons[0].window(U0), mons[1].window(U1)
    same_window = same_trace and np.array_equal(U0, U1) and np.array_equal(np.asarray(Hw0, np.int64), Hw1)
    ref_markov.stationary(Hw0[:, :100])
    _markov.stationary(Hw1[:, :100])
    (S0, P0), s_ref = _timed(ref_markov.stationary, Hw0)
    (S1, P1), s_new = _timed(_markov.stationary, Hw1)
    states_equal = bool(np.array_equal(S0, S1))
    max_diff = float(np.abs(P0 - P1).max()) if states_equal else np.nan
    (_, _), f_ref = _timed(ref_markov.frequency, Hw0)
    (_, _), f_new = _timed(_markov.frequency, Hw1)
    for name, s, st, fr in (("v0.4.0", secs[0], s_ref, f_ref), ("current", secs[1], s_new, f_new)):
        rows.append(dict(case="constant step k=12", impl=name, n=n, iterations=epochs, monitor_s=s,
                         monitor_ms_per_iter=s / epochs * 1e3, engine_s=step_s, monitor_share=s / (s + step_s),
                         unstable=len(U1), identical_decisions=same_window, long_run_s=st,
                         long_run_us_per_point=st / max(len(U1), 1) * 1e6, frequency_s=fr,
                         long_run_states_equal=states_equal, long_run_max_diff=max_diff))
    print(rows[-2:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "markov_speed.csv"), index=False)
    return df


if __name__ == "__main__":
    print(run().round(4).to_string(index=False))
