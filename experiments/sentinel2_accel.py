"""In-run acceleration on the twelve Sentinel-2 scenes: SQUAREM on the center chain of exact K-means.

experiments/accelerate.py found on synthetic data that SQUAREM (squared rank-one
extrapolation of the centers, Varadhan & Roland 2008), safeguarded so that the SSE never
rises, reaches a Lloyd fixed point in 8-56 % fewer iterations. Here, per scene (same k-means++
seed as experiments/sentinel2_tail.py, k = 12):

    plain     Hamerly to convergence
    squarem   the same with SQUAREM proposals every second step, each kept only if the SSE of the
              centers with the labels they induce does not rise (else a plain step)

and for both runs the stop of the 0.6 default path (stop_rule="unstable", no label history)
judged on its iterates: iteration, time, and the share of pixels whose label differs from the
plain converged map and from the run's own converged map. Times count the engine (and its
proposals and fallbacks) and the monitor; the movers are taken from the label change of each
step outside the timing, as the engines report them.

    python -m experiments.sentinel2_accel    # -> experiments/results/sentinel2_accel.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.accelerate import AcceleratedHamerly
from experiments.sentinel2_tail import K, RESULTS, SEED, load
from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import compute_centers, inertia


def run_one(X, C0, method, max_iter=3000):
    eng = AcceleratedHamerly(X, C0, method="squarem") if method == "squarem" else HamerlyEngine(X, C0)
    mon = _rules.StateMonitor(len(X), K, _rules.StopParams(), exact=True, rule="unstable", history=False)
    t_s = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    secs = time.perf_counter() - t_s
    prev = eng.labels.copy()
    stop, it = None, 0
    while it < max_iter:
        t_s = time.perf_counter()
        ch = eng.step()
        secs += time.perf_counter() - t_s
        it += 1
        movers = np.flatnonzero(eng.labels != prev)            # what an engine would report (not timed)
        prev[movers] = eng.labels[movers]
        if stop is None:
            t_s = time.perf_counter()
            mon.push(eng.labels, eng.centers, len(movers), movers=movers)
            s, _, _ = mon.assess()
            secs += time.perf_counter() - t_s
            if s:
                stop = dict(t=it, seconds=secs, labels=eng.labels.copy())
        if ch == 0:
            break
    return dict(iterations=it, seconds=secs, converged=ch == 0, labels=eng.labels.copy(), stop=stop,
                passes=getattr(eng, "passes", it + 1), rejected=getattr(eng, "rejected", 0))


def sse(X, labels, C):
    return float(inertia(X, compute_centers(X, labels, C)[0], labels))


def run(only=None):
    from sklearn.cluster import kmeans_plusplus

    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    out = os.path.join(RESULTS, "sentinel2_accel.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["site"] for r in rows}
    for _, s in scenes.iterrows():
        if (only and not any(s.site.startswith(o) for o in only)) or s.site in done:
            continue
        cube, valid = load(s.url)
        X = np.ascontiguousarray(cube[valid], dtype=np.float64)
        del cube, valid
        C0, _ = kmeans_plusplus(X, K, random_state=SEED)
        res = {m: run_one(X, C0, m) for m in ("plain", "squarem")}
        ref = res["plain"]["labels"]
        sse_ref = sse(X, ref, C0)
        for m, r in res.items():
            st = r["stop"]
            rows.append(dict(
                site=s.site, method=m, iterations=r["iterations"], passes=r["passes"], rejected=r["rejected"],
                seconds=r["seconds"], time_ratio=r["seconds"] / res["plain"]["seconds"], converged=r["converged"],
                same_partition=float((r["labels"] == ref).mean()), sse_vs_plain=sse(X, r["labels"], C0) / sse_ref - 1,
                stop_iter=st["t"] if st else np.nan, stop_time_ratio=st["seconds"] / res["plain"]["seconds"] if st else np.nan,
                stop_error_vs_plain=float((st["labels"] != ref).mean()) if st else np.nan,
                stop_error_vs_own=float((st["labels"] != r["labels"]).mean()) if st else np.nan))
            pd.DataFrame(rows).to_csv(out, index=False)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        del X, res, ref
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
