"""How unique is the converged map? Exact K-means from slightly perturbed seeds on Sentinel-2 scenes.

The label error of the stopping rules (experiments/sentinel2_tail.py) is measured against one
converged run. If a tiny change of the starting centers already leads to another converged
partition, part of that error is a floor that no stopping rule can go below. For each scene:

    reference   the study's run (same k-means++ seed), with the Markov-K-means and scikit-learn
                tol stops (tail_run)
    perturbed   the same seed plus Gaussian noise of eps x (feature std) on every center,
                run with Hamerly until no label changes

and per perturbed run: initial labels changed by the noise, iterations, share of pixels whose
converged label differs from the reference, the same for the Markov-K-means and scikit-learn
tol stops, and the SSE of each partition relative to the reference.

    python -m experiments.sentinel2_perturb     # -> experiments/results/sentinel2_perturb.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import K, RESULTS, SEED, load, tail_run

SITES = ("Seoul", "Sundarbans", "Amazon", "Nile Delta", "Central Valley")
EPS = (1e-6, 1e-4)
REPEATS = 3


def sse(X, labels, k):
    from markov_kmeans._kernels import compute_centers, inertia

    C = compute_centers(X, labels, np.zeros((k, X.shape[1])))[0]
    return float(inertia(X, C, labels))


def converge(X, C0, max_iter=3000):
    from markov_kmeans._engines import HamerlyEngine

    eng = HamerlyEngine(X, C0)
    first = eng.labels.copy()
    t = 0
    while t < max_iter:
        t += 1
        if eng.step() == 0:
            break
    return eng.labels.copy(), first, t


def run(sites=SITES):
    from sklearn.cluster import kmeans_plusplus

    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    out = os.path.join(RESULTS, "sentinel2_perturb.csv")
    rows = []
    for name in sites:
        s = scenes[scenes.site.str.startswith(name)].iloc[0]
        t0 = time.perf_counter()
        cube, valid = load(s.url)
        X = np.ascontiguousarray(cube[valid], dtype=np.float64)
        del cube, valid
        n = len(X)
        _, _, _, snaps = tail_run(X, keep_snaps=True)
        ref = snaps["full"].astype(np.int64)
        stops = {r: snaps[r].astype(np.int64) for r in ("markov_auto", "sklearn_tol")}
        sse_ref = sse(X, ref, K)
        C0, _ = kmeans_plusplus(X, K, random_state=SEED)
        _, first0, _ = converge(X, C0, max_iter=0)
        std = X.std(0)
        for eps in EPS:
            for rep in range(REPEATS):
                rng = np.random.default_rng(1000 + rep)
                C = C0 + eps * std * rng.standard_normal(C0.shape)
                lab, first, iters = converge(X, C)
                row = dict(site=s.site, n=n, eps=eps, rep=rep, init_changed=int((first != first0).sum()),
                           iterations=iters, differs_from_ref=float((lab != ref).mean()),
                           sse_vs_ref=sse(X, lab, K) / sse_ref - 1)
                for r, lab_r in stops.items():
                    row[f"{r}_vs_ref"] = float((lab_r != ref).mean())
                    row[f"{r}_vs_this"] = float((lab_r != lab).mean())
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False)
                print(s.site, f"eps={eps:g} rep={rep}", f"init {row['init_changed']}", f"it {iters}",
                      f"differs {row['differs_from_ref']:.4%}", f"sse {row['sse_vs_ref']:+.2e}",
                      f"markov vs this {row['markov_auto_vs_this']:.4%}", flush=True)
        print(s.site, f"{time.perf_counter() - t0:.0f}s", flush=True)
        del X, snaps, ref, stops
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(tuple(sys.argv[1:]) or SITES)
