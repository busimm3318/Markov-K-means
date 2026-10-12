"""Accelerating the center chain of exact K-means: fewer iterations to the same kind of fixed point.

Lloyd's iteration C_{t+1} = G(C_t) (G: assign every point to its nearest center, then take
the centroids) is a fixed-point iteration, i.e. the center chain that Markov-K-means already
watches. Four classical accelerations of such chains are tried on top of the Hamerly engine:

    lloyd       plain iteration (reference)
    anderson    Anderson acceleration, type II, memory m (Walker & Ni 2011; for K-means:
                Zhang et al. 2018, arXiv:1805.10638)
    squarem     squared rank-one extrapolation, scheme S3 (Varadhan & Roland 2008)
    overrelax   adaptive over-relaxation C + w (G(C) - C), w grows while the energy falls and
                resets to 1 when it rises (Salakhutdinov & Roweis 2003)
    chain       every `period` iterations, jump to the long-run mean of the AR(1) chain fitted
                to the recent centers (the 0.4 pseudo representative, used during the run)

All share one safeguard: a proposal is kept only if the energy (SSE of the centers with the
labels they induce) does not rise; otherwise the step falls back to the plain Lloyd step from
the last kept state. Lloyd steps never raise the energy, so every method decreases it
monotonically and stops at a Lloyd fixed point (labels unchanged under the centroids). The
energy comes from the per-cluster sums that the centroid step computes anyway:
SSE(C, L) = sum|x|^2 - 2 sum_j C_j . S_j + sum_j n_j |C_j|^2, an O(kd) extra.

    python -m experiments.accelerate synthetic      # 4 exact trajectories of pareto.py, n = 10^6, k = 50
    python -m experiments.accelerate tile           # one Sentinel-2 tile at 60 m, k = 256
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import _assign, _half_nearest_center, _update_bounds, center_drift, compute_centers

METHODS = ("lloyd", "anderson", "squarem", "overrelax", "chain")


class AcceleratedHamerly(HamerlyEngine):
    """Hamerly engine whose center update is accelerated, with an energy safeguard."""

    def __init__(self, X, C0, method="anderson", memory=5, period=10, w_growth=1.2, w_max=2.0):
        super().__init__(X, C0)
        self.method = method
        self.memory = int(memory)
        self.period = int(period)
        self.w_growth, self.w_max, self.w = w_growth, w_max, 1.0
        self.xsq = float(np.einsum("ij,ij->", X, X))
        self.hist = []              # kept states: (C, G)
        self.E_keep = np.inf
        self.G_keep = None
        self.pending = False        # the current centers are an accelerated proposal awaiting its energy
        self.k_steps = 0
        self.passes = 1             # assignment passes (the initial one included)
        self.proposals = 0
        self.rejected = 0

    # -- helpers
    def _energy_G(self):
        G, counts = compute_centers(self.X, self.labels, self.centers)
        C = self.centers
        S = G * counts[:, None]
        E = self.xsq - 2.0 * float((C * S).sum()) + float((counts * (C * C).sum(1)).sum())
        return E, G

    def _move_to(self, C_new):
        drift = center_drift(self.centers, C_new)
        self.centers = np.ascontiguousarray(C_new)
        _update_bounds(self.labels, self.u, self.l, drift)
        changed, nd = _assign(self.X, self.centers, self.labels, self.u, self.l, _half_nearest_center(self.centers))
        self.n_dist += int(nd)
        self.passes += 1
        return int(changed)

    def _keep(self, E, G):
        self.E_keep, self.G_keep = E, G
        self.hist.append((self.centers.copy(), G))
        if len(self.hist) > max(self.memory + 1, self.period + 1, 3):
            self.hist.pop(0)

    def _reset(self):
        self.hist = self.hist[-1:]
        self.w = 1.0

    # -- proposals
    def _propose(self):
        """Next centers from the kept states, and whether they are an accelerated proposal."""
        C, G = self.hist[-1]
        m = self.method
        if m == "lloyd" or len(self.hist) < 2:
            return G, False
        if m == "anderson":
            h = self.hist[-(self.memory + 1):]
            F = np.stack([(g - c).ravel() for c, g in h], 1)
            Gs = np.stack([g.ravel() for _, g in h], 1)
            dF, dG = np.diff(F, axis=1), np.diff(Gs, axis=1)
            f = F[:, -1]
            gram = dF.T @ dF
            gram += 1e-10 * np.trace(gram) * np.eye(gram.shape[0])
            try:
                gamma = np.linalg.solve(gram, dF.T @ f)
            except np.linalg.LinAlgError:
                return G, False
            return (G.ravel() - dG @ gamma).reshape(G.shape), True
        if m == "squarem":
            if self.k_steps % 2 == 0:          # odd half of the cycle: a plain Lloyd step
                return G, False
            (x0, x1), (_, x2) = (self.hist[-2][0], C), (C, G)
            r, v = x1 - x0, x2 - 2 * x1 + x0
            nv = np.sqrt((v * v).sum())
            if nv == 0:
                return G, False
            alpha = min(-np.sqrt((r * r).sum()) / nv, -1.0)
            return x0 - 2 * alpha * r + alpha * alpha * v, True
        if m == "overrelax":
            if self.w <= 1.0:                   # a plain step; the next one is over-relaxed
                self.w = self.w_growth
                return G, False
            return C + self.w * (G - C), True
        if m == "chain":
            if self.k_steps % self.period or len(self.hist) < self.period + 1:
                return G, False
            Cw = np.stack([c for c, _ in self.hist[-(self.period + 1):]] + [G])
            Cs, _ = _rules.long_run_centers(Cw)
            return Cs, True
        raise ValueError(m)

    # -- one iteration
    def step(self):
        self.k_steps += 1
        E, G = self._energy_G()
        changed = 0
        if self.pending and E > self.E_keep + 1e-12 * self.xsq:
            # the proposal raised the energy: take the plain Lloyd step from the last kept state
            self.rejected += 1
            self._reset()
            changed += self._move_to(self.G_keep)
            E, G = self._energy_G()
        elif self.pending and self.method == "overrelax":
            self.w = min(self.w * self.w_growth, self.w_max)
        self.pending = False
        self._keep(E, G)
        C_next, accel = self._propose()
        self.proposals += int(accel)
        moved = self._move_to(C_next)
        changed += moved
        self.pending = accel
        if moved == 0 and accel:
            # labels unchanged under the proposal: judge it now, then settle with a Lloyd step
            E2, G2 = self._energy_G()
            if E2 > self.E_keep + 1e-12 * self.xsq:
                self.rejected += 1
                self._reset()
                changed += self._move_to(self.G_keep)
                E2, G2 = self._energy_G()
            self.pending = False
            self._keep(E2, G2)
            changed += self._move_to(G2)
        return changed

    def energy(self):
        return self._energy_G()[0]


def run_to_convergence(X, C0, method, max_iter=10_000, **kw):
    eng = AcceleratedHamerly(X, C0, method=method, **kw) if method != "plain" else HamerlyEngine(X, C0)
    t0 = time.perf_counter()
    it = 0
    trace = []
    while it < max_iter:
        ch = eng.step()
        it += 1
        if it % 25 == 0 or ch == 0:
            trace.append((it, time.perf_counter() - t0, ch))
        if ch == 0:
            break
    secs = time.perf_counter() - t0
    G, counts = compute_centers(X, eng.labels, eng.centers)
    E = float(np.einsum("ij,ij->", X, X) - 2 * (eng.centers * G * counts[:, None]).sum()
              + (counts * (eng.centers ** 2).sum(1)).sum())
    out = dict(method=method, iterations=it, seconds=secs, converged=ch == 0, sse=E, n_dist=eng.n_dist,
               passes=getattr(eng, "passes", it + 1), proposals=getattr(eng, "proposals", 0),
               rejected=getattr(eng, "rejected", 0))
    return out, eng.labels.copy(), trace


def compare(X, C0, name, methods=METHODS, **kw):
    rows, labels = [], {}
    for m in methods:
        r, lab, _ = run_to_convergence(X, C0, m, **kw)
        r["case"] = name
        rows.append(r)
        labels[m] = lab
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    base = df.set_index("method").loc["lloyd"]
    df["time_ratio"] = df.seconds / base.seconds
    df["iter_ratio"] = df.iterations / base.iterations
    df["sse_vs_lloyd"] = df.sse / base.sse - 1
    df["same_partition_share"] = [float((labels[m] == labels["lloyd"]).mean()) for m in df.method]
    return df


def synthetic():
    from experiments.pareto import INSTANCES, K, name_of, trajectory
    from kmeans_accel import datasets

    out = []
    for inst in INSTANCES:
        if inst[0] != "exact":
            continue
        traj, X = trajectory(*inst)
        if X is None:
            X, _, _ = datasets.gmm(1_000_000, 16, K, sep=inst[1], seed=inst[2])
        out.append(compare(np.ascontiguousarray(X), traj["C"][0], name_of(*inst)))
    return pd.concat(out, ignore_index=True)


TILES = {"seoul60": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/52/S/CG/2024/11/"
                     "S2A_52SCG_20241112_0_L2A/"}


def tile(k=256, methods=("lloyd", "squarem", "chain", "anderson")):
    """One Sentinel-2 tile at 60 m (3.3 million pixels), k-means++ seeding on a 200k-pixel sample."""
    from sklearn.cluster import kmeans_plusplus

    from experiments.sentinel2_wide import load_tile

    cube, valid = load_tile(TILES["seoul60"])
    X = np.ascontiguousarray(cube[valid], dtype=np.float64)
    rng = np.random.default_rng(0)
    C0, _ = kmeans_plusplus(X[rng.choice(len(X), 200_000, replace=False)], k, random_state=0)
    return compare(X, C0, f"sentinel2-60m-k{k}", methods=methods)


if __name__ == "__main__":
    from experiments.pareto import RESULTS

    cmd = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if cmd == "synthetic":
        df = synthetic()
        df.to_csv(RESULTS / "accelerate_synthetic.csv", index=False)
        print(df[["case", "method", "iterations", "passes", "seconds", "time_ratio", "sse_vs_lloyd", "rejected"]]
              .round(4).to_string(index=False))
    elif cmd == "tile":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        df = tile(k)
        df.to_csv(RESULTS / f"accelerate_tile_k{k}.csv", index=False)
        print(df[["case", "method", "iterations", "passes", "seconds", "time_ratio", "sse_vs_lloyd", "rejected",
                  "same_partition_share"]].round(5).to_string(index=False))
