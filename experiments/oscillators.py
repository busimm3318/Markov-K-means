"""Per-point study of the points whose Markov long-run distribution is fractional.

The T0 sweep (experiments/markov_tail.py) only stores aggregates.  The continuous
reading "π = [0.5, 0.5] means half in each cluster" is tested here point by point: for
every T0 the unstable points whose chain gives a fractional π are dumped together with
what actually happened later — the converged label (exact regime), the long-run label
occupancy over the last 50 epochs (mini-batch), the converged model's own posterior,
the generative posterior, and the geometric margin at T0.

    python -m experiments.oscillators --regime minibatch --n 2000000 --k 50 --sep 1.5
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.cluster import kmeans_plusplus

from experiments.markov_tail import RESULTS, T0S, W, MBRecorder, Recorder
from kmeans_accel import datasets, tail
from kmeans_accel import markov_assign as mk
from kmeans_accel.hamerly import hamerly
from kmeans_accel.minibatch_epochs import minibatch_epochs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["exact", "minibatch"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--sep", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    a = p.parse_args(argv)
    tag = f"{a.regime}-n{a.n}-d{a.d}-k{a.k}-sep{a.sep}-s{a.seed}"
    X, z, means = datasets.gmm(a.n, a.d, a.k, sep=a.sep, seed=a.seed)
    C0, _ = kmeans_plusplus(X, a.k, random_state=a.seed)
    t_keep = max(T0S)
    if a.regime == "exact":
        rec = Recorder(a.n, a.k, t_keep)
        res = hamerly(X, C0, max_iter=3000, callback=rec)
        ref_labels, ref_C, T_conv = res.labels.astype(np.int64), res.centers, res.n_iter
    else:
        rec = MBRecorder(a.n, a.k, t_keep, a.epochs)
        ref_C, _ = minibatch_epochs(X, C0, a.epochs, seed=a.seed, callback=rec)
        ref_labels = rec.modal_labels()
        T_conv = a.epochs
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: reference done, T_conv={T_conv}", flush=True)
    model_params = tail.gmm_params(X, ref_labels, ref_C)
    out = {k_: [] for k_ in ("T0", "idx", "a", "b", "pi_a", "freq_a", "label_T0", "ref", "occ_a", "occ_b",
                             "model_a", "model_b", "true_a", "true_b", "margin", "n_switch")}
    from scipy.optimize import linear_sum_assignment
    r_, c_ = linear_sum_assignment(np.linalg.norm(ref_C[:, None] - means[None], axis=2))
    fit2true = np.empty(a.k, dtype=np.int64)
    fit2true[r_] = c_
    for T0 in T0S:
        if T0 >= T_conv:
            break
        L = rec.H[T0].astype(np.int64)
        win = rec.H[T0 - W:T0 + 1]
        U = np.flatnonzero((win != rec.H[T0]).any(0)).astype(np.int64)
        Hw = win[:, U].astype(np.int64)
        S, P = mk.stationary(Hw)
        frac = P.max(1) < 1 - 1e-9
        Uf = U[frac]
        Sf, Pf = S[frac], P[frac]
        Ff = mk.frequency(Hw[:, frac])
        # the two states with the largest long-run mass (oscillators visit 2 states almost always)
        order = np.argsort(-Pf, axis=1)
        rows = np.arange(len(Uf))
        sa, sb = Sf[rows, order[:, 0]], Sf[rows, order[:, 1]]
        pa = Pf[rows, order[:, 0]]
        fa = np.array([Ff[1][r][list(Ff[0][r]).index(sa[r])] for r in rows]) if len(Uf) else np.zeros(0)
        C_T = rec.centers[T0]
        D = tail._sqdist_subset(X, Uf, C_T)
        Ds = np.sort(D, axis=1)
        margin = (np.sqrt(Ds[:, 1]) - np.sqrt(Ds[:, 0])) / (np.sqrt(Ds[:, 1]) + np.sqrt(Ds[:, 0]))
        Mp = tail.soft_gmm(X, ref_labels, ref_C, Uf, params=model_params)
        Tp = tail.true_posterior(X, Uf, means)
        if a.regime == "minibatch":
            occ = rec.occupancy(Uf, a.k)
            oa, ob = occ[rows, sa], occ[rows, sb]
        else:
            oa = (ref_labels[Uf] == sa).astype(float)
            ob = (ref_labels[Uf] == sb).astype(float)
        out["T0"].append(np.full(len(Uf), T0))
        out["idx"].append(Uf)
        out["a"].append(sa)
        out["b"].append(sb)
        out["pi_a"].append(pa)
        out["freq_a"].append(fa)
        out["label_T0"].append(L[Uf])
        out["ref"].append(ref_labels[Uf])
        out["occ_a"].append(oa)
        out["occ_b"].append(ob)
        out["model_a"].append(Mp[rows, sa])
        out["model_b"].append(Mp[rows, sb])
        out["true_a"].append(Tp[rows, fit2true[sa]])
        out["true_b"].append(Tp[rows, fit2true[sb]])
        out["margin"].append(margin)
        out["n_switch"].append((Hw[1:, frac] != Hw[:-1, frac]).sum(0))
        print(f"  T0={T0}: |U|={len(U)} fractional={len(Uf)} ({len(Uf) / max(len(U), 1):.2%} of U)", flush=True)
    arrays = {k_: (np.concatenate(v) if v else np.zeros(0)) for k_, v in out.items()}
    path = RESULTS / f"oscillators_{tag}.npz"
    np.savez_compressed(path, **arrays, T_conv=T_conv)
    print(f"saved {path} ({len(arrays['idx'])} point-cutoff pairs)")


if __name__ == "__main__":
    main()
