"""Lloyd's algorithm: the reference every exact method must reproduce.

``lloyd``      — scalar numba kernel, n*k distance evaluations per iteration.
``lloyd_gemm`` — BLAS formulation ||x||^2 - 2 x.c + ||c||^2 in chunks; same work,
                 far better arithmetic intensity, but not bit-exact on near-ties.
"""
from __future__ import annotations

import time

import numpy as np

from .core import KMeansResult, check_inputs, compute_centers, inertia, nearest_all

ALGORITHMS = {
    "lloyd": {"kind": "exact", "fit": "lloyd", "ref": "Lloyd 1982 (IEEE TIT); Forgy 1965"},
    "lloyd_gemm": {"kind": "exact-gemm", "fit": "lloyd_gemm", "ref": "BLAS-3 distance expansion"},
}


def lloyd(X, init_centers, max_iter=300):
    t0 = time.perf_counter()
    X, C = check_inputs(X, init_centers)
    n, k = X.shape[0], C.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    nearest_all(X, C, labels)
    n_dist = n * k
    n_iter = 0
    converged = False
    for _ in range(max_iter):
        C, _ = compute_centers(X, labels, C)
        n_iter += 1
        changed = nearest_all(X, C, labels)
        n_dist += n * k
        if changed == 0:
            converged = True
            break
    return KMeansResult(C, labels, inertia(X, C, labels), n_iter, converged, n_dist,
                        seconds=time.perf_counter() - t0)


def _assign_gemm(X, x_sq, C, chunk):
    c_sq = np.einsum("ij,ij->i", C, C)
    labels = np.empty(X.shape[0], dtype=np.int64)
    for s in range(0, X.shape[0], chunk):
        D = X[s:s + chunk] @ C.T
        D *= -2.0
        D += c_sq
        D += x_sq[s:s + chunk, None]
        labels[s:s + chunk] = np.argmin(D, axis=1)
    return labels


def lloyd_gemm(X, init_centers, max_iter=300, chunk=4096):
    t0 = time.perf_counter()
    X, C = check_inputs(X, init_centers)
    n, k = X.shape[0], C.shape[0]
    x_sq = np.einsum("ij,ij->i", X, X)
    labels = _assign_gemm(X, x_sq, C, chunk)
    n_dist = n * k
    n_iter = 0
    converged = False
    for _ in range(max_iter):
        C, _ = compute_centers(X, labels, C)
        n_iter += 1
        new = _assign_gemm(X, x_sq, C, chunk)
        n_dist += n * k
        if np.array_equal(new, labels):
            converged = True
            break
        labels = new
    return KMeansResult(C, labels, inertia(X, C, labels), n_iter, converged, n_dist,
                        seconds=time.perf_counter() - t0)
