"""Markov-chain assignment of the few non-converging points vs. alternatives.

For every instance (synthetic GMM, ground truth known) one reference trajectory is run
to convergence while each point's label history is recorded:

* regime ``exact``      — Lloyd's algorithm (run with Hamerly's exact acceleration, which
                          produces the identical trajectory);
* regime ``minibatch``  — Sculley mini-batch K-means, one iteration = one epoch.

Only instances whose trajectory needs >= 100 iterations are kept.  At each cut-off
T0 = 10, 20, ..., 90 every strategy starts from the *identical* state (labels and
centers at T0, label window of the unstable set U) and is scored for compute, error
against the converged solution / ground truth, and quality of its soft memberships.

    python -m experiments.markov_tail --regime exact --n 10000000 --d 16 --k 50 --sep 2.0 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import roc_auc_score

from kmeans_accel import datasets, tail
from kmeans_accel import markov_assign as mk
from kmeans_accel.core import compute_centers, inertia
from kmeans_accel.hamerly import hamerly
from kmeans_accel.lloyd import lloyd, lloyd_gemm
from kmeans_accel.minibatch_epochs import minibatch_epochs

RESULTS = Path(__file__).resolve().parent / "results"
T0S = list(range(10, 100, 10))
W = 10            # window that defines U and feeds the chains
W_MAX = 20        # longest window any variant reads
MIN_CONV = 100    # instances must need at least this many iterations
REF_WINDOW = 50   # mini-batch: soft reference = occupancy over the last REF_WINDOW epochs
EVAL_MAX = 200_000  # soft metrics are scored on a seeded random sample of U when |U| is larger


def strategies(regime):
    """name -> (callable, output is soft).  Same set and same inputs for every T0."""
    fixed = regime == "minibatch"   # mini-batch: keep the stable representatives C_T
    s = {
        "stop": (partial(tail.stop, fixed=fixed), False),
        "reassign_U": (partial(tail.reassign_u, fixed=fixed), False),
        "majority": (partial(tail.majority, fixed=fixed), False),
        "frequency_soft": (partial(tail.majority, fixed=fixed, soft=True), True),
        "markov": (partial(tail.markov, fixed=fixed), False),
        "markov_soft": (partial(tail.markov, fixed=fixed, soft=True), True),
        "markov_a0.5": (partial(tail.markov, fixed=fixed, alpha=0.5), False),
        "markov_a0.5_soft": (partial(tail.markov, fixed=fixed, alpha=0.5, soft=True), True),
        "markov_pooled": (partial(tail.markov, fixed=fixed, pooled=2.0), False),
        "markov_pooled_soft": (partial(tail.markov, fixed=fixed, pooled=2.0, soft=True), True),
        "markov_w5": (partial(tail.markov, fixed=fixed, window=5), False),
        "markov_w20": (partial(tail.markov, fixed=fixed, window=20), False),
    }
    if regime == "exact":
        s["active_set_U"] = (tail.active_set, False)
        s["hartigan_U"] = (tail.hartigan_u, False)
    else:
        s["reassign_all"] = (tail.reassign_all, False)
    return s


# ----------------------------------------------------------------------------- trajectories

class Recorder:
    """Labels for t <= t_keep, centers, per-iteration cost/changes, last change per point."""

    def __init__(self, n, k, t_keep):
        self.dtype = np.uint8 if k <= 255 else np.uint16
        self.H = np.zeros((t_keep + 1, n), dtype=self.dtype)
        self.t_keep = t_keep
        self.centers = {}
        self.n_dist, self.seconds, self.changes = [], [], []
        self.last_change = np.zeros(n, dtype=np.int16)
        self.prev = None

    def __call__(self, t, labels, C, nd, sec):
        lab = labels.astype(self.dtype)
        if self.prev is not None:
            ch = lab != self.prev
            self.last_change[ch] = t
            self.changes.append(int(ch.sum()))
        if t <= self.t_keep:
            self.H[t] = lab
            self.centers[t] = C.copy()
        self.prev = lab
        self.n_dist.append(nd)
        self.seconds.append(sec)


class MBRecorder(Recorder):
    """Also counts each point's labels over the last REF_WINDOW epochs (dense base + sparse rest)."""

    def __init__(self, n, k, t_keep, epochs):
        super().__init__(n, k, t_keep)
        self.ref_start = epochs - REF_WINDOW + 1
        self.base = None
        self.base_count = None
        self.other = {}

    def __call__(self, t, labels, C, nd, sec):
        super().__call__(t, labels, C, nd, sec)
        if t == self.ref_start:
            self.base = self.prev.copy()
            self.base_count = np.zeros(len(labels), dtype=np.int16)
        if t >= self.ref_start:
            same = self.prev == self.base
            self.base_count[same] += 1
            for i in np.flatnonzero(~same):
                key = (int(i), int(self.prev[i]))
                self.other[key] = self.other.get(key, 0) + 1

    def occupancy(self, idx, k):
        """(len(idx), k) long-run occupancy of the given points."""
        R = np.zeros((len(idx), k))
        R[np.arange(len(idx)), self.base[idx].astype(np.int64)] = self.base_count[idx] / REF_WINDOW
        pos = {int(i): r for r, i in enumerate(idx)}
        for (i, lab), c in self.other.items():
            if i in pos:
                R[pos[i], lab] += c / REF_WINDOW
        return R

    def modal_labels(self):
        lab = self.base.astype(np.int64)
        best = self.base_count.astype(np.int64)
        for (i, l_), c in self.other.items():
            if c > best[i]:
                best[i], lab[i] = c, l_
        return lab


# ----------------------------------------------------------------------------- metrics

def ari(a, b, k):
    M = np.bincount(a.astype(np.int64) * k + b.astype(np.int64), minlength=k * k).astype(float)
    M = M.reshape(k, k)
    comb = lambda x: x * (x - 1) / 2
    s_ij = comb(M).sum()
    s_a, s_b = comb(M.sum(1)).sum(), comb(M.sum(0)).sum()
    exp = s_a * s_b / comb(M.sum())
    return float((s_ij - exp) / (0.5 * (s_a + s_b) - exp))


def hard_metrics(X, lab, C, ref, fit2true, z):
    dis = lab != ref["labels"]
    err = np.linalg.norm(C - ref["centers"], axis=1) / ref["center_scale"]
    return {
        "n_disagree": int(dis.sum()),
        "disagree_rate": float(dis.mean()),
        "rel_sse": float(inertia(X, C, lab) / ref["sse"] - 1.0),
        "center_err": float(err.mean()),
        "center_err_max": float(err.max()),
        "acc_truth": float((fit2true[lab] == z).mean()),
        "ari_ref": ari(lab, ref["labels"], len(C)),
    }


def _auroc(err, score):
    return float(roc_auc_score(err, score)) if 0 < err.sum() < len(err) else float("nan")


def _ece(conf, correct):
    bins = np.minimum((conf * 10).astype(int), 9)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(10) if (bins == b).any()))


def soft_metrics(Wfit, ctx):
    """Quality of (|U|, k) memberships over fitted clusters, evaluated on the points of U.

    References: the converged labels (calibration / error detection), the converged
    model's own posterior (plug-in GMM at the reference centers), the generative truth on
    the points whose clusters are matched to true components, and, for mini-batch, the
    long-run label occupancy.
    """
    ref_U = ctx["ref_U"]
    m = len(ref_U)
    if m == 0:
        return {}
    rows = np.arange(m)
    oh_r = np.zeros_like(Wfit)
    oh_r[rows, ref_U] = 1.0
    conf = Wfit.max(1)
    pred = Wfit.argmax(1)
    ent = -(Wfit * np.log(np.where(Wfit > 0, Wfit, 1.0))).sum(1)
    out = {
        "brier_ref": float(((Wfit - oh_r) ** 2).sum(1).mean()),
        "p_ref_label": float(Wfit[rows, ref_U].mean()),
        "ece_ref": _ece(conf, pred == ref_U),
        "auroc_err_ref": _auroc(pred != ref_U, 1 - conf),
        "entropy": float(ent.mean()),
        "frac_fractional": float((conf < 1 - 1e-9).mean()),
        "tv_model_post": float(0.5 * np.abs(Wfit - ctx["model_post"]).sum(1).mean()),
    }
    mt = ctx["matched"]
    out["n_U_matched"] = int(mt.sum())
    if mt.any():
        Wt = np.zeros((int(mt.sum()), Wfit.shape[1]))
        Wt[:, ctx["fit2true"]] = Wfit[mt]
        z = ctx["z_U"][mt]
        oh_z = np.zeros_like(Wt)
        oh_z[np.arange(len(z)), z] = 1.0
        ct = Wt.max(1)
        ok = Wt.argmax(1) == z
        out.update({
            "tv_true_post": float(0.5 * np.abs(Wt - ctx["post_true"][mt]).sum(1).mean()),
            "brier_truth": float(((Wt - oh_z) ** 2).sum(1).mean()),
            "ece_truth": _ece(ct, ok),
            "auroc_err_truth": _auroc(~ok, 1 - ct),
        })
    if ctx.get("ref_soft") is not None:
        out["tv_ref_soft"] = float(0.5 * np.abs(Wfit - ctx["ref_soft"]).sum(1).mean())
    return out


# ----------------------------------------------------------------------------- driver

def write(path, row):
    row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
    new = not path.exists()
    fields = list(row) if new else next(csv.reader(path.open()))
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)



def lloyd_iteration_seconds(X, C0):
    """Per-iteration wall time of plain Lloyd at this size: BLAS (GEMM) and scalar numba."""
    small = np.ascontiguousarray(X[:2000])
    lloyd(small, C0, max_iter=1)
    lloyd_gemm(small, C0, max_iter=1)
    t = time.perf_counter()
    lloyd_gemm(X, C0, max_iter=1)          # 2 assignment passes + 1 center update
    gemm = (time.perf_counter() - t) / 2
    m = min(len(X), 1_000_000)
    sub = np.ascontiguousarray(X[:m])
    t = time.perf_counter()
    lloyd(sub, C0, max_iter=1)
    scalar = (time.perf_counter() - t) / 2 * len(X) / m
    return gemm, scalar


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["exact", "minibatch"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--sep", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out", default="")
    p.add_argument("--min-conv", type=int, default=MIN_CONV)
    p.add_argument("--eval-max", type=int, default=EVAL_MAX)
    a = p.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (a.out or f"markov_tail_{a.regime}.csv")
    inst_id = f"{a.regime}-n{a.n}-d{a.d}-k{a.k}-sep{a.sep}-s{a.seed}"

    def log(*m):
        print(f"[{time.strftime('%H:%M:%S')}] {inst_id}:", *m, flush=True)

    X, z, means = datasets.gmm(a.n, a.d, a.k, sep=a.sep, seed=a.seed)
    t = time.perf_counter()
    C0, _ = kmeans_plusplus(X, a.k, random_state=a.seed)
    log(f"data + k-means++ seeding ({time.perf_counter() - t:.1f}s)")
    t_keep = max(T0S)
    if a.regime == "exact":
        rec = Recorder(a.n, a.k, t_keep)
        res = hamerly(X, C0, max_iter=3000, callback=rec)
        ref_labels, T_conv, converged = res.labels.astype(np.int64), res.n_iter, res.converged
        ref_C = res.centers
    else:
        rec = MBRecorder(a.n, a.k, t_keep, a.epochs)
        ref_C, _ = minibatch_epochs(X, C0, a.epochs, seed=a.seed, callback=rec)
        ch_ = np.array(rec.changes)
        T_conv = int(np.flatnonzero(ch_)[-1]) + 1 if ch_.any() else 0
        converged = T_conv < a.epochs - REF_WINDOW
        ref_labels = rec.modal_labels()
    log(f"reference trajectory: T_conv={T_conv} converged={converged} ({sum(rec.seconds):.1f}s)")
    meta = dict(instance=inst_id, regime=a.regime, n=a.n, d=a.d, k=a.k, sep=a.sep, seed=a.seed,
                T_conv=T_conv, converged=converged)
    if T_conv < a.min_conv:
        log(f"SKIPPED: converges in {T_conv} < {a.min_conv} iterations")
        write(RESULTS / "markov_tail_skipped.csv", meta)
        return

    gemm_s, scalar_s = lloyd_iteration_seconds(X, C0)
    nd_cum = np.cumsum(np.array(rec.n_dist, dtype=float))
    sec_cum = np.cumsum(np.array(rec.seconds))
    ch = np.array([0] + rec.changes)
    D = np.linalg.norm(ref_C[:, None] - ref_C[None], axis=2)
    np.fill_diagonal(D, np.inf)
    ref = {"labels": ref_labels, "centers": ref_C, "sse": inertia(X, ref_C, ref_labels),
           "center_scale": float(D.min(1).mean())}
    r_, c_ = linear_sum_assignment(np.linalg.norm(ref_C[:, None] - means[None], axis=2))
    fit2true = np.empty(a.k, dtype=np.int64)
    fit2true[r_] = c_
    ref_acc_truth = float((fit2true[ref_labels] == z).mean())
    model_params = tail.gmm_params(X, ref_labels, ref_C)
    matched_cluster = np.linalg.norm(ref_C - means[fit2true], axis=1) < 0.5
    log(f"lloyd per-iteration: gemm {gemm_s:.3f}s, scalar {scalar_s:.3f}s; "
        f"ref acc vs truth {ref_acc_truth:.4f}")

    strat = strategies(a.regime)
    for T0 in T0S:
        if T0 >= T_conv:
            break
        L = rec.H[T0].astype(np.int64)
        C_T = rec.centers[T0]
        win = rec.H[max(0, T0 - W):T0 + 1]
        U = np.flatnonzero((win != rec.H[T0]).any(0)).astype(np.int64)
        V = rec.last_change > T0
        Hw = rec.H[max(0, T0 - W_MAX):T0 + 1][:, U].astype(np.int64)
        if len(U) > a.eval_max:
            rng = np.random.default_rng([a.seed, T0])
            eval_rows = np.sort(rng.choice(len(U), a.eval_max, replace=False))
        else:
            eval_rows = np.arange(len(U))
        Ue = U[eval_rows]
        post_true = tail.true_posterior(X, Ue, means)
        ref_U = ref_labels[Ue]
        ctx = {"ref_U": ref_U, "z_U": z[Ue], "fit2true": fit2true, "post_true": post_true,
               "model_post": tail.soft_gmm(X, ref_labels, ref_C, Ue, params=model_params),
               "matched": matched_cluster[ref_U] & matched_cluster[L[Ue]],
               "ref_soft": rec.occupancy(Ue, a.k) if a.regime == "minibatch" else None}
        common = dict(meta, T0=T0, n_U=len(U), n_U_eval=len(Ue), frac_U=len(U) / a.n, n_V=int(V.sum()),
                      n_U_and_V=int(V[U].sum()), changes_at_T0=int(ch[T0]),
                      U_disagree_at_T0=int((L[U] != ref_labels[U]).sum()),
                      prefix_base_dist=nd_cum[T0], prefix_base_sec=sec_cum[T0],
                      full_base_dist=nd_cum[-1], full_base_sec=sec_cum[-1],
                      lloyd_dist_iter=float(a.n) * a.k, lloyd_gemm_sec_iter=gemm_s,
                      lloyd_scalar_sec_iter=scalar_s, ref_acc_truth=ref_acc_truth,
                      n_matched_clusters=int(matched_cluster.sum()))

        def emit(name, lab, C, Wsoft, n_dist, passes, seconds, extra=None):
            row = dict(common, method=name, extra_dist=n_dist, extra_center_passes=passes,
                       extra_seconds=seconds)
            if lab is not None:
                row.update(hard_metrics(X, lab, C, ref, fit2true, z))
            row.update(soft_metrics(Wsoft, ctx))
            row["extra"] = extra or {}
            write(out, row)

        onehot = np.eye(a.k)
        for name, (fn, is_soft) in strat.items():
            Hin = Hw if name == "markov_w20" else Hw[-(W + 1):]
            r = fn(X, L, C_T, U, Hin)
            Wsoft = (mk.to_dense(r.soft[0][eval_rows], r.soft[1][eval_rows], a.k) if is_soft
                     else onehot[r.labels[Ue]])
            emit(name, r.labels, r.centers, Wsoft, r.n_dist, r.center_passes, r.seconds, r.extra)

        # soft comparators: memberships from distances to the current representatives
        C_star = C_T if a.regime == "minibatch" else compute_centers(X, L, C_T)[0]
        gparams = tail.gmm_params(X, L, C_star)
        for name, kind in (("fcm_soft", "fcm"), ("gmm_plugin_soft", "gmm")):
            t = time.perf_counter()
            labU, Wsoft = tail.soft_comparator(kind, X, C_star, U, eval_rows,
                                               params=gparams if kind == "gmm" else None)
            lab = L.copy()
            lab[U] = labU
            emit(name, lab, C_star, Wsoft, len(U) * a.k, 0 if a.regime == "minibatch" else 1,
                 time.perf_counter() - t)

        # references: running the base algorithm to convergence, and the true posterior
        emit("full_convergence", ref_labels, ref_C, onehot[ref_U], float(nd_cum[-1] - nd_cum[T0]),
             T_conv - T0, float(sec_cum[-1] - sec_cum[T0]))
        emit("true_posterior_oracle", None, None, post_true[:, fit2true], 0, 0, 0.0)
        emit("model_posterior_oracle", None, None, ctx["model_post"], 0, 0, 0.0)
        log(f"T0={T0}: |U|={len(U)} ({len(U) / a.n:.4%}) |V|={int(V.sum())} |U&V|={int(V[U].sum())} "
            f"U wrong at T0={common['U_disagree_at_T0']}")


if __name__ == "__main__":
    main()
