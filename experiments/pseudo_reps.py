"""Pseudo representatives (0.4) against history settlement only (0.3) at the automatic stop.

For each of the 10 trajectories of experiments/pareto.py (n = 10^6, d = 16, k = 50), the
automatic rule's stop is replayed with default settings and both finishes are applied to
the same state. The pseudo step runs on the base algorithm's own state: Hamerly is replayed
to the stop (bit-identical iterates) so the extra assignment pass is pruned and timed as in
MarkovKMeans.fit; mini-batch assigns by brute force, as fit does.

    python -m experiments.pseudo_reps        # experiments/results/pseudo_reps.csv, tables in docs/results.md
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.analyze_markov_tail import fill_report, md
from experiments.auto_rule import reference, replay_stop
from experiments.pareto import INSTANCES, K, RESULTS, name_of, trajectory, warm_up
from experiments.time_for_error import _Brute
from kmeans_accel import datasets
from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import compute_centers
from markov_kmeans.estimator import pseudo_finish

W = _rules.StopParams().window


def _half_nearest(C):
    D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(2))
    np.fill_diagonal(D, np.inf)
    return D.min(1) / 2


def analyse(regime, sep, seed, eta):
    traj, X = trajectory(regime, sep, seed, eta)
    H, C, secs = traj["H"], traj["C"], traj["secs"]
    T = len(H)
    exact = regime == "exact"
    ref, _ = reference(regime, traj, K)
    if X is None:
        X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
    t, reason, _, st = replay_stop(traj, _rules.StopParams(), exact, "auto", K)
    row = dict(instance=name_of(regime, sep, seed, eta), regime=regime, t=t, T=T, reason=reason)
    L = H[t].astype(np.int64)
    if reason == "converged" or t < 2:
        return row
    if exact:
        eng = HamerlyEngine(X, C[0])
        for _ in range(t):
            eng.step()
        assert np.array_equal(eng.labels, H[t]) and np.array_equal(eng.centers, C[t])
    else:
        eng = _Brute(X, C[t])
    lo = max(0, t - W)
    U = np.flatnonzero((H[lo + 1:t + 1] != H[lo:t]).any(0))
    t0 = time.perf_counter()
    Hw = np.ascontiguousarray(H[lo:t + 1][:, U].astype(np.int64))
    regime_name = "oscillation" if reason == "oscillation" else "drift"
    lab_U, S, P, osc = _rules.settle(Hw, L[U], rule="auto", n_clusters=K, regime=regime_name)
    lab3 = L.copy()
    lab3[U] = lab_U
    settle_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lab4, U4, S4, P4, _, Cs, rho, moves, accepted, nd = pseudo_finish(
        X, eng, SimpleNamespace(centers_window=lambda: C[lo:t + 1]), L, lab3, U, S, P, osc)
    pseudo_s = time.perf_counter() - t0

    full = secs[T - 1]
    w0, w3, w4 = L != ref, lab3 != ref, lab4 != ref
    inU, in4 = np.zeros(len(L), bool), np.zeros(len(L), bool)
    inU[U], in4[U4] = True, True
    # centers: how far the current and the pseudo centers are from the reference partition's centroids
    C_ref = compute_centers(X, ref, C[t])[0]
    scale = _half_nearest(C_ref)
    row.update(
        accepted=bool(accepted), rho_median=float(np.median(rho)), rho_max_share=float((rho >= 0.95 - 1e-12).mean()),
        error_stop=float(w0.mean()), error_03=float(w3.mean()), error_04=float(w4.mean()),
        fixed=int((w3 & ~w4).sum()), broken=int((~w3 & w4).sum()),
        moves=len(moves), moves_right=float((~w4[moves]).mean()) if len(moves) else np.nan,
        U_frac=float(inU.mean()), undecided_frac=float(in4.mean()),
        recall_03=float((w3 & inU).sum() / max(w3.sum(), 1)), recall_04=float((w4 & in4).sum() / max(w4.sum(), 1)),
        cost_03=(secs[t] + st["monitor_seconds"] + settle_s) / full,
        cost_04=(secs[t] + st["monitor_seconds"] + settle_s + pseudo_s) / full,
        pseudo_share=pseudo_s / full, pseudo_dist_share=nd / (len(L) * K),
        center_err_now=float(np.median(np.sqrt(((C[t] - C_ref) ** 2).sum(1)) / scale)),
        center_err_pseudo=float(np.median(np.sqrt(((Cs - C_ref) ** 2).sum(1)) / scale)))
    if accepted and len(moves):
        # membership of the predicted movers: weight on the new label against how often it is right
        r = np.searchsorted(U4, moves)
        p_new = P4[r, 0]
        right = ~w4[moves]
        row.update(moves_p_new=float(p_new.mean()))
        for lo_, hi_ in ((0.5, 0.75), (0.75, 0.9), (0.9, 1.01)):
            sel = (p_new >= lo_) & (p_new < hi_)
            row[f"right_p{lo_:g}"] = float(right[sel].mean()) if sel.any() else np.nan
            row[f"n_p{lo_:g}"] = int(sel.sum())
    return row


RNAME = {"exact": "정확 Lloyd", "minibatch": "mini-batch 1/count", "minibatch_const": "mini-batch 상수 스텝"}


def report(df):
    T = {}
    d = df.dropna(subset=["error_03"]).copy()
    d.index = [f"{RNAME[r]} · {i.split('-', 1)[1]}" for r, i in zip(d.regime, d.instance)]
    t1 = d[["t", "error_stop", "error_03", "error_04", "fixed", "broken", "cost_03", "cost_04"]].copy()
    t1.columns = ["정지 반복", "오류: 멈춘 그대로", "오류: 0.3 (이력 정리)", "오류: 0.4 (의사 대표점)", "고친 점", "망친 점",
                  "시간: 0.3", "시간: 0.4"]
    T["pseudo_default"] = md(t1, {"정지 반복": "{:.0f}", "오류: 멈춘 그대로": "{:.3%}", "오류: 0.3 (이력 정리)": "{:.3%}",
                                  "오류: 0.4 (의사 대표점)": "{:.3%}", "고친 점": "{:,.0f}", "망친 점": "{:,.0f}",
                                  "시간: 0.3": "{:.1%}", "시간: 0.4": "{:.1%}"}, index_name="사례")
    t2 = d[["accepted", "rho_median", "center_err_now", "center_err_pseudo", "U_frac", "undecided_frac", "recall_03",
            "recall_04", "moves_right"]].copy()
    t2["accepted"] = t2.accepted.map({True: "예", False: "아니오"})
    t2.columns = ["채택", "ρ 중앙값", "대표점 오차: 현재", "대표점 오차: 의사", "미결정: 0.3", "미결정: 0.4",
                  "틀린 점 중 미결정: 0.3", "틀린 점 중 미결정: 0.4", "옮긴 점 중 맞은 비율"]
    T["pseudo_detail"] = md(t2, {"채택": "{}", "ρ 중앙값": "{:.2f}", "대표점 오차: 현재": "{:.4f}",
                                 "대표점 오차: 의사": "{:.4f}", "미결정: 0.3": "{:.2%}", "미결정: 0.4": "{:.2%}",
                                 "틀린 점 중 미결정: 0.3": "{:.1%}", "틀린 점 중 미결정: 0.4": "{:.1%}",
                                 "옮긴 점 중 맞은 비율": "{:.0%}"}, index_name="사례")
    names = {"moves_p_new": "새 군집 몫 (평균)", "right_p0.5": "맞은 비율: 몫 0.5–0.75",
             "right_p0.75": "맞은 비율: 몫 0.75–0.9", "right_p0.9": "맞은 비율: 몫 0.9–1"}
    cols = [c for c in names if c in d]
    if "moves_p_new" in cols:
        t3 = d[cols].dropna(subset=["moves_p_new"]).rename(columns=names)
        T["pseudo_membership"] = md(t3, "{:.0%}", index_name="사례")
    return T


def main(argv=()):
    if "--report" in argv:
        df = pd.read_csv(RESULTS / "pseudo_reps.csv")
    else:
        warm_up()
        df = pd.DataFrame([analyse(*inst) for inst in INSTANCES])
        df.to_csv(RESULTS / "pseudo_reps.csv", index=False)
    T = report(df)
    fill_report(T)
    for k, v in T.items():
        print(f"== {k}\n{v}")
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
