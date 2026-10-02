"""Application study 2: EMA vector-quantization codebooks (the VQ-VAE / VQGAN tokenizer setting).

A VQ tokenizer maps every latent vector to its nearest codebook entry; the codebook is
trained with exponential moving averages (van den Oord et al. 2017, Razavi et al. 2019):

    N_j <- g N_j + (1 - g) n_j,   M_j <- g M_j + (1 - g) sum_{x -> j} x,   c_j = M_j / N_j

per mini-batch, i.e. mini-batch K-means with a constant step, whose labels keep jittering
at the boundaries.  The encoder is frozen here so that the vectors are fixed:

    latent   MobileNetV2 (ImageNet weights, Keras) features after block_12 (stride 16,
             96 channels) of 100 photos resized to 2048 px on the long side
    patch    4x4 RGB pixel patches (48 values, identity encoder) of the same photos at
             768 px; the reconstruction error of a patch is the image error, so PSNR applies

One trajectory per setting (EPOCHS epochs, label of every vector at every epoch).  At each
cut-off T0 = 10, ..., 90 and at the stop of the markov_kmeans auto rule, with the codebook
C_T0 of that moment:

* tokens: unstable vectors U (label changed in the last W epochs) are settled by keeping
  the current label, the nearest code of C_T0, the Markov rule or the window frequency, and
  compared with the modal token of the last 50 epochs (reference);
* soft tokens: the reconstruction sum_j p_j c_j for the vectors of U, with p the Markov
  law pi, the window occupancy, a softmax over distances (temperature tuned on half of U,
  scored on the other half) or the best mix of the two nearest codes (oracle), against the
  hard nearest code;
* ambiguity: AUROC of 1 - max p for "token at T0 differs from the reference".

    python -m experiments.vq_codebook [--settings latent patch] [--n-patch 1000000]
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from markov_kmeans import _rules
from markov_kmeans._markov import frequency, stationary

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
EPOCHS = 200
REF_WINDOW = 50
T0S = tuple(range(10, 100, 10))
W = 10


# ----------------------------------------------------------------------------- features

def latent_features(ids, long_side=2048):
    f = DATA / f"vq_latent_mnv2_b12_{long_side}.npy"
    if f.exists():
        return np.load(f)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras

    m = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    enc = keras.Model(m.input, m.get_layer("block_12_add").output)
    out = []
    for i in ids:
        im = Image.open(DATA / "longcap100" / f"{i}.jpeg").convert("RGB")
        s = long_side / max(im.size)
        w, h = (max(16, round(im.size[0] * s / 16) * 16), max(16, round(im.size[1] * s / 16) * 16))
        x = np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32)[None] / 127.5 - 1.0
        y = enc.predict(x, verbose=0)[0]
        out.append(y.reshape(-1, y.shape[-1]))
        print(f"  latent {i}: {y.shape}", flush=True)
    X = np.concatenate(out).astype(np.float32)
    np.save(f, X)
    return X


def patch_features(ids, long_side=768, p=4):
    f = DATA / f"vq_patch{p}_{long_side}.npy"
    if f.exists():
        return np.load(f)
    out = []
    for i in ids:
        im = Image.open(DATA / "longcap100" / f"{i}.jpeg").convert("RGB")
        s = long_side / max(im.size)
        w, h = (round(im.size[0] * s / p) * p, round(im.size[1] * s / p) * p)
        a = np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32)
        a = a.reshape(h // p, p, w // p, p, 3).transpose(0, 2, 1, 3, 4).reshape(-1, p * p * 3)
        out.append(a)
    X = np.concatenate(out)
    np.save(f, X)
    return X


# ----------------------------------------------------------------------------- EMA engine

class EMAVQEngine:
    """EMA codebook updates per mini-batch; one step = one epoch over a random permutation.

    Same interface as the markov_kmeans engines (labels, centers, n_dist, step()).  A
    vector's label for the epoch is the code it gets when its batch is processed.
    """

    exact = False

    def __init__(self, X, C0, batch_size=4096, decay=0.99, eps=1e-5, seed=0):
        self.X = X
        self.K = C0.shape[0]
        self.batch_size, self.decay, self.eps = batch_size, decay, eps
        self.rng = np.random.default_rng(seed)
        self.centers = np.array(C0, dtype=np.float32)
        self.N = np.full(self.K, batch_size / self.K)
        self.M = self.centers.astype(np.float64) * self.N[:, None]
        self.x_sq = np.einsum("ij,ij->i", X, X)
        n = X.shape[0]
        self.labels = np.empty(n, dtype=np.int64)
        for s in range(0, n, 1 << 15):
            idx = np.arange(s, min(n, s + (1 << 15)))
            self.labels[idx] = self._nearest(idx)
        self.n_dist = n * self.K

    def _nearest(self, idx, C=None):
        C = self.centers if C is None else C
        D = self.X[idx] @ C.T
        D *= -2.0
        D += np.einsum("ij,ij->i", C, C)
        D += self.x_sq[idx, None]
        return D.argmin(1)

    def step(self):
        prev = self.labels.copy()
        perm = self.rng.permutation(self.X.shape[0])
        g, K = self.decay, self.K
        for s in range(0, len(perm), self.batch_size):
            idx = perm[s:s + self.batch_size]
            lab = self._nearest(idx)
            self.labels[idx] = lab
            onehot = np.zeros((K, len(idx)), dtype=np.float32)
            onehot[lab, np.arange(len(idx))] = 1.0
            self.N = g * self.N + (1 - g) * onehot.sum(1)
            self.M = g * self.M + (1 - g) * (onehot @ self.X[idx])
            tot = self.N.sum()
            Nsm = (self.N + self.eps) / (tot + K * self.eps) * tot
            self.centers = (self.M / Nsm[:, None]).astype(np.float32)
        self.n_dist += self.X.shape[0] * K
        return int((self.labels != prev).sum())


def record(X, C0, seed, epochs):
    eng = EMAVQEngine(X, C0, seed=seed)
    n = X.shape[0]
    H = np.empty((epochs + 1, n), dtype=np.uint16)
    C = np.empty((epochs + 1,) + C0.shape, dtype=np.float32)
    H[0], C[0] = eng.labels, eng.centers
    counts, secs = [], [0.0]
    for t in range(1, epochs + 1):
        t0 = time.perf_counter()
        counts.append(eng.step())
        secs.append(secs[-1] + time.perf_counter() - t0)
        H[t], C[t] = eng.labels, eng.centers
        if t % 10 == 0:
            print(f"    epoch {t}: changed {counts[-1]} ({counts[-1] / n:.4f}) {secs[-1]:.0f}s", flush=True)
    return dict(H=H, C=C, counts=np.array(counts), secs=np.array(secs))


# ----------------------------------------------------------------------------- evaluation

def auroc(score, positive):
    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(score)
    n1 = positive.sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(score) - n1)))


def topk_dist(X, C, m=8, chunk=1 << 14):
    """Indices and squared distances of the m nearest codes of every row of X."""
    c_sq = np.einsum("ij,ij->i", C, C)
    I = np.empty((len(X), m), dtype=np.int64)
    D = np.empty((len(X), m), dtype=np.float64)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d = (x @ C.T) * -2.0 + c_sq + np.einsum("ij,ij->i", x, x)[:, None]
        part = np.argpartition(d, m, axis=1)[:, :m]
        dp = np.take_along_axis(d, part, 1)
        o = np.argsort(dp, 1)
        I[s:s + chunk] = np.take_along_axis(part, o, 1)
        D[s:s + chunk] = np.maximum(np.take_along_axis(dp, o, 1), 0.0)
    return I, D


def soft_recon(S, P, C):
    R = np.zeros((len(S), C.shape[1]))
    for j in range(S.shape[1]):
        ok = S[:, j] >= 0
        R[ok] += P[ok, j, None] * C[S[ok, j]]
    return R


def mse_rows(A, B):
    return ((A.astype(np.float64) - B) ** 2).mean(1)


def evaluate_cut(setting, X, traj, T, ref_label, ref_occ, rng, policy, extra=None):
    H, C_all = traj["H"], traj["C"]
    n, K = H.shape[1], C_all.shape[1]
    C = C_all[T].astype(np.float64)
    L = H[T].astype(np.int64)
    last = np.full(n, -1, dtype=np.int32)
    for s in range(max(1, T - W + 1), T + 1):
        last[H[s] != H[s - 1]] = s
    U = np.flatnonzero(last > T - W)
    row = dict(setting=setting, policy=policy, T=T, n=n, K=K, n_U=len(U), frac_U=len(U) / n)
    # hard tokens of every vector under C_T (what a stopped tokenizer outputs)
    I_all, D_all = topk_dist(X, C.astype(np.float32), m=1)
    nearest = I_all[:, 0]
    row["mse_all_hard"] = float(D_all[:, 0].mean() / X.shape[1])
    row["token_disagree_nearest"] = float((nearest != ref_label).mean())
    row["token_disagree_current"] = float((L != ref_label).mean())
    if len(U) == 0:
        return row
    XU = X[U].astype(np.float64)
    Hw = np.ascontiguousarray(H[T - W:T + 1][:, U].astype(np.int64))
    S_m, P_m = stationary(Hw)
    S_f, P_f = frequency(Hw)
    lab_m = _rules.settle(Hw, L[U], rule="markov", n_clusters=K)[0]
    lab_f = _rules.settle(Hw, L[U], rule="frequency", n_clusters=K)[0]
    regime = (extra or {}).get("regime")
    if regime is None:   # a fixed cut-off: classify the window as MarkovKMeans does
        regime = "oscillation" if (_rules.switches(Hw) >= 2).mean() >= _rules.StopParams().osc_share else "drift"
    lab_a = _rules.settle(Hw, L[U], rule="auto", n_clusters=K, regime=regime)[0]
    ref_U = ref_label[U]
    for nm, lab in (("current", L[U]), ("nearest", nearest[U]), ("markov", lab_m), ("frequency", lab_f),
                    ("auto", lab_a)):
        row[f"U_err_{nm}"] = float((lab != ref_U).mean())
    # soft tokens: reconstruction error on U
    I, D = topk_dist(XU.astype(np.float32), C.astype(np.float32), m=8)
    hard = D[:, 0] / X.shape[1]
    row["mse_U_hard"] = float(hard.mean())
    for nm, (S, P) in (("pi", (S_m, P_m)), ("occupancy", (S_f, P_f))):
        row[f"mse_U_{nm}"] = float(mse_rows(XU, soft_recon(S, P, C)).mean())
        row[f"frac_fractional_{nm}"] = float((P.max(1) < 1 - 1e-9).mean())
    # distance softmax over the 8 nearest codes, temperature tuned on one half of U
    half = rng.random(len(U)) < 0.5
    scale = np.median(D[:, 1] - D[:, 0]) + 1e-12
    best = None
    for tau in scale * np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0]):
        Wt = np.exp(-(D - D[:, :1]) / tau)
        Wt /= Wt.sum(1, keepdims=True)
        R = np.einsum("ij,ijd->id", Wt, C[I])
        e = mse_rows(XU, R)
        if best is None or e[half].mean() < best[0]:
            best = (e[half].mean(), tau, e, Wt)
    _, tau, e_soft, Wt = best
    row["mse_U_distsoft"] = float(e_soft[~half].mean())
    row["mse_U_hard_heldout"] = float(hard[~half].mean())
    row["mse_U_pi_heldout"] = float(mse_rows(XU[~half], soft_recon(S_m[~half], P_m[~half], C)).mean())
    row["distsoft_tau_rel"] = float(tau / scale)
    # oracle: best mix of the two nearest codes
    c1, c2 = C[I[:, 0]], C[I[:, 1]]
    dv = c1 - c2
    w = np.clip(((XU - c2) * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-12), 0, 1)
    row["mse_U_top2_oracle"] = float(mse_rows(XU, w[:, None] * c1 + (1 - w[:, None]) * c2).mean())
    # whole-set effect of soft tokens on U (patch setting: PSNR of the image patches)
    tot_hard = D_all[:, 0].sum() / X.shape[1]
    row["mse_all_pi"] = float((tot_hard - hard.sum() + mse_rows(XU, soft_recon(S_m, P_m, C)).sum()) / n)
    # ambiguity detection for "token at T differs from the reference"
    wrong = L[U] != ref_U
    row["auroc_pi"] = auroc(1 - P_m.max(1), wrong)
    row["auroc_occupancy"] = auroc(1 - P_f.max(1), wrong)
    row["auroc_distsoft"] = auroc(1 - Wt.max(1), wrong)
    row["auroc_margin"] = auroc(-(D[:, 1] - D[:, 0]), wrong)
    # agreement of the soft tokens with the long-run occupancy of the reference window
    occ = ref_occ[U]
    for nm, (S, P) in (("pi", (S_m, P_m)), ("occupancy", (S_f, P_f))):
        tv = np.ones(len(U))
        for j in range(S.shape[1]):
            ok = S[:, j] >= 0
            tv[ok] -= np.minimum(P[ok, j], occ[ok, S[ok, j]])
        row[f"tv_{nm}"] = float(tv.mean())
    row.update(extra or {})
    return row


def reference(H, K):
    tail = H[-REF_WINDOW:]
    n = H.shape[1]
    occ = np.zeros((n, K), dtype=np.float32)
    for r in tail:
        occ[np.arange(n), r] += 1.0 / REF_WINDOW
    return occ.argmax(1), occ


def write(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


FIELDS = ["setting", "policy", "T", "stop_reason", "n", "K", "d", "n_U", "frac_U", "time_frac", "mse_all_hard",
          "mse_all_pi", "psnr_all_hard", "psnr_all_pi", "token_disagree_current", "token_disagree_nearest",
          "U_err_current", "U_err_nearest", "U_err_markov", "U_err_frequency", "U_err_auto", "mse_U_hard",
          "mse_U_pi", "mse_U_occupancy", "mse_U_hard_heldout", "mse_U_pi_heldout", "mse_U_distsoft",
          "distsoft_tau_rel", "mse_U_top2_oracle", "frac_fractional_pi", "frac_fractional_occupancy",
          "auroc_pi", "auroc_occupancy", "auroc_distsoft", "auroc_margin", "tv_pi", "tv_occupancy",
          "state_unstable_frac", "state_center_travel", "state_oscillating_share"]


def run_setting(setting, X, K, seed, out, epochs=EPOCHS):
    from sklearn.cluster import kmeans_plusplus

    rng = np.random.default_rng(seed)
    n, d = X.shape
    sample = X[rng.choice(n, size=min(n, 100_000), replace=False)]
    C0, _ = kmeans_plusplus(sample.astype(np.float64), K, random_state=seed)
    print(f"[{time.strftime('%H:%M:%S')}] {setting}: n={n} d={d} K={K}", flush=True)
    traj = record(X, C0.astype(np.float32), seed, epochs)
    np.savez(DATA / f"vq_traj_{setting}_K{K}.npz", counts=traj["counts"], secs=traj["secs"])
    ref_label, ref_occ = reference(traj["H"], K)
    full_secs = traj["secs"][-1]
    rows = []
    for T in T0S:
        rows.append(evaluate_cut(setting, X, traj, T, ref_label, ref_occ, rng, f"T0={T}"))
    # the markov_kmeans auto rule replayed on the same trajectory
    params = _rules.StopParams()
    mon = _rules.StateMonitor(n, K, params, exact=False)
    mon.push(traj["H"][0], traj["C"][0])
    t_auto, reason, state = epochs, "max_iter", {}
    for t in range(1, epochs + 1):
        mon.push(traj["H"][t], traj["C"][t], int(traj["counts"][t - 1]))
        stop, regime, state = mon.assess()
        if stop:
            t_auto = t
            reason = {"converged": "converged", "oscillation": "oscillation",
                      "drift": "few_unstable" if state.get("unstable_frac", 1) <= params.unstable_tol
                      else "stable_drift"}[regime]
            break
    extra = {"stop_reason": reason, "regime": "oscillation" if reason == "oscillation" else "drift",
             "state_unstable_frac": state.get("unstable_frac"), "state_center_travel": state.get("center_travel"),
             "state_oscillating_share": state.get("oscillating_share")}
    if t_auto <= epochs - REF_WINDOW:
        rows.append(evaluate_cut(setting, X, traj, t_auto, ref_label, ref_occ, rng, "auto", extra))
    else:
        rows.append(dict(setting=setting, policy="auto", T=t_auto, n=n, K=K, **extra))
    for r in rows:
        r["d"] = d
        r["time_frac"] = traj["secs"][r["T"]] / full_secs
        if setting == "patch" and "mse_all_hard" in r:
            r["psnr_all_hard"] = 10 * np.log10(255.0 ** 2 / r["mse_all_hard"])
            r["psnr_all_pi"] = 10 * np.log10(255.0 ** 2 / r["mse_all_pi"]) if "mse_all_pi" in r else np.nan
        write(out, r)
        keys = ("T", "frac_U", "U_err_current", "U_err_markov", "U_err_frequency", "mse_U_hard", "mse_U_pi",
                "mse_U_distsoft", "mse_U_top2_oracle", "auroc_pi", "auroc_distsoft")
        print("  " + r["policy"] + " " + " ".join(f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k)}"
                                               for k in keys), flush=True)
    del traj


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="+", default=["latent", "patch"])
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--n-patch", type=int, default=1_000_000)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out", default="vq_codebook.csv")
    a = p.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = list(range(1, 101))
    for setting in a.settings:
        if setting == "latent":
            X = latent_features(ids)
        else:
            X = patch_features(ids)
            if len(X) > a.n_patch:
                X = X[np.sort(np.random.default_rng(0).choice(len(X), a.n_patch, replace=False))]
        X = np.ascontiguousarray(X, dtype=np.float32)
        run_setting(setting, X, a.K, 0, RESULTS / a.out, a.epochs)


if __name__ == "__main__":
    main()
