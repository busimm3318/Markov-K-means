"""Application study 1: K-means color quantization (palette) of high-resolution photographs.

Every pixel of a photo is a vector in RGB (n = 2-21 million per photo).  One exact run
(Hamerly bounds, iterates identical to Lloyd) is driven to convergence while the stopping
policies below are evaluated on the same trajectory, at the iteration where each would
have stopped:

    auto          the markov_kmeans StateMonitor (movement / representatives / partitions),
                  unstable points settled by markov_kmeans._rules.settle
    v0.1.0        stop when |U| / n <= 1e-3 (the 0.1.0 rule), Markov settlement
    sklearn_tol   scikit-learn's rule: sum of squared center shifts <= 1e-4 * mean variance
    fixed_T25     faiss' default of 25 iterations
    fixed_T50
    full          convergence (no label changes) or 1000 iterations
and two subsampling practices that fit the palette on a sample and assign every pixel once:
    sample_faiss  256 points per center (faiss' max_points_per_centroid), 25 iterations
    sample_100k   100 000 pixels, run to convergence

The palette of a stopped run is the centroid of its labels (one extra pass), as a user
would output it.  Image quality is PSNR against the original; ``psnr_blur`` compares
Gaussian-blurred images (sigma = 1 px, a crude model of viewing distance) and is used to
test the continuous reading: unstable pixels drawn at random from their soft membership
(``pi_dither``) versus their hard label.

    python -m experiments.color_quant [--images 1-100] [--k 16 64 256]
Images: data/longcap100/<i>.jpeg (Google Cloud public sample bucket, see --download).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import kmeans_plusplus

from markov_kmeans import _rules
from markov_kmeans._engines import HamerlyEngine
from markov_kmeans._kernels import compute_centers, nearest_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "longcap100"
RESULTS = Path(__file__).resolve().parent / "results"
URL = "https://storage.googleapis.com/cloud-samples-data/ai-platform/generative_ai/media_files/longcap100/{i}.jpeg"
MAX_ITER = 1000
CURVE = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000)


def download(ids):
    import urllib.request

    DATA.mkdir(parents=True, exist_ok=True)
    for i in ids:
        f = DATA / f"{i}.jpeg"
        if not f.exists():
            urllib.request.urlretrieve(URL.format(i=i), f)


def load(i):
    im = np.asarray(Image.open(DATA / f"{i}.jpeg").convert("RGB"))
    return im


def psnr(mse):
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def quality(img, X, labels, C, blur=True):
    """PSNR of the quantized image (palette C, labels) and of its blurred version."""
    Q = C[labels]
    mse = float(((Q - X) ** 2).mean())
    out = {"psnr": psnr(mse)}
    if blur:
        h, w, _ = img.shape
        qb = np.stack([gaussian_filter(Q[:, c].reshape(h, w), 1.0) for c in range(3)], -1)
        out["psnr_blur"] = psnr(float(((qb - img_blur(img)) ** 2).mean()))
    return out


_BLUR = {}


def img_blur(img):
    key = id(img)
    if key not in _BLUR:
        _BLUR.clear()
        _BLUR[key] = np.stack([gaussian_filter(img[..., c].astype(np.float64), 1.0) for c in range(3)], -1)
    return _BLUR[key]


def palette(X, labels, C):
    return compute_centers(X, labels, C)[0]


def snapshot(name, img, X, eng, mon, t, secs, ref_labels_fn, extra=None, settle_rule=None, regime="drift",
             rng=None):
    """Labels at iteration t (U settled when asked), centroid palette, quality."""
    L = eng.labels.copy()
    labels = L
    U = np.empty(0, dtype=np.int64)
    row = dict(policy=name, T=t, n_dist=int(eng.n_dist), seconds=secs)
    pi_dither = None
    if settle_rule is not None:
        U = mon.unstable().astype(np.int64)
        if len(U):
            Hw = mon.window(U)
            lab_U, S, P, osc = _rules.settle(Hw, L[U], rule=settle_rule, n_clusters=eng.centers.shape[0],
                                             regime=regime)
            labels = L.copy()
            labels[U] = lab_U
            row["frac_fractional"] = float((P.max(1) < 1 - 1e-9).mean())
            # continuous reading: draw each unstable pixel from its soft membership
            cum = np.cumsum(P, 1)
            r = rng.random(len(U))[:, None]
            pick = np.minimum((r > cum).sum(1), S.shape[1] - 1)
            drawn = S[np.arange(len(U)), pick]
            pi_dither = (U, np.where(drawn < 0, labels[U], drawn))
    C = palette(X, labels, eng.centers)
    row.update(quality(img, X, labels, C))
    row["n_U"] = len(U)
    if pi_dither is not None:
        lab_d = labels.copy()
        lab_d[pi_dither[0]] = pi_dither[1]
        q = quality(img, X, lab_d, C)
        row["psnr_pi_dither"], row["psnr_blur_pi_dither"] = q["psnr"], q["psnr_blur"]
    row["_labels"] = labels
    row.update(extra or {})
    return row


def run_image(i, k, out, seed=0):
    img = load(i).astype(np.float64)
    h, w, _ = img.shape
    X = np.ascontiguousarray(img.reshape(-1, 3))
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(n, size=min(n, 200_000), replace=False)]
    C0, _ = kmeans_plusplus(sample, k, random_state=seed)
    inst = dict(image=i, height=h, width=w, n=n, k=k, seed=seed)
    print(f"[{time.strftime('%H:%M:%S')}] {inst}", flush=True)

    rows, curve = [], []
    eng = HamerlyEngine(X, C0)
    params = _rules.StopParams()
    mon = _rules.StateMonitor(n, k, params, exact=True, rule="auto")
    mon.push(eng.labels, eng.centers)
    var_tol = 1e-4 * float(X.var(0).mean())
    done = {"auto": False, "v0.1.0": False, "sklearn_tol": False}
    secs, t = 0.0, 0
    while t < MAX_ITER:
        t0 = time.perf_counter()
        C_prev = eng.centers.copy()
        changed = eng.step()
        mon.push(eng.labels, eng.centers, changed)
        secs += time.perf_counter() - t0
        t += 1
        stop, regime, state = mon.assess()
        if stop and not done["auto"]:
            done["auto"] = True
            reason = {"converged": "converged", "oscillation": "oscillation",
                      "drift": "few_unstable" if state.get("unstable_frac", 1) <= params.unstable_tol
                      else "stable_drift"}[regime]
            rows.append(snapshot("auto", img, X, eng, mon, t, secs, None, {"stop_reason": reason},
                                 settle_rule=None if regime == "converged" else "auto", regime=regime, rng=rng))
        u = state.get("unstable_frac")
        if not done["v0.1.0"] and (changed == 0 or (u is not None and u <= params.unstable_tol)):
            done["v0.1.0"] = True
            rows.append(snapshot("v0.1.0", img, X, eng, mon, t, secs, None,
                                 settle_rule=None if changed == 0 else "markov", rng=rng))
        if not done["sklearn_tol"] and float(((eng.centers - C_prev) ** 2).sum()) <= var_tol:
            done["sklearn_tol"] = True
            rows.append(snapshot("sklearn_tol", img, X, eng, mon, t, secs, None))
        if t in (25, 50):
            rows.append(snapshot(f"fixed_T{t}", img, X, eng, mon, t, secs, None))
        if t in CURVE:
            q = quality(img, X, eng.labels, palette(X, eng.labels, eng.centers), blur=False)
            curve.append(dict(T=t, n_dist=int(eng.n_dist), seconds=secs, psnr=q["psnr"], changed=changed))
        if changed == 0:
            break
    # policies that never triggered before convergence stop where the full run stops
    for name in ("auto", "v0.1.0", "sklearn_tol"):
        if not done[name]:
            rows.append(snapshot(name, img, X, eng, mon, t, secs, None, {"stop_reason": "converged"}))
    for T in (25, 50):
        if t < T:
            rows.append(snapshot(f"fixed_T{T}", img, X, eng, mon, t, secs, None))
    full = snapshot("full", img, X, eng, mon, t, secs, None, {"converged": changed == 0})
    rows.append(full)
    ref = full["_labels"]

    # subsampling practices: fit on a sample from the same init, one assignment pass
    for name, m, iters in (("sample_faiss", min(n, 256 * k), 25), ("sample_100k", min(n, 100_000), MAX_ITER)):
        t0 = time.perf_counter()
        Xs = np.ascontiguousarray(X[rng.choice(n, size=m, replace=False)])
        e = HamerlyEngine(Xs, C0)
        it = 0
        while it < iters:
            it += 1
            if e.step() == 0:
                break
        C = compute_centers(Xs, e.labels, e.centers)[0]
        labels = np.empty(n, dtype=np.int64)
        nearest_all(X, C, labels)
        sec = time.perf_counter() - t0
        row = dict(policy=name, T=it, n_dist=int(e.n_dist + n * k), seconds=sec, n_U=0, _labels=labels)
        row.update(quality(img, X, labels, C))
        rows.append(row)

    for r in rows:
        lab = r.pop("_labels")
        r["disagree_rate"] = float((lab != ref).mean())
        r["dist_frac"] = r["n_dist"] / full["n_dist"]
        r["time_frac"] = r["seconds"] / full["seconds"]
        r["d_psnr"] = r["psnr"] - full["psnr"]
        r["d_psnr_blur"] = r.get("psnr_blur", np.nan) - full["psnr_blur"]
        r["T_full"] = full["T"]
        write(out, {**inst, **r})
        print(f"  {r['policy']:12s} T={r['T']:4d} cost={r['dist_frac']:.3f}/{r['time_frac']:.3f} "
              f"psnr={r['psnr']:.3f} (d={r['d_psnr']:+.4f}) disagree={r['disagree_rate']:.5f} |U|={r.get('n_U', 0)}"
              + (f" pi_dither_blur={r['psnr_blur_pi_dither'] - r['psnr_blur']:+.5f}" if "psnr_blur_pi_dither" in r else ""),
              flush=True)
    for c in curve:
        write(out.with_name(out.stem + "_curve.csv"), {**inst, **c, "T_full": full["T"], "psnr_full": full["psnr"]})


FIELDS = ["image", "height", "width", "n", "k", "seed", "policy", "T", "T_full", "stop_reason", "converged", "n_U",
          "frac_fractional", "n_dist", "seconds", "dist_frac", "time_frac", "psnr", "psnr_blur", "d_psnr",
          "d_psnr_blur", "psnr_pi_dither", "psnr_blur_pi_dither", "disagree_rate"]


def write(path, row):
    new = not path.exists()
    fields = FIELDS if "policy" in row else list(row)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def parse_ids(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="1-100")
    p.add_argument("--k", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--out", default="color_quant.csv")
    p.add_argument("--download", action="store_true")
    a = p.parse_args(argv)
    ids = parse_ids(a.images)
    if a.download:
        download(ids)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / a.out
    for i in ids:
        for k in a.k:
            run_image(i, k, out)


if __name__ == "__main__":
    main()
