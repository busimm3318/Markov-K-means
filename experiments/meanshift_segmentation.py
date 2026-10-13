"""Does the Markov-K-means toolkit shorten mean-shift image segmentation (bilateral-filter weights)?

Mean-shift filtering (Comaniciu & Meer 2002) moves every pixel's joint spatial-range point
y = (x, y, r, g, b) to the mode of the kernel density of the image around it, y_{k+1} = m(y_k),
where m is the kernel-weighted mean over a window of the image. With the Gaussian product
kernel exp(-|dx|^2 / 2 hs^2 - |dc|^2 / 2 hr^2) one step is a bilateral-filter evaluation whose
centre also moves in space. The filtered image is the converged range part; segments are the
connected groups of neighbouring pixels whose converged colours lie within hr / 2.

Pixels are independent (the data are the fixed input image), so the cost is the number of window
evaluations (each ~ (2R + 1)^2 kernel values). What Markov-K-means brings, transferred here:

    plain          stop each pixel when its own shift |m(y) - y| < eps (hs, hr units), as usual
    extrapolate    every few steps fit the pixel's path as an AR(1) chain (the pseudo
                   representative: limit = y + d rho / (1 - rho), rho from the last two steps,
                   clipped to [0, rho_max]), jump there and keep it only if the kernel density
                   there is not below the current one (the SSE safeguard of K-means: a mean-shift
                   step never lowers the density); the jump costs one extra evaluation
    cap            the global stop: stop all pixels at the first iteration T at which at most
                   tau of the pixels are still moving (the monitor of the stragglers), with or
                   without a final extrapolation for the pixels still moving

Label histories (the other half of Markov-K-means) have no counterpart: a mean-shift path climbs
the density monotonically and does not oscillate.

Accuracy is measured against the same procedure run to a tight eps (the "converged" image) on
(a) the filtered colours and (b) the segmentation (adjusted Rand index; the floor is the index
between two converged references, eps 1e-3 and 1e-4).

Pre-registered criterion for "a significant time reduction": at least 25 % fewer window
evaluations than the cheapest plain setting that reaches the same segmentation quality (ARI within
0.01 of the reference floor), on at least two of the three images.

    python -m experiments.meanshift_segmentation           # -> experiments/results/meanshift_*.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

from experiments.sentinel2_tail import RESULTS

HS, HR = 5.0, 16.0                       # spatial bandwidth (pixels), range bandwidth (8-bit levels)
EPS_REF, EPS_REF2 = 1e-3, 1e-4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "meanshift_images.npz")


# ----------------------------------------------------------------------------- kernels

@njit(cache=True)
def _eval(img, R, ihs2, ihr2, p, m):
    """Kernel density (up to a constant) at the joint point p; writes the weighted mean into m."""
    H, W = img.shape[0], img.shape[1]
    ix, iy = int(p[0] + 0.5), int(p[1] + 0.5)
    x0, x1 = max(ix - R, 0), min(ix + R, W - 1)
    y0, y1 = max(iy - R, 0), min(iy + R, H - 1)
    sw = sx = sy = s0 = s1 = s2 = 0.0
    for qy in range(y0, y1 + 1):
        dy = qy - p[1]
        wy = dy * dy * ihs2
        for qx in range(x0, x1 + 1):
            dx = qx - p[0]
            a0 = img[qy, qx, 0]
            a1 = img[qy, qx, 1]
            a2 = img[qy, qx, 2]
            r0, r1, r2 = a0 - p[2], a1 - p[3], a2 - p[4]
            d = dx * dx * ihs2 + wy + (r0 * r0 + r1 * r1 + r2 * r2) * ihr2
            if d < 30.0:
                w = np.exp(-0.5 * d)
                sw += w
                sx += w * qx
                sy += w * qy
                s0 += w * a0
                s1 += w * a1
                s2 += w * a2
    if sw <= 1e-300:
        for j in range(5):
            m[j] = p[j]
        return 0.0
    m[0], m[1], m[2], m[3], m[4] = sx / sw, sy / sw, s0 / sw, s1 / sw, s2 / sw
    return sw


@njit(cache=True)
def _dot(a, b, ihs, ihr):
    return ((a[0] * b[0] + a[1] * b[1]) * ihs * ihs + (a[2] * b[2] + a[3] * b[3] + a[4] * b[4]) * ihr * ihr)


@njit(cache=True)
def _try_extrapolate(img, R, ihs, ihr, cur, h1, h2, dens, eps, rho_max, ye, me):
    """AR(1) limit of the path (h2, h1, cur); (accepted, density, evaluations spent)."""
    d1 = cur - h1
    d0 = h1 - h2
    den = _dot(d0, d0, ihs, ihr)
    if den <= 0.0:
        return False, dens, 0
    rho = min(max(_dot(d1, d0, ihs, ihr) / den, 0.0), rho_max)
    jump = d1 * (rho / (1.0 - rho))
    if np.sqrt(_dot(jump, jump, ihs, ihr)) <= eps:      # a plain step or two from converging: not worth it
        return False, dens, 0
    for j in range(5):
        ye[j] = cur[j] + jump[j]
    ye[0] = min(max(ye[0], 0.0), img.shape[1] - 1.0)
    ye[1] = min(max(ye[1], 0.0), img.shape[0] - 1.0)
    de = _eval(img, R, ihs * ihs, ihr * ihr, ye, me)
    return de >= dens, de, 1


@njit(parallel=True, cache=True)
def mean_shift(img, hs, hr, eps, max_iter, every, rho_max, final_extrap):
    """Converged joint points (H, W, 5), window evaluations and plain steps per pixel."""
    H, W = img.shape[0], img.shape[1]
    R = int(2.5 * hs + 0.5)
    ihs, ihr = 1.0 / hs, 1.0 / hr
    out = np.empty((H, W, 5))
    evals = np.zeros((H, W), np.int32)
    iters = np.zeros((H, W), np.int32)
    for pix in prange(H * W):
        i, j = pix // W, pix % W
        cur, nxt = np.empty(5), np.empty(5)
        h1, h2, ye, me = np.empty(5), np.empty(5), np.empty(5), np.empty(5)
        cur[0], cur[1] = j, i
        for c in range(3):
            cur[2 + c] = img[i, j, c]
        dens = _eval(img, R, ihs * ihs, ihr * ihr, cur, nxt)
        nev, k, nh = 1, 0, 0
        while True:
            dlt = nxt - cur
            if np.sqrt(_dot(dlt, dlt, ihs, ihr)) < eps:
                break
            if k >= max_iter:
                if final_extrap and nh == 2:
                    ok, de, spent = _try_extrapolate(img, R, ihs, ihr, cur, h1, h2, dens, eps, rho_max, ye, me)
                    nev += spent
                    if ok:
                        nxt[:] = me
                break
            h2[:] = h1
            h1[:] = cur
            nh = min(nh + 1, 2)
            cur[:] = nxt
            k += 1
            dens = _eval(img, R, ihs * ihs, ihr * ihr, cur, nxt)
            nev += 1
            if every > 0 and nh == 2 and k % every == 0:
                ok, de, spent = _try_extrapolate(img, R, ihs, ihr, cur, h1, h2, dens, eps, rho_max, ye, me)
                nev += spent
                if ok:
                    cur[:] = ye
                    nxt[:] = me
                    dens = de
                    nh = 0
        out[i, j, :] = nxt
        evals[i, j] = nev
        iters[i, j] = k
    return out, evals, iters


# ----------------------------------------------------------------------------- segmentation, metrics

def segment(out, hr=HR):
    """Connected groups of 4-neighbours whose converged colours differ by less than hr / 2."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    c = out[..., 2:5]
    H, W = c.shape[:2]
    idx = np.arange(H * W).reshape(H, W)
    near_h = np.sqrt(((c[:, 1:] - c[:, :-1]) ** 2).sum(-1)) < hr * 0.5
    near_v = np.sqrt(((c[1:] - c[:-1]) ** 2).sum(-1)) < hr * 0.5
    a = np.concatenate([idx[:, :-1][near_h], idx[:-1][near_v]])
    b = np.concatenate([idx[:, 1:][near_h], idx[1:][near_v]])
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(H * W, H * W))
    n, lab = connected_components(g, directed=False)
    return n, lab


def compare(out, ref_out, lab, ref_lab):
    from sklearn.metrics import adjusted_rand_score

    dc = np.sqrt(((out[..., 2:5] - ref_out[..., 2:5]) ** 2).sum(-1))
    dj = np.sqrt((((out[..., :2] - ref_out[..., :2]) / HS) ** 2).sum(-1) + ((dc / HR) ** 2))
    return dict(color_err_mean=float(dc.mean()), color_err_p99=float(np.percentile(dc, 99)),
                joint_err_p99=float(np.percentile(dj, 99)), share_over_1_level=float((dc > 1.0).mean()),
                ari=float(adjusted_rand_score(ref_lab, lab)))


# ----------------------------------------------------------------------------- images

def _window(url, row, col, size):
    import rasterio
    from rasterio.windows import Window

    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    chans = []
    for b in ("B04", "B03", "B02"):
        with rasterio.open("/vsicurl/" + url + b + ".tif") as src:
            chans.append(src.read(1, window=Window(col, row, size, size)).astype(np.float64))
    rgb = np.stack(chans, -1) / 10000.0
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / (hi - lo), 0, 1) * 255.0


def images(size=256):
    """{name: (size, size, 3) float64 in 0..255}: a natural photograph and two Sentinel-2 10 m windows."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if all(z[k].shape[0] >= size for k in z.files):
            return {k: z[k][:size, :size] for k in z.files}
    from sklearn.datasets import load_sample_images

    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    url = {s.split(" (")[0]: u for s, u in zip(scenes.site, scenes.url)}
    china = load_sample_images().images[0].astype(np.float64)
    r0, c0 = (china.shape[0] - 384) // 2, (china.shape[1] - 384) // 2
    out = {"photo": china[r0:r0 + 384, c0:c0 + 384]}
    out["seoul"] = _window(url["Seoul"], 4400, 3600, 384)
    out["iowa"] = _window(url["Iowa"], 4400, 4400, 384)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return {k: v[:size, :size] for k, v in out.items()}


# ----------------------------------------------------------------------------- the study

PLAIN_EPS = (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3)
ACCEL_EPS = (0.3, 0.1, 0.03, 0.01)
TAUS = (0.1, 0.03, 0.01, 3e-3)


def timed(img, eps, max_iter=5000, every=0, rho_max=0.95, final_extrap=False):
    t0 = time.perf_counter()
    out, ev, it = mean_shift(img, HS, HR, eps, max_iter, every, rho_max, final_extrap)
    return out, ev, it, time.perf_counter() - t0


def run(size=256, names=None):
    imgs = images(size)
    mean_shift(imgs["photo"][:16, :16].copy(), HS, HR, 0.1, 50, 3, 0.95, True)         # compile
    rows, tails = [], []
    for name, img in imgs.items():
        if names and name not in names:
            continue
        img = np.ascontiguousarray(img)
        ref, ev_ref, it_ref, s_ref = timed(img, EPS_REF)
        ref2, _, _, _ = timed(img, EPS_REF2)
        n_ref, lab_ref = segment(ref)
        floor = compare(ref2, ref, segment(ref2)[1], lab_ref)["ari"]
        n_pix = img.shape[0] * img.shape[1]
        tails.append(dict(image=name, pixels=n_pix, mean_iters=float(it_ref.mean()), p50=float(np.median(it_ref)),
                          p90=float(np.percentile(it_ref, 90)), p99=float(np.percentile(it_ref, 99)),
                          max=int(it_ref.max()), over10=float((it_ref > 10).mean()), over30=float((it_ref > 30).mean()),
                          over100=float((it_ref > 100).mean()), evals_ref=int(ev_ref.sum()), seconds_ref=s_ref,
                          segments_ref=int(n_ref), ari_floor=floor))
        print(tails[-1], flush=True)

        def record(method, out, ev, it, secs, **kw):
            n, lab = segment(out)
            rows.append(dict(image=name, method=method, evals=int(ev.sum()), evals_vs_ref=float(ev.sum() / ev_ref.sum()),
                             seconds=secs, mean_iters=float(it.mean()), segments=int(n), ari_floor=floor,
                             **kw, **compare(out, ref, lab, lab_ref)))
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

        for eps in PLAIN_EPS:
            record("plain", *timed(img, eps), eps=eps)
        for eps in ACCEL_EPS:
            for rho_max in (0.8, 0.95):
                record("extrapolate", *timed(img, eps, every=3, rho_max=rho_max), eps=eps, rho_max=rho_max, every=3)
        for tau in TAUS:
            cap = int(np.argmax([(it_ref > T).mean() <= tau for T in range(int(it_ref.max()) + 2)]))
            record("cap", *timed(img, EPS_REF, max_iter=cap), eps=EPS_REF, tau=tau, cap=cap)
            record("cap+extrapolate", *timed(img, EPS_REF, max_iter=cap, rho_max=0.95, final_extrap=True),
                   eps=EPS_REF, tau=tau, cap=cap, rho_max=0.95)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "meanshift_runs.csv"), index=False)
        pd.DataFrame(tails).to_csv(os.path.join(RESULTS, "meanshift_tails.csv"), index=False)
    return pd.DataFrame(rows), pd.DataFrame(tails)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 256, sys.argv[2:] or None)
