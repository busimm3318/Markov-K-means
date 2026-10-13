"""Application study 4: mean-shift image segmentation built on the bilateral filter.

Mean-shift segmentation (Comaniciu & Meer 2002) moves the point (row, col, L, u, v) of every
pixel uphill on the joint spatial-range density. One step replaces the point by the average of
the pixels in its spatial window, weighted by a spatial kernel and by a range kernel on the
colour difference: the bilateral-filter weights, evaluated at the moving point (Barash &
Comaniciu 2004). When the points have stopped, 8-neighbours whose colours lie within hr / 2 are
joined and regions smaller than MIN_SIZE pixels are merged into the adjacent region of closest
mean colour (as in EDISON).

Two forms of the iteration:

    ms    mean-shift filtering. The image stays fixed and each pixel's point climbs to its mode
          on its own. A point whose normalised shift sqrt(|dx|^2 / hs^2 + |dc|^2 / hr^2) falls
          below EPS_REF has converged and leaves the active set, so an iteration costs only the
          points still moving.
    ibf   iterated bilateral filter (the blurring form, positions fixed). The image itself is
          filtered again and again, so every pass recomputes every pixel; converged when no
          colour changes by more than EPS_REF * hr.

Kernels: "flat" (uniform spatial disk of radius hs and range ball of radius hr, as EDISON and
OpenCV use) and "gauss" (Gaussians with sigma = h / 2, truncated at h). Colours are CIE L*u*v*.

One reference run per image and setting goes on until convergence (or MAX_ITER). Every stopping
rule is evaluated on the same iterates:

    full          convergence: the reference segmentation
    edison        ms: a point stops once its normalised shift is below 0.1, at most 100
                  iterations (EDISON: squared normalised shift < 0.01, LIMIT 100)
    eps_<e>       ms: a point stops once its normalised shift is below e (no cap);
                  ibf: the run stops once no colour moved more than e * hr in a pass
    eps_<e>_ar1   ms: the same per-point stop, the point replaced by the long-run mean of its
                  last W + 1 positions read as an AR(1) chain (the pseudo representative of one
                  point)
    T<t>          a fixed number of iterations (OpenCV pyrMeanShiftFiltering: 5)
    mk_stop       the Markov-K-means monitor's stop, the points kept where they are
    mk            Markov-K-means: the same stop, then
                  1. pseudo representatives: the path of every point still moving over the
                     window, fitted as an AR(1) chain (markov_kmeans._rules.long_run_centers,
                     coordinates scaled by (hs, hr), rho clipped at RHO_MAX), replaced by its
                     long-run mean; safeguard (the analogue of the SSE test): ms keeps a
                     point's pseudo representative only where the kernel density is not lower
                     than at its current point; ibf keeps the extrapolated image only if its
                     fixed-point residual |BF(I) - I|^2 is not larger;
                  2. label chains: the undecided pixels (segment changed within the window)
                     are settled by markov_kmeans._rules.settle (regime-aware); a pixel the
                     settlement moves takes the mean colour of its settled segment.

The monitor (SegmentMonitor) is markov_kmeans._rules.StateMonitor's judgement with the
segments as clusters. Labels are the segments of the current colours (8-neighbours within hr /
2), carried from one iteration to the next by largest overlap, so a pixel "changes cluster"
when its segment changes. Representatives are the mean colours of the segments of at least
MIN_SIZE pixels, their drift measured in units of hr / 2 (the joining distance, the analogue of
half the distance to the nearest other center); partitions are those segments' sizes and
memberships. The movement, representative and partition tests and every threshold are the
package defaults.

Error: the share of pixels whose segment differs from the reference segmentation, segments
matched one to one greedily by largest overlap. Time: the measured filtering time (pixels still
active, 4 threads) plus the segmentation step (the same for every rule), and for mk the
monitor and the finishing step; mk_free leaves out the monitor's cost (a lower bound for any
monitor). The full-pass cost model counts iterations: the time of an implementation that
recomputes every pixel at every iteration.

    python -m experiments.meanshift_seg download
    python -m experiments.meanshift_seg run --form ms --kernel flat gauss --setting fine --images 0:200:2
    python -m experiments.meanshift_seg run --form ibf --kernel flat gauss --setting fine --images 0:200:10
    python -m experiments.meanshift_seg report
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from markov_kmeans import _rules

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bsds500"
RESULTS = ROOT / "experiments" / "results"
BSDS = "https://github.com/BIDS/BSDS500"

SETTINGS = {"fine": (7.0, 6.5), "coarse": (16.0, 19.0)}     # (hs, hr): EDISON's default, and a coarse one
MIN_SIZE = 20
EPS_REF = 1e-3
MAX_ITER = {"ms": 500, "ibf": 2000}
RHO_MAX = 0.95
EPS = {"ms": (0.1, 0.03, 0.01, 0.003), "ibf": (0.1, 0.03, 0.01, 0.003)}
EPS_AR = (0.03, 0.01, 0.003)
TS = {"ms": (5, 10, 20, 50, 100, 200), "ibf": (5, 10, 20, 50, 100, 200, 500, 1000)}
CHECK = sorted(set(range(1, 31)) | set(range(35, 101, 5)) | set(range(110, 501, 10)) | set(range(550, 2001, 50)))
PARAMS = _rules.StopParams()


# ----------------------------------------------------------------------------- data

def download():
    """BSDS500 test images (200 JPEGs, 481 x 321) from the BIDS mirror, by a sparse clone."""
    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "bsds")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", BSDS, repo], check=True)
        subprocess.run(["git", "-C", repo, "sparse-checkout", "set", "BSDS500/data/images/test"], check=True)
        src = Path(repo) / "BSDS500" / "data" / "images" / "test"
        for f in sorted(src.glob("*.jpg")):
            shutil.copy(f, DATA / f.name)
    print(f"{len(list(DATA.glob('*.jpg')))} images in {DATA}")


def image_ids():
    return sorted(f.stem for f in DATA.glob("*.jpg"))


def load(name):
    from PIL import Image
    from skimage.color import rgb2luv

    rgb = np.asarray(Image.open(DATA / f"{name}.jpg").convert("RGB"))
    return np.ascontiguousarray(rgb2luv(rgb), dtype=np.float64)


# ----------------------------------------------------------------------------- kernels

@njit(cache=True, inline="always")
def _weight(ds2, dr2, gauss):
    return np.exp(-2.0 * (ds2 + dr2)) if gauss else 1.0


@njit(cache=True, parallel=True)
def ms_step(img, Y, act, hs, hr, gauss, out, shift):
    """One mean-shift step of the points Y[act] (row, col, L, u, v) on the fixed image."""
    H, W, _ = img.shape
    for a in prange(act.shape[0]):
        p = act[a]
        cy, cx, c0, c1, c2 = Y[p, 0], Y[p, 1], Y[p, 2], Y[p, 3], Y[p, 4]
        sw = sy = sx = s0 = s1 = s2 = 0.0
        for i in range(max(0, int(np.ceil(cy - hs))), min(H - 1, int(np.floor(cy + hs))) + 1):
            dy = (i - cy) / hs
            for j in range(max(0, int(np.ceil(cx - hs))), min(W - 1, int(np.floor(cx + hs))) + 1):
                dx = (j - cx) / hs
                ds2 = dy * dy + dx * dx
                if ds2 > 1.0:
                    continue
                d0, d1, d2 = img[i, j, 0] - c0, img[i, j, 1] - c1, img[i, j, 2] - c2
                dr2 = (d0 * d0 + d1 * d1 + d2 * d2) / (hr * hr)
                if dr2 > 1.0:
                    continue
                w = _weight(ds2, dr2, gauss)
                sw += w
                sy += w * i
                sx += w * j
                s0 += w * img[i, j, 0]
                s1 += w * img[i, j, 1]
                s2 += w * img[i, j, 2]
        if sw > 0.0:
            out[a, 0], out[a, 1], out[a, 2], out[a, 3], out[a, 4] = sy / sw, sx / sw, s0 / sw, s1 / sw, s2 / sw
        else:
            out[a, 0], out[a, 1], out[a, 2], out[a, 3], out[a, 4] = cy, cx, c0, c1, c2
        shift[a] = np.sqrt(((out[a, 0] - cy) ** 2 + (out[a, 1] - cx) ** 2) / (hs * hs)
                           + ((out[a, 2] - c0) ** 2 + (out[a, 3] - c1) ** 2 + (out[a, 4] - c2) ** 2) / (hr * hr))


@njit(cache=True, parallel=True)
def kde(img, P, hs, hr, gauss):
    """Unnormalised kernel density of the image's joint-domain points at the points P (m, 5)."""
    H, W, _ = img.shape
    out = np.empty(P.shape[0])
    for a in prange(P.shape[0]):
        cy, cx, c0, c1, c2 = P[a, 0], P[a, 1], P[a, 2], P[a, 3], P[a, 4]
        s = 0.0
        for i in range(max(0, int(np.ceil(cy - hs))), min(H - 1, int(np.floor(cy + hs))) + 1):
            dy = (i - cy) / hs
            for j in range(max(0, int(np.ceil(cx - hs))), min(W - 1, int(np.floor(cx + hs))) + 1):
                dx = (j - cx) / hs
                ds2 = dy * dy + dx * dx
                if ds2 > 1.0:
                    continue
                d0, d1, d2 = img[i, j, 0] - c0, img[i, j, 1] - c1, img[i, j, 2] - c2
                dr2 = (d0 * d0 + d1 * d1 + d2 * d2) / (hr * hr)
                if dr2 > 1.0:
                    continue
                if gauss:   # the density of the Gaussian kernel; the flat kernel's shadow is Epanechnikov
                    s += np.exp(-2.0 * (ds2 + dr2))
                else:
                    s += (1.0 - ds2) * (1.0 - dr2)
        out[a] = s
    return out


@njit(cache=True, parallel=True)
def bf_pass(src, dst, hs, hr, gauss, shift):
    """One bilateral filter pass of the whole image (positions fixed); shift in units of hr."""
    H, W, _ = src.shape
    r = int(np.floor(hs))
    for p in prange(H * W):
        y, x = p // W, p % W
        c0, c1, c2 = src[y, x, 0], src[y, x, 1], src[y, x, 2]
        sw = s0 = s1 = s2 = 0.0
        for i in range(max(0, y - r), min(H - 1, y + r) + 1):
            dy = (i - y) / hs
            for j in range(max(0, x - r), min(W - 1, x + r) + 1):
                dx = (j - x) / hs
                ds2 = dy * dy + dx * dx
                if ds2 > 1.0:
                    continue
                d0, d1, d2 = src[i, j, 0] - c0, src[i, j, 1] - c1, src[i, j, 2] - c2
                dr2 = (d0 * d0 + d1 * d1 + d2 * d2) / (hr * hr)
                if dr2 > 1.0:
                    continue
                w = _weight(ds2, dr2, gauss)
                sw += w
                s0 += w * src[i, j, 0]
                s1 += w * src[i, j, 1]
                s2 += w * src[i, j, 2]
        dst[y, x, 0], dst[y, x, 1], dst[y, x, 2] = s0 / sw, s1 / sw, s2 / sw
        shift[p] = np.sqrt((s0 / sw - c0) ** 2 + (s1 / sw - c1) ** 2 + (s2 / sw - c2) ** 2) / hr


@njit(cache=True, parallel=True)
def ar1_limits(ring, rows, idx, scale, rho_max):
    """Long-run mean of each point's AR(1) chain over the ring rows (oldest first).

    The formula of markov_kmeans._rules.long_run_centers (one rho per point, clipped to
    [0, rho_max], least squares over all coordinates scaled by ``scale``), compiled so that
    it costs little when applied to the points stopping at every iteration.
    """
    m, d = rows.shape[0] - 1, ring.shape[2]
    out = np.empty((idx.shape[0], d))
    for a in prange(idx.shape[0]):
        p = idx[a]
        mx = np.zeros(d)
        my = np.zeros(d)
        for s in range(m):
            for c in range(d):
                mx[c] += ring[rows[s], p, c] / scale[c]
                my[c] += ring[rows[s + 1], p, c] / scale[c]
        num = den = 0.0
        for c in range(d):
            mx[c] /= m
            my[c] /= m
        for s in range(m):
            for c in range(d):
                u = ring[rows[s], p, c] / scale[c] - mx[c]
                num += u * (ring[rows[s + 1], p, c] / scale[c] - my[c])
                den += u * u
        rho = num / den if den > 0.0 else 0.0
        rho = min(max(rho, 0.0), rho_max)
        for c in range(d):
            out[a, c] = (my[c] - rho * mx[c]) / (1.0 - rho) * scale[c]
    return out


# ----------------------------------------------------------------------------- segmentation

@njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(cache=True)
def _union(par, a, b):
    a, b = _find(par, a), _find(par, b)
    if a != b:
        if a < b:
            par[b] = a
        else:
            par[a] = b


@njit(cache=True)
def components(R, H, W, thr2):
    """8-neighbour components of the colours R (n, 3) joined below squared distance thr2.

    Returns the root of every pixel: the smallest pixel index of its component.
    """
    n = H * W
    par = np.arange(n)
    for y in range(H):
        for x in range(W):
            p = y * W + x
            for k in range(4):
                yy, xx = (y, x + 1) if k == 0 else (y + 1, x - 1 + (k - 1))
                if yy >= H or xx < 0 or xx >= W:
                    continue
                q = yy * W + xx
                d = (R[p, 0] - R[q, 0]) ** 2 + (R[p, 1] - R[q, 1]) ** 2 + (R[p, 2] - R[q, 2]) ** 2
                if d < thr2:
                    _union(par, p, q)
    out = np.empty(n, dtype=np.int64)
    for p in range(n):
        out[p] = _find(par, p)
    return out


@njit(cache=True)
def _compact(lab):
    m = np.full(lab.shape[0], -1, dtype=np.int64)
    out = np.empty(lab.shape[0], dtype=np.int64)
    r = 0
    for p in range(lab.shape[0]):
        if m[lab[p]] < 0:
            m[lab[p]] = r
            r += 1
        out[p] = m[lab[p]]
    return out, r


@njit(cache=True)
def _prune(lab, nreg, R, H, W, min_size):
    """Merge every region below min_size into the 4-adjacent region of closest mean colour, until none is left."""
    n = H * W
    while True:
        size = np.zeros(nreg, dtype=np.int64)
        mean = np.zeros((nreg, 3))
        for p in range(n):
            size[lab[p]] += 1
            for c in range(3):
                mean[lab[p], c] += R[p, c]
        for r in range(nreg):
            for c in range(3):
                mean[r, c] /= size[r]
        best = np.full(nreg, -1, dtype=np.int64)
        bestd = np.full(nreg, np.inf)
        small = False
        for y in range(H):
            for x in range(W):
                a = lab[y * W + x]
                for k in range(2):
                    yy, xx = (y, x + 1) if k == 0 else (y + 1, x)
                    if yy >= H or xx >= W:
                        continue
                    b = lab[yy * W + xx]
                    if a == b:
                        continue
                    d = (mean[a, 0] - mean[b, 0]) ** 2 + (mean[a, 1] - mean[b, 1]) ** 2 + (mean[a, 2] - mean[b, 2]) ** 2
                    for u, v in ((a, b), (b, a)):
                        if size[u] < min_size:
                            small = True
                            if d < bestd[u] or (d == bestd[u] and v < best[u]):
                                bestd[u] = d
                                best[u] = v
        if not small:
            return lab, nreg
        par = np.arange(nreg)
        for r in range(nreg):
            if size[r] < min_size and best[r] >= 0:
                _union(par, r, best[r])
        for p in range(n):
            lab[p] = _find(par, lab[p])
        lab, nreg = _compact(lab)


def segment(R, H, W, hr):
    """Segments of the filtered colours R (n, 3): join within hr / 2, then prune regions below MIN_SIZE."""
    R = np.ascontiguousarray(R)
    lab, nreg = _compact(components(R, H, W, (hr / 2.0) ** 2))
    lab, nreg = _prune(lab, nreg, R, H, W, MIN_SIZE)
    return lab.astype(np.int32), int(nreg)


def differing(a, b):
    """Share of pixels whose segment differs: segments matched one to one, greedily by largest overlap."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    nb = int(b.max()) + 1
    key, count = np.unique(a * nb + b, return_counts=True)
    order = np.argsort(-count, kind="stable")
    used_a, used_b = set(), set()
    matched = 0
    for kk, c in zip(key[order], count[order]):
        i, j = int(kk // nb), int(kk % nb)
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += int(c)
    return 1.0 - matched / a.shape[0]


# ----------------------------------------------------------------------------- the monitor

@njit(cache=True)
def _track(prev, roots, next_id):
    """Carry segment ids over one iteration.

    Each new segment (pixels sharing a root) takes the previous id held by most of its pixels
    (majority vote), unless another new segment holds more of that previous segment; the
    others get fresh ids. Returns (ids, next_id).
    """
    n = roots.shape[0]
    cand = np.full(n, -1, dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    for p in range(n):
        r, v = roots[p], prev[p]
        if cnt[r] == 0:
            cand[r] = v
            cnt[r] = 1
        elif cand[r] == v:
            cnt[r] += 1
        else:
            cnt[r] -= 1
    ov = np.zeros(n, dtype=np.int64)
    for p in range(n):
        if prev[p] == cand[roots[p]]:
            ov[roots[p]] += 1
    best_ov = np.zeros(next_id, dtype=np.int64)
    best_r = np.full(next_id, -1, dtype=np.int64)
    for p in range(n):
        if roots[p] == p and ov[p] > best_ov[cand[p]]:
            best_ov[cand[p]] = ov[p]
            best_r[cand[p]] = p
    new = np.empty(n, dtype=np.int64)
    for p in range(n):
        if roots[p] == p:
            if best_r[cand[p]] == p:
                new[p] = cand[p]
            else:
                new[p] = next_id
                next_id += 1
    ids = np.empty(n, dtype=np.int64)
    for p in range(n):
        ids[p] = new[roots[p]]
    return ids, next_id


class SegmentMonitor:
    """markov_kmeans' StateMonitor judgement for a segmentation run: the clusters are the segments.

    ``push(colours)`` labels the pixels by the segments of their current colours (tracked ids),
    records the label ring buffer, the movers, and the mean colours of the large segments;
    ``assess(moving)`` returns (stop, regime, state) with the decision logic of
    ``StateMonitor.assess`` and the package's default thresholds.
    """

    def __init__(self, H, W, hr, params=PARAMS, min_size=MIN_SIZE):
        self.H_, self.W_, self.n = H, W, H * W
        self.thr2 = (hr / 2.0) ** 2
        self.scale = hr / 2.0
        self.p = params
        self.min_size = min_size
        Wn = params.window
        self.ring = np.empty((Wn + 1, self.n), dtype=np.int64)
        self.last_change = np.full(self.n, -1, dtype=np.int64)
        self.means = [None] * (Wn + 1)       # (ids, mean colours) of the large segments
        self.counts, self.drift, self.trace = [], [], []
        self.t = -1
        self.next_id = 0
        self.ids = None

    def push(self, R):
        Wn = self.p.window
        self.t += 1
        t = self.t
        roots = components(np.ascontiguousarray(R), self.H_, self.W_, self.thr2)
        if t == 0:
            ids, self.next_id = _compact(roots)
        else:
            ids, self.next_id = _track(self.ids, roots, self.next_id)
            moved = np.flatnonzero(ids != self.ids)
            self.last_change[moved] = t
            self.counts.append(len(moved))
        self.ids = ids
        self.ring[t % (Wn + 1)] = ids
        size = np.bincount(ids, minlength=self.next_id)
        big = np.flatnonzero(size >= self.min_size)
        M = np.stack([np.bincount(ids, weights=R[:, c], minlength=self.next_id)[big] for c in range(3)], 1)
        M /= size[big][:, None]
        self.means[t % (Wn + 1)] = (big, M)
        if t > 0:
            self.drift.append(self._move(self.means[(t - 1) % (Wn + 1)], (big, M)))

    def _move(self, old, new):
        """Largest move of a large segment's mean colour between two records, in units of hr / 2."""
        (ia, Ma), (ib, Mb) = old, new
        common, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        if len(common) == 0:
            return 0.0
        return float(np.sqrt(((Ma[xa] - Mb[xb]) ** 2).sum(1)).max() / self.scale)

    def travel(self):
        Wn, t = self.p.window, self.t
        if t < Wn or Wn < 2:
            return float("inf")
        step = float(np.mean(self.drift[-Wn:]))
        if step <= 0:
            return 0.0
        return self._move(self.means[(t - Wn) % (Wn + 1)], self.means[t % (Wn + 1)]) / step

    def unstable(self):
        return np.flatnonzero(self.last_change > self.t - self.p.window)

    def window(self, idx):
        Wn, t = self.p.window, self.t
        rows = np.array([s % (Wn + 1) for s in range(max(0, t - Wn), t + 1)])
        return self.ring[rows[:, None], np.asarray(idx)[None, :]]

    def assess(self, moving):
        p, t, Wn = self.p, self.t, self.p.window
        state = {"t": t, "changed": self.counts[-1] if self.counts else None}
        if moving == 0:
            return self._log(True, "converged", state)
        if t < max(p.min_iter, Wn):
            return self._log(False, "running", state)
        U = self.unstable()
        u = len(U) / self.n
        state["unstable_frac"] = u
        if u <= p.unstable_tol:
            return self._log(True, "drift", state)
        if u > p.max_unstable:
            return self._log(False, "running", state)
        # representatives: mean colours of the large segments
        d = self.drift[-1]
        d_mean = float(np.mean(self.drift[-Wn:]))
        travel = self.travel()
        reps = d_mean <= p.center_tol or (_rules.plateau(self.drift, Wn, p.plateau_ratio)
                                          and d <= p.center_noise_tol and travel <= p.travel_ratio)
        state.update(center_drift=d, center_drift_window=d_mean, center_travel=travel, representatives_stable=bool(reps))
        if not reps:
            return self._log(False, "running", state)
        # partitions: the large segments
        cur = self.ids
        size = np.bincount(cur, minlength=self.next_id)
        old = np.bincount(self.ring[(t - Wn) % (Wn + 1)], minlength=self.next_id)
        big = size >= self.min_size
        cl_unstable = float((np.bincount(cur[U], minlength=self.next_id)[big] / size[big]).max()) if big.any() else 0.0
        size_change = float((np.abs(size[big] - old[big]) / np.maximum(old[big], 1)).max()) if big.any() else 0.0
        Hw = self.window(U)
        flow = _rules.net_flow(Hw, self.next_id) if len(U) else 0.0
        parts = size_change <= p.size_tol and (cl_unstable <= p.cluster_tol or flow <= p.flow_tol)
        state.update(cluster_unstable=cl_unstable, size_change=size_change, net_flow=flow, partitions_stable=bool(parts))
        if not parts:
            return self._log(False, "running", state)
        # movement
        osc = float((_rules.switches(Hw) >= 2).mean()) if len(U) else 0.0
        rho = _rules.projected_changes(self.counts, Wn, self.n)
        flat = _rules.plateau(self.counts, Wn, p.plateau_ratio)
        state.update(oscillating_share=osc, projected_changes=rho, counts_plateau=bool(flat))
        if osc >= p.osc_share and flat:
            return self._log(True, "oscillation", state)
        if rho <= p.change_tol:
            return self._log(True, "drift", state)
        return self._log(False, "running", state)

    def _log(self, stop, regime, state):
        state["decision"] = regime if stop else "continue"
        self.trace.append(state)
        return stop, regime, state


def settle_colours(mon, regime, colours):
    """Label chains: settle the undecided pixels; a pixel moved to another segment takes its mean colour.

    Returns (pixels moved, unstable share, oscillating share).
    """
    U = mon.unstable()
    if len(U) == 0:
        return 0, 0.0, 0.0
    Hw = mon.window(U)
    cur = mon.ids[U]
    lab, _, _, osc = _rules.settle(Hw, cur, rule="auto", regime=regime)
    size = np.bincount(mon.ids, minlength=mon.next_id)
    move = np.flatnonzero((lab != cur) & (lab < mon.next_id))
    move = move[size[lab[move]] > 0]           # the settled segment must still exist
    if len(move):
        tgt = lab[move]
        for c in range(3):
            m = np.bincount(mon.ids, weights=colours[:, c], minlength=mon.next_id)
            colours[U[move], c] = m[tgt] / size[tgt]
    return len(move), len(U) / mon.n, float(osc.mean())


def _rows(t, Wn):
    return np.array([s % (Wn + 1) for s in range(max(0, t - Wn), t + 1)])


def _decision(mon):
    tr = mon.trace[-1]
    if tr["decision"] == "drift":
        return "few_unstable" if tr.get("unstable_frac", 1.0) <= mon.p.unstable_tol else "stable_drift"
    return tr["decision"]


# ----------------------------------------------------------------------------- one run of each form

def run_ms(img, hs, hr, kernel):
    """Reference mean-shift run with every rule evaluated on its iterates. Returns (rules, curve, info)."""
    H, W, _ = img.shape
    n, Wn, gauss = H * W, PARAMS.window, kernel == "gauss"
    yy, xx = np.mgrid[0:H, 0:W]
    Y = np.column_stack([yy.ravel(), xx.ravel(), img.reshape(n, 3)]).astype(np.float64)
    scale = np.array([hs, hs, hr, hr, hr])
    ring = np.empty((Wn + 1, n, 5))
    ring[0] = Y
    act = np.arange(n)
    it_ref = np.full(n, MAX_ITER["ms"], dtype=np.int64)
    eps = EPS["ms"]
    done = {e: np.zeros(n, dtype=bool) for e in eps}
    it_e = {e: np.full(n, MAX_ITER["ms"], dtype=np.int64) for e in eps}
    frozen = {e: Y.copy() for e in eps}
    frozen_ar = {e: Y.copy() for e in EPS_AR}
    ar_s = {e: 0.0 for e in EPS_AR}
    Y100 = None
    step_s, mon_s, moving_n = [0.0], [0.0], [n]
    labels_at = {}
    mk = None
    t0 = time.perf_counter()
    mon = SegmentMonitor(H, W, hr)
    mon.push(Y[:, 2:])
    mon_s[0] = time.perf_counter() - t0
    for t in range(1, MAX_ITER["ms"] + 1):
        out = np.empty((len(act), 5))
        sh = np.empty(len(act))
        t0 = time.perf_counter()
        ms_step(img, Y, act, hs, hr, gauss, out, sh)
        Y[act] = out
        step_s.append(time.perf_counter() - t0)
        moving_n.append(len(act))
        ring[t % (Wn + 1)] = Y
        for e in eps:
            new = act[(sh < e) & ~done[e][act]]
            if len(new) == 0:
                continue
            done[e][new] = True
            it_e[e][new] = t
            frozen[e][new] = Y[new]
            if e in EPS_AR:
                ta = time.perf_counter()
                if t >= 2:
                    frozen_ar[e][new] = ar1_limits(ring, _rows(t, Wn), new, scale, RHO_MAX)
                else:
                    frozen_ar[e][new] = Y[new]
                ar_s[e] += time.perf_counter() - ta
        if t == 100:
            Y100 = Y.copy()
        conv = sh < EPS_REF
        it_ref[act[conv]] = t
        act = act[~conv]
        t0 = time.perf_counter()
        mon.push(Y[:, 2:])
        if mk is None:
            stop, regime, _ = mon.assess(len(act))
        mon_s.append(time.perf_counter() - t0)
        if mk is None and stop:
            mk = finish_ms(img, Y, ring, t, act, mon, regime, hs, hr, gauss, scale)
        if t in CHECK or len(act) == 0:
            labels_at[t] = segment(Y[:, 2:], H, W, hr)[0]
        if len(act) == 0:
            break
    T = t
    if mk is None:              # MAX_ITER reached before the monitor stopped
        mon.trace[-1]["decision"] = "max_iter"
        mk = finish_ms(img, Y, ring, T, act, mon, "drift", hs, hr, gauss, scale)
    for e in eps:               # points that never got below e before MAX_ITER stop where they are
        frozen[e][~done[e]] = Y[~done[e]]
        if e in EPS_AR:
            frozen_ar[e][~done[e]] = Y[~done[e]]
    t0 = time.perf_counter()
    ref, nref = segment(Y[:, 2:], H, W, hr)
    seg_s = time.perf_counter() - t0
    step_s, mon_s, moving_n = np.array(step_s), np.array(mon_s), np.array(moving_n)
    full_s = step_s.sum()

    def active_time(its):
        """Filtering time of a rule that stops point p after its[p] iterations (same iterates, fewer points)."""
        a = np.bincount(np.minimum(its, T), minlength=T + 1)
        alive = n - np.concatenate([[0], np.cumsum(a)[:-1]])       # points still computed at iteration t
        alive = alive[1:T + 1]
        return float((step_s[1:T + 1] * alive / np.maximum(moving_n[1:T + 1], 1)).sum())

    rules = []

    def add(rule, labels, filt_s, iters, extra=None):
        rules.append(dict(rule=rule, iterations=int(iters), seconds=filt_s + seg_s, time_frac=(filt_s + seg_s) / (full_s + seg_s),
                          passes_frac=iters / T, error=differing(labels, ref), segments=int(labels.max()) + 1,
                          **(extra or {})))

    add("full", ref, full_s, T)
    ed = np.minimum(it_e[0.1], 100)
    Yed = np.where((it_e[0.1] <= 100)[:, None], frozen[0.1], Y100 if Y100 is not None else Y)
    add("edison", segment(Yed[:, 2:], H, W, hr)[0], active_time(ed), ed.max())
    for e in eps:
        its = np.minimum(it_e[e], it_ref)
        add(f"eps_{e:g}", segment(frozen[e][:, 2:], H, W, hr)[0], active_time(its), its.max())
        if e in EPS_AR:
            add(f"eps_{e:g}_ar1", segment(frozen_ar[e][:, 2:], H, W, hr)[0], active_time(its) + ar_s[e], its.max())
    for tt in TS["ms"]:
        tt_ = min(tt, T)
        add(f"T{tt}", labels_at[tt_] if tt_ in labels_at else ref, float(step_s[1:tt_ + 1].sum()), tt_)
    info = dict(n=n, T=T, full_s=full_s, seg_s=seg_s, segments_ref=nref, monitor_s=float(mon_s.sum()),
                converged=bool(len(act) == 0))
    if mk is not None:
        ts = mk["t"]
        filt = float(step_s[1:ts + 1].sum())
        mon_t = float(mon_s[:ts + 1].sum())
        add("mk_stop", segment(mk["Y_stop"][:, 2:], H, W, hr)[0], filt + mon_t, ts)
        lab_mk = segment(mk["Y"][:, 2:], H, W, hr)[0]
        add("mk", lab_mk, filt + mon_t + mk["seconds"], ts)
        add("mk_free", lab_mk, filt + mk["seconds"], ts)
        info.update({k: v for k, v in mk.items() if k not in ("Y", "Y_stop")}, mk_monitor_s=mon_t)
    curve = _curve(labels_at, ref, step_s, mon_s, moving_n, mon, n)
    return rules, curve, info


def finish_ms(img, Y, ring, t, act, mon, regime, hs, hr, gauss, scale):
    """Markov-K-means' finish of a mean-shift run stopped at t (timed)."""
    t0 = time.perf_counter()
    Wn = PARAMS.window
    Ye = Y.copy()
    info = dict(t=t, reason=_decision(mon), regime=regime, moving_frac=len(act) / Y.shape[0], rho_median=np.nan,
                accepted_frac=np.nan)
    if len(act) and t >= 2:
        Cs, rho = _rules.long_run_centers(ring[_rows(t, Wn)[:, None], act[None, :]] / scale, RHO_MAX)
        cand = np.ascontiguousarray(Cs * scale)
        keep = kde(img, cand, hs, hr, gauss) >= kde(img, np.ascontiguousarray(Y[act]), hs, hr, gauss)
        Ye[act[keep]] = cand[keep]
        info.update(rho_median=float(np.median(rho)), accepted_frac=float(keep.mean()))
    colours = Ye[:, 2:].copy()
    moved, u, osc = settle_colours(mon, regime, colours)
    Ye[:, 2:] = colours
    info.update(settled_moved=moved, unstable_frac=u, oscillating_share=osc, seconds=time.perf_counter() - t0,
                Y=Ye, Y_stop=Y.copy())
    return info


def run_ibf(img, hs, hr, kernel):
    """Reference iterated-bilateral-filter run with every rule evaluated on its iterates."""
    H, W, _ = img.shape
    n, Wn, gauss = H * W, PARAMS.window, kernel == "gauss"
    src, dst = img.copy(), np.empty_like(img)
    ring = np.empty((Wn + 1, n, 3))
    ring[0] = src.reshape(n, 3)
    step_s, mon_s, moving_n = [0.0], [0.0], [n]
    labels_at, gtol = {}, {}
    mk = None
    t0 = time.perf_counter()
    mon = SegmentMonitor(H, W, hr)
    mon.push(src.reshape(n, 3))
    mon_s[0] = time.perf_counter() - t0
    sh = np.empty(n)
    for t in range(1, MAX_ITER["ibf"] + 1):
        t0 = time.perf_counter()
        bf_pass(src, dst, hs, hr, gauss, sh)
        src, dst = dst, src
        step_s.append(time.perf_counter() - t0)
        R = src.reshape(n, 3)
        ring[t % (Wn + 1)] = R
        moving = int((sh >= EPS_REF).sum())
        moving_n.append(moving)
        mx = float(sh.max())
        for e in EPS["ibf"]:
            if e not in gtol and mx < e:
                gtol[e] = (t, segment(R, H, W, hr)[0])
        t0 = time.perf_counter()
        mon.push(R)
        if mk is None:
            stop, regime, _ = mon.assess(moving)
        mon_s.append(time.perf_counter() - t0)
        if mk is None and stop:
            mk = finish_ibf(src, ring, t, sh, mon, regime, hs, hr, gauss)
        if t in CHECK or moving == 0:
            labels_at[t] = segment(R, H, W, hr)[0]
        if moving == 0:
            break
    T = t
    if mk is None:              # MAX_ITER reached before the monitor stopped
        mon.trace[-1]["decision"] = "max_iter"
        mk = finish_ibf(src, ring, T, sh, mon, "drift", hs, hr, gauss)
    t0 = time.perf_counter()
    ref, nref = segment(src.reshape(n, 3), H, W, hr)
    seg_s = time.perf_counter() - t0
    step_s, mon_s, moving_n = np.array(step_s), np.array(mon_s), np.array(moving_n)
    full_s = step_s.sum()
    rules = []

    def add(rule, labels, filt_s, iters):
        rules.append(dict(rule=rule, iterations=int(iters), seconds=filt_s + seg_s, time_frac=(filt_s + seg_s) / (full_s + seg_s),
                          passes_frac=iters / T, error=differing(labels, ref), segments=int(labels.max()) + 1))

    add("full", ref, full_s, T)
    for e in EPS["ibf"]:
        te, lab = gtol.get(e, (T, ref))
        add(f"eps_{e:g}", lab, float(step_s[1:te + 1].sum()), te)
    for tt in TS["ibf"]:
        tt_ = min(tt, T)
        add(f"T{tt}", labels_at[tt_] if tt_ in labels_at else ref, float(step_s[1:tt_ + 1].sum()), tt_)
    info = dict(n=n, T=T, full_s=full_s, seg_s=seg_s, segments_ref=nref, monitor_s=float(mon_s.sum()),
                converged=bool(moving_n[-1] == 0))
    if mk is not None:
        ts = mk["t"]
        filt = float(step_s[1:ts + 1].sum())
        mon_t = float(mon_s[:ts + 1].sum())
        add("mk_stop", segment(mk["R_stop"], H, W, hr)[0], filt + mon_t, ts)
        lab_mk = segment(mk["R"], H, W, hr)[0]
        add("mk", lab_mk, filt + mon_t + mk["seconds"], ts)
        add("mk_free", lab_mk, filt + mk["seconds"], ts)
        info.update({k: v for k, v in mk.items() if k not in ("R", "R_stop")}, mk_monitor_s=mon_t)
    curve = _curve(labels_at, ref, step_s, mon_s, moving_n, mon, n)
    return rules, curve, info


def finish_ibf(src, ring, t, sh, mon, regime, hs, hr, gauss):
    """Markov-K-means' finish of an iterated-bilateral-filter run stopped at t (timed)."""
    t0 = time.perf_counter()
    H, W, _ = src.shape
    n, Wn = H * W, PARAMS.window
    R = src.reshape(n, 3)
    act = np.flatnonzero(sh >= EPS_REF)
    Re = R.copy()
    info = dict(t=t, reason=_decision(mon), regime=regime, moving_frac=len(act) / n, rho_median=np.nan,
                accepted_frac=0.0)
    if len(act) and t >= 2:
        Cs, rho = _rules.long_run_centers(ring[_rows(t, Wn)[:, None], act[None, :]] / hr, RHO_MAX)
        Re[act] = Cs * hr
        out, res_cur, res_new = np.empty_like(src), np.empty(n), np.empty(n)
        bf_pass(src, out, hs, hr, gauss, res_cur)                     # fixed-point residual of the current image
        bf_pass(Re.reshape(H, W, 3), out, hs, hr, gauss, res_new)     # and of the extrapolated one
        accepted = float((res_new ** 2).sum()) <= float((res_cur ** 2).sum())
        if not accepted:
            Re = R.copy()
        info.update(rho_median=float(np.median(rho)), accepted_frac=float(accepted))
    moved, u, osc = settle_colours(mon, regime, Re)
    info.update(settled_moved=moved, unstable_frac=u, oscillating_share=osc, seconds=time.perf_counter() - t0,
                R=Re, R_stop=R.copy())
    return info


def _curve(labels_at, ref, step_s, mon_s, moving_n, mon, n):
    """Per checkpoint: work done so far, points still moving, segment changes, error of stopping there."""
    T = len(step_s) - 1
    cum = np.cumsum(step_s)
    counts = np.concatenate([[0], mon.counts])
    out = []
    for t in sorted(labels_at):
        out.append(dict(t=t, time_frac=cum[t] / cum[T], passes_frac=t / T, moving_frac=moving_n[t] / n,
                        changed_frac=counts[t] / n, monitor_frac=float(mon_s[1:t + 1].sum()) / cum[T],
                        error=differing(labels_at[t], ref)))
    return out


# ----------------------------------------------------------------------------- driver

def _append(path, rows):
    if not rows:
        return
    rows = [{k: (float(f"{v:.6g}") if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows]
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _done(path, key):
    if not path.exists():
        return set()
    import pandas as pd

    d = pd.read_csv(path, dtype={"image": str})
    return set(map(tuple, d[list(key)].astype(str).values))


def warm_up(forms, kernels):
    """Compile every kernel on a small crop so that no run's timing includes compilation."""
    img = load(image_ids()[0])[:48, :64].copy()
    ring = np.random.default_rng(0).normal(size=(PARAMS.window + 1, 50, 5)).cumsum(0)
    rows, idx, scale = np.arange(PARAMS.window + 1), np.arange(50), np.array([2.0, 2.0, 3.0, 3.0, 3.0])
    Cs, _ = _rules.long_run_centers(ring / scale, RHO_MAX)
    assert np.allclose(ar1_limits(ring, rows, idx, scale, RHO_MAX), Cs * scale)
    for form in forms:
        for kernel in kernels:
            (run_ms if form == "ms" else run_ibf)(img, *SETTINGS["fine"], kernel)


def run(forms, kernels, settings, sl):
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = image_ids()[sl]
    warm_up(forms, kernels)
    fr, fc, fi = RESULTS / "meanshift_rules.csv", RESULTS / "meanshift_curves.csv", RESULTS / "meanshift_runs.csv"
    done = _done(fi, ("image", "form", "kernel", "setting"))
    for name in names:
        img = None
        for form in forms:
            for setting in settings:
                for kernel in kernels:
                    if (name, form, kernel, setting) in done:
                        continue
                    img = load(name) if img is None else img
                    hs, hr = SETTINGS[setting]
                    t0 = time.perf_counter()
                    rules, curve, info = (run_ms if form == "ms" else run_ibf)(img, hs, hr, kernel)
                    key = dict(image=name, form=form, kernel=kernel, setting=setting, hs=hs, hr=hr)
                    _append(fr, [{**key, **r} for r in rules])
                    _append(fc, [{**key, **c} for c in curve])
                    info = {**key, **info, "wall_s": time.perf_counter() - t0}
                    _append(fi, [info])
                    r = {x["rule"]: x for x in rules}
                    mk = r.get("mk", {})
                    print(f"{name} {form} {kernel} {setting}: T={info['T']} full={info['full_s']:.2f}s "
                          f"mk t={mk.get('iterations')} time={mk.get('time_frac', np.nan):.2f} "
                          f"err={mk.get('error', np.nan):.4f} ({info.get('reason')}) "
                          f"edison err={r.get('edison', {}).get('error', np.nan):.4f} wall={info['wall_s']:.0f}s",
                          flush=True)


def _slice(s):
    parts = [int(x) if x else None for x in s.split(":")]
    return slice(*parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("download", "run", "report"))
    ap.add_argument("--form", nargs="+", default=["ms"], choices=("ms", "ibf"))
    ap.add_argument("--kernel", nargs="+", default=["flat", "gauss"], choices=("flat", "gauss"))
    ap.add_argument("--setting", nargs="+", default=["fine"], choices=tuple(SETTINGS))
    ap.add_argument("--images", default="0:200:2", help="slice of the sorted image ids, start:stop:step")
    a = ap.parse_args()
    if a.cmd == "download":
        download()
    elif a.cmd == "run":
        run(a.form, a.kernel, a.setting, _slice(a.images))
    else:
        from experiments.meanshift_report import report

        report()


if __name__ == "__main__":
    main()
