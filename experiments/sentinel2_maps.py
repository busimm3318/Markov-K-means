"""Maps of the twelve Sentinel-2 scenes of experiments/sentinel2_tail.py: what the clustering looks like.

Each scene is clustered again exactly as in the study (same seeding, so the same iterates and
stops), and scikit-learn's MiniBatchKMeans is run on it with its defaults. Two rows of panels
are drawn at 100 m (every fifth 20 m pixel):

    top     the clusters each method returns, every pixel in the colour of its cluster's
            center: converged run, Markov-K-means (default), scikit-learn tol, changed < 2 % or
            10 iterations (ENVI), changed < 5 % or 10 iterations, MiniBatchKMeans
    bottom  true colour (B04, B03, B02, 2-98 % stretch), then in red the pixels whose label
            differs from the converged run; MiniBatchKMeans reaches another partition, so its
            clusters are first matched one to one with the converged ones (Hungarian matching)

    python -m experiments.sentinel2_maps     # -> docs/figures/sentinel2_maps/<site>.jpg, sentinel2_maps.csv
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import RESULTS, SIZE, load, tail_run

STEP = 5                                   # 20 m -> 100 m for display
OUT = os.path.join(os.path.dirname(RESULTS), "..", "docs", "figures", "sentinel2_maps")


def stretch(rgb, valid, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(rgb[valid], [2, 98], axis=0)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1), lo, hi


def grid_of(labels, valid, fill=-1):
    g = np.full(valid.shape, fill, dtype=np.int32)
    g[valid] = labels
    return g


def dilate(mask, r=1):
    """Grow isolated pixels so that a single differing pixel stays visible after downsampling."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


METHODS = [("full", "Converged (reference)"), ("markov_auto", "Markov-K-means 0.4.1"),
           ("sklearn_tol", "scikit-learn tol"), ("envi_2pct_10", "Changed < 2 % or 10 it. (ENVI)"),
           ("pct5_10", "Changed < 5 % or 10 it."), ("minibatch", "MiniBatchKMeans (defaults)")]


def figure(name, cube, valid, snaps, centers, rules, path, mb=None):
    """``centers``: per method, the centers its clusters are coloured with; ``mb``: MiniBatchKMeans summary."""
    from experiments.analyze_markov_tail import INK, INK2, SURFACE, plt

    v = valid[::STEP, ::STEP]
    rgb, lo, hi = stretch(cube[::STEP, ::STEP, [2, 1, 0]], v)
    rgb[~v] = 1.0
    full = grid_of(snaps["full"], valid)
    top, bottom = [], [("True colour", rgb)]
    for key, title in METHODS:
        if key not in snaps:
            continue
        ccol = stretch(centers[key][:, [2, 1, 0]], None, lo, hi)[0]      # a cluster's colour: its center
        g = grid_of(snaps[key], valid)
        gs = g[::STEP, ::STEP]
        img = np.ones(gs.shape + (3,))
        img[gs >= 0] = ccol[gs[gs >= 0]]
        if key == "minibatch":
            sub = f"SSE +{mb['sse_excess']:.2%} vs converged"
        elif key == "full":
            sub = f"{int(rules.loc[key, 'iterations'])} iterations"
        else:
            sub = f"{int(rules.loc[key, 'iterations'])} it., SSE +{rules.loc[key, 'sse_excess']:.3%}"
        top.append((f"{title}\n{sub}", img))
        if key == "full":
            continue
        diff = (g != (mb["matched_full"] if key == "minibatch" else full)) & valid
        d = dilate(diff, 2)[::STEP, ::STEP]
        dimg = np.where(v[..., None], 0.86, 1.0) * np.ones(v.shape + (3,))
        dimg[d] = (0.85, 0.12, 0.10)
        share = diff.sum() / valid.sum()
        bottom.append((f"Differs: {title.split(' (')[0] if key != 'envi_2pct_10' else 'ENVI default'}\n"
                       f"{share:.3%} of pixels" + (" (clusters matched)" if key == "minibatch" else ""), dimg))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    ncol = len(top)
    fig, axes = plt.subplots(2, ncol, figsize=(4.0 * ncol, 9.6))
    for row, panels in ((0, top), (1, bottom)):
        for ax, (title, img) in zip(axes[row], panels):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=10.5, color=INK, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(name, x=0.01, ha="left", fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


def minibatch(X, ref, k, seed):
    """scikit-learn MiniBatchKMeans with its defaults; its clusters matched one to one with ``ref``'s."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import MiniBatchKMeans

    t0 = time.perf_counter()
    mb = MiniBatchKMeans(n_clusters=k, random_state=seed).fit(X)
    secs = time.perf_counter() - t0
    conf = np.zeros((k, k), dtype=np.int64)
    np.add.at(conf, (mb.labels_, ref), 1)
    rows, cols = linear_sum_assignment(-conf)
    to_ref = np.empty(k, dtype=np.int64)
    to_ref[rows] = cols
    return mb, secs, to_ref


def run():
    from markov_kmeans._kernels import compute_centers

    os.makedirs(OUT, exist_ok=True)
    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    rows = []
    for _, s in scenes.iterrows():
        t0 = time.perf_counter()
        cube, valid = load(s.url)
        X = np.ascontiguousarray(cube[valid], dtype=np.float64)
        rules, _, extra, snaps = tail_run(X, keep_snaps=True)
        rules = rules.set_index("rule")
        k = extra["k"]
        ref = snaps["full"].astype(np.int64)
        cref = compute_centers(X, ref, np.zeros((k, X.shape[1])))[0]
        centers = {key: compute_centers(X, snaps[key].astype(np.int64), cref)[0] for key, _ in METHODS if key in snaps}
        mb, mb_s, to_ref = minibatch(X, ref, k, 0)
        snaps["minibatch"] = mb.labels_.astype(np.int64)
        centers["minibatch"] = mb.cluster_centers_
        sse_ref = float(((X - cref[ref]) ** 2).sum())
        # the converged labels in MiniBatchKMeans' numbering, through the one-to-one matching
        mbinfo = dict(sse_excess=float(mb.inertia_) / sse_ref - 1, seconds=mb_s,
                      matched_full=grid_of(np.argsort(to_ref)[ref], valid))
        mismatch = float((np.argsort(to_ref)[ref] != snaps["minibatch"]).mean())
        del X
        short = s.site.split(" (")[0].replace(" ", "_").lower()
        figure(s.site, cube, valid, snaps, centers, rules, os.path.join(OUT, f"{short}.jpg"), mbinfo)
        rows.append(dict(site=s.site, file=f"{short}.jpg", tile=s["tile"], date=str(s["datetime"])[:10],
                         n=extra["n"], total_iter=extra["total_iter"], markov_iter=int(rules.loc["markov_auto", "iterations"]),
                         markov_error=rules.loc["markov_auto", "error"], envi_error=rules.loc["envi_2pct_10", "error"],
                         pct5_error=rules.loc["pct5_10", "error"], sklearn_error=rules.loc["sklearn_tol", "error"],
                         minibatch_mismatch=mismatch, minibatch_sse_excess=mbinfo["sse_excess"],
                         minibatch_s=mb_s, minibatch_time_frac=mb_s / extra["total_s"]))
        print(s.site, f"{time.perf_counter() - t0:.0f}s", rows[-1]["markov_error"], flush=True)
        del cube, valid, snaps
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "sentinel2_maps.csv"), index=False)


if __name__ == "__main__":
    run()
