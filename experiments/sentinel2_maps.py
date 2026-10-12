"""Maps of the twelve Sentinel-2 scenes of experiments/sentinel2_tail.py: what the clustering looks like.

Each scene is clustered again exactly as in the study (same seeding, so the same iterates and
stops), and five panels are drawn at 100 m (every fifth 20 m pixel):

    true colour                     B04, B03, B02 reflectance, 2-98 % stretch
    converged clusters              every pixel in the colour of its cluster's center
    Markov-K-means (default)        the same, for the labels Markov-K-means returns
    differs: Markov-K-means         pixels whose label differs from the converged run, in red
    differs: changed < 2 % or 10    the same for the remote-sensing default (ENVI)

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


def figure(name, cube, valid, snaps, centers, rules, path):
    from experiments.analyze_markov_tail import INK, INK2, SURFACE, plt

    v = valid[::STEP, ::STEP]
    rgb, lo, hi = stretch(cube[::STEP, ::STEP, [2, 1, 0]], v)
    rgb[~v] = 1.0
    ccol = stretch(centers[:, [2, 1, 0]], None, lo, hi)[0]       # a cluster's colour: its center, same stretch
    panels = [("True colour", rgb)]
    for key, title in (("full", "Converged clusters"), ("markov_auto", "Markov-K-means 0.4.1 (default)")):
        g = grid_of(snaps[key], valid)[::STEP, ::STEP]
        img = np.ones(g.shape + (3,))
        img[g >= 0] = ccol[g[g >= 0]]
        panels.append((f"{title}\n{int(rules.loc[key, 'iterations'])} iterations", img))
    full = grid_of(snaps["full"], valid)
    for key, title in (("markov_auto", "Differs: Markov-K-means"), ("envi_2pct_10", "Differs: changed < 2 % or 10 it.")):
        diff = (grid_of(snaps[key], valid) != full) & valid
        d = dilate(diff, 2)[::STEP, ::STEP]
        img = np.where(v[..., None], 0.86, 1.0) * np.ones(v.shape + (3,))
        img[d] = (0.85, 0.12, 0.10)
        panels.append((f"{title}\n{rules.loc[key, 'error']:.3%} of pixels", img))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.9))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(name, x=0.01, ha="left", fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


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
        centers = compute_centers(X, snaps["full"].astype(np.int64), np.zeros((extra["k"], X.shape[1])))[0]
        del X
        short = s.site.split(" (")[0].replace(" ", "_").lower()
        figure(s.site, cube, valid, snaps, centers, rules, os.path.join(OUT, f"{short}.jpg"))
        rows.append(dict(site=s.site, file=f"{short}.jpg", tile=s["tile"], date=str(s["datetime"])[:10],
                         n=extra["n"], total_iter=extra["total_iter"], markov_iter=int(rules.loc["markov_auto", "iterations"]),
                         markov_error=rules.loc["markov_auto", "error"], envi_error=rules.loc["envi_2pct_10", "error"],
                         sklearn_error=rules.loc["sklearn_tol", "error"]))
        print(s.site, f"{time.perf_counter() - t0:.0f}s", rows[-1]["markov_error"], flush=True)
        del cube, valid, snaps
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "sentinel2_maps.csv"), index=False)


if __name__ == "__main__":
    run()
