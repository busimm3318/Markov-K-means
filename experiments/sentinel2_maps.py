"""Maps of the twelve Sentinel-2 scenes of experiments/sentinel2_tail.py: what the clustering looks like.

Each scene is clustered again exactly as in the study (same seeding, so the same iterates and
stops), and scikit-learn's MiniBatchKMeans is run on it with its defaults. Two rows of panels
are drawn at 100 m (every fifth 20 m pixel):

    top     the clusters each method returns, every pixel in the colour of its cluster's
            center: converged run, Markov-K-means (default), scikit-learn tol, changed < 2 % or
            10 iterations (ENVI), changed < 5 % or 10 iterations, MiniBatchKMeans
    bottom  true colour (B04, B03, B02, 2-98 % stretch), then each method's map again with the
            pixels whose label agrees with the converged run in grey tones and the pixels whose
            label differs in red; MiniBatchKMeans reaches another partition, so its clusters are
            first matched one to one with the converged ones (Hungarian matching)

The downsampled maps are kept in experiments/results/sentinel2_maps/<site>.npz (not tracked by
git), so the figures can be drawn again without clustering.

    python -m experiments.sentinel2_maps             # -> docs/figures/sentinel2_maps/<site>.jpg, sentinel2_maps.csv
    python -m experiments.sentinel2_maps --redraw    # figures again from the stored maps
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import RESULTS, load, tail_run

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


def block_count(mask):
    """Number of the pixels of each STEP x STEP block (100 m) where ``mask`` is true."""
    h, w = mask.shape[0] // STEP * STEP, mask.shape[1] // STEP * STEP
    return mask[:h, :w].reshape(h // STEP, STEP, w // STEP, STEP).sum((1, 3)).astype(np.uint8)


WRONG = np.array([0.86, 0.06, 0.04])        # a pixel whose label differs from the converged run


def luminance(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


def grey(img, v, lo, hi):
    """``img`` in light grey tones (luminance ``lo``-``hi`` to 0.4-0.95), so red stands out on any land cover."""
    out = np.repeat((0.4 + 0.55 * np.clip((luminance(img) - lo) / max(hi - lo, 1e-6), 0, 1))[..., None], 3, axis=2)
    out[~v] = 1.0
    return out


METHODS = [("full", "Converged (reference)"), ("markov_auto", "Markov-K-means 0.4.1"),
           ("sklearn_tol", "scikit-learn tol"), ("envi_2pct_10", "Changed < 2 % or 10 it. (ENVI)"),
           ("pct5_10", "Changed < 5 % or 10 it."), ("minibatch", "MiniBatchKMeans (defaults)")]


def maps(name, cube, valid, snaps, centers, rules, mb=None):
    """The 100 m maps and panel titles of one scene, as the arrays ``draw`` takes.

    ``centers``: per method, the centers its clusters are coloured with; ``mb``: MiniBatchKMeans
    summary (SSE excess, converged labels in its numbering).
    """
    v = valid[::STEP, ::STEP]
    rgb, lo, hi = stretch(cube[::STEP, ::STEP, [2, 1, 0]], v)
    rgb[~v] = 1.0
    store = {"rgb": (rgb * 255).round().astype(np.uint8), "valid": v}
    full = grid_of(snaps["full"], valid)
    titles = {"name": name}
    for key, title in METHODS:
        if key not in snaps:
            continue
        g = grid_of(snaps[key], valid)
        store[f"clusters_{key}"] = g[::STEP, ::STEP].astype(np.int16)
        store[f"colours_{key}"] = stretch(centers[key][:, [2, 1, 0]], None, lo, hi)[0].astype(np.float32)
        if key == "minibatch":
            sub = f"SSE +{mb['sse_excess']:.2%} vs converged"
        elif key == "full":
            sub = f"{int(rules.loc[key, 'iterations'])} iterations"
        else:
            sub = f"{int(rules.loc[key, 'iterations'])} it., SSE +{rules.loc[key, 'sse_excess']:.3%}"
        titles[key] = f"{title}\n{sub}"
        if key == "full":
            continue
        diff = (g != (mb["matched_full"] if key == "minibatch" else full)) & valid
        store[f"wrong_{key}"] = diff[::STEP, ::STEP]
        store[f"wrong_blocks_{key}"] = block_count(diff)
        titles[f"wrong_{key}"] = (f"Red: differs from converged\n{diff.sum() / valid.sum():.3%} of pixels"
                                  + (" (clusters matched)" if key == "minibatch" else ""))
    store["titles"] = np.array(json.dumps(titles))
    return store


def draw(store, path, dpi=100):
    """Top row: each method's clusters in their centers' colours; bottom row: true colour, then each
    method's map in grey tones with the pixels that differ from the converged run in red."""
    from experiments.analyze_markov_tail import INK, INK2, SURFACE, plt

    titles = json.loads(str(store["titles"]))
    v = store["valid"]
    top, bottom = [], [("True colour", store["rgb"] / 255.0)]
    for key, _ in METHODS:
        if f"clusters_{key}" not in store:
            continue
        gs, ccol = store[f"clusters_{key}"], store[f"colours_{key}"]
        img = np.ones(gs.shape + (3,))
        img[gs >= 0] = ccol[gs[gs >= 0]]
        top.append((titles[key], img))
        if key == "full":
            lo, hi = np.percentile(luminance(img[v]), [1, 99])     # one grey scale for the scene
            continue
        dimg = grey(img, v, lo, hi)
        dimg[store[f"wrong_{key}"]] = WRONG
        bottom.append((titles[f"wrong_{key}"], dimg))
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
    fig.suptitle(titles["name"], x=0.01, ha="left", fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    fig.savefig(path, dpi=dpi, pil_kwargs={"quality": 85, "optimize": True})
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
    os.makedirs(os.path.join(RESULTS, "sentinel2_maps"), exist_ok=True)
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
        store = maps(s.site, cube, valid, snaps, centers, rules, mbinfo)
        np.savez_compressed(os.path.join(RESULTS, "sentinel2_maps", f"{short}.npz"), **store)
        draw(store, os.path.join(OUT, f"{short}.jpg"))
        rows.append(dict(site=s.site, file=f"{short}.jpg", tile=s["tile"], date=str(s["datetime"])[:10],
                         n=extra["n"], total_iter=extra["total_iter"], markov_iter=int(rules.loc["markov_auto", "iterations"]),
                         markov_error=rules.loc["markov_auto", "error"], envi_error=rules.loc["envi_2pct_10", "error"],
                         pct5_error=rules.loc["pct5_10", "error"], sklearn_error=rules.loc["sklearn_tol", "error"],
                         minibatch_mismatch=mismatch, minibatch_sse_excess=mbinfo["sse_excess"],
                         minibatch_s=mb_s, minibatch_time_frac=mb_s / extra["total_s"]))
        print(s.site, f"{time.perf_counter() - t0:.0f}s", rows[-1]["markov_error"], flush=True)
        del cube, valid, snaps
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "sentinel2_maps.csv"), index=False)


def redraw(out=OUT, dpi=100):
    """The figures again from the stored maps (no clustering)."""
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(RESULTS, "sentinel2_maps", "*.npz"))):
        with np.load(f) as z:
            draw(dict(z), os.path.join(out, os.path.basename(f)[:-4] + ".jpg"), dpi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true", help="draw the figures from the stored maps")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()
    redraw(a.out, a.dpi) if a.redraw else run()
