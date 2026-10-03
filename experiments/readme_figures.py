"""Figures for the README: how the method works, and where the time of a run goes.

    python -m experiments.readme_figures          # docs/figures/readme_*.png, social_preview.png (+ experiments/results/readme_tail.csv)
    python -m experiments.readme_figures --plot   # redraw from the saved CSV
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from experiments.analyze_markov_tail import FIG, GRID, INK, INK2, SERIES, SURFACE, plt
from experiments.auto_rule import reference, replay_stop
from experiments.pareto import K, RESULTS, first_t, settle_labels, trajectory
from kmeans_accel import datasets
from markov_kmeans import _rules

BLUE, ORANGE = SERIES[0], SERIES[1]
TINT = "#e9f1fb"
WHITE = "#ffffff"


def _box(ax, x0, y0, w, h, title, body, accent=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=TINT if accent else WHITE, edgecolor=BLUE if accent else INK2,
                                linewidth=1.6 if accent else 1.1))
    ax.text(x0 + w / 2, y0 + h - 0.22, title, ha="center", va="top", fontsize=12, fontweight="bold", color=INK)
    ax.text(x0 + w / 2, y0 + h - 0.58, body, ha="center", va="top", fontsize=10.5, color=INK2, linespacing=1.35)


def _arrow(ax, pts, label=None, at=None, color=INK2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=1.4, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                                                   mutation_scale=14, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*at, label, ha="center", va="bottom", fontsize=10.5, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def mechanism():
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.9)
    ax.axis("off")
    y1, h1 = 2.75, 1.45
    _box(ax, 0.1, y1, 2.0, h1, "Base step", "one iteration of\nHamerly, Lloyd\nor mini-batch")
    _box(ax, 2.55, y1, 2.0, h1, "Label history", "last W+1 labels\nof every point")
    _box(ax, 5.0, y1, 2.55, h1, "State monitor", "who still changes?\ncenters travel or jitter?\nclusters reorganise or swap?",
         accent=True)
    cx, cy, dx, dy = 8.85, y1 + h1 / 2, 0.8, 0.62
    ax.add_patch(Polygon([(cx - dx, cy), (cx, cy + dy), (cx + dx, cy), (cx, cy - dy)], closed=True,
                         facecolor=TINT, edgecolor=BLUE, linewidth=1.6))
    ax.text(cx, cy, "all\nquiet?", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    _arrow(ax, [(2.1, cy), (2.55, cy)])
    _arrow(ax, [(4.55, cy), (5.0, cy)])
    _arrow(ax, [(7.55, cy), (cx - dx, cy)])
    top = y1 + h1 + 0.42
    _arrow(ax, [(cx, cy + dy), (cx, top), (1.1, top), (1.1, y1 + h1)], "no: next iteration", (5.0, top + 0.02))

    y2, h2 = 0.12, 1.55
    mid = y2 + h2 + 0.5
    _arrow(ax, [(cx, cy - dy), (cx, mid), (1.45, mid), (1.45, y2 + h2)], "yes: stop", (5.0, mid + 0.02))
    _box(ax, 0.1, y2, 2.9, h2, "Name the regime", "drift: points moved,\nnow settling\noscillation: points\nkeep switching")
    _box(ax, 3.55, y2, 3.0, h2, "Settle undecided points", "each from its own\nlabel history\n(see the figure below)",
         accent=True)
    _box(ax, 7.1, y2, 2.8, h2, "Output", "hard labels\nsoft memberships\nlist of undecided points")
    _arrow(ax, [(3.0, y2 + h2 / 2), (3.55, y2 + h2 / 2)])
    _arrow(ax, [(6.55, y2 + h2 / 2), (7.1, y2 + h2 / 2)])
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "readme_mechanism.png", dpi=160)
    plt.close(fig)


ROWS = [("Settled long ago", "AAAAAAAAAAA", "left as it is"),
        ("Moved once (drift)", "AAAAABBBBBB", "keeps its new cluster"),
        ("Keeps switching (oscillation)", "ABABBABAABA", "share of time in each;\nhard label: the larger")]


def settle_figure():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    col = {"A": BLUE, "B": ORANGE}
    cw, ch, x0 = 0.36, 0.5, 2.75
    ax.text(x0, 2.95, "label history, oldest → newest", fontsize=10.5, color=INK2, va="center")
    ax.text(7.35, 2.95, "membership", fontsize=10.5, color=INK2, va="center")
    for r, (name, seq, note) in enumerate(ROWS):
        y = 2.2 - r * 0.85
        ax.text(0.05, y + ch / 2, name, fontsize=11, color=INK, va="center")
        for i, c in enumerate(seq):
            ax.add_patch(Rectangle((x0 + i * cw, y), cw - 0.04, ch, facecolor=col[c], edgecolor="none"))
            ax.text(x0 + i * cw + (cw - 0.04) / 2, y + ch / 2, c, ha="center", va="center", fontsize=9.5,
                    color=WHITE, fontweight="bold")
        _arrow(ax, [(x0 + len(seq) * cw + 0.1, y + ch / 2), (7.25, y + ch / 2)])
        share = seq.count("A") / len(seq)
        if r == 0:
            share = 1.0
        elif r == 1:
            share = 0.0
        bw = 1.1
        ax.add_patch(Rectangle((7.35, y), bw * share, ch, facecolor=BLUE, edgecolor="none"))
        ax.add_patch(Rectangle((7.35 + bw * share, y), bw * (1 - share), ch, facecolor=ORANGE, edgecolor="none"))
        ax.text(7.35 + bw + 0.15, y + ch / 2, note, fontsize=10, color=INK2, va="center", linespacing=1.25)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "readme_settle.png", dpi=160)
    plt.close(fig)


TAIL = [("exact", 1.5, 1, 0.0, "Exact Lloyd / Hamerly: a long tail"),
        ("minibatch_const", 1.5, 0, 1e-3, "Mini-batch, constant step: a plateau")]


def tail_data():
    rows = []
    for regime, sep, seed, eta, _ in TAIL:
        traj, X = trajectory(regime, sep, seed, eta)
        H, C, counts = traj["H"], traj["C"], traj["counts"]
        T, n = H.shape
        exact = regime == "exact"
        ref, _ = reference(regime, traj, K)
        t_auto, _, _, _ = replay_stop(traj, _rules.StopParams(), exact, "auto", K)
        if X is None:
            X, _, _ = datasets.gmm(1_000_000, 16, K, sep=sep, seed=seed)
        var = float(np.var(X, axis=0).mean())
        shift = np.r_[np.inf, ((C[1:] - C[:-1]) ** 2).sum((1, 2))]
        t_sk = first_t(lambda t_: shift[t_] <= 1e-4 * var, T)
        for t in range(1, T):
            settled, _ = settle_labels(H, t, None, "auto")
            rows.append(dict(regime=regime, t=t, changing=counts[t - 1] / n, keep=float((H[t] != ref).mean()),
                             settle=float((settled != ref).mean()), t_auto=t_auto, t_sklearn=t_sk, n_iter=T))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "readme_tail.csv", index=False)
    return df


def tail_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (regime, *_, title) in zip(axes, TAIL):
        d = df[df.regime == regime]
        pct = lambda s: s.where(s > 0) * 100
        ax.plot(d.t, pct(d.changing), color=INK2, lw=1.4, label="points changing cluster in this iteration")
        ax.plot(d.t, pct(d.keep), color=ORANGE, lw=2.0, label="wrong labels if stopped here")
        ax.plot(d.t, pct(d.settle), color=BLUE, lw=2.0, ls="--", label="wrong labels if stopped here and settled")
        t_auto, t_sk, T = int(d.t_auto.iloc[0]), int(d.t_sklearn.iloc[0]), int(d.n_iter.iloc[0])
        ax.axvline(t_auto, color=BLUE, lw=1.6)
        ax.text(t_auto, 1.02, " Markov-K-means stops", transform=ax.get_xaxis_transform(), color=BLUE, fontsize=9.5,
                ha="left", va="bottom")
        if t_sk < T - 1:
            ax.axvline(t_sk, color=INK2, lw=1.2, ls=":")
            ax.text(t_sk, 1.02, "scikit-learn tol stops ", transform=ax.get_xaxis_transform(), color=INK2,
                    fontsize=9.5, ha="right", va="bottom")
        else:
            ax.text(0.98, 0.97, "scikit-learn tol never stops", transform=ax.transAxes, color=INK2, fontsize=9.5,
                    ha="right", va="top")
        ax.set_yscale("log")
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=22)
        ax.set_xlabel("iteration (pass over the data)", color=INK2, fontsize=10)
        ax.set_ylabel("share of points (%, log)", color=INK2, fontsize=10)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "readme_tail.png", dpi=160)
    plt.close(fig)


def social_preview():
    """1280 x 640 card for the repository's social preview (Settings > General > Social preview)."""
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=SURFACE)
    fig.text(0.045, 0.88, "Markov-K-means", fontsize=46, fontweight="bold", color=INK, va="center")
    fig.text(0.045, 0.765, "Stop K-means when only boundary points still switch clusters,", fontsize=21, color=INK2,
             va="center")
    fig.text(0.045, 0.705, "and settle them from their own label history.", fontsize=21, color=INK2, va="center")
    fig.text(0.955, 0.88, "pip install markov-kmeans", fontsize=17, color=BLUE, ha="right", va="center",
             family="DejaVu Sans Mono")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.6])
    ax.imshow(mpimg.imread(FIG / "readme_settle.png"))
    ax.axis("off")
    fig.savefig(FIG / "social_preview.png", dpi=100, facecolor=SURFACE)
    plt.close(fig)


def main(argv=()):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    FIG.mkdir(parents=True, exist_ok=True)
    mechanism()
    settle_figure()
    social_preview()
    df = pd.read_csv(RESULTS / "readme_tail.csv") if "--plot" in argv else tail_data()
    tail_figure(df)


if __name__ == "__main__":
    main(sys.argv[1:])
