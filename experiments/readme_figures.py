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
    _box(ax, 2.55, y1, 2.0, h1, "History", "last W+1 labels\nof every point and\npositions of centers")
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

    y2, h2 = 0.12, 1.62
    mid = y2 + h2 + 0.45
    w, gap = 2.19, 0.35
    xs = [0.1 + i * (w + gap) for i in range(4)]
    _arrow(ax, [(cx, cy - dy), (cx, mid), (xs[0] + w / 2, mid), (xs[0] + w / 2, y2 + h2)], "yes: stop",
           (5.0, mid + 0.02))
    _box(ax, xs[0], y2, w, h2, "Center chains", "each center's path:\nwhere it is heading\n= pseudo representative",
         accent=True)
    _box(ax, xs[1], y2, w, h2, "Reassign", "every point to its\nnearest pseudo rep.;\nkept if SSE not higher")
    _box(ax, xs[2], y2, w, h2, "Label chains", "undecided points:\nmembership from\ntheir own history",
         accent=True)
    _box(ax, xs[3], y2, w, h2, "Output", "hard labels\nsoft memberships\nundecided points")
    for i in range(3):
        _arrow(ax, [(xs[i] + w, y2 + h2 / 2), (xs[i + 1], y2 + h2 / 2)])
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "readme_mechanism.png", dpi=160)
    plt.close(fig)


def _bisector(ax, a, b, color, ls, label=None):
    """Draw the boundary between centers a and b (their perpendicular bisector) across the panel."""
    m, d = (a + b) / 2, b - a
    u = np.array([-d[1], d[0]]) / np.hypot(*d)
    pts = np.array([m - 4 * u, m + 4 * u])
    ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=1.8, label=label, zorder=2)


def _side(P, a, b):
    """True where a point is nearer to b than to a."""
    return ((P - b) ** 2).sum(1) < ((P - a) ** 2).sum(1)


def pseudo_figure():
    """Center chain -> pseudo representative, in a drift and in an oscillation (geometry, not data)."""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    a_star, b_star = np.array([3.6, 2.0]), np.array([6.4, 2.0])
    P = np.c_[rng.uniform(3.4, 6.6, 140), rng.uniform(0.3, 3.7, 140)]
    # drift: A converges geometrically towards its fixed point, B has stopped
    s = np.arange(2, 10)
    pathA = a_star + 0.75 ** s[:, None] * np.array([-5.0, -2.6])
    paths = {"drift": (pathA, np.repeat(b_star[None], len(s), 0))}
    # oscillation: both jitter around fixed points
    ja, jb = rng.normal(scale=0.22, size=(9, 2)), rng.normal(scale=0.22, size=(9, 2))
    ja[-1], jb[-1] = [0.45, 0.3], [0.4, -0.25]          # the last jitter happens to lean right
    paths["oscillation"] = (a_star + ja, b_star + jb)
    titles = {"drift": "Drift: the chain is extrapolated to its fixed point",
              "oscillation": "Oscillation: the chain's mean, not its last jitter"}
    for ax, key in zip(axes, ("drift", "oscillation")):
        pa, pb = paths[key]
        Cw = np.stack([pa, pb], 1)
        Cs, _ = _rules.long_run_centers(Cw)
        a_t, b_t = pa[-1], pb[-1]
        now, later = _side(P, a_t, b_t), _side(P, Cs[0], Cs[1])
        move = now != later
        for lab, col in ((False, BLUE), (True, ORANGE)):
            q = P[(now == lab) & ~move]
            ax.scatter(q[:, 0], q[:, 1], s=14, color=col, alpha=0.35, edgecolor="none", zorder=1)
            q = P[(later == lab) & move]
            ax.scatter(q[:, 0], q[:, 1], s=46, facecolor=col, edgecolor=INK, linewidth=1.1, zorder=4)
        for path, col in ((pa, BLUE), (pb, ORANGE)):
            ax.plot(path[:, 0], path[:, 1], color=col, lw=1, alpha=0.6, zorder=3)
            ax.scatter(path[:-1, 0], path[:-1, 1], s=16, color=col, alpha=0.5, zorder=3)
            ax.scatter([path[-1, 0]], [path[-1, 1]], s=70, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=5)
        ax.scatter(Cs[:, 0], Cs[:, 1], marker="*", s=300, color=[BLUE, ORANGE], edgecolor=INK, linewidth=1, zorder=6)
        _bisector(ax, a_t, b_t, INK2, ":", "boundary now")
        _bisector(ax, Cs[0], Cs[1], INK, "-", "boundary at the pseudo representatives")
        ax.set_xlim(0.6, 8.0)
        ax.set_ylim(-0.5, 4.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(titles[key], fontsize=11.5, color=INK, loc="left")
    h = [plt.Line2D([], [], color=INK2, marker="o", ms=4, lw=1, alpha=0.7),
         plt.Line2D([], [], ls="none", marker="o", ms=8, color=INK2, markeredgecolor=SURFACE),
         plt.Line2D([], [], ls="none", marker="*", ms=15, color=INK2, markeredgecolor=INK),
         plt.Line2D([], [], color=INK2, ls=":", lw=1.8), plt.Line2D([], [], color=INK, lw=1.8),
         plt.Line2D([], [], ls="none", marker="o", ms=7, color=INK2, markeredgecolor=INK)]
    l = ["center path in the window", "center now", "pseudo representative (long-run mean)", "boundary now",
         "boundary at the pseudo representatives", "point moved now (colour: new cluster)"]
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "readme_pseudo.png", dpi=160)
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
    pseudo_figure()
    settle_figure()
    social_preview()
    df = pd.read_csv(RESULTS / "readme_tail.csv") if "--plot" in argv else tail_data()
    tail_figure(df)


if __name__ == "__main__":
    main(sys.argv[1:])
