"""Sentinel-2 over a wide area with many clusters: the southern Korean peninsula at 60 m, k = 256.

Mosaic: the 20 MGRS 100 km squares of UTM zone 52 with easting 200-600 km and northing
3800-4300 km (columns B-E, rows D-H): the mainland of South Korea, the seas around it and
the strip just north of the DMZ, about 400 km x 500 km. For each square the 2024
acquisition with the most usable pixels (least nodata and cloud) is read at 60 m (the
same ten bands as experiments/sentinel2_tail.py, averaged from the 10/20 m grids), cropped
to its 100 km square so that neighbouring tiles do not overlap, and masked with the scene
classification (no data, saturated, cloud, cloud shadow, cirrus).

One exact K-means run (Hamerly bounds with parallel assignment, k-means++ seeding on a
one-million-pixel sample) goes on until no pixel changes or MAX_ITER, and the stopping
rules of sentinel2_tail are judged on its iterates. scikit-learn's MiniBatchKMeans with
its defaults, the usual choice for data this size, is run separately on the same pixels:
it reaches a different partition, so only its SSE and time are compared.

    python -m experiments.sentinel2_wide select   # -> experiments/results/sentinel2_wide_scenes.csv
    python -m experiments.sentinel2_wide run      # -> sentinel2_wide_{rules,curves,runs}.csv, sentinel2_wide_maps.npz
    python -m experiments.sentinel2_wide report   # -> docs/figures/fig12_sentinel2_wide.png, tables in docs/results.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import BAD_SCL, BANDS, BUCKET, RESULTS, RULES, _get, items, tail_run

ZONE, BAND = 52, "S"
COLS, ROWS = "BCDE", "HGFED"                 # west to east, north to south
YEAR = 2024
PX = 1667                                    # 100 km at 60 m
K = 256
SEED = 0
MAX_ITER = 3000
CHECKPOINTS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000)
BLOCK = 17                                   # ~1 km blocks for the error maps
F = {s: os.path.join(RESULTS, f"sentinel2_wide_{s}.csv") for s in ("scenes", "rules", "curves", "runs")}


def select(year=YEAR):
    """For every square, the acquisition with the largest share of usable pixels."""
    def best_of(square):
        best = None
        for it in items(ZONE, BAND, square, year):
            iid = it.rstrip("/").split("/")[-1]
            try:
                p = json.loads(_get(BUCKET + it + iid + ".json"))["properties"]
            except Exception:
                continue
            nod, cc = p.get("s2:nodata_pixel_percentage", 100), p.get("eo:cloud_cover", 100)
            usable = (1 - nod / 100) * (1 - cc / 100)
            if best is None or usable > best["usable"]:
                best = dict(square=square, tile=f"{ZONE}{BAND}{square}", item=iid, url=BUCKET + it, usable=usable,
                            cloud_cover=cc, nodata=nod, datetime=p.get("datetime"))
        print(square, best and (best["item"], round(best["usable"], 3)), flush=True)
        return best

    squares = [c + r for r in ROWS for c in COLS]
    with ThreadPoolExecutor(8) as ex:
        rows = [b for b in ex.map(best_of, squares) if b]
    df = pd.DataFrame(rows)
    df.to_csv(F["scenes"], index=False)
    return df


def load_tile(url):
    """(PX, PX, 10) reflectance at 60 m (float32) and the validity mask, cropped to the 100 km square."""
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    import rasterio
    from rasterio.enums import Resampling

    def read(name, resampling):
        for i in range(4):
            try:
                with rasterio.open("/vsicurl/" + url + name + ".tif") as src:
                    return src.read(1, out_shape=(1830, 1830), resampling=resampling)[:PX, :PX]
            except Exception:
                if i == 3:
                    raise
                time.sleep(2 ** i)

    cube = np.empty((PX, PX, len(BANDS)), dtype=np.float32)
    for j, b in enumerate(BANDS):
        cube[:, :, j] = read(b, Resampling.average)
    scl = read("SCL", Resampling.nearest)
    valid = ~np.isin(scl, BAD_SCL) & (cube > 0).all(2)
    cube /= 10000.0
    return cube, valid


def mosaic(scenes):
    """Pixels of the whole mosaic (float64, row-major over the mosaic grid), validity grid and an RGB preview."""
    H, W = len(ROWS) * PX, len(COLS) * PX
    valid = np.zeros((H, W), dtype=bool)
    rgb = np.zeros((H // BLOCK, W // BLOCK, 3), dtype=np.float32)
    parts = []
    by_square = scenes.set_index("square")
    for r, row in enumerate(ROWS):
        for c, col in enumerate(COLS):
            if col + row not in by_square.index:
                continue
            t0 = time.perf_counter()
            cube, v = load_tile(by_square.loc[col + row, "url"])
            valid[r * PX:(r + 1) * PX, c * PX:(c + 1) * PX] = v
            parts.append(((r, c), cube, v))
            print(f"  {col + row}: {v.mean():.1%} usable, {time.perf_counter() - t0:.0f}s", flush=True)
    # pixels in mosaic row-major order: gather tile rows band by band
    X = np.empty((int(valid.sum()), len(BANDS)), dtype=np.float64)
    pos = np.zeros((H, W), dtype=np.int64)
    pos[valid] = np.arange(valid.sum())
    for (r, c), cube, v in parts:
        X[pos[r * PX:(r + 1) * PX, c * PX:(c + 1) * PX][v]] = cube[v]
        sub = cube[:PX // BLOCK * BLOCK, :PX // BLOCK * BLOCK, [2, 1, 0]]
        blk = sub.reshape(PX // BLOCK, BLOCK, PX // BLOCK, BLOCK, 3).mean((1, 3))
        rr, cc = r * PX // BLOCK, c * PX // BLOCK
        rgb[rr:rr + blk.shape[0], cc:cc + blk.shape[1]] = blk
    return X, valid, rgb


def block_share(mask_px, valid):
    """Share of the valid pixels of each ~1 km block where mask_px (over valid pixels) is true."""
    H, W = valid.shape
    grid = np.zeros((H, W), dtype=np.float32)
    grid[valid] = mask_px
    h, w = H // BLOCK * BLOCK, W // BLOCK * BLOCK
    num = grid[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum((1, 3))
    den = valid[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum((1, 3))
    return np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)


def run():
    from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

    scenes = pd.read_csv(F["scenes"])
    t0 = time.perf_counter()
    X, valid, rgb = mosaic(scenes)
    load_s = time.perf_counter() - t0
    n = len(X)
    print(f"mosaic: {valid.shape}, n={n:,} usable pixels ({valid.mean():.1%}), loaded in {load_s:.0f}s", flush=True)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    C0, _ = kmeans_plusplus(X[np.sort(rng.choice(n, 1_000_000, replace=False))], K, random_state=SEED)
    seed_s = time.perf_counter() - t0
    rules, curve, extra, snaps = tail_run(X, k=K, seed=SEED, max_iter=MAX_ITER, C0=C0, checkpoints=CHECKPOINTS,
                                          keep_snaps=True)
    extra.update(seed_sample_s=seed_s, load_s=load_s, height=valid.shape[0], width=valid.shape[1])
    ref = snaps["full"]
    maps = {"rgb": rgb, "valid": block_share(np.ones(n, dtype=bool), valid)}
    for r in RULES[:-1]:
        maps[r] = block_share(snaps[r] != ref, valid)
    np.savez_compressed(os.path.join(RESULTS, "sentinel2_wide_maps.npz"), **maps)
    rules.to_csv(F["rules"], index=False)
    curve.to_csv(F["curves"], index=False)
    print(rules.round(5).to_string(index=False), flush=True)
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in extra.items()}, flush=True)
    del snaps, ref

    # the usual large-data alternative, with its defaults
    t0 = time.perf_counter()
    mb = MiniBatchKMeans(n_clusters=K, random_state=SEED).fit(X)
    mb_s = time.perf_counter() - t0
    extra.update(minibatch_s=mb_s, minibatch_time_frac=mb_s / extra["total_s"], minibatch_iter=int(mb.n_iter_),
                 minibatch_steps=int(mb.n_steps_), minibatch_sse_excess=float(mb.inertia_) / extra["sse_ref"] - 1)
    print({k_: extra[k_] for k_ in extra if k_.startswith("minibatch")}, flush=True)
    pd.DataFrame([extra]).to_csv(F["runs"], index=False)


# ----------------------------------------------------------------------------- report

def fixed_stops(curve):
    """Time (share of the full run) and error of stopping after each checkpoint iteration."""
    c = curve.dropna(subset=["error"]).copy()
    c["time_frac"] = c.cum_s / curve.cum_s.iloc[-1]
    return c[["t", "time_frac", "error"]]


def report():
    from matplotlib.colors import LogNorm

    from experiments.analyze_markov_tail import FIG, GRID, INK, INK2, SERIES, SURFACE, _style, fill_report, md, plt
    from experiments.sentinel2_tail import RULE_KO, RULE_NAMES

    rules = pd.read_csv(F["rules"]).set_index("rule")
    curve = pd.read_csv(F["curves"])
    run_ = pd.read_csv(F["runs"]).iloc[0]
    scenes = pd.read_csv(F["scenes"])
    maps = np.load(os.path.join(RESULTS, "sentinel2_wide_maps.npz"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    style = {"envi_2pct_10": (SERIES[1], "s"), "pct5_10": (SERIES[3], "D"), "pct2": (SERIES[4], "v"),
             "sklearn_tol": (SERIES[2], "^"), "markov_v03": (INK2, "P"), "markov_auto": (SERIES[0], "o")}

    # figure 12: the run and the rules
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ax.plot(curve.t, curve.changed_share.clip(lower=1e-9) * 100, color=INK2, lw=1.2)
    for rule, (col, mk) in style.items():
        t_ = int(rules.loc[rule, "iterations"])
        y = curve[curve.t == t_].changed_share.clip(lower=1e-9).iloc[0] * 100
        ax.scatter([t_], [y], color=col, marker=mk, s=70, zorder=4, edgecolor=SURFACE, linewidth=1,
                   label=RULE_NAMES[rule])
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"Pixels changing cluster per iteration (k = {int(run_.k)}, {run_.n / 1e6:.0f} million pixels)",
           "pixels changing cluster (%, log)", "iteration (log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    ax = axes[1]
    fixed = fixed_stops(curve)
    ax.plot(fixed.time_frac, fixed.error * 100, color=GRID, lw=2.2, marker="o", ms=4, mfc=INK2, mec=INK2, zorder=2,
            label="fixed number of iterations (labels: iterations)")
    for _, f in fixed[fixed.t.isin([10, 100, 300, 1000, 2000])].iterrows():
        ax.annotate(f"{int(f.t)}", (f.time_frac, f.error * 100), textcoords="offset points", xytext=(4, 5),
                    fontsize=8.5, color=INK2)
    for rule, (col, mk) in style.items():
        r = rules.loc[rule]
        ax.scatter([r.time_frac], [max(r.error, 1e-6) * 100], color=col, marker=mk, s=150, edgecolor=INK,
                   linewidth=1.1, zorder=5, label=RULE_NAMES[rule])
    ax.set_yscale("log")
    _style(ax, "Time and error of each stopping rule", "labels differing from the converged run (%, log)",
           "time / run to convergence")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_sentinel2_wide.png", dpi=150)
    plt.close(fig)

    # figure 13: where the wrong labels are
    panels = [("rgb", "True colour (60 m mosaic, 20 tiles)"), ("envi_2pct_10", RULE_NAMES["envi_2pct_10"]),
              ("sklearn_tol", RULE_NAMES["sklearn_tol"]), ("markov_auto", RULE_NAMES["markov_auto"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    for ax, (key, title) in zip(axes, panels):
        if key == "rgb":
            img = np.clip(maps["rgb"] / np.nanpercentile(maps["rgb"], 98), 0, 1)
            ax.imshow(img)
        else:
            share = np.where(maps["valid"] > 0, maps[key], np.nan)
            im = ax.imshow(np.where(share > 0, share, np.nan), cmap="magma_r", norm=norm, interpolation="nearest")
            ax.imshow(np.where(np.isnan(share), np.nan, np.where(share > 0, np.nan, 0.0)), cmap="Greys", vmin=0, vmax=8,
                      interpolation="nearest")
            err = rules.loc[key, "error"]
            title = f"{title}\nwrong: {err:.2%} of pixels"
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    cb = fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.01)
    cb.set_label("share of wrong labels in each ~1 km block (blank: none)", color=INK2)
    fig.savefig(FIG / "fig13_sentinel2_wide_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    T = {}
    rows = {}
    for rule in RULES:
        r = rules.loc[rule]
        rows[RULE_KO[rule]] = {"정지 반복": f"{int(r.iterations):,}", "시간": f"{r.time_frac:.0%}",
                               "오차": f"{r.error:.2%}", "SSE 초과": f"{r.sse_excess:.3%}"}
    for _, f in fixed[fixed.t.isin([300, 700, 1000, 1500, 2000])].iterrows():
        rows[f"고정 {int(f.t):,}회"] = {"정지 반복": f"{int(f.t):,}", "시간": f"{f.time_frac:.0%}",
                                       "오차": f"{f.error:.2%}", "SSE 초과": "–"}
    if "minibatch_s" in run_:
        rows["scikit-learn MiniBatchKMeans (기본값, 다른 알고리즘)"] = {
            "정지 반복": f"미니배치 {int(run_.minibatch_steps)}개", "시간": f"{run_.minibatch_time_frac:.1%}",
            "오차": "–", "SSE 초과": f"{run_.minibatch_sse_excess:.1%}"}
    T["sentinel2_wide_rules"] = md(pd.DataFrame(rows).T, "{}", index_name="정지 규칙")
    info = {"화소 수": f"{run_.n / 1e6:.1f}백만 ({int(run_.height):,} × {int(run_.width):,} 격자의 {run_.n / (run_.height * run_.width):.1%})",
            "수렴": f"{int(run_.total_iter):,}회, {run_.total_s / 60:.0f}분" + ("" if run_.converged else f" (상한 {MAX_ITER})"),
            "Markov 정지": f"{int(rules.loc['markov_auto', 'iterations'])}회 ({run_.markov_reason})",
            "감시기 비용": f"끝까지 돌린 시간의 {run_.monitor_s / run_.total_s:.1%}",
            "의사 대표점 단계": f"{run_.pseudo_s / run_.total_s:.1%} (채택: {'예' if run_.pseudo_accepted else '아니오'}, ρ 중앙값 {run_.rho_median:.2f})",
            "옮긴 화소": f"{run_.moves_frac:.2%}, 그중 맞은 비율 {run_.moves_right:.0%}",
            "고친 / 망친 화소": f"{int(run_.fixed):,} / {int(run_.broken):,}",
            "틀린 화소 중 미결정 목록에 든 비율": f"0.3: {run_.U_recall:.1%}, 0.4: {run_.undecided_recall:.1%}"}
    T["sentinel2_wide_run"] = md(pd.DataFrame({"값": info}), "{}", index_name="항목")
    sc = scenes[["tile", "datetime", "cloud_cover", "nodata"]].copy()
    sc["datetime"] = sc.datetime.str[:10]
    sc = sc.set_index("tile")
    sc.columns = ["촬영일", "구름 (%)", "결측 (%)"]
    T["sentinel2_wide_scenes"] = md(sc, {"촬영일": "{}", "구름 (%)": "{:.2f}", "결측 (%)": "{:.2f}"}, index_name="타일")
    fill_report(T)
    for k_, v in T.items():
        print(f"== {k_}\n{v}")
    return T


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"select": select, "run": run, "report": report}[cmd]()
