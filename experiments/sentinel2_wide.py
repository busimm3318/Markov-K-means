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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"select": select, "run": run}.get(cmd, lambda: None)()
    if cmd == "report":
        from experiments.sentinel2_wide_report import report
        report()
