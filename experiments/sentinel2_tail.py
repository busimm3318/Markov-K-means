"""Convergence tail of K-means on Sentinel-2 scenes, and the stopping rules people use on them.

Each scene is one 110 km MGRS tile (the least cloudy 2024 Sentinel-2 L2A acquisition, read
from the public sentinel-cogs bucket) at 20 m: 10 surface-reflectance bands (B02, B03, B04,
B08 averaged from 10 m; B05, B06, B07, B8A, B11, B12), with pixels the scene classification
flags as no data, saturated, cloud, cloud shadow or cirrus dropped.

One exact K-means run per scene (Hamerly bounds, k-means++ seeding) goes on until no pixel
changes (or MAX_ITER). Every stopping rule is evaluated on the same iterates, so the rules
differ only in where they stop:

    envi_2pct_10   changed share < 2 % or 10 iterations (ENVI ISODATA defaults)
    pct5_10        changed share < 5 % or 10 iterations (common tutorial setting)
    pct2           changed share < 2 %, no iteration cap
    sklearn_tol    squared center shift <= 1e-4 x mean feature variance (scikit-learn default)
    markov_auto    MarkovKMeans defaults: StateMonitor, then settlement from label histories
    full           convergence: the reference partition

The 60 m study asks what a fractional membership means on real pixels: the 20 m cube is
averaged to 60 m, clustered with MarkovKMeans, and each 60 m pixel's undecided status and
soft membership are compared with the mixture of its nine 20 m sub-pixels.

    python -m experiments.sentinel2_tail select   # -> experiments/results/sentinel2_scenes.csv
    python -m experiments.sentinel2_tail run      # -> sentinel2_rules.csv, sentinel2_curves.csv, sentinel2_mixing.csv
    python -m experiments.sentinel2_tail report   # -> docs/figures/fig10_sentinel2_tail.png, tables
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
PREFIX = "sentinel-s2-l2a-cogs"
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
BAD_SCL = (0, 1, 3, 8, 9, 10)   # no data, saturated/defective, cloud shadow, cloud medium/high, cirrus
SIZE = 5490                     # 20 m grid of a 110 km tile
K = 12
SEED = 0
MAX_ITER = 1500
CHECKPOINTS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500)

# (name, latitude, longitude): one tile per land-cover setting
SITES = [
    ("Seoul (urban, forest, river)", 37.55, 126.98),
    ("Iowa (row crops)", 42.03, -93.62),
    ("Central Valley (irrigated farms)", 36.75, -119.8),
    ("Amazon (rainforest, river)", -3.1, -60.0),
    ("Sahara (sand and rock desert)", 27.0, 3.0),
    ("Nile Delta (farms, desert, cities)", 30.8, 31.0),
    ("Netherlands (polders, water, towns)", 52.5, 5.5),
    ("Swiss Alps (mountains, snow, glaciers)", 46.5, 8.0),
    ("Finland (boreal forest, lakes)", 62.0, 26.0),
    ("Ganges Plain (smallholder farms)", 26.5, 81.0),
    ("Central Australia (arid savanna)", -23.7, 133.9),
    ("Sundarbans (mangroves, estuaries)", 22.0, 89.2),
]


def _get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def tile_of(lat, lon):
    import mgrs

    s = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)
    m = re.match(r"(\d+)([A-Z])([A-Z]{2})", s)
    return int(m.group(1)), m.group(2), m.group(3)


def items(zone, band, square, year):
    out = []
    for month in range(1, 13):
        prefix = f"{PREFIX}/{zone}/{band}/{square}/{year}/{month}/"
        xml = _get(f"{BUCKET}?list-type=2&delimiter=/&prefix={prefix}").decode()
        out += re.findall(r"<Prefix>(" + re.escape(prefix) + r"[^<]+/)</Prefix>", xml)
    return out


def select(year=2024, max_nodata=1.0):
    """Least cloudy acquisition of each site's tile with (almost) no missing pixels."""
    rows = []
    for name, lat, lon in SITES:
        zone, band, square = tile_of(lat, lon)
        best = None
        for it in items(zone, band, square, year):
            iid = it.rstrip("/").split("/")[-1]
            try:
                p = json.loads(_get(BUCKET + it + iid + ".json"))["properties"]
            except Exception:
                continue
            if p.get("s2:nodata_pixel_percentage", 100) > max_nodata:
                continue
            c = p.get("eo:cloud_cover", 100)
            if best is None or c < best["cloud_cover"]:
                best = dict(site=name, tile=f"{zone}{band}{square}", item=iid, url=BUCKET + it,
                            cloud_cover=c, nodata=p.get("s2:nodata_pixel_percentage"), datetime=p.get("datetime"))
        print(name, best and (best["item"], round(best["cloud_cover"], 2)), flush=True)
        if best:
            rows.append(best)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"), index=False)
    return df


def load(url):
    """(SIZE, SIZE, 10) reflectance cube (float32) and the validity mask at 20 m."""
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    import rasterio
    from rasterio.enums import Resampling

    def read(name, resampling):
        for i in range(4):
            try:
                with rasterio.open("/vsicurl/" + url + name + ".tif") as src:
                    return src.read(1, out_shape=(SIZE, SIZE), resampling=resampling)
            except Exception:
                if i == 3:
                    raise
                time.sleep(2 ** i)

    cube = np.empty((SIZE, SIZE, len(BANDS)), dtype=np.float32)
    for j, b in enumerate(BANDS):
        cube[:, :, j] = read(b, Resampling.average)
    scl = read("SCL", Resampling.nearest)
    valid = ~np.isin(scl, BAD_SCL) & (cube > 0).all(2)
    cube /= 10000.0
    return cube, valid


# ----------------------------------------------------------------------------- the tail run

RULES = ("envi_2pct_10", "pct5_10", "pct2", "sklearn_tol", "markov_auto", "full")


def tail_run(X, k=K, seed=SEED, max_iter=MAX_ITER):
    from sklearn.cluster import kmeans_plusplus

    from markov_kmeans import _rules
    from markov_kmeans._engines import HamerlyEngine
    from markov_kmeans._kernels import compute_centers, inertia

    n = X.shape[0]
    t_init = time.perf_counter()
    C0, _ = kmeans_plusplus(X, k, random_state=seed)
    seed_s = time.perf_counter() - t_init
    t_s = time.perf_counter()
    eng = HamerlyEngine(X, C0)
    assign0_s = time.perf_counter() - t_s
    tol = 1e-4 * float(X.var(0).mean())
    mon = _rules.StateMonitor(n, k, _rules.StopParams(), exact=True)
    t_s = time.perf_counter()
    mon.push(eng.labels, eng.centers)
    mon_s = time.perf_counter() - t_s

    curve, snaps, fired, extra = [], {}, {}, {}
    cum = assign0_s
    t = 0
    markov = None
    while t < max_iter:
        C_old = eng.centers.copy()
        t_s = time.perf_counter()
        changed = eng.step()
        step_s = time.perf_counter() - t_s
        t += 1
        cum += step_s
        share = changed / n
        shift = float(((eng.centers - C_old) ** 2).sum())
        curve.append(dict(t=t, changed=changed, changed_share=share, shift_rel=shift / tol * 1e-4,
                          step_s=step_s, cum_s=cum))
        if t in CHECKPOINTS:
            snaps[t] = eng.labels.astype(np.uint8)

        def fire(rule):
            if rule not in fired:
                fired[rule] = (t, cum)
                snaps[rule] = eng.labels.astype(np.uint8)

        if share < 0.02 or t >= 10:
            fire("envi_2pct_10")
        if share < 0.05 or t >= 10:
            fire("pct5_10")
        if share < 0.02:
            fire("pct2")
        if shift <= tol or changed == 0:
            fire("sklearn_tol")
        if markov is None:
            t_s = time.perf_counter()
            mon.push(eng.labels, eng.centers, changed)
            stop, regime, state = mon.assess()
            mon_s += time.perf_counter() - t_s
            if stop:
                t_s = time.perf_counter()
                L = eng.labels.copy()
                if regime == "converged":
                    U = np.empty(0, dtype=np.int64)
                    lab = L
                else:
                    U = mon.unstable().astype(np.int64)
                    Hw = mon.window(U)
                    lab_U, _, _, _ = _rules.settle(Hw, L[U], rule="auto", n_clusters=k, regime=regime)
                    lab = L.copy()
                    lab[U] = lab_U
                settle_s = time.perf_counter() - t_s
                markov = dict(t=t, regime=regime, reason=("few_unstable" if regime == "drift" and
                                                          state.get("unstable_frac", 1) <= mon.params.unstable_tol
                                                          else {"drift": "stable_drift"}.get(regime, regime)),
                              U=U, settle_s=settle_s, mon_s=mon_s, keep=L.astype(np.uint8))
                fired["markov_auto"] = (t, cum + mon_s + settle_s)
                snaps["markov_auto"] = lab.astype(np.uint8)
                del mon      # the monitor is not needed past its decision
        if changed == 0:
            break
    fired["full"] = (t, cum)
    snaps["full"] = eng.labels.astype(np.uint8)
    for r in RULES:          # rules that never fired stop where the run stopped
        if r not in fired:
            fired[r] = (t, cum)
            snaps[r] = eng.labels.astype(np.uint8)

    ref = snaps["full"]
    C_ref = compute_centers(X, ref.astype(np.int64), eng.centers)[0]
    sse_ref = inertia(X, C_ref, ref.astype(np.int64))
    rows = []
    for r in RULES:
        lab = snaps[r].astype(np.int64)
        C = compute_centers(X, lab, eng.centers)[0]
        rows.append(dict(rule=r, iterations=fired[r][0], seconds=fired[r][1], time_frac=fired[r][1] / cum,
                         error=float((snaps[r] != ref).mean()), sse_excess=inertia(X, C, lab) / sse_ref - 1))
    for c in curve:
        c["error"] = float((snaps[c["t"]] != ref).mean()) if c["t"] in snaps else np.nan
    if markov is not None:
        U, wrong = markov["U"], snaps["markov_auto"] != ref
        wrong_keep = markov["keep"] != ref
        inU = np.zeros(n, dtype=bool)
        inU[U] = True
        extra = dict(markov_reason=markov["reason"], markov_regime=markov["regime"], U_frac=len(U) / n,
                     U_precision=float(wrong[inU].mean()) if len(U) else np.nan,
                     U_recall=float((wrong & inU).sum() / max(wrong.sum(), 1)),
                     U_error_keep=float(wrong_keep[inU].mean()) if len(U) else np.nan,
                     U_error_markov=float(wrong[inU].mean()) if len(U) else np.nan,
                     monitor_s=markov["mon_s"], settle_s=markov["settle_s"])
    extra.update(n=n, k=k, converged=bool(curve[-1]["changed"] == 0), total_iter=t, total_s=cum, seed_s=seed_s,
                 assign0_s=assign0_s)
    return pd.DataFrame(rows), pd.DataFrame(curve), extra


# ----------------------------------------------------------------------------- 60 m mixing study

def mixing(cube, valid, k=K, seed=SEED):
    """Undecided 60 m pixels and soft memberships vs the mixture of their 20 m sub-pixels."""
    from sklearn.metrics import roc_auc_score

    from markov_kmeans import MarkovKMeans

    m = SIZE // 3
    c = cube[:3 * m, :3 * m].reshape(m, 3, m, 3, -1)
    ok = valid[:3 * m, :3 * m].reshape(m, 3, m, 3).all((1, 3))
    X60 = c.mean((1, 3))[ok].astype(np.float64)
    sub = c.transpose(0, 2, 1, 3, 4)[ok].reshape(-1, cube.shape[2])                       # (9 n60, d), float32
    mk = MarkovKMeans(k, random_state=seed).fit(X60)
    C = mk.cluster_centers_
    # sub-pixel classes under the same centers, in chunks
    lab_sub = np.empty(len(sub), dtype=np.int64)
    c_sq = (C ** 2).sum(1)
    for s0 in range(0, len(sub), 1 << 20):
        xb = sub[s0:s0 + (1 << 20)].astype(np.float64)
        lab_sub[s0:s0 + (1 << 20)] = (c_sq - 2 * xb @ C.T).argmin(1)
    lab_sub = lab_sub.reshape(-1, 9)
    share = np.zeros((len(X60), k))
    np.add.at(share, (np.repeat(np.arange(len(X60)), 9), lab_sub.ravel()), 1.0 / 9)
    purity = share.max(1)
    mixed_any = purity < 1.0 - 1e-9
    mixed = purity <= 6 / 9 + 1e-9          # strong mixing: the majority class covers at most 6 of 9 sub-pixels
    D = mk.transform(X60)
    D.sort(1)
    ratio = D[:, 0] / np.maximum(D[:, 1], 1e-12)      # geometric ambiguity: near 1 = between two centers
    inU = np.zeros(len(X60), dtype=bool)
    inU[mk.unstable_indices_] = True
    unc = np.zeros(len(X60))
    unc[mk.unstable_indices_] = mk.uncertainty_
    out = dict(n60=len(X60), stop_reason=mk.stop_reason_, regime=mk.regime_, iterations=mk.n_iter_,
               U_frac=float(inU.mean()), mixed_any_frac=float(mixed_any.mean()), mixed_frac=float(mixed.mean()),
               mixed_frac_in_U=float(mixed[inU].mean()) if inU.any() else np.nan,
               mixed_frac_outside_U=float(mixed[~inU].mean()),
               auroc_U=float(roc_auc_score(mixed, inU)) if inU.any() else np.nan,
               auroc_uncertainty=float(roc_auc_score(mixed, unc)) if inU.any() else np.nan,
               auroc_distance_ratio=float(roc_auc_score(mixed, ratio)))
    if inU.any():
        # among undecided pixels with a fractional membership: does pi of the label track the sub-pixel share?
        M = mk.membership_.tocsr()
        idx = mk.unstable_indices_
        pi_lab = np.asarray(M[idx, mk.labels_[idx]]).ravel()
        frac_idx = idx[pi_lab < 1 - 1e-9]
        if len(frac_idx) > 20:
            pi_f = np.asarray(M[frac_idx, mk.labels_[frac_idx]]).ravel()
            sh_f = share[frac_idx, mk.labels_[frac_idx]]
            out.update(n_fractional=len(frac_idx), corr_pi_share=float(np.corrcoef(pi_f, sh_f)[0, 1]),
                       mean_pi=float(pi_f.mean()), mean_share=float(sh_f.mean()))
        else:
            out.update(n_fractional=len(frac_idx))
    return out


# ----------------------------------------------------------------------------- driver

def run(only=None):
    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    f_rules, f_curves, f_mix, f_runs = (os.path.join(RESULTS, f"sentinel2_{s}.csv")
                                        for s in ("rules", "curves", "mixing", "runs"))
    done = set(pd.read_csv(f_runs).site) if os.path.exists(f_runs) else set()
    for _, s in scenes.iterrows():
        if (only and s.site not in only) or s.site in done:
            continue
        t0 = time.perf_counter()
        cube, valid = load(s.url)
        load_s = time.perf_counter() - t0
        X = np.ascontiguousarray(cube[valid], dtype=np.float64)
        print(f"{s.site}: n={len(X):,} valid={valid.mean():.1%} loaded in {load_s:.0f}s", flush=True)
        rules, curve, extra = tail_run(X)
        del X
        mix = mixing(cube, valid)
        del cube, valid
        for df, f in ((rules, f_rules), (curve, f_curves)):
            df.insert(0, "site", s.site)
            df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        pd.DataFrame([dict(site=s.site, **mix)]).to_csv(f_mix, mode="a", header=not os.path.exists(f_mix), index=False)
        pd.DataFrame([dict(site=s.site, tile=s.tile, item=s.item, cloud_cover=s.cloud_cover, valid_load_s=load_s,
                           **extra)]).to_csv(f_runs, mode="a", header=not os.path.exists(f_runs), index=False)
        print(rules.round(5).to_string(index=False), flush=True)
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in {**extra, **mix}.items()}, flush=True)


# ----------------------------------------------------------------------------- report

RULE_NAMES = {"envi_2pct_10": "changed < 2 % or 10 iterations (ENVI)", "pct5_10": "changed < 5 % or 10 iterations",
              "pct2": "changed < 2 %", "sklearn_tol": "scikit-learn tol", "markov_auto": "Markov-K-means (auto)",
              "full": "to convergence"}
RULE_KO = {"envi_2pct_10": "변경 < 2 % 또는 10회 (ENVI 기본값)", "pct5_10": "변경 < 5 % 또는 10회",
           "pct2": "변경 < 2 % (상한 없음)", "sklearn_tol": "scikit-learn tol (기본값)",
           "markov_auto": "Markov-K-means (기본값)", "full": "끝까지 (기준)"}


def report():
    from experiments.analyze_markov_tail import FIG, GRID, INK, INK2, SERIES, SURFACE, _style, fill_report, md, plt

    rules = pd.read_csv(os.path.join(RESULTS, "sentinel2_rules.csv"))
    curves = pd.read_csv(os.path.join(RESULTS, "sentinel2_curves.csv"))
    runs = pd.read_csv(os.path.join(RESULTS, "sentinel2_runs.csv")).set_index("site")
    mix = pd.read_csv(os.path.join(RESULTS, "sentinel2_mixing.csv")).set_index("site")
    sites = list(runs.index)
    short = {s_: s_.split(" (")[0] for s_ in sites}
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    for s_ in sites:
        c = curves[curves.site == s_]
        ax.plot(c.t, c.changed_share.clip(lower=1e-9) * 100, color=INK2, lw=1, alpha=0.55)
        r = rules[(rules.site == s_)].set_index("rule")
        for rule, col, mk in (("envi_2pct_10", SERIES[1], "s"), ("markov_auto", SERIES[0], "o")):
            t_ = int(r.loc[rule, "iterations"])
            y = c[c.t == t_].changed_share.clip(lower=1e-9).iloc[0] * 100
            ax.scatter([t_], [y], color=col, marker=mk, s=40, zorder=4, edgecolor=SURFACE, linewidth=1)
    ax.axhline(2, color=SERIES[1], lw=1, ls="--")
    ax.axhline(5, color=SERIES[3], lw=1, ls="--")
    ax.axvline(10, color=GRID, lw=1.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter([], [], color=SERIES[1], marker="s", label="ENVI rule stops (2 % or 10 iterations)")
    ax.scatter([], [], color=SERIES[0], marker="o", label="Markov-K-means stops")
    ax.plot([], [], color=INK2, lw=1, label="one Sentinel-2 scene")
    _style(ax, "Pixels changing cluster per iteration (12 scenes, k = 12)", "pixels changing cluster (%, log)",
           "iteration (log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")

    ax = axes[1]
    style = {"envi_2pct_10": (SERIES[1], "s"), "pct5_10": (SERIES[3], "D"), "pct2": (SERIES[4], "v"),
             "sklearn_tol": (SERIES[2], "^"), "markov_auto": (SERIES[0], "o")}
    for rule, (col, mk) in style.items():
        d = rules[rules.rule == rule]
        ax.scatter(d.time_frac, d.error.clip(lower=1e-6) * 100, color=col, marker=mk, s=28, alpha=0.6,
                   edgecolor="none")
        ax.scatter([d.time_frac.median()], [d.error.median() * 100], color=col, marker=mk, s=150,
                   edgecolor=INK, linewidth=1.2, zorder=5, label=RULE_NAMES[rule])
    ax.set_yscale("log")
    _style(ax, "Time and error of each stopping rule", "labels differing from the converged run (%, log)",
           "time / run to convergence (large marker: median over scenes)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_sentinel2_tail.png", dpi=150)
    plt.close(fig)

    T = {}
    rows = {}
    for s_ in sites:
        r = rules[rules.site == s_].set_index("rule")
        c = curves[curves.site == s_].set_index("t")
        cap = bool(r.loc["envi_2pct_10", "iterations"] == 10 and c.loc[10, "changed_share"] >= 0.02) if 10 in c.index else False
        rows[short[s_]] = {"화소 수 (백만)": runs.loc[s_, "n"] / 1e6, "수렴 반복": runs.loc[s_, "total_iter"],
                           "수렴 여부": "예" if runs.loc[s_, "converged"] else "상한",
                           "ENVI 정지": f"{int(r.loc['envi_2pct_10', 'iterations'])}" + (" (상한)" if cap else ""),
                           "2 % 도달": int(r.loc["pct2", "iterations"]),
                           "Markov 정지": f"{int(r.loc['markov_auto', 'iterations'])} ({runs.loc[s_, 'markov_reason']})"}
    T["sentinel2_scenes"] = md(pd.DataFrame(rows).T, {"화소 수 (백만)": "{:.1f}", "수렴 반복": "{:.0f}", "수렴 여부": "{}",
                                                      "ENVI 정지": "{}", "2 % 도달": "{:.0f}", "Markov 정지": "{}"},
                               index_name="장면")
    agg = {}
    for rule in RULES:
        d = rules[rules.rule == rule]
        agg[RULE_KO[rule]] = {"시간 (중앙값)": d.time_frac.median(),
                              "시간 (범위)": f"{d.time_frac.min():.0%}–{d.time_frac.max():.0%}",
                              "오차 (중앙값)": d.error.median(), "오차 (최대)": d.error.max(),
                              "SSE 초과 (중앙값)": d.sse_excess.median()}
    T["sentinel2_rules"] = md(pd.DataFrame(agg).T, {"시간 (중앙값)": "{:.0%}", "시간 (범위)": "{}", "오차 (중앙값)": "{:.2%}",
                                                    "오차 (최대)": "{:.2%}", "SSE 초과 (중앙값)": "{:.3%}"},
                              index_name="정지 규칙")
    u = runs[["U_frac", "U_precision", "U_recall", "U_error_keep", "U_error_markov"]].copy()
    u.index = [short[s_] for s_ in u.index]
    u.columns = ["미결정 비율", "미결정 중 틀린 비율", "틀린 점 중 미결정 비율", "미결정: 유지 오차", "미결정: 정리 오차"]
    T["sentinel2_unstable"] = md(u, "{:.2%}", index_name="장면")
    m = mix[["U_frac", "mixed_frac_in_U", "mixed_frac_outside_U", "auroc_U", "auroc_uncertainty", "auroc_distance_ratio",
             "corr_pi_share"]].copy()
    m.index = [short[s_] for s_ in m.index]
    m.columns = ["미결정 비율", "혼합 화소 비율: 미결정", "혼합 화소 비율: 나머지", "AUROC: 미결정 여부", "AUROC: 1 − max π",
                 "AUROC: 거리 비", "상관: π와 하위 화소 비율"]
    T["sentinel2_mixing"] = md(m, {c_: ("{:.2f}" if c_.startswith(("AUROC", "상관")) else "{:.1%}") for c_ in m.columns},
                               index_name="장면")
    fill_report(T)
    for k_, v in T.items():
        print(f"== {k_}\n{v}")
    return T


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "select":
        select()
    elif cmd == "run":
        run(sys.argv[2:] or None)
    elif cmd == "report":
        report()
