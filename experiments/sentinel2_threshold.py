"""One changed-share threshold for every scene: how close does it come to Markov-K-means?

The Pérez-Ortega rule stops at the first iteration where the share of pixels that changed
cluster falls below a threshold. Every threshold of a sweep is applied to all twelve scenes of
experiments/sentinel2_tail.py (and to the wide-area mosaic of experiments/sentinel2_wide.py)
from the saved curves, with no new clustering:

    time    engine time at the stop / engine time of the full run (the rule itself costs nothing)
    error   share of pixels whose label differs from the converged map; between the saved
            checkpoints it is bounded by the label changes still to come (a pixel wrong at the
            stop must change at least once more). The bound matches the measured error at the
            checkpoints to about 1 % on the local scenes; with k = 256 pixels switch back and forth
            and it is 1.4-1.5 times too high, so ``error_est`` scales it by the measured / bound
            ratio interpolated between the checkpoints

    python -m experiments.sentinel2_threshold    # -> experiments/results/sentinel2_threshold.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import RESULTS

THRESHOLDS = (1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5)


def sweep(curve, site, thresholds=THRESHOLDS):
    g = curve.sort_values("t").reset_index(drop=True)
    n = float(g.changed.iloc[0] / g.changed_share.iloc[0])
    later = (g.changed[::-1].cumsum()[::-1].shift(-1).fillna(0) / n).to_numpy()
    total = float(g.cum_s.iloc[-1])
    chk = g.dropna(subset=["error"]).set_index("t").error
    ok = (chk > 0) & (pd.Series(later, index=g.t)[chk.index] > 0)
    ratio_t = chk.index[ok].to_numpy()
    ratio = (chk[ok] / pd.Series(later, index=g.t)[chk.index[ok]]).to_numpy()
    rows = []
    for th in thresholds:
        below = np.flatnonzero(g.changed_share.to_numpy() < th)
        i = int(below[0]) if len(below) else len(g) - 1
        t = int(g.t[i])
        rows.append(dict(site=site, threshold=th, iterations=t, time_frac=g.cum_s[i] / total,
                         error_bound=later[i], error_est=later[i] * np.interp(t, ratio_t, ratio),
                         error_measured=chk.get(t, np.nan)))
    return rows


def bound_tightness(curve):
    """Median ratio bound / measured error over the checkpoints from iteration 30 on."""
    g = curve.sort_values("t").reset_index(drop=True)
    n = float(g.changed.iloc[0] / g.changed_share.iloc[0])
    later = g.changed[::-1].cumsum()[::-1].shift(-1).fillna(0) / n
    k = g.error.notna() & (g.t >= 30) & (g.error > 0)
    return float((later[k] / g.error[k]).median())


def run():
    rows, tight = [], []
    curves = pd.read_csv(os.path.join(RESULTS, "sentinel2_curves.csv"))
    for site, g in curves.groupby("site", sort=False):
        rows += sweep(g, site)
        tight.append(bound_tightness(g))
    wide = pd.read_csv(os.path.join(RESULTS, "sentinel2_wide_curves.csv"))
    rows += [dict(r, scope="wide") for r in sweep(wide, "Southern Korean peninsula (k = 256)")]
    df = pd.DataFrame(rows)
    df["scope"] = df["scope"].fillna("local") if "scope" in df else "local"
    df.to_csv(os.path.join(RESULTS, "sentinel2_threshold.csv"), index=False)
    print("bound / measured error at checkpoints:", np.round(tight, 3))
    loc = df[df.scope == "local"]
    print(loc.groupby("threshold", sort=False).agg(time=("time_frac", "median"), error=("error_est", "median"),
                                                   worst=("error_est", "max")).round(4).to_string())
    print(df[df.scope == "wide"].round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    run()
