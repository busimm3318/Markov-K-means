"""The 0.6 default path against the 0.5 default on the twelve Sentinel-2 scenes, in one run.

tail_run judges every stopping rule on the same exact K-means iterates; besides the 0.5 default
(markov_auto: the full monitor, history settlement, pseudo representatives) it runs the 0.6 path
(markov_06: stop_rule="unstable" with a monitor that keeps no label history, labels kept in a
drift, pseudo representatives) with its own monitor and timing. The 60 m mixing study of
experiments/sentinel2_tail.py is not repeated, and its result files are left untouched.

    python -m experiments.sentinel2_simple    # -> experiments/results/sentinel2_simple_{rules,runs}.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.sentinel2_tail import RESULTS, load, tail_run


def run(only=None):
    scenes = pd.read_csv(os.path.join(RESULTS, "sentinel2_scenes.csv"))
    f_rules, f_runs = (os.path.join(RESULTS, f"sentinel2_simple_{s}.csv") for s in ("rules", "runs"))
    done = set(pd.read_csv(f_runs).site) if os.path.exists(f_runs) else set()
    for _, s in scenes.iterrows():
        if (only and s.site not in only) or s.site in done:
            continue
        t0 = time.perf_counter()
        cube, valid = load(s.url)
        X = np.ascontiguousarray(cube[valid], dtype=np.float64)
        del cube, valid
        rules, _, extra = tail_run(X)
        del X
        rules.insert(0, "site", s.site)
        rules.to_csv(f_rules, mode="a", header=not os.path.exists(f_rules), index=False)
        pd.DataFrame([dict(site=s.site, **extra)]).to_csv(f_runs, mode="a", header=not os.path.exists(f_runs),
                                                          index=False)
        r = rules.set_index("rule")
        print(s.site, f"{time.perf_counter() - t0:.0f}s",
              {x: (int(r.loc[x, "iterations"]), round(r.loc[x, "time_frac"], 4), round(r.loc[x, "error"], 5))
               for x in ("markov_auto", "markov_06")},
              f"monitor {extra['monitor_s']:.1f}s vs {extra.get('monitor06_s', float('nan')):.1f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
