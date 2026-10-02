"""Benchmark harness for the K-means efficiency testbed.

Usage:
    python -m benchmarks.bench exact     [--quick]
    python -m benchmarks.bench seeding   [--quick]
    python -m benchmarks.bench approx    [--quick]
    python -m benchmarks.bench chain     [--quick]   # MCMC chain-length study
    python -m benchmarks.bench systems   [--quick]

Every run is single-threaded (numba kernels are serial; BLAS/OpenMP limited to one
thread through threadpoolctl) unless a row says otherwise, and timings are the
minimum over ``--repeats`` runs after a JIT warm-up.  Rows are appended to
``benchmarks/results/<experiment>.csv`` as they finish, so partial runs are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from kmeans_accel import datasets
from kmeans_accel.lloyd import lloyd
from kmeans_accel.registry import discover

RESULTS = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------------- helpers

def parse_spec(spec):
    """'afkmc2:chain_length=50' -> ('afkmc2', {'chain_length': 50})."""
    name, _, rest = spec.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        key, val = kv.split("=")
        try:
            params[key] = json.loads(val)
        except json.JSONDecodeError:
            params[key] = val
    return name, params


def make_data(cfg):
    kind = cfg["data"]
    if kind == "blobs":
        return datasets.blobs(cfg["n"], cfg["d"], cfg["k_true"], sep=cfg.get("sep", 4.0), seed=cfg.get("seed", 0))
    if kind == "uniform":
        return datasets.uniform(cfg["n"], cfg["d"], seed=cfg.get("seed", 0))
    return datasets.load(kind, n=cfg.get("n"), seed=cfg.get("seed", 0))


def data_label(cfg):
    if cfg["data"] == "blobs":
        return f"blobs(n={cfg['n']},d={cfg['d']},k*={cfg['k_true']},sep={cfg.get('sep', 4.0)})"
    if cfg["data"] == "uniform":
        return f"uniform(n={cfg['n']},d={cfg['d']})"
    return f"{cfg['data']}(n={cfg.get('n') or 'all'})"


def timed(fn, repeats):
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


class Writer:
    def __init__(self, name):
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.path = RESULTS / f"{name}.csv"
        self.fields = None

    def write(self, row):
        row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
        new = not self.path.exists()
        if self.fields is None:
            self.fields = list(row)
            if not new:
                with self.path.open() as f:
                    header = next(csv.reader(f), None)
                if header:
                    self.fields = header
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(" ".join(f"{k}={row[k]}" for k in ("dataset", "k", "method", "seconds", "n_dist_ratio", "rel_inertia") if k in row), flush=True)


def warmup(registry):
    X = datasets.blobs(300, 3, 4, seed=0)
    C0 = datasets.init_random(X, 5, seed=0)
    lloyd(X, C0)
    for name, meta in registry.items():
        try:
            if meta["kind"] == "seeding":
                meta["fn"](X, 5, seed=0)
            else:
                meta["fn"](X, C0, max_iter=5) if meta["kind"] != "approx" else meta["fn"](X, C0, seed=0)
        except Exception:
            print(f"warm-up failed for {name}", file=sys.stderr)
            traceback.print_exc()


def env_info():
    import numba
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
            "sklearn": sklearn.__version__, "cpu": platform.processor() or platform.machine(),
            "cpus": os.cpu_count()}


# ----------------------------------------------------------------------------- experiments

EXACT_GRID = [
    *[dict(data="blobs", n=50000, d=d, k_true=50, sep=2.0, k=k) for d in (2, 8, 32, 128) for k in (10, 50, 200)],
    *[dict(data="uniform", n=50000, d=d, k=k) for d in (2, 16) for k in (10, 50, 200)],
    dict(data="birch1", k=100), dict(data="birch3", k=100),
    dict(data="fashion_mnist", n=20000, k=10), dict(data="fashion_mnist", n=20000, k=50),
    dict(data="fashion_mnist", n=20000, k=200),
]
EXACT_QUICK = [
    dict(data="blobs", n=10000, d=8, k_true=20, sep=2.0, k=50),
    dict(data="uniform", n=10000, d=2, k=50),
    dict(data="fashion_mnist", n=3000, k=20),
]


def sklearn_fit(algorithm):
    from sklearn.cluster import KMeans

    def fit(X, C0, max_iter=300):
        km = KMeans(n_clusters=C0.shape[0], init=C0, n_init=1, max_iter=max_iter, tol=0.0,
                    algorithm=algorithm).fit(X)
        return km
    return fit


def run_exact(args):
    registry, errors = discover()
    if errors:
        print("import errors:", errors, file=sys.stderr)
    methods = [n for n, m in registry.items() if m["kind"] in ("exact", "exact-gemm")]
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(",")]
    warmup({n: registry[n] for n in methods})
    w = Writer("exact" + args.suffix)
    for cfg in (EXACT_QUICK if args.quick else EXACT_GRID):
        X = make_data(cfg)
        k = cfg["k"]
        for init_seed in range(args.inits):
            C0 = datasets.init_random(X, k, seed=init_seed)
            ref, ref_t = timed(lambda: lloyd(X, C0, max_iter=args.max_iter), args.repeats)
            base = dict(dataset=data_label(cfg), n=X.shape[0], d=X.shape[1], k=k, init_seed=init_seed,
                        n_iter=ref.n_iter, lloyd_seconds=ref_t)
            for name in ["sklearn_lloyd", "sklearn_elkan", *methods]:
                try:
                    if name.startswith("sklearn_"):
                        km, t = timed(lambda: sklearn_fit(name[8:])(X, C0, args.max_iter), args.repeats)
                        row = dict(method=name, seconds=t, method_n_iter=km.n_iter_,
                                   rel_inertia=km.inertia_ / ref.inertia, n_dist_ratio="", same_labels=bool(np.array_equal(km.labels_, ref.labels)))
                    else:
                        res, t = timed(lambda: registry[name]["fn"](X, C0, max_iter=args.max_iter), args.repeats)
                        row = dict(method=name, seconds=t, method_n_iter=res.n_iter,
                                   rel_inertia=res.inertia / ref.inertia, n_dist=res.n_dist,
                                   n_dist_ratio=res.n_dist / ref.n_dist, n_aux_dist=res.n_aux_dist,
                                   same_labels=bool(np.array_equal(res.labels, ref.labels)))
                    row["speedup_vs_lloyd"] = ref_t / row["seconds"]
                    w.write({**base, **row})
                except Exception as exc:
                    w.write({**base, "method": name, "error": repr(exc)})


SEED_GRID = [
    dict(data="blobs", n=50000, d=16, k_true=100, sep=2.0, k=100),
    dict(data="blobs", n=50000, d=16, k_true=100, sep=2.0, k=500),
    dict(data="birch1", k=100), dict(data="birch3", k=100),
    dict(data="fashion_mnist", n=60000, k=50), dict(data="fashion_mnist", n=60000, k=200),
    dict(data="uniform", n=50000, d=8, k=100),
]
SEED_QUICK = [dict(data="blobs", n=10000, d=8, k_true=30, sep=2.0, k=30), dict(data="fashion_mnist", n=5000, k=20)]
SEED_METHODS = ["forgy", "kmeanspp", "greedy_kmeanspp", "kmeanspp_raff", "kmeans_parallel",
                "kmc2:chain_length=20", "kmc2:chain_length=200", "afkmc2:chain_length=20",
                "afkmc2:chain_length=50", "afkmc2:chain_length=200"]


def _final_lloyd(X, C, registry, max_iter):
    fast = registry.get("hamerly") or registry.get("elkan")
    return (fast["fn"] if fast else lloyd)(X, C, max_iter=max_iter)


def run_seeding(args, methods=None, grid=None, name="seeding"):
    registry, _ = discover()
    specs = (args.methods.split(",") if args.methods else methods or SEED_METHODS)
    specs = [s for s in specs if parse_spec(s)[0] in registry]
    warmup({parse_spec(s)[0]: registry[parse_spec(s)[0]] for s in specs})
    w = Writer(name + args.suffix)
    for cfg in grid or (SEED_QUICK if args.quick else SEED_GRID):
        X = make_data(cfg)
        k = cfg["k"]
        for spec in specs:
            algo, params = parse_spec(spec)
            for seed in range(args.inits):
                base = dict(dataset=data_label(cfg), n=X.shape[0], d=X.shape[1], k=k, method=spec, seed=seed)
                try:
                    sr, t = timed(lambda: registry[algo]["fn"](X, k, seed=seed, **params), args.repeats)
                    row = dict(seconds=t, n_dist=sr.n_dist, n_dist_per_nk=sr.n_dist / (X.shape[0] * k),
                               seed_cost=sr.extra.get("seed_cost"))
                    if not args.no_lloyd:
                        fin = _final_lloyd(X, sr.centers, registry, args.max_iter)
                        row.update(final_inertia=fin.inertia, final_n_iter=fin.n_iter,
                                   final_n_dist=fin.n_dist, final_seconds=fin.seconds)
                    w.write({**base, **row})
                except Exception as exc:
                    w.write({**base, "error": repr(exc)})


CHAIN_LENGTHS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def run_chain(args):
    specs = [f"{a}:chain_length={m}" for a in ("kmc2", "afkmc2") for m in CHAIN_LENGTHS] + ["kmeanspp", "forgy"]
    grid = SEED_QUICK if args.quick else [
        dict(data="blobs", n=50000, d=16, k_true=100, sep=2.0, k=100),
        dict(data="birch1", k=100),
        dict(data="fashion_mnist", n=60000, k=100),
    ]
    run_seeding(args, methods=specs, grid=grid, name="chain")


APPROX_GRID = [
    dict(data="blobs", n=200000, d=16, k_true=100, sep=2.0, k=100),
    dict(data="birch1", k=100),
    dict(data="fashion_mnist", n=60000, k=50),
    dict(data="uniform", n=100000, d=8, k=100),
]
APPROX_QUICK = [dict(data="blobs", n=20000, d=8, k_true=30, sep=2.0, k=30)]


def run_approx(args):
    registry, _ = discover()
    specs = args.methods.split(",") if args.methods else [n for n, m in registry.items() if m["kind"] == "approx"]
    warmup({parse_spec(s)[0]: registry[parse_spec(s)[0]] for s in specs if parse_spec(s)[0] in registry})
    w = Writer("approx" + args.suffix)
    fast = registry.get("hamerly", {"fn": lloyd})["fn"]
    for cfg in APPROX_QUICK if args.quick else APPROX_GRID:
        X = make_data(cfg)
        k = cfg["k"]
        for init_seed in range(args.inits):
            C0 = datasets.init_random(X, k, seed=init_seed)
            ref, ref_t = timed(lambda: fast(X, C0, max_iter=args.max_iter), 1)
            lref = lloyd(X, C0, max_iter=args.max_iter) if args.lloyd_ref else None
            base = dict(dataset=data_label(cfg), n=X.shape[0], d=X.shape[1], k=k, init_seed=init_seed,
                        exact_seconds=ref_t, exact_n_iter=ref.n_iter, exact_n_dist=ref.n_dist)
            for spec in specs:
                algo, params = parse_spec(spec)
                try:
                    res, t = timed(lambda: registry[algo]["fn"](X, C0, seed=init_seed, **params), args.repeats)
                    w.write({**base, "method": spec, "seconds": t, "method_n_iter": res.n_iter,
                             "rel_inertia": res.inertia / ref.inertia, "n_dist": res.n_dist,
                             "n_dist_ratio": res.n_dist / ref.n_dist,
                             "lloyd_n_dist_ratio": (res.n_dist / lref.n_dist) if lref else "",
                             "speedup_vs_exact_fast": ref_t / t,
                             "extra": {k2: v for k2, v in res.extra.items() if np.isscalar(v)}})
                except Exception as exc:
                    w.write({**base, "method": spec, "error": repr(exc)})


def run_systems(args):
    """Lloyd variants: scalar numba vs BLAS GEMM vs scikit-learn, 1 and all threads."""
    w = Writer("systems" + args.suffix)
    grid = EXACT_QUICK if args.quick else [
        dict(data="blobs", n=100000, d=d, k_true=50, sep=2.0, k=k) for d in (4, 32, 256) for k in (16, 256)
    ] + [dict(data="fashion_mnist", n=60000, k=100)]
    registry, _ = discover()
    warmup({n: registry[n] for n in ("lloyd", "lloyd_gemm")})
    for cfg in grid:
        X = make_data(cfg)
        C0 = datasets.init_random(X, cfg["k"], seed=0)
        iters = 10  # fixed iteration count: per-iteration throughput is what matters here
        for threads in (1, os.cpu_count()):
            with threadpool_limits(limits=threads):
                for name, fn in (("lloyd_numba", lambda: lloyd(X, C0, max_iter=iters)),
                                 ("lloyd_gemm", lambda: registry["lloyd_gemm"]["fn"](X, C0, max_iter=iters)),
                                 ("sklearn_lloyd", lambda: sklearn_fit("lloyd")(X, C0, iters)),
                                 ("sklearn_elkan", lambda: sklearn_fit("elkan")(X, C0, iters))):
                    if name == "lloyd_numba" and threads != 1:
                        continue
                    _, t = timed(fn, args.repeats)
                    w.write(dict(dataset=data_label(cfg), n=X.shape[0], d=X.shape[1], k=cfg["k"], method=name,
                                 threads=threads, iters=iters, seconds=t, seconds_per_iter=t / iters,
                                 gflops_eff=3 * X.shape[0] * cfg["k"] * X.shape[1] * iters / t / 1e9))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["exact", "seeding", "approx", "chain", "systems", "env"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inits", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--methods", default="")
    p.add_argument("--suffix", default="")
    p.add_argument("--no-lloyd", action="store_true")
    p.add_argument("--lloyd-ref", action="store_true")
    args = p.parse_args(argv)
    if args.experiment == "env":
        print(json.dumps(env_info(), indent=2))
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "env.json").write_text(json.dumps(env_info(), indent=2))
    limit = None if args.experiment == "systems" else 1
    with threadpool_limits(limits=limit):
        {"exact": run_exact, "seeding": run_seeding, "approx": run_approx,
         "chain": run_chain, "systems": run_systems}[args.experiment](args)


if __name__ == "__main__":
    main()
