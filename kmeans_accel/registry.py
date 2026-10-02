"""Auto-discovery of algorithms.

Every module in the package may expose ``ALGORITHMS = {name: meta}`` where ``meta`` has
``kind`` ("exact" | "exact-gemm" | "approx" | "seeding"), ``fit`` (name of the
callable in that module) and ``ref`` (citation).  Modules never edit a shared list,
so methods can be added independently.

Signatures by kind:
    exact / exact-gemm : fit(X, init_centers, max_iter=300, **params) -> KMeansResult
    approx             : fit(X, init_centers, max_iter=..., seed=0, **params) -> KMeansResult
    seeding            : fit(X, k, seed=0, **params) -> SeedResult
"""
from __future__ import annotations

import importlib
import pkgutil

_SKIP = {"kmeans_accel.registry", "kmeans_accel.core", "kmeans_accel.datasets"}


def discover(strict: bool = False):
    import kmeans_accel

    registry, errors = {}, {}
    for info in pkgutil.walk_packages(kmeans_accel.__path__, "kmeans_accel."):
        if info.name in _SKIP:
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception as exc:  # keep one broken module from hiding the others
            if strict:
                raise
            errors[info.name] = repr(exc)
            continue
        for name, meta in getattr(mod, "ALGORITHMS", {}).items():
            if name in registry:
                raise ValueError(f"duplicate algorithm name {name!r} in {info.name}")
            registry[name] = dict(meta, name=name, fn=getattr(mod, meta["fit"]), module=info.name)
    return registry, errors


def by_kind(*kinds):
    registry, _ = discover()
    return {n: m for n, m in registry.items() if m["kind"] in kinds}
