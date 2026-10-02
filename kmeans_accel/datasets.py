"""Synthetic generators and cached loaders for real benchmark datasets.

Real datasets are downloaded once into ``$KMEANS_DATA_DIR`` (default ``<repo>/data``).
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("KMEANS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

_URLS = {
    "fashion_mnist": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
    "birch1": "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/birch-rg1.arff",
    "birch2": "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/birch-rg2.arff",
    "birch3": "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/birch-rg3.arff",
}


# ----------------------------------------------------------------------------- synthetic

def blobs(n, d, k_true, sep=4.0, seed=0):
    """Isotropic unit-variance Gaussian mixture; centers ~ N(0, sep^2 I).

    Larger ``sep`` means better separated clusters (where bound-based pruning shines).
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=sep, size=(k_true, d))
    comp = rng.integers(k_true, size=n)
    return np.ascontiguousarray(means[comp] + rng.normal(size=(n, d)))


def uniform(n, d, seed=0):
    """Unclustered data: the hard case for every pruning method."""
    return np.random.default_rng(seed).random((n, d))


def init_random(X, k, seed=0):
    """k distinct data points chosen uniformly (Forgy); shared default init for exact methods."""
    rng = np.random.default_rng(seed)
    return np.array(X[rng.choice(X.shape[0], size=k, replace=False)], dtype=np.float64)


# ----------------------------------------------------------------------------- real

def _fetch(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = _URLS[name]
    path = DATA_DIR / url.rsplit("/", 1)[1]
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _subsample(X, n, seed):
    if n is None or n >= X.shape[0]:
        return np.ascontiguousarray(X, dtype=np.float64)
    idx = np.random.default_rng(seed).choice(X.shape[0], size=n, replace=False)
    return np.ascontiguousarray(X[np.sort(idx)], dtype=np.float64)


def fashion_mnist(n=None, seed=0):
    """Fashion-MNIST training images, 60000 x 784, pixel values scaled to [0, 1]."""
    cache = DATA_DIR / "fashion_mnist.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        with gzip.open(_fetch("fashion_mnist"), "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        X = raw.reshape(-1, 784).astype(np.float32) / 255.0
        np.save(cache, X)
    return _subsample(X, n, seed)


def birch(which=1, n=None, seed=0):
    """Joensuu BIRCH sets (100000 x 2, 100 true clusters): 1 = grid, 2 = sine, 3 = random."""
    name = f"birch{which}"
    cache = DATA_DIR / f"{name}.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        rows, data = [], False
        for line in _fetch(name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                data = True
                continue
            if data:
                rows.append([float(v) for v in line.split(",")[:2]])
        X = np.asarray(rows)
        np.save(cache, X)
    X = (X - X.mean(0)) / X.std(0)
    return _subsample(X, n, seed)


def digits():
    """scikit-learn's bundled 8x8 digits, 1797 x 64 (no download)."""
    from sklearn.datasets import load_digits

    return np.ascontiguousarray(load_digits().data, dtype=np.float64)


def load(name, n=None, seed=0):
    if name == "fashion_mnist":
        return fashion_mnist(n, seed)
    if name.startswith("birch"):
        return birch(int(name[-1]), n, seed)
    if name == "digits":
        return digits()
    raise KeyError(name)
