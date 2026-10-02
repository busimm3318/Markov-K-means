#!/usr/bin/env bash
# Reproduces experiments/results/markov_tail_*.csv (see docs/results.md).
# Exact-regime runs are single-threaded so Hamerly and BLAS Lloyd timings are comparable.
set -u
cd "$(dirname "$0")/.."
queue="${1:-all}"
exact() {
  for cfg in "50 1.5" "50 2.0" "100 1.5" "100 2.0"; do
    set -- $cfg
    for seed in 0 1; do
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.markov_tail --regime exact \
        --n 10000000 --d 16 --k "$1" --sep "$2" --seed "$seed"
    done
  done
}
extreme() {  # ~10 GB peak: run alone
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.markov_tail --regime exact \
    --n 30000000 --d 16 --k 50 --sep 2.0 --seed 0
}
minibatch() {
  for cfg in "50 1.5" "50 2.0" "100 1.5" "100 2.0"; do
    set -- $cfg
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m experiments.markov_tail --regime minibatch \
      --n 10000000 --d 16 --k "$1" --sep "$2" --seed 0 --epochs 200
  done
}
highdim() {
  for cfg in "128 50 0.35" "128 50 0.5"; do
    set -- $cfg
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.markov_tail --regime exact \
      --n 1000000 --d "$1" --k "$2" --sep "$3" --seed 0
  done
}
case "$queue" in
  exact) exact ;; minibatch) minibatch ;; extreme) extreme ;; highdim) highdim ;;
  all) exact; minibatch; extreme; highdim ;;
esac
