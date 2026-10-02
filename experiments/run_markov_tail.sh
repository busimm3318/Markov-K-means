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
const() {  # constant-step mini-batch: a few points oscillate for ever (eta from a pilot at n=1e6)
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m experiments.markov_tail --regime minibatch_const \
    --n 10000000 --d 16 --k 50 --sep 1.5 --seed 0 --epochs 200 --eta 2e-4
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m experiments.oscillators --regime minibatch_const \
    --n 10000000 --d 16 --k 50 --sep 1.5 --seed 0 --epochs 200 --eta 2e-4
}
oscillators() {  # per-point dumps for the first n=1e7 instance of each regime
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.oscillators --regime minibatch --n 10000000 --k 50 --sep 1.5
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.oscillators --regime exact --n 10000000 --k 50 --sep 1.5
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
  # d=128 separations chosen by a pilot at n=1e6 (sep 0.25 / 0.35 converged in < 100 iterations)
  for cfg in "128 50 0.5" "128 50 0.7"; do
    set -- $cfg
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.markov_tail --regime exact \
      --n 2000000 --d "$1" --k "$2" --sep "$3" --seed 0
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m experiments.markov_tail --regime minibatch \
      --n 2000000 --d "$1" --k "$2" --sep "$3" --seed 0 --epochs 200
  done
}
case "$queue" in
  exact) exact ;; minibatch) minibatch ;; const) const ;; oscillators) oscillators ;;
  extreme) extreme ;; highdim) highdim ;;
  all) exact; minibatch; oscillators; const; highdim; extreme ;;
esac
