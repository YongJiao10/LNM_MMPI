#!/usr/bin/env bash
set -euo pipefail

PROJECT="/Users/jy/Documents/LNM_CCA_project"
WORKTREE="${PROJECT}/.worktrees/lnm-code-nested-cv"
PYTHON="/opt/miniconda3/envs/cca/bin/python"
OUT_DIR="${PROJECT}/review_prep/nested_results_review_0706/seed1111_10x5_fdc_1000perm_0709"

mkdir -p "${OUT_DIR}" "${WORKTREE}/.mplconfig"

export MPLCONFIGDIR="${WORKTREE}/.mplconfig"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${WORKTREE}"

exec "${PYTHON}" scripts/formal_nested_cv_rcca_permutation.py \
  --out-dir "${OUT_DIR}" \
  --sfc fc \
  --outer-folds 10 \
  --inner-folds 5 \
  --selection-metric mean_r \
  --seed 1111 \
  --n-perm 1000 \
  --perm-seed 0 \
  --n-jobs 4 \
  --parallel-backend thread \
  --flush-every 25
