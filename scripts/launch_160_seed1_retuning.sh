#!/bin/bash
# Use the baseline's pinned runtime; launch this script in a remote screen.
set -euo pipefail
TASK_ROOT=/Users/yojiao/Documents/LNM_CCA_compute401
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py314
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/lnm22_mpl
export XDG_CACHE_HOME=/private/tmp/lnm22_cache
cd "$TASK_ROOT/LNM_MMPI_code"
exec /usr/bin/caffeinate -i "$TASK_ROOT/runtime/bin/python" -u \
    scripts/run_160_seed1_retuning.py run --scales "$1" >> "$2/run.log" 2>&1
