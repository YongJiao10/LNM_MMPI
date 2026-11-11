#!/bin/bash
#$ -N lnm22_fdc_direct
#$ -pe smp 1
#$ -l mem_free=16G,h_vmem=24G,h_rt=08:00:00
#$ -j y
#$ -q UI,PINC
#$ -o /Users/yojiao/lnm_22scale_239_20261030/results
set -euo pipefail
# Match the original upstream numerical environment on Argon.
source /Users/yojiao/miniconda3/etc/profile.d/conda.sh
conda activate py313
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_ROOT=/Users/yojiao/lnm_22scale_239_20261030
python -u "$TASK_ROOT/source/rebuild_fdc_gsp1000.py" \
  --cohort "$TASK_ROOT/source/complete_RF_22_without_HLP.csv" \
  --out "$TASK_ROOT/results/fdc_direct_gsp1000" "$@"
