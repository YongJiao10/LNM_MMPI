#!/bin/bash
# Generic SGE job script template for lesion FC analysis

#$ -N fc_lesion                          # Job name
#$ -pe smp 50                            # Number of CPU cores (slots)
#$ -l mem_free=4G,h_vmem=100G            # Memory: request per-core free mem & hard limit
#$ -j y                                  # Merge stderr into stdout
#$ -q all.q,UI,PINC                      # Queues to consider (comma-separated); comment out if not fixed
#$ -o /Users/yojiao/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out   # Log path (create logs/ first)
#$ -t 1-160                             # Array job range (e.g., 1 to 269)
#$ -tc 20                                # Max concurrent tasks (for array jobs)

# ---- Environment ----
source ~/.bashrc                         # Load user env (PATH, conda, modules)
conda activate py313                     # Activate conda env

# ---- Thread/BLAS control ----
export N_SUBJ_PROCS=$NSLOTS              # Parallel workers = slots assigned
export OMP_NUM_THREADS=1                 # One thread per process for OpenMP
export MKL_NUM_THREADS=1                 # One thread per process for MKL
export OPENBLAS_NUM_THREADS=1            # One thread per process for OpenBLAS
export MKL_DYNAMIC=FALSE                 # Disable MKL dynamic adjustment

# ---- Run analysis ----
python -u /Users/yojiao/SFC_project/LNM_voxel/FC_lesion_voxel.py               # -u = unbuffered stdout (real-time logs)
