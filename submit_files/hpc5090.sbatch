#!/bin/bash
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -e
source /fs6/software/modules/bashrc
module load GCC/12.2.0
module load CUDA/12.9
source /fs6/software/python/3.13_miniforge3-26.1.1/etc/profile.d/conda.sh
conda activate unifiedgroupgen

# The submitter selects an immutable source snapshot with --chdir.
export UGG_PROJECT_ROOT="$PWD"
export PROJECT_ROOT="$PWD"
export HYDRA_JOBS="$PWD/outputs"
export WANDB_DIR="$PWD/wandb"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DGLBACKEND=pytorch
export MPLBACKEND=Agg
hostname
printf 'Git commit at submission: %s\n' "${UGG_GIT_COMMIT:-not recorded}"
yhrun python -u submit_files/hpc5090.py run "$@"
