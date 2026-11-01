#!/bin/bash
#SBATCH -J ugg_gpu_check
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -D /fs6/home/zhongrsh/workspace/UnifiedGroupGen
#SBATCH -o /fs6/home/zhongrsh/workspace/UnifiedGroupGen/intermediate/checks/5090/gpu-smoke-%j.out

set -e
source /fs6/software/modules/bashrc
module load GCC/12.2.0
module load CUDA/12.9
source /fs6/software/python/3.13_miniforge3-26.1.1/etc/profile.d/conda.sh
conda activate unifiedgroupgen

export OMP_NUM_THREADS=1
hostname
printf 'Git commit at submission: %s\n' "${UGG_GIT_COMMIT:-not recorded}"
yhrun python -u submit_files/gpu_smoke.py
