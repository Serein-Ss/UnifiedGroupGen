#!/bin/bash
set -e
set -o pipefail
source /fs6/software/modules/bashrc
module load GCC/12.2.0
module load CUDA/12.9
source /fs6/software/python/3.13_miniforge3-26.1.1/etc/profile.d/conda.sh
conda activate unifiedgroupgen

export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 MPLBACKEND=Agg
exec yhrun "$1" -B -u "$2" worker --plan "$3" --job "$4"

