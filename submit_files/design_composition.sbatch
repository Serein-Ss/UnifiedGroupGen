#!/bin/bash
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Submit with --chdir=<campaign>/source so this job also uses frozen source.
# Arguments: absolute_checkpoint exact_composition_json space_or_layer unique_output_directory
set -e
test "$#" -eq 4
source /fs6/software/modules/bashrc
module load GCC/12.2.0
module load CUDA/12.9
source /fs6/software/python/3.13_miniforge3-26.1.1/etc/profile.d/conda.sh
conda activate unifiedgroupgen
export UGG_PROJECT_ROOT="$PWD"
export PROJECT_ROOT="$PWD"
export HYDRA_JOBS="$PWD/outputs"
export WANDB_DIR="$PWD/wandb"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
printf 'Git commit at submission: %s\n' "${UGG_GIT_COMMIT:-not recorded}"
yhrun python -u -m unifiedgroupgen.generate device=cuda threads=4 seed=142 \
    "generation.checkpoint=$1" "generation.composition=$2" "generation.kind=$3" \
    "generation.output=$4" 'generation.properties={}' generation.guidance=0 \
    generation.count=256 generation.steps=64 generation.solver=heun \
    generation.precision=fp32 generation.batch_size=16
