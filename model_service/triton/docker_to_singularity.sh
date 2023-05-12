#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/docker_to_singularity.%j.out
#SBATCH --error=logs/docker_to_singularity.%j.err


module load singularity-ce/3.8.2

# export SINGULARITY_CACHEDIR=/ssd005/projects/llm/triton/ft_pytriton_gpt/.singularity
# export SINGULARITY_TMPDIR=/ssd005/projects/llm/triton/ft_pytriton_gpt/tmp
export SINGULARITY_CACHEDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/.singularity
export SINGULARITY_CACHEDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/tmp
singularity pull --disable-cache \
    /scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/triton_with_ft_complete.sif \
    docker://xeon27/triton_with_ft_complete:22.03