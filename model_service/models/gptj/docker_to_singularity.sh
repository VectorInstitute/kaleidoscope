#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/docker_to_singularity.%j.out
#SBATCH --error=logs/docker_to_singularity.%j.err


SINGULARITY_IMG_DIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete
DOCKER_IMG=docker://xeon27/triton_with_ft_complete:22.03

module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=$SINGULARITY_IMG_DIR/.singularity
export SINGULARITY_TMPDIR=$SINGULARITY_IMG_DIR/tmp


singularity pull --disable-cache \
    $SINGULARITY_IMG_DIR/triton_with_ft_complete.sif \
    $DOCKER_IMG