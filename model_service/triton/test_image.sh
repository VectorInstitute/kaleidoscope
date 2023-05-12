#!/bin/bash
#SBATCH --job-name=test-image
#SBATCH --mem=32G
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/test-image.%j.out
#SBATCH --error=logs/test-image.%j.err


source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=/ssd005/projects/llm/triton/ft_pytriton_gpt/.singularity
export SINGULARITY_TMPDIR=/ssd005/projects/llm/triton/ft_pytriton_gpt/tmp

singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
            /ssd005/projects/llm/triton/ft_pytriton_gpt/ft_pytriton_gpt.sif \
            python -c "import torch;print(torch.__version__)"
            # python -c "import torch;print(torch.__version__)"
            # nvcc --version
            # ls -l /lib/x86_64-linux-gnu/libpthread.so.0


# COMMAND_TO_RUN
# ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
# ls -l /lib/x86_64-linux-gnu/libpthread.so.0
