#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=client-multi-node-GPT-J
#SBATCH --mem=64G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/run_client_script.%j.out
#SBATCH --error=logs/run_client_script.%j.err


SINGULARITY_IMG_DIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=$SINGULARITY_IMG_DIR/.singularity
export SINGULARITY_TMPDIR=$SINGULARITY_IMG_DIR/tmp

export PYTHONPATH=${PYTHONPATH}:/h/odige/triton_multi_node/kaleidoscope

singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
                $SINGULARITY_IMG_DIR/triton_with_ft_complete.sif \
                python client_script.py