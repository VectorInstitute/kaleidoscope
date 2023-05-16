#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=multi-node-GPT-J
#SBATCH --mem=64G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/triton_GPT-J_service.%j.out
#SBATCH --error=logs/triton_GPT-J_service.%j.err
# --qos=llm


# model_service_dir=$1
# gateway_host=$2
# gateway_port=$3

SINGULARITY_IMG_DIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete
MODEL_DIR=${PWD}
NUM_GPUS=4

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=$SINGULARITY_IMG_DIR/.singularity
export SINGULARITY_TMPDIR=$SINGULARITY_IMG_DIR/tmp

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1,2,3


singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
                $SINGULARITY_IMG_DIR/triton_with_ft_complete.sif \
                /opt/tritonserver/bin/tritonserver \
                        --model-repository=$MODEL_DIR/triton_model_store/gptj_$NUM_GPUS