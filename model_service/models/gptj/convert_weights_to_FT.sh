#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/convert_weights_to_FT.%j.out
#SBATCH --error=logs/convert_weights_to_FT.%j.err


SINGULARITY_IMG_DIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete
MODEL_WEIGHTS_DIR=/scratch/ssd002/projects/opt_test/gpt-j-6b-slim
NUM_SHARDS=4

module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=$SINGULARITY_IMG_DIR/.singularity
export SINGULARITY_TMPDIR=$SINGULARITY_IMG_DIR/tmp


# srun -q llm -p rtx6000 -N 1 --gres=gpu:1 --mem=32G -c 8 \
singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
    $SINGULARITY_IMG_DIR/triton_with_ft_complete.sif \
    python /ft_workspace/FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
        --output-dir $MODEL_WEIGHTS_DIR \
        --ckpt-dir $MODEL_WEIGHTS_DIR/step_383500/ \
        --n-inference-gpus $NUM_SHARDS