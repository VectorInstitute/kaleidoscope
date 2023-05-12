#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/convert_weights_to_FT.%j.out
#SBATCH --error=logs/convert_weights_to_FT.%j.err


module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/.singularity
export SINGULARITY_CACHEDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/tmp

srun -q llm -p rtx6000 -N 1 --gres=gpu:1 --mem=32G -c 8 \
    singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
    /scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/triton_with_ft_complete.sif \
    python /ft_workspace/FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
        --output-dir /scratch/ssd002/projects/opt_test/gpt-j-6b-slim \
        --ckpt-dir /scratch/ssd002/projects/opt_test/gpt-j-6b-slim/step_383500/ \
        --n-inference-gpus 2

# /scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/triton_with_ft_complete.sif \

#srun -q llm -p a40 -N 1 --gres=gpu:1 --mem=32G -c 8 \
#    python3 ./FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
#        --output-dir /ssd005/projects/llm/gpt-j-6b \
#        --ckpt-dir /ssd005/projects/llm/gpt-j-6b/step_383500/ \
#        --n-inference-gpus 1
