#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --output=models/GPT-J/logs/GPT-J_service.%j.out
#SBATCH --error=models/GPT-J/logs/GPT-J_service.%j.err
#--qos=llm
#--cpus-per-task=8

model_service_dir=$1
gateway_host=$2
gateway_port=$3

SINGULARITY_IMG_DIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete
MODEL_DIR=${PWD}
MODEL_CHKP_DIR=/scratch/ssd002/projects/opt_test/gpt-j-6b-slim/2-gpu/

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export NCCL_IB_DISABLE=1

srun -p rtx6000 -N 1 --gres=gpu:2 --mem=64G \
    singularity exec --nv --bind $MODEL_CHKP_DIR:/ft_workspace/model_checkpoint/ \
        $SINGULARITY_IMG_DIR/triton_ft_gptj_2.sif \
        python3 -s $model_service_dir/model_service.py --model_type GPT-J --model_path $MODEL_DIR --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port
