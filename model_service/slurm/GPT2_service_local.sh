#!/bin/bash
#SBATCH --output=/data/GPT2_service.%j.out
#SBATCH --error=/data/GPT2_service.%j.out


$model_path=$1
model_path="/model/path"
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

python3 /app/model_service.py --model_type GPT-2 --model_path $model_path --model_instance_id $SLURM_JOB_NAME