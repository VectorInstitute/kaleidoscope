#!/bin/bash
#SBATCH --job-name=GPT2_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --qos=high
#SBATCH --partition=t4v2
#SBATCH --gres=gpu:1
#SBATCH --output=GPT2_service.%j.out
#SBATCH --error=GPT2_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

source /opt/lmod/lmod/init/profile
module load cuda-11.3
module load python/3.8.0
python3 $model_service_dir/model_service.py --model_type GPT2 --model_path gpt2 --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port
