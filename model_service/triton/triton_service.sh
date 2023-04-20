#!/bin/bash
#SBATCH --job-name=GPT2_triton_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=GPT2_triton_service.%j.out
#SBATCH --error=GPT2_triton_service.%j.err

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=localhost
singularity exec --nv --bind /ssd005,/scratch,/checkpoint /ssd005/projects/llm/triton.sif /usr/bin/python3 -s ~/marshallw/kaleidoscope/model_service/model_service.py --model_type GPT2 --model_path /checkpoint/opt_test/gpt2 --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 9001
