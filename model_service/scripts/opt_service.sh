#!/bin/bash
#SBATCH --job-name=opt_triton_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=opt_triton_service.%j.out
#SBATCH --error=opt_triton_service.%j.err

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=localhost
singularity exec --nv --bind /ssd003,/ssd005,/scratch,/checkpoint /ssd005/projects/llm/opt-6.7b-latest.sif /usr/bin/python3 -s ../model_service.py --model_type opt --model_variant 6.7b --model_path /hardcoded/in/opt/container --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 9001 --master_host localhost --master_port 8003
