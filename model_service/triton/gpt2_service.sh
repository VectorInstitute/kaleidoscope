#!/bin/bash
#SBATCH --job-name=gpt2_triton_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=gpt2_triton_service.%j.out
#SBATCH --error=gpt2_triton_service.%j.err

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
singularity exec --nv --bind /scratch/ssd004,/ssd005,/scratch,/checkpoint /ssd005/projects/llm/triton/pytriton_base.sif /usr/bin/python3 -s ../model_service.py --model_type gpt2 --model_variant None --model_path /checkpoint/opt_test/gpt2 --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 0 --master_host localhost --master_port 8003
