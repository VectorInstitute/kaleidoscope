#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=t4v2
#SBATCH --qos=high
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/triton_OPT-6.7b_service.%j.out
#SBATCH --error=logs/triton_OPT-6.7b_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

# TODO: Implement passing in the model_path
#model_path=$1
model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR="localhost"
singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 /ssd005/projects/llm/triton/triton_with_ft.sif /usr/bin/python3 -s  ~/scratch/kaleidoscope-triton/model_service/model_service.py --model_type GPT-J --model_path /not/implemented/yet --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 9001
