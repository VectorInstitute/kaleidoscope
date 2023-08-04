#!/bin/bash
#SBATCH --job-name=GPT2_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --qos=high
#SBATCH --partition=t4v2
#SBATCH --gres=gpu:1
#SBATCH --output=gpt2_service.%j.out
#SBATCH --error=gpt2_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

export MASTER_ADDR=localhost
export MASTER_HOST="$(hostname --fqdn)"
#export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export MASTER_PORT=8080

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

# Send registration request to gateway 
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_HOST"':8080"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

singularity exec --nv --bind /checkpoint,/ssd003,/ssd005 /ssd005/projects/llm/triton/pytriton_base.sif python3 $model_service_dir/model_service.py --model_type gpt2 --model_variant None --model_path /ssd005/projects/llm/gpt2 --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_HOST --master_port $MASTER_PORT
