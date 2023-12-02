#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --mem=16G
#SBATCH --time=3-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=sdxl_service.%j.out
#SBATCH --error=sdxl_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

export MASTER_ADDR=localhost
export MASTER_HOST="$(hostname --fqdn)"
export MASTER_PORT=51345

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

# Send registration request to gateway 
#curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_HOST"':'"$MASTER_PORT"'"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

#singularity exec --nv --bind /checkpoint,/fs01,/ssd005 /ssd005/projects/llm/triton/pytriton_base.sif python3 $model_service_dir/model_service.py --model_type sdxl --model_variant None --model_path /ssd005/projects/llm/sdxl --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_HOST --master_port $MASTER_PORT



singularity exec --nv --bind /scratch/ssd004,/ssd005,/scratch,/checkpoint /ssd005/projects/llm/triton/pytriton_base.sif /usr/bin/python3 -s ../../model_service.py --model_type sdxl --model_variant None --model_path /model-weights/sdxl-turbo --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 0 --master_host localhost --master_port 8003
