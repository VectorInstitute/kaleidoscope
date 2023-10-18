#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64G
#SBATCH --partition=t4v2
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=opt-6.7b_service.%j.out
#SBATCH --error=opt-6.7b_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

# TODO: Implement passing in the model_path
#model_path=$1
model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Send registration request to gateway
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':51345"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

singularity exec --nv --bind /checkpoint,/scratch,/fs01,/ssd005 /ssd005/projects/llm/opt-6.7b-latest.sif /usr/bin/python3 -s $model_service_dir/model_service.py --model_type opt --model_variant 6.7b --model_path $model_path --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_ADDR --master_port 51345
