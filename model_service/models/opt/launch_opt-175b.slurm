#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=167G
#SBATCH --qos=llm
#SBATCH --partition=a40
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=opt-175b_service.%j.out
#SBATCH --error=opt-175b_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

# TODO: Implement passing in the model_path
#model_path=$1
model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export NCCL_IB_DISABLE=1

# Send registration request to gateway
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':51345"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

srun -q llm -p a40 -N 4 --gres=gpu:4 --mem=167G -c 32 singularity exec --nv --bind /checkpoint,/scratch,/ssd005,/fs01 /ssd005/projects/llm/opt-175b-latest.sif /usr/bin/python3 -s $model_service_dir/model_service.py --model_type opt --model_variant 175b --model_path $model_path --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_ADDR --master_port 51345
