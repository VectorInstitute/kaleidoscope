#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --time=3-00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=phi3-medium_128k_instruct_service.%j.out
#SBATCH --error=phi3-medium_128k_instruct_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3
model_path="/model-weights/Phi-3-medium-128k-instruct"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Send registration request to gateway
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':51345"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

# TODO: Fix the below command to use torchrun instead of python3.
# I only used python3 so I could pass the -s argument and exclude my home directory from sys.path. This is not supposed in torchrun. I don't understand why this wasn't needed for other models.
srun -q llm \
	-p a40 \
	-N 1 \
	--gres=gpu:1 \
	--mem=0 \
	-c 16 \
	singularity exec \
	--nv \
	--bind /checkpoint,/scratch,/fs01,/model-weights \
	/ssd005/projects/llm/phi3-latest.sif \
    python3 -s \
	${model_service_dir}/model_service.py \
	--model_type phi3 \
	--model_variant medium_128k_instruct \
	--model_path $model_path \
	--model_instance_id $SLURM_JOB_NAME \
	--gateway_host $gateway_host \
	--gateway_port $gateway_port \
	--master_host $MASTER_ADDR \
	--master_port 51345
