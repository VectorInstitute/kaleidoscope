#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --time=3-00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=llama3-8b_instruct_service.%j.out
#SBATCH --error=llama3-8b_instruct_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3
model_path="/model-weights/Meta-Llama-3-8B-Instruct/original"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Send registration request to gateway
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':51345"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

srun -q llm \
	-p a40 \
	-N 1 \
	--gres=gpu:1 \
	--mem=0 \
	-c 16 \
	singularity exec \
	--nv \
	--bind /checkpoint,/scratch,/fs01,/model-weights \
	/ssd005/projects/llm/llama3-latest.sif \
	torchrun \
	${model_service_dir}/model_service.py \
	--model_type llama3 \
	--model_variant 8b_instruct \
	--model_path $model_path \
	--model_instance_id $SLURM_JOB_NAME \
	--gateway_host $gateway_host \
	--gateway_port $gateway_port \
	--master_host $MASTER_ADDR \
	--master_port 51345
