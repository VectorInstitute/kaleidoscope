#!/bin/bash
#SBATCH --partition=a40
#--qos=llm
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=llama2-7b_service.%j.out
#SBATCH --error=llama2-7b_service.%j.err

# model_service_dir=$1
# gateway_host=$2
# gateway_port=$3
# model_path="/model-weights/Llama-2-7b-chat"

export PYTHONPATH=$PYTHONPATH:/h/odige/triton_multi_node/kaleidoscope

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')

# # Send registration request to gateway
# curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':51345"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register

srun -p a40 \
	-N "${SLURM_JOB_NUM_NODES}" \
	--gres=gpu:1 \
	--mem=32G \
	-c 16 \
	singularity exec \
	--nv \
	--bind /checkpoint,/scratch,/fs01,/model-weights \
	/ssd005/projects/llm/llama2-latest.sif \
	torchrun \
	--nnodes 1 \
	--nproc_per_node 1 \
	~/triton_multi_node/kaleidoscope/model_service/triton/llama2_client.py
