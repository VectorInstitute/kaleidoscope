#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=0
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=llama2-hf-7b_service.%j.out
#SBATCH --error=llama2-hf-7b_service.%j.err

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_CPUS_ON_NODE"=$SLURM_CPUS_ON_NODE
echo "SLURM_PROCID"=$SLURM_PROCID

model_service_dir=$1
gateway_host=$2
gateway_port=$3

model_chkp_dir="/model-weights/Llama-2-7b-chat-hf"
model_path="/model_checkpoint"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

NODES=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
NODES_ARRAY=(${NODES})
HEAD_NODE=${NODES_ARRAY[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)
MASTER_PORT=8855

# export CUDA_VISIBLE_DEVICES=0

GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
echo "NNODES"=$NNODES
echo "NUM_PROCESSES"=$NUM_PROCESSES

# Send registration request to gateway 
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':50126"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register
echo $MASTER_ADDR

# Use --containall flag to avoid using host python env
singularity exec --containall --nv --bind /checkpoint,/ssd003,/ssd005,$model_chkp_dir:$model_path /ssd005/projects/llm/llama2-hf-ds.sif \
    accelerate launch \
		--main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank $SLURM_PROCID \
        --num_processes $NUM_PROCESSES \
        --num_machines $NNODES \
		--use_deepspeed \
		--zero_stage 3 \
		--zero3_init_flag true \
            $model_service_dir/model_service.py \
                --model_type llama2_hf --model_variant 7b --model_path $model_path \
                --model_instance_id $SLURM_JOB_NAME \
                --gateway_host $gateway_host --gateway_port $gateway_port \
				--master_host $MASTER_ADDR --master_port 50126
