#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=0
#SBATCH --partition=rtx6000
#--qos=llm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#--gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=falcon-7b_service.%j.out
#SBATCH --error=falcon-7b_service.%j.err

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_CPUS_ON_NODE"=$SLURM_CPUS_ON_NODE
echo "SLURM_PROCID"=$SLURM_PROCID

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

model_service_dir=$1
gateway_host=$2
gateway_port=$3

model_chkp_dir="/checkpoint/opt_test/original/falcon-7b-hf/falcon-7b"
model_path="/model_checkpoint"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

NODES=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
NODES_ARRAY=(${NODES})
HEAD_NODE=${NODES_ARRAY[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)
MASTER_PORT=8855

export PYTHONPATH=/usr/bin/python3

GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
echo "NNODES"=$NNODES
echo "NUM_PROCESSES"=$NUM_PROCESSES


# # create the hostfile
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
# nodes_array=($nodes)

# hostfile="$(TMPDIR='/tmp' mktemp)"
# for ((i = 0; i < $SLURM_NNODES; i++)); do
#   node_i=${nodes_array[$i]}
#   echo "WRITING WORKER $i at $node_i to $hostfile"
#   echo "$node_i slots=$SLURM_GPUS_ON_NODE" >> $hostfile
# done

# echo "FINISHED WRITING hostfile $hostfile:"
# cat $hostfile


export SINGULARITY="singularity exec \
                    --nv --bind /checkpoint,/scratch,/ssd003,/ssd005,$model_chkp_dir:$model_path /ssd005/projects/llm/falcon-hf-ds.sif \
                  "

# export LAUNCHER="accelerate launch \
#         --config_file $model_service_dir/models/falcon/accelerate_config_multi_node.yaml \
#         --main_process_ip $MASTER_ADDR \
#         --main_process_port $MASTER_PORT \
#         --machine_rank \$SLURM_PROCID \
#         --num_processes $NUM_PROCESSES \
#         --num_machines $NNODES \
#         "

export LAUNCHER="accelerate launch \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank \$SLURM_PROCID \
        --num_processes $NUM_PROCESSES \
        --num_machines $NNODES \
        --use_deepspeed \
        --zero_stage 3 \
        --zero3_init_flag true \
        --deepspeed_multinode_launcher standard \
        "

# export LAUNCHER="torchrun \
#         --nnodes $NNODES \
#         --nproc-per-node $GPUS_PER_NODE \
#         --node-rank \$SLURM_PROCID \
#         --rdzv_id 6969 \
#         --rdzv_backend c10d \
#         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#         "

# --master-addr $MASTER_ADDR \
# --master-port $MASTER_PORT \

# --role $SLURMD_NODENAME: \
# --rdzv_conf rdzv_backend=c10d \


export SCRIPT="$model_service_dir/model_service.py \
            --model_type falcon --model_variant 7b --model_path $model_path --model_instance_id $SLURM_JOB_NAME \
            --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_ADDR --master_port 50116 \
            "

export CMD="$SINGULARITY $LAUNCHER $SCRIPT"
# export CMD="$SINGULARITY $LAUNCHER $SCRIPT --deepspeed $model_service_dir/models/falcon/ds_config.json"

# Send registration request to gateway 
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':50116"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register
echo $MASTER_ADDR

(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &

srun --mem=0 --gres=gpu:1 -c $SLURM_CPUS_ON_NODE -N 2 -n 2 bash -c "$CMD"