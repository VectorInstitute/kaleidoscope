#!/bin/bash
#SBATCH --time=3-00:00
#SBATCH --mem=64G
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --output=falcon-40b_service.%j.out
#SBATCH --error=falcon-40b_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

model_chkp_dir="/checkpoint/opt_test/original/falcon-40b-hf/falcon-40b"
model_path="/model_checkpoint"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export CUDA_VISIBLE_DEVICES=0,1 #,2,3 # Add 2 and 3 if changed to 4 gpus
# export NCCL_IB_DISABLE=1

# Send registration request to gateway 
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':50112"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register
echo $MASTER_ADDR

singularity exec --nv --bind /checkpoint,/scratch,/fs01,/ssd005,$model_chkp_dir:$model_path /ssd005/projects/llm/falcon-hf.sif \
/usr/bin/python3 -s \
    $model_service_dir/model_service.py --model_type falcon --model_variant 40b --model_path $model_path --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_ADDR --master_port 50112
