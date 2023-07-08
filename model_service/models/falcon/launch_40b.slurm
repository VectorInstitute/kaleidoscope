#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=167G
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --output=falcon-40b_service.%j.out
#SBATCH --error=falcon-40b_service.%j.err

model_service_dir=$1
gateway_host=$2
gateway_port=$3

model_path="/checkpoint/opt_test/original/falcon-40b-hf/falcon-40b"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export CUDA_VISIBLE_DEVICES=0,1 #,2,3
# export NCCL_IB_DISABLE=1

# Send registration request to gateway 
curl -X POST -H "Content-Type: application/json" -d '{"host": "'"$MASTER_ADDR"':50112"}' http://$gateway_host:$gateway_port/models/instances/$SLURM_JOB_NAME/register
echo $MASTER_ADDR

(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &

# srun -p a40 -N 1 --gres=gpu:4 --mem=167G -c 32 \
singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 /ssd005/projects/llm/falcon-40b.sif \
/usr/bin/python3 -s \
    $model_service_dir/model_service.py --model_type falcon --model_variant 40b --model_path $model_path --model_instance_id $SLURM_JOB_NAME --gateway_host $gateway_host --gateway_port $gateway_port --master_host $MASTER_ADDR --master_port 50112