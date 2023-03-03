#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=150G
#SBATCH --qos=llm
#SBATCH --partition=a40
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --output=OPT-175B_service.%j.out
#SBATCH --error=OPT-175B_service.%j.err

model_service_dir=$1

# TODO: Implement passing in the model_path
#model_path=$1
model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export NCCL_IB_DISABLE=1
srun -q llm -p a40 -N 4 --gres=gpu:4 --mem=150G -c 40 singularity exec --nv --bind /checkpoint,/scratch,/ssd005 /ssd005/projects/llm/opt-175b-a40.sif /usr/bin/python3 -s $model_service_dir/model_service.py --model_type OPT-175B --model_path $model_path --model_instance_id $SLURM_JOB_NAME
