#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=OPT-6.7b_service.%j.out
#SBATCH --error=OPT-6.7b_service.%j.err

model_service_dir=$1

# TODO: Implement passing in the model_path
#model_path=$1
model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
export MASTER_ADDR=localhost
singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 /ssd005/projects/llm/opt-6.7b-latest.sif /usr/bin/python3 -s $model_service_dir/model_service.py --model_type OPT-6.7B --model_path $model_path --model_instance_id $SLURM_JOB_NAME
