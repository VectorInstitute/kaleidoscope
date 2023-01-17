#!/bin/bash
#SBATCH --job-name=OPT_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --qos=llm
#SBATCH --partition=rtx6000,a40
#SBATCH --gres=gpu:2
#SBATCH --output=OPT_service.%j.out
#SBATCH --error=OPT_service.%j.err

module load singularity-ce/3.8.2
export MASTER_ADDR=localhost
singularity exec --nv --bind /checkpoint,/scratch,/ssd003 /ssd003/projects/aieng/opt-125-latest.sif /usr/bin/python3 -s ~/lingua/model_service/model_service.py --model_type OPT --model_path /checkpoint/opt_test/original/OPT-175B
