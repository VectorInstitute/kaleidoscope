#!/bin/bash
#SBATCH --job-name=opt_125m_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --output=opt_125m_service.%j.out
#SBATCH --error=opt_125m_service.%j.err

cd /h/coatsworth/scratch/slurm/model_service/LLM
module load singularity-ce/3.8.2
export MASTER_ADDR=localhost
singularity exec --nv --bind /checkpoint,/scratch opt-125-latest.sif /usr/bin/python3 -s /h/coatsworth/scratch/slurm/model_service/LLM/model_service/model_service.py --model_type opt_125m
