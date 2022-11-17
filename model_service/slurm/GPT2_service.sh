#!/bin/bash
#SBATCH --job-name=GPT2_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --partition=opsdev
#SBATCH --gres=gpu:1
#SBATCH --output=GPT2_service.%j.out
#SBATCH --error=GPT2_service.%j.err

cd ~/scratch/slurm/model_service/LLM/model_service
module load cuda-11.3
python3 model_service.py --model_type GPT2
