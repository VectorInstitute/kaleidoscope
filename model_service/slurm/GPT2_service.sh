#!/bin/bash
#SBATCH --job-name=GPT2_service
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --partition=rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --output=GPT2_service.%j.out
#SBATCH --error=GPT2_service.%j.err

cd ~/lingua/model_service
module load cuda-11.3
module load python/3.8.0
python3 model_service.py --model_type GPT2
