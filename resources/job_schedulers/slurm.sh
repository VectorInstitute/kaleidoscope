#!/bin/bash

model_type=$1

if [[ "$model_type" == "GPT2" ]]; then
    scp ~/lingua/model_service/slurm/GPT2_service.sh vremote:~
    ssh vremote sbatch ~/GPT2_service.sh
elif [[ "$model_type" == "OPT" ]]; then
    scp ~/lingua/model_service/slurm/OPT_service.sh vremote:~
    ssh vremote sbatch ~/OPT_service.sh
fi

ssh vremote python3 ~/lingua/model_service/model_service.py --launch-model GPT2