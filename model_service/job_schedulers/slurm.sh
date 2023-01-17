#!/bin/bash

model_type=$1

if [[ "$model_type" == "GPT2" ]]; then
    sbatch ~/lingua/model_service/slurm/GPT2_service.sh
elif [[ "$model_type" == "OPT" ]]; then
    sbatch ~/lingua/model_service/slurm/OPT_service.sh
fi
