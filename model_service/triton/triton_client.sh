#!/bin/bash

server_host="localhost"
if [ ! -z "$1" ]; then
    server_host=$1
fi

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
echo "Sending request to http://$server_host:8003"
singularity exec --nv --bind /ssd005,/scratch /ssd005/projects/llm/triton.sif /usr/bin/python3 -s ~/scratch/kaleidoscope-triton/model_service/triton/client.py --url http://$server_host:8003
