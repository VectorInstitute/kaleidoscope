#!/bin/bash

server_host="localhost"
server_port=8003

if [ ! -z "$1" ]; then
    server_host=$1
else
    echo "No hostname provided, defaulting to localhost"
fi

if [ ! -z "$2" ]; then
    server_port=$2
fi

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
echo "Sending request to http://$server_host:$server_port"
singularity exec --bind /ssd005,/scratch /ssd005/projects/llm/triton/pytriton_base.sif /usr/bin/python3 -s ~/scratch/triton-refactor/model_service/triton/gpt2_client.py --url http://$server_host:$server_port
