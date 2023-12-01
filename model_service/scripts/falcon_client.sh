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
singularity exec --bind /ssd005 /ssd005/projects/llm/falcon-hf.sif /usr/bin/python3 -s ~/triton_multi_node/kaleidoscope/model_service/triton/falcon_client.py --url http://$server_host:$server_port

# singularity exec --bind /scratch /scratch/ssd002/projects/opt_test/triton/pytriton_falcon/pytriton_falcon.sif /usr/bin/python3 -s ~/triton_multi_node/kaleidoscope/model_service/triton/falcon_client.py --url http://$server_host:$server_port

