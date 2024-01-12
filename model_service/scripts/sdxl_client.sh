#!/bin/bash

server_host="localhost"
server_port=8003

if [ ! -z "$1" ]; then
    server_host=$1
else
    echo "No hostname provided, defaulting to localhost"
fi

if [ ! -z "$2" ]; then
    server_host=$2
fi

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
echo "Sending request to http://$server_host:$server_port"
singularity exec --bind /ssd005,/scratch /ssd005/projects/llm/sdxl-latest.sif /usr/bin/python3 -s sdxl_client.py --url http://$server_host:$server_port
