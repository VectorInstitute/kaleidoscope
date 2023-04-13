#!/bin/bash

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2
singularity exec --nv --bind /ssd005,/scratch /ssd005/projects/llm/triton.sif /usr/bin/python3 -s ~/scratch/kaleidoscope-triton/model_service/triton/client.py --url http://localhost:8003
