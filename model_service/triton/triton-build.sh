#!/bin/bash

echo "Starting build at $(date)"

docker build . --no-cache -t "triton"

echo "Converting to singularity container at $(date)"

singularity build triton.sif docker-daemon://triton

echo "Build completed at $(date)"

