#!/bin/bash

echo "Starting build at $(date)"

docker build . --no-cache -t "pytriton_base"

echo "Converting to singularity container at $(date)"

singularity build pytriton_base.sif docker-daemon://pytriton_base:latest

echo "Build completed at $(date)"
