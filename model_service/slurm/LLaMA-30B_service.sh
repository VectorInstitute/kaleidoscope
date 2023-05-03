#!/bin/bash
#SBATCH --job-name=prompt_llama
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --partition=a40
#SBATCH --qos=llm
#SBATCH --output=LLaMA-30B_service.%j.out
#SBATCH --error=LLaMA-30B_service.%j.err

echo "###############################"
echo "######## SBATCH SCRIPT ########"
echo "###############################"

# Setup dirs
TARGET_FOLDER=$1	# Path to LLaMA ckpt weights
ROOT_DIR=$(pwd)
LOG_DIR="${ROOT_DIR}/logs"
ENV_DIR="${ROOT_DIR}/llama_env"

# Set location of host and access port
NODES=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
NODES_ARRAY=(${NODES})
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)

# Redirected stdout + stderror filepaths
OUTFILE="${LOG_DIR}/output-%j-%t.out"

# Worker script to run with args
WORKER_SETUP_SCRIPT="${ROOT_DIR}/worker_script.sh"
PORT=42069
MODEL_SIZE="30B"

## `torchrun` args
NNODES="${SLURM_JOB_NUM_NODES}"
WORKERS_PER_NODE="${SLURM_GPUS_ON_NODE}"
RDVZ_ID=6969
RDVZ_BACKEND="c10d"
RDVZ_ENDPOINT="${HEAD_NODE_IP}:${PORT}"

## Python llama script args
#PYTHON_SCRIPT="${ROOT_DIR}/llama/example.py"
PYTHON_SCRIPT="${ROOT_DIR}/host_model.py"
MAX_BATCH_SIZE=8
MAX_SEQ_LEN=256
CKPT_DIR="${TARGET_FOLDER}/${MODEL_SIZE}"
TOKENIZER_PATH="${TARGET_FOLDER}/tokenizer.model"

# Set env variables
export LOGLEVEL=INFO
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# Print environment and other variables
echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "OUTFILE=${OUTFILE}"
echo "WORKER_SETUP_SCRIPT=${WORKER_SETUP_SCRIPT}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "TARGET_FOLDER=${TARGET_FOLDER}"
echo "NNODES=${NNODES}"
echo "WORKERS_PER_NODE=${WORKERS_PER_NODE}"
echo "RDVZ_ID=${RDVZ_ID}"
echo "RDVZ_BACKEND=${RDVZ_BACKEND}"
echo "RDVZ_ENDPOINT=${RDVZ_ENDPOINT}"
echo "PYTHON_SCRIPT=${PYTHON_SCRIPT}"
echo "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
echo "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"

read -r -d '' cmd << EOF
bash ${WORKER_SETUP_SCRIPT} \
${WORKERS_PER_NODE} \
${ENV_DIR} \
${NNODES} \
${MAX_BATCH_SIZE} \
${MAX_SEQ_LEN} \
${PYTHON_SCRIPT} \
${CKPT_DIR} \
${TOKENIZER_PATH} \
${RDVZ_ID} \
${RDVZ_BACKEND} \
${RDVZ_ENDPOINT}
EOF

echo "Running command:"
echo "${cmd}"

/opt/slurm/bin/srun -N "${SLURM_JOB_NUM_NODES}" -l -o "${OUTFILE}" -e "${OUTFILE}" bash -c "${cmd}"