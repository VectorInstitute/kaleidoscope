#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=multi-node-GPT-J
#SBATCH --mem=64G
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/triton_GPT-J-6B_service.%j.out
#SBATCH --error=logs/triton_GPT-J-6B_service.%j.err


# model_service_dir=$1
# gateway_host=$2
# gateway_port=$3

# # TODO: Implement passing in the model_path
# #model_path=$1
# model_path="/model/path"

source /opt/lmod/lmod/init/profile
module load singularity-ce/3.8.2

export SINGULARITY_CACHEDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/.singularity
export SINGULARITY_TMPDIR=/scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/tmp

export CUDA_VISIBLE_DEVICES=0,1
export BASE_DIR=/ssd003/home/odige/triton_multi_node/kaleidoscope/model_service/triton

# srun -q llm -p t4v2 -N 1 --gres=gpu:2 --mem=64G -c 8 \
singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
        /scratch/ssd002/projects/opt_test/triton/triton_with_ft_complete/triton_with_ft_complete.sif \
        /opt/tritonserver/bin/tritonserver \
        --model-repository=${BASE_DIR}/ft_workspace/triton-model-store/gptj




# # set default for NCCL backend
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')
# # export MASTER_PORT="8003"

# # set NCCL comm vars - only for t4v2 and rtx6000
# # export NCCL_SOCKET_IFNAME=bond0 # t4v2
# # export NCCL_SOCKET_IFNAME=ens1f0 # a40
# export NCCL_DEBUG=DEBUG 
# # export NCCL_IB_DISABLE=1 
# # export NCCL_IBEXT_DISABLE=1 
# # export NCCL_NET=Socket

# # singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 /ssd005/projects/llm/triton/triton_with_ft.sif /usr/bin/python3 -s  ~/scratch/kaleidoscope-triton/model_service/model_service.py --model_type GPT-J --model_path /not/implemented/yet --model_instance_id test --gateway_host 127.0.0.1 --gateway_port 9001

# singularity exec --nv --bind /checkpoint,/scratch,/ssd003,/ssd005 \
#             /ssd005/projects/llm/triton/ft_pytriton_gpt/ft_pytriton_gpt.sif \
#             mpirun -n 2 --allow-run-as-root \
#             python /ssd003/home/odige/triton_multi_node/kaleidoscope/model_service/model_service.py \
#                 --model_type GPT-J \
#                 --model_path /ssd005/projects/llm/gpt-j-6b/ \
#                 --model_instance_id test \
#                 --gateway_host 127.0.0.1 --gateway_port 9001 --master_port 29100
#             # /ssd005/projects/llm/triton/triton_with_ft.sif \