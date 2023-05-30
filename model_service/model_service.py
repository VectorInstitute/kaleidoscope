import argparse
import logging
import numpy as np
import os
import requests
import socket
import sys
import torch
import subprocess
import pathlib

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


# Globals
AVAILABLE_MODELS = ["OPT-175B", "OPT-6.7B", "GPT2", "GPT-J"]
TRITON_HTTP_PORT = 8000
logger = logging.getLogger("kaleidoscope.model_service")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

# We only want to load the model library that's being requested, not all of them
# TODO: Is there a way to make this happen automatically, without separate entries?


def initialize_model(model_type):
    if model_type == "OPT-175B" or model_type == "OPT-6.7B":
        from models import OPT
        return OPT.OPT()
    elif model_type == "GPT2":
        from models.GPT2 import model as GPT2
        return GPT2.GPT2()


# Signal handler to send a remove request to the gateway, if this service is killed by the system
def signal_handler(sig, frame):
    global model_type
    send_remove_request(model_type)
    sys.exit(0)


def send_remove_request(model_type, gateway_host):
    remove_url = f"http://{gateway_host}/models/{model_type}/remove"
    try:
        response = requests.delete(remove_url)
    except requests.ConnectionError as e:
        logger.info(f"Connection error: {e}")
    except:
        logger.info(f"Unknown error contacting gateway service at {gateway_host}")


def register_model_instance(model_instance_id, model_host, gateway_host):

    logger.info(f"Preparing model registration request")
    register_url = (
        f"http://{gateway_host}/models/instances/{model_instance_id}/register"
    )
    register_data = {"host": model_host}
    logger.info(
        f"Sending model registration request to {register_url} with data: {register_data}"
    )
    try:
        response = requests.post(register_url, json=register_data)
        # HTTP error codes between 450 and 500 are custom to the kaleidoscope gateway
        if int(response.status_code) >= 450 and int(response.status_code) < 500:
            raise requests.HTTPError(response.content.decode("utf-8"))
    # If we fail to contact the gateway service, print an error but continue running anyway
    # TODO: HTTPError probably isn't the best way to catch custom errors
    except requests.HTTPError as e:
        logger.info(e)
    except requests.ConnectionError as e:
        logger.info(f"Connection error: {e}")
    except:
        logger.info(f"Unknown error contacting gateway service at {gateway_host}")


def activate_model_instance(model_instance_id, gateway_host):
    activation_url = (
        f"http://{gateway_host}/models/instances/{model_instance_id}/activate"
    )
    logger.info(f"Sending model activation request to {activation_url}")
    try:
        response = requests.post(activation_url)
        logger.info(f"Model activation response: {response.text}")
    except Exception as err:
        logger.info(f"Model instance activation failed with error: {err}")
        logger.info(f"Continuing to load model anyway, but it will not be accessible to any gateway services")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        help="Model type selected in the list: " + ", ".join(AVAILABLE_MODELS),
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Path to pre-trained model"
    )
    parser.add_argument("--model_instance_id", required=True, type=str)
    parser.add_argument(
        "--gateway_host", required=False, type=str, help="Hostname of gateway service", default="llm.cluster.local"
    )
    parser.add_argument(
        "--gateway_port", required=False, type=int, help="Port of gateway service", default=3001
    )
    parser.add_argument(
        "--master_port", required=False, type=int, help="Port for device communication", default=29400
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input arguments
    if args.model_type not in AVAILABLE_MODELS:
        logger.error(
            f"Error: model type {args.model_type} is not supported. Please use one of the following: {', '.join(AVAILABLE_MODELS)}"
        )
        sys.exit(1)

    gateway_host = f"{args.gateway_host}:{args.gateway_port}"

    # Determine the distributed training rank (if applicable)
    rank = 0
    if 'SLURM_PROCID' in os.environ:
        try:
            rank = int(os.environ['SLURM_PROCID'])
        except:
            pass

    logger.info(f"Loading model service with rank {rank}")

    # Determine the IP address for the head node of this model
    try:
        master_addr = os.environ['MASTER_ADDR']
    except:
        master_addr = "localhost"
        logger.info("MASTER_ADDR not set, defaulting to localhost")

    # # Find an ephemeral port to use for this model service - TODO - required when using Triton or PyTriton?
    # sock = socket.socket()
    # sock.bind(('', 0))
    # model_port = sock.getsockname()[1]
    # sock.close()

    model_port = 8003 #TRITON_HTTP_PORT # Port used by triton for HTTP requests
    model_host = f"{master_addr}:{model_port}"

    # Models that only run on a single node should advertise their IP address instead of "localhost"
    if master_addr == "localhost":
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        model_host = f"{ip_addr}:{model_port}"
    
    if rank == 0:
        logger.info(model_host)

    model_instance_id = args.model_instance_id
    # register model
    if rank == 0:
        register_model_instance(model_instance_id, model_host, gateway_host)

    if args.model_type == "GPT-J":
        # If directly loading model in triton
        if rank  == 0:
            os.system(
                f"CUDA_VISIBLE_DEVICES=0,1 " + \
                f"/opt/tritonserver/bin/tritonserver " + \
                f"--model-repository={args.model_path}/models/GPT-J/triton_model_store/gptj_2"
            )

    else:
        # If using pytriton
        # Setup a global model instance
        global model, model_type
        model = initialize_model(args.model_type)
        model_type = args.model_type

        # set master port
        os.environ["MASTER_PORT"] = str(args.master_port)

        # Load the model into GPU memory
        logger.info(f"Loading model into device {args.device}")
        logger.info(f"Loading model from model path {args.model_path}")
        model.load(args.device, args.model_path)

    # Register signal handlers
    #signal.signal(signal.SIGINT, signal_handler)
    #signal.signal(signal.SIGTERM, signal_handler)

    # Activate the model.
    # TODO: Is there any way we can do this after starting the Triton service?
    logger.info(f"Calling model activation function...")
    activate_model_instance(model_instance_id, gateway_host)

    # Now start the service. This will block until user hits Ctrl+C or the process gets killed by the system
    print("Starting Triton service...")
    if args.model_type != "GPT-J":
        triton_config = TritonConfig(http_address="0.0.0.0", http_port=8003, log_verbose=4)
        with Triton(config=triton_config) as triton:
            triton.bind(
                model_name=model_type,
                infer_func=model.generate,
                inputs=[
                    Tensor(name="prompts", dtype=bytes, shape=(1,))
                ],
                outputs=[
                    Tensor(name="sequences", dtype=bytes, shape=(-1,)),
                ],
                config=ModelConfig(max_batch_size=128),
            )
            logger.info("Starting model service, press Ctrl+C to exit")
            triton.serve()


if __name__ == "__main__":
    main()
