import argparse
import logging
import time

import numpy as np

# from pytriton.client import ModelClient
from model_service.models.llama2.model import Model

from web.utils.triton import TritonClient
from enum import Enum


logger = logging.getLogger("triton.llama2_client")


class Task(Enum):
    """Task enum"""
    GENERATE = 0
    GET_ACTIVATIONS = 1
    EDIT_ACTIVATIONS = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    
    model_type, model_variant = "llama2", "7b_chat"
    model_path = "/model-weights/Llama-2-7b-chat"
    llama2_model = Model(model_type, model_variant)
    llama2_model.load(model_path)

    # get activations
    batch_size = 8
    num_tokens = 16
    layer = "layers.20"
    prompts = [["William Shakespeare was a great writer"]]*batch_size
    prompts = np.char.encode(np.array(prompts), "utf-8")
    modules = [[layer]]*batch_size
    modules = np.char.encode(np.array(modules), "utf-8")

    def _param(dtype, value):
        if bool(value):
            return np.ones((batch_size, 1), dtype=dtype) * value
        else:
            return np.zeros((batch_size, 1), dtype=dtype)
    
    inputs = {
        "prompts": prompts,
        "modules": modules,
        "max_tokens": _param(np.int64, num_tokens),
    }

    # activations = llama2_model.get_activations(inputs)
    # print(activations)

    # Benchmarking over iterations
    num_iters = 1000
    mean_window = 50
    time_taken = []
    mean_time_taken = []
    for idx in range(num_iters):
        start_time = time.time()
        activations = llama2_model.get_activations(inputs)
        time_taken.append(time.time() - start_time)
        print(f"Time taken for iteration {idx+1}: {time_taken[-1]}")
        if (idx + 1) % mean_window == 0:
            mean_time_taken.append(np.mean(time_taken[-mean_window:]))
            print(f"Mean time taken for last {mean_window} iterations: {mean_time_taken[-1]}")
    print(f"Mean time taken every {mean_window} iterations: {mean_time_taken}")
    

if __name__ == "__main__":
    main()
