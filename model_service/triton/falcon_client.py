#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import time

import numpy as np

from pytriton.client import ModelClient

from web.utils.triton import TritonClient


logger = logging.getLogger("triton.falcon_client")


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
   
    num_tokens = 32
    
    # Using TritonClient
    prompts = ["William Shakespeare was a great writer"]*8 
    inputs = {
            "prompts": prompts,
            "max_tokens": num_tokens
            }
    model_name = "falcon-7b"
    batch_size = len(prompts)
    host = args.url.lstrip("http://")
    
    triton_client = TritonClient(host)
    start_time = time.time()
    generation = triton_client.infer(model_name, inputs, task="generation")
    print(generation)
    time_taken = time.time() - start_time

    # Common logging for both methods
    logger.info(f"Total time taken: {time_taken:.2f} secs")
    token_per_sec = (num_tokens*batch_size)/time_taken
    logger.info(f"tokens/sec: {token_per_sec:.2f}")


if __name__ == "__main__":
    main()
