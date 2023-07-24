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

    # sequence = np.array(
    #     [
    #         ["Show me the meaning of "],
    #         ["I would love to learn cook the Asian street food"],
    #         ["Carnival in Rio de Janeiro"],
    #         ["William Shakespeare was a great writer"],
    #     ]
    # )

    sequence = np.array(
        [
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
            ["William Shakespeare was a great writer"],
        ]
    )

    sequence = np.char.encode(sequence, "utf-8")
    logger.info(f"Sequence: {sequence}")

    batch_size = sequence.shape[0]
    def _param(dtype, value):
        if bool(value):
            return np.ones((batch_size, 1), dtype=dtype) * value
        else:
            return np.zeros((batch_size, 1), dtype=dtype)
    
    num_tokens = 8
    gen_params = {
        "max_tokens": _param(np.int64, num_tokens),
        "do_sample": _param(np.bool_, False),
        "temperature": _param(np.float64, 0.7),
    }
    
    model_name = "falcon-40b_generation"

    # logger.info(f"Waiting for response...")
    # start_time = time.time()
    # with ModelClient(args.url, model_name, init_timeout_s=args.init_timeout_s) as client:
    #     for req_idx in range(1, args.iterations + 1):
    #         logger.info(f"Sending request ({req_idx}).")
    #         result_dict = client.infer_batch(
    #             prompts=sequence, **gen_params)
    #         logger.info(f"Result: {result_dict} for request ({req_idx}).")
    # time_taken = time.time() - start_time
    # logger.info(f"Total time taken: {time_taken:.2f} secs")
    # token_per_sec = (num_tokens*batch_size)/time_taken
    # logger.info(f"tokens/sec: {token_per_sec:.2f}")


    # benchmark
    n_runs = 5
    run_times = []
    for run_idx in range(n_runs):
        start_time = time.time()
        with ModelClient(args.url, model_name, init_timeout_s=args.init_timeout_s) as client:
            for req_idx in range(1, args.iterations + 1):
                logger.info(f"Sending request ({req_idx}).")
                result_dict = client.infer_batch(
                    prompts=sequence, **gen_params)
        run_times.append(time.time() - start_time)
    mean_token_per_sec = np.mean([(num_tokens*batch_size)/elm for elm in run_times])
    logger.info(f"seq_len: {num_tokens}, tokens/sec: {mean_token_per_sec:.2f}")

if __name__ == "__main__":
    main()
