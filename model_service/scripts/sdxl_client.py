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
import base64
import logging
import numpy as np
from pathlib import Path
from pytriton.client import ModelClient

logger = logging.getLogger("triton.sdxl_client")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://gpu113.cluster.local:8080",
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
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    sequences = np.array(
        [
            ["Photo of a cat riding a horse on the surface of Jupiter"]
        ]
    )
    task = np.array([1])
    task = task.reshape(-1, 1)
    sequences = np.char.encode(sequences, "utf-8")
    logger.info(f"Sequences: {sequences}")

    #result_dict = client.infer_batch(sequences)
    #logger.info(f"Result: {result_dict}")

    with ModelClient(args.url, "sdxl", init_timeout_s=args.init_timeout_s) as client:
        for req_idx in range(1, 2):
            logger.info(f"Sending request ({req_idx}) with sequences={sequences}")
            result_dict = client.infer_batch(task, sequences)
            logger.info(f"Result keys: {result_dict.keys()} for request ({req_idx}).")
    
            output_file = f"sdxl-output-{req_idx}.png"
            with open(output_file, "wb") as f:
                f.write(base64.decodebytes(result_dict["images"][0]))



if __name__ == "__main__":
    main()
