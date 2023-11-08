import argparse
import logging
import time

import numpy as np

import requests

logger = logging.getLogger("ray.falcon_client")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Ray server (ex. grpc://localhost:8001)."
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

    sequence = [["William Shakespeare was a great writer"]]
    logger.info(f"Sequence: {sequence}")

    batch_size = len(sequence)
    def _param(value):
        return [[value]]*batch_size
    
    num_tokens = 8
    gen_params = {
        "max_tokens": _param(num_tokens),
        "do_sample": _param(False),
        "temperature": _param(0.7),
    }
    
    model_name = "falcon-7b_generation"
    params = {"prompts": sequence}
    params.update(gen_params)

    response = requests.get(args.url, params=params).json()

    logger.info(type(response))
    logger.info(response)


if __name__ == "__main__":
    main()
