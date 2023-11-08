import argparse
import logging
import time

import numpy as np
import ray

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

    @ray.remote
    def send_query(text):
        params = {"prompts": text}
        num_tokens = 8
        gen_params = {
            "max_tokens": num_tokens,
            "do_sample": False,
            "temperature": 1.0,
        }
        params.update(gen_params)
        response = requests.get(args.url, params=params).json()
        return response
    
    batch_size=1
    sequence = ["William Shakespeare"]*batch_size

    results = ray.get([send_query.remote(text) for text in sequence])
    logger.info(results)


if __name__ == "__main__":
    main()
