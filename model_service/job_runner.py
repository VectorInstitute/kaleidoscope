import argparse
import subprocess
import torch

from config import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, help="Model type not supported")
    args = parser.parse_args()

    try:
        scheduler_output = subprocess.check_output(f"{JOB_SCHEDULER_BIN} {args.model_type}", shell=True).decode('utf-8')
        print(f"Job scheduler returned following output: {scheduler_output}")
    except Exception as err:
        print(f"Job scheduler failed: {err}")
    

if __name__ == "__main__":
    main()
