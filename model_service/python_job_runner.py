import argparse
import pathlib
import subprocess
import torch

from config import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, type=str, help="Action for job runner to perform")
    parser.add_argument("--model_instance_id", required=True, type=str, help="Model type not supported")
    parser.add_argument("--model_type", type=str, help="Type of model requested")
    parser.add_argument("--model_path", type=str, help="Model type not supported")
    args = parser.parse_args()

    cwd = pathlib.Path(__file__).parent.resolve()

    if args.action == "launch":
        if not args.model_type:
            print("Argument --model_type must be specified to launch a job")
            return
        elif not args.model_path:
            print("Argument --model_path must be specified to launch a job")
            return

        try:
            scheduler_output = subprocess.check_output(f"python3 /model_service/model_service.py --model_type {args.model_type} --model_path {args.model_path} --model_instance_id {args.model_instance_id}", shell=True).decode('utf-8')
            print(f"Result: {scheduler_output}")
        except Exception as err:
            print(f"Job scheduler failed: {err}")

    elif args.action == "get_status":
        try:
            status_output = subprocess.check_output(f"squeue --noheader --name {args.model_instance_id}", shell=True).decode('utf-8')
            print(f"{status_output}")
        except Exception as err:
            print(f"Job status failed: {err}")
    

if __name__ == "__main__":
    main()
