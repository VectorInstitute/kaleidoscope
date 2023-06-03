#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess


MODEL_CONFIG = {
    "models": [
        {
            "name": "opt6.7b",
            "type": "opt",
            "launch_path": "opt/launch_opt6.7b.slurm"
        },
        {
            "name": "opt175b",
            "type": "opt",
            "launch_path": "opt/launch_opt175b.slurm"
        },
        {
            "name": "gpt2",
            "type": "gpt2",
            "launch_path": "gpt2/launch_gpt2.slurm"
        }
    ] 
}

def launch_job(args):
    for arg in ["model_name", "gateway_host", "gateway_port"]:
        if not getattr(args, arg):
            raise ValueError(f"Argument --{arg} must be specified to launch a job")
        
    print(MODEL_CONFIG)
    # ToDo: No validation of model_type, model_variant
    launch_path = [model['launch_path'] for model in MODEL_CONFIG['models'] if model['name'] == args.model_name][0]
    print(launch_path)
    try:
        cwd = pathlib.Path(__file__).parent.resolve()
        scheduler_cmd = f'sbatch --job-name={args.model_instance_id} {cwd}/models/{launch_path} {cwd} {args.gateway_host} {args.gateway_port}'
        print(f"Scheduler command: {scheduler_cmd}")
        scheduler_output = subprocess.check_output(
            scheduler_cmd, shell=True
        ).decode("utf-8")
        print(f"{scheduler_output}")
    except Exception as err:
        print(f"Job scheduler failed: {err}")

def shutdown_job(args):
    try:
        scheduler_cmd = f"scancel --jobname={args.model_instance_id}"
        print(f"Scheduler command: {scheduler_cmd}")
        scheduler_output = subprocess.check_output(
            scheduler_cmd, shell=True
        ).decode("utf-8")
        print(f"{scheduler_output}")
    except Exception as err:
        print(f"Job scheduler failed: {err}")

def get_job_status(args):
    try:
        status_output = subprocess.check_output(
            f"squeue --noheader --name {args.model_instance_id}", shell=True
        ).decode("utf-8")
        print(f"{status_output}")
    except Exception as err:
        print(f"Job status failed: {err}")

def get_model_config():
    metadata = []
    cwd = os.path.dirname(os.path.realpath(__file__))

    with open(f"{cwd}/config.json", "r") as config:
         metadata.append(json.load(config))
    print(metadata)

job_manager_actions = {
    'launch': launch_job,
    'shutdown': shutdown_job,
    'get_status': get_job_status,
    'get_model_config': get_model_config,
}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, type=str, help="Action for job manager to perform")
    parser.add_argument( "--model_instance_id", required=True, type=str, help="Model type not supported")
    parser.add_argument("--model_name", type=str, help="Type of model requested")
    parser.add_argument("--model_type", type=str, help="Type of model requested")
    parser.add_argument("--gateway_host", type=str, help="Hostname of gateway service")
    parser.add_argument("--gateway_port", type=int, help="Port of gateway service")
    args = parser.parse_args()

    job_manager_actions[args.action](args)

if __name__ == "__main__":
    main()
