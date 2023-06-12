#!/usr/bin/env python3
"""Module to submit system level jobs"""
import argparse
import json
import os
import pathlib
import subprocess


def main():
    """Schedule system level jobs"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        type=str,
        help="Action for job runner to perform",
    )
    parser.add_argument(
        "--model_instance_id",
        required=True,
        type=str,
        help="Model type not supported",
    )
    parser.add_argument("--model_type", type=str, help="Type of model requested")
    parser.add_argument("--model_variant", type=str, help="Variant of model requested")
    parser.add_argument("--model_path", type=str, help="Model type not supported")
    parser.add_argument("--gateway_host", type=str, help="Hostname of gateway service")
    parser.add_argument("--gateway_port", type=int, help="Port of gateway service")
    args = parser.parse_args()

    cwd = pathlib.Path(__file__).parent.resolve()

    if args.action == "launch":
        if not args.model_type:
            print("Argument --model_type must be specified to launch a job")
            return
        if not args.model_variant:
            print("Argument --model_variant must be specified to launch a job")
            return
        elif not args.model_path:
            print("Argument --model_path must be specified to launch a job")
            return
        elif not args.gateway_host:
            print("Argument --gateway_host must be specified to launch a job")
            return
        elif not args.gateway_port:
            print("Argument --gateway_port must be specified to launch a job")
            return

        try:
            process = subprocess.Popen(
                [
                    'python3',
                    f'{cwd}/model_service.py',
                    '--model_type', f'{args.model_type}',
                    '--model_variant', f'{args.model_variant}',
                    '--model_path', f'{args.model_path}',
                    '--model_instance_id', f'{args.model_instance_id}',
                    '--gateway_host', f'{args.gateway_host}',
                    '--gateway_port', f'{args.gateway_port}',
                    '--master_host', 'localhost',
                    '--master_port', '8080'
                ],
                start_new_session=True,
            )
            print(f"Started model service under PID {process.pid}")
        except Exception as err:
            print(f"Job scheduler failed: {err}")

    if args.action == "get_status":
        try:
            # TODO: Find a better way to determine if the model_service process is active
            status_output = subprocess.check_output(
                f"ps aux | grep {args.model_instance_id} | grep -v get_status | grep -v grep",
                shell=True,
            ).decode("utf-8")
            print(f"{status_output}")
        except Exception as err:
            # If the command fails, don't send any response, this will indicate failure
            print(f"Model status error: {err}")

    elif args.action == "get_model_config":
        # Look at every subdirectory under the /models directory, and grab config.json files
        metadata = []
        cwd = os.path.dirname(os.path.realpath(__file__))
        for subdir in os.listdir(f"{cwd}/models"):
            if os.path.isdir(os.path.join(f"{cwd}/models", subdir)):
                try:
                    with open(f"{cwd}/models/{subdir}/config.json", "r") as config:
                        metadata.append(json.load(config))
                except:
                    pass
        print(metadata)

if __name__ == "__main__":
    main()
