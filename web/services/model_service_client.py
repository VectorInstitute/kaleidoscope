from __future__ import annotations
from flask import current_app
import subprocess
from typing import Dict, List

import requests

import models
from config import Config


def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    current_app.logger.info(f"Launching model: {model_name}")
    # TODO: For now we are assuming that local jobs are launched via python, non-local jobs are launched via ssh. We should check the JOB_SCHEDULER_BIN. 
    # If the job is being scheduled locally, just run the command directly
    if Config.JOB_SCHEDULER_HOST == "localhost":
        try:
            local_command = f"python3 {Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id}"
            current_app.logger.info(f"Local launch job command: {local_command}")
            local_output = subprocess.check_output(local_command, shell=True).decode("utf-8")
            current_app.logger.info(f"Local launch job output: [{local_output}]")
        except Exception as err:
            current_app.logger.error(f"Failed to issue local command to job runner: {err}")
    # Otherwise, send the launch command via ssh to the job scheduler host
    else:
        try:
            ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id}"
            current_app.logger.info(f"Launch SSH command: {ssh_command}")
            ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
            current_app.logger.info(f"SSH launch job output: [{ssh_output}]")
        except Exception as err:
            current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")
    return


def generate(host: str, generation_id: int, prompt: str,  generation_config: Dict) -> Dict:
    
    current_app.logger.info("generating")

    body = {
        "prompt": [prompt],
        **generation_config
    }

    current_app.logger.info(f"body {body}")

    response = requests.post(
        f"http://{host}/generate",
        json=body
    )

    current_app.logger.info(response)

    response_body = response.json()
    current_app.logger.info(response_body)
    return response_body

def generate_activations(host: str, generation_id: int, prompt: str, module_names: List[str], generation_config: Dict) -> Dict:
    
    current_app.logger.info("activations")

    body = {
        "prompt": [prompt],
        "module_names": module_names,
        **generation_config
    }

    current_app.logger.info(f"body {body}")

    response = requests.post(
        f"http://{host}/get_activations",
        json=body
    )

    current_app.logger.info(response)

    response_body = response.json()
    current_app.logger.info(response_body)
    return response_body

def get_module_names(host: str) -> Dict:

    response = requests.get(
        f"http://{host}/module_names",
    )

    current_app.logger.info(response)

    response_body = response.json()
    current_app.logger.info(response_body)
    return response_body


def verify_model_health(host: str) -> bool:
    try:
        response = requests.get(
            f"http://{host}/health"
        )
        return response.status_code == 200
    except Exception as err:
        print(f"Model health verification error:: {err}")
        return False


def verify_job_health(model_instance_id: str) -> bool:
    if Config.JOB_SCHEDULER_HOST == "localhost":
        try:
            # How do we check health status of a local model?
            local_command = f"curl localhost:9001/health"
            print(f"Get status local command: {local_command}")
            local_output = subprocess.check_output(local_command, shell=True).decode("utf-8")
            print(f"Get status local output: [{local_output}]")

            # If we didn't get any output from SSH, the job doesn't exist
            if not ssh_output.strip(' \n'):
                return False

            # For now, assume that any output means the job is healthy
            return True
        except Exception as err:
            print(f"Failed to issue SSH command to job runner: {err}")
            return False
    else:
        try:
            ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action get_status --model_instance_id {model_instance_id}"
            #print(f"Get status SSH command: {ssh_command}")
            ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
            #print(f"SSH get status output: [{ssh_output}]")

            # If we didn't get any output from SSH, the job doesn't exist
            if not ssh_output.strip(' \n'):
                return False

            # For now, assume that any output means the job is healthy
            return True
        except Exception as err:
            print(f"Failed to issue SSH command to job runner: {err}")
            return False

