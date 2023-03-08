from __future__ import annotations
from flask import current_app
import subprocess
from typing import Dict, List

import requests

import models
from config import Config


def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_REMOTE_BIN} --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id}"
        current_app.logger.info(f"Launch SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        current_app.logger.info(f"SSH launch job output: [{ssh_output}]")
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")
    return


def generate(
    host: str, generation_id: int, prompts: List[str],  generation_config: Dict
) -> Dict:
    
    current_app.logger.info("generating")

    body = {"prompt": prompts, **generation_config}

    current_app.logger.info(f"body {body}")

    response = requests.post(f"http://{host}/generate", json=body)

    current_app.logger.info(response)

    response_body = response.json()
    current_app.logger.info(response_body)
    return response_body

def generate_activations(
    host: str, 
    generation_id: int, 
    prompts: List[str], 
    module_names: List[str], 
    generation_config: Dict
) -> Dict:
    
    current_app.logger.info("activations")

    body = {"prompt": prompts, "module_names": module_names, **generation_config}

    current_app.logger.info(f"body {body}")

    response = requests.post(f"http://{host}/get_activations", json=body)

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
        response = requests.get(f"http://{host}/health")
        return response.status_code == 200
    except Exception as err:
        print(f"Model health verification error:: {err}")
        return False


def verify_job_health(model_instance_id: str) -> bool:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_REMOTE_BIN} --action get_status --model_instance_id {model_instance_id}"
        # print(f"Get status SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        # print(f"SSH get status output: [{ssh_output}]")

        # If we didn't get any output from SSH, the job doesn't exist
        if not ssh_output.strip(" \n"):
            return False

        # For now, assume that any output means the job is healthy
        return True
    except Exception as err:
        print(f"Failed to issue SSH command to job runner: {err}")
        return False
