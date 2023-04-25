"""Module for model service client"""
from __future__ import annotations
import subprocess
from typing import Dict, List
from flask import current_app

import requests

from config import Config


def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    """Launch a model based on a defined path"""
    current_app.logger.info(f"Preparing to launch model: {model_name}")
    try:
        ssh_command = f"""ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} \
        {Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_name} \
        --model_path {model_path} --model_instance_id {model_instance_id} \
        --gateway_host {Config.GATEWAY_ADVERTISED_HOST} --gateway_port {Config.GATEWAY_PORT}"""
        current_app.logger.info(f"Launch SSH command: {ssh_command}")

        # System job scheduler needs ssh to keep running in the background
        if Config.JOB_SCHEDULER == "system":
            result = subprocess.Popen(ssh_command, shell=True, close_fds=True)
            current_app.logger.info(f"SSH launched system job with PID {result.pid}")
        # For all other job schedulers, wait for the ssh command to return
        else:
            ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
            current_app.logger.info(f"SSH launch job output: [{ssh_output}]")

    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")


def generate(host: str, generation_id: int, prompts: List[str], generation_config: Dict) -> Dict:
    """Generate into JSON format"""

    current_app.logger.info(f"Sending generation request to http://{host}/generate")

    body = {"prompt": prompts, **generation_config}

    current_app.logger.info(f"Generation request body: {body}")

    response = requests.post(f"http://{host}/generate", json=body)

    response_body = response.json()
    return response_body


def generate_activations(
    host: str,
    generation_id: int,
    prompts: List[str],
    module_names: List[str],
    generation_config: Dict,
) -> Dict:
    """Generate intermediate activations"""
    current_app.logger.info("activations")

    body = {
        "prompt": prompts,
        "module_names": module_names,
        **generation_config,
    }

    current_app.logger.info(f"body {body}")

    response = requests.post(f"http://{host}/get_activations", json=body)

    response_body = response.json()
    return response_body


def get_module_names(host: str) -> Dict:
    """Retrieve module names"""
    response = requests.get(
        f"http://{host}/module_names",
    )

    current_app.logger.info(response)

    response_body = response.json()
    current_app.logger.info(response_body)
    return response_body


def verify_model_health(host: str) -> bool:
    """Verify the health of a model"""
    try:
        response = requests.get(f"http://{host}/health")
        return response.status_code == 200
    except Exception as err:
        print(f"Model health verification error:: {err}")
        return False


def verify_job_health(model_instance_id: str) -> bool:
    """Verify if the tasks is healthy"""
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} \
        {Config.JOB_SCHEDULER_BIN} --action get_status --model_instance_id {model_instance_id}"
        # print(f"Get job health SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        # print(f"SSH get job health output: [{ssh_output}]")

        # If we didn't get any output from SSH, the job doesn't exist
        if not ssh_output.strip(" \n"):
            current_app.logger.info("No output from ssh, the model doesn't exist")
            return False

        # For now, assume that any output means the job is healthy
        print("The model is healthy")
        return True
    except Exception as err:
        print(f"Failed to issue SSH command to job runner: {err}")
        return False
