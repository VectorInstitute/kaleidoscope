from __future__ import annotations
from flask import current_app
import subprocess
from typing import Dict

import requests

import models
from config import Config


def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 /h/llm/lingua/model_service/job_runner.py --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id}"
        #current_app.logger.info(f"Launch SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        current_app.logger.info(f"SSH launch job output: [{ssh_output}]")
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")
    return


def generate(host: str, generation_id: int, prompt: str, generation_args: Dict) -> Dict:
    body = {
        "id": generation_id,
        **generation_args
    }

    response = requests.post(
        f"http://{host}:5000/generate",
        json=body
    )

    response_body = response.json()
    return response_body


def verify_model_health(host: str) -> bool:
    response = requests.get(
        f"http://{host}:5000/health"
    )
    return response.status_code == 200


def verify_job_health(model_instance_id: str) -> bool:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 /h/llm/lingua/model_service/job_runner.py --action get_status --model_instance_id {model_instance_id}"
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