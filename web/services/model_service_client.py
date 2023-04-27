from __future__ import annotations
from flask import current_app
import subprocess
from typing import Dict, List
from fabric import Connection

import requests

import models
from config import Config


class JobScheduler:
    def run(self, command: str, type: str) -> str:
        scheduler = get_scheduler(type)
        return scheduler.run(command)
    
def get_scheduler(type: str) -> JobScheduler:
    if type == "system":
        return _system_job_scheduler
    elif type == "ssh":
        return _ssh_job_scheduler
    else:
        raise ValueError(type)
    
def _system_job_scheduler(command: str) -> None:
    command = " ".join([f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST}", command])
    result = subprocess.Popen(command, shell=True, close_fds=True)
    return result

def open_ssh_connection() -> Connection:
    return Connection(
        host=Config.JOB_SCHEDULER_HOST,
        user=Config.JOB_SCHEDULER_USER,
        connect_kwargs={"key_filename": Config.MODEL_SERVICE_KEY},
    )

def _ssh_job_scheduler(command: str) -> None:
    with open_ssh_connection() as c:
        result = c.run(command)
        if result.ok:
            return result.stdout
        else:
            raise RuntimeError(result.stderr)



def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    current_app.logger.info(f"Preparing to launch model: {model_name}")
    try:
        command = f"""{Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id} --gateway_host {Config.GATEWAY_ADVERTISED_HOST} --gateway_port {Config.GATEWAY_PORT}"""
        job_scheduler = JobScheduler()
        job_scheduler.run(command, Config.JOB_SCHEDULER)
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")
    return

def shutdown(model_instance_id: str) -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action shutdown --model_instance_id {model_instance_id}"
        current_app.logger.info(f"Shutdown SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        current_app.logger.info(f"SSH shutdown job output: [{ssh_output}]")
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job runner: {err}")
    return


def generate(
    host: str, generation_id: int, prompts: List[str], generation_config: Dict
) -> Dict:
    
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

    current_app.logger.info("activations")

    body = {"prompt": prompts, "module_names": module_names, **generation_config}

    current_app.logger.info(f"body {body}")

    response = requests.post(f"http://{host}/get_activations", json=body)

    response_body = response.json()
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

        command = f"{Config.JOB_SCHEDULER_BIN} --action get_status --model_instance_id {model_instance_id}"
        job_scheduler = JobScheduler()
        result = job_scheduler.run(command, Config.JOB_SCHEDULER)

        if not result.strip(' \n'):
            current_app.logger.info("No output from ssh, the model doesn't exist")
            return False
        
        return True

    except Exception as err:
        print(f"Failed to issue SSH command to job runner: {err}")
        return False

