"""Module for model service client"""
from __future__ import annotations
import ast
from flask import current_app
import json
import numpy as np
import requests
import subprocess
from typing import Callable, Dict, List, Optional

from config import Config


from utils.triton import Task, TritonClient


def get_available_models() -> List:
    available_models = []
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action get_available_models --model_instance_id 0"
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        available_models = ast.literal_eval(ssh_output)
    except Exception as err:
        print(f"Failed to issue SSH command to job manager: {err}")
    return available_models

def launch(model_instance_id: str, model_name: str) -> None:
    current_app.logger.info(f"Model service client: launching {model_name} with ID {model_instance_id}")
    try:
        ssh_command = f"""ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} {Config.JOB_SCHEDULER_BIN} --action launch --model_name {model_name} --model_instance_id {model_instance_id} --gateway_host {Config.GATEWAY_ADVERTISED_HOST} --gateway_port {Config.GATEWAY_PORT}"""
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
        current_app.logger.error(f"Failed to issue SSH command to job manager: {err}")

    return

def shutdown(model_instance_id: str) -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action shutdown --model_instance_id {model_instance_id}"
        current_app.logger.info(f"Shutdown SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        current_app.logger.info(f"SSH shutdown job output: [{ssh_output}]")
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job manager: {err}")
    return

def verify_job_health(model_instance_id: str) -> bool:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} {Config.JOB_SCHEDULER_BIN} --action get_status --model_instance_id {model_instance_id}"
        #print(f"Get job health SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        #print(f"SSH get job health output: [{ssh_output}]")

        # If we didn't get any output from SSH, the job doesn't exist
        if not ssh_output.strip(' \n'):
            current_app.logger.info("No output from ssh, the job doesn't exist")
            return False

        # For now, assume that any output means the job is healthy
        print("The job is healthy")
        return True
    except Exception as err:
        print(f"Failed to issue SSH command to job manager: {err}")
        return False

def verify_model_instance_active(host: str, model_name: str) -> bool:
    try:
        triton_client = TritonClient(host)
        return triton_client.is_model_ready(model_name)

    except Exception as err:
        current_app.logger.error(f"Model active check failed: {err}")
        return False

def verify_model_health(host: str, model_name: str) -> bool:
    try:
        triton_client = TritonClient(host)
        return triton_client.is_model_ready(model_name)

    except Exception as err:
        current_app.logger.error(f"Model health failed check: {err}")
        return False

def shutdown(model_instance_id: str) -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action shutdown --model_instance_id {model_instance_id}"
        current_app.logger.info(f"Shutdown SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        current_app.logger.info(f"SSH shutdown job output: [{ssh_output}]")
    except Exception as err:
        current_app.logger.error(f"Failed to issue SSH command to job manager: {err}")
    return

def generate(host: str, model_name: str, inputs: Dict) -> Dict:

    triton_client = TritonClient(host)
    generation = triton_client.infer(model_name, inputs, task=Task.GENERATE)
    return generation
