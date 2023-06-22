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


from utils.triton import TritonClient

# UPDATE (1406): Inserted model_type arg to fetch config for specific model. Set default as None to preserve original function.
def get_model_config(model_type: str=None) -> List:
    config = []
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action get_model_config --model_instance_id 0"
        #print(f"Get model config SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        #print(f"Get model config SSH output: {ssh_output}")
        config = ast.literal_eval(ssh_output)

        if model_type is not None:
            for model_cfg in config:
                if model_cfg["type"] == model_type:
                    config = [model_cfg]
    except Exception as err:
        print(f"Failed to issue SSH command to job manager: {err}")
    return config

def launch(model_instance_id: str, model_type: str, model_variant: str, model_path: str) -> None:
    current_app.logger.info(f"Model service client: launching {model_type} with ID {model_instance_id}")
    try:
        ssh_command = f"""ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} {Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_type} --model_variant {model_variant} --model_instance_id {model_instance_id} --model_path {model_path} --gateway_host {Config.GATEWAY_ADVERTISED_HOST} --gateway_port {Config.GATEWAY_PORT}"""
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


# TODO: Implement with Triton
def edit_activations(
    host: str,
    generation_id: int,
    prompts: List[str],
    modules: Dict[str, Optional[Callable]],
    generation_config: Dict,
) -> Dict:
    body = {"prompt": prompts, "modules": modules, **generation_config}
    current_app.logger.info(f"Sending edit activations request, body: {body}")
    response = requests.post(f"http://{host}/edit_activations", json=body)
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
        return triton_client.is_model_ready(model_name, task="generation")

    except Exception as err:
        current_app.logger.error(f"Model active check failed: {err}")
        return False

def verify_model_health(host: str, model_name: str) -> bool:
    try:
        triton_client = TritonClient(host)
        return triton_client.is_model_ready(model_name, task="generation")

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
    generation = triton_client.infer(model_name, inputs, task="generation")
    return generation

    # # Only for GPT-J
    # MODEl_GPTJ_FASTERTRANSFORMER = "ensemble"

    # client = httpclient.InferenceServerClient(host,
    #                                           concurrency=1,
    #                                           verbose=False)

    # inputs = [[elm] for elm in prompts]
    # param_config = json.load(open("../../models/GPT-J/config.json", "r")) # TODO - Query model service to fetch param config
    # param_config = update_param_cfg(param_config, generation_config)
    # from pprint import pprint
    # pprint(param_config)
    # inputs = prepare_inputs(inputs, param_config)

    # result = client.infer(MODEl_GPTJ_FASTERTRANSFORMER, inputs)
    # output0 = result.as_numpy("OUTPUT_0")
    # print(output0.shape)
    # print(output0)

def generate_activations(host: str, model_name: str, inputs: Dict) -> Dict:

    triton_client = TritonClient(host)
    return triton_client.infer(model_name, inputs, task="activations")


