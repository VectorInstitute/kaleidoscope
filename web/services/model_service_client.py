from __future__ import annotations
from flask import current_app
import subprocess
from typing import Dict, List

import requests

# import models
from config import Config

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def launch(model_instance_id: str, model_name: str, model_path: str) -> None:
    current_app.logger.info(f"Preparing to launch model: {model_name}")
    try:
        ssh_command = f"""ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} {Config.JOB_SCHEDULER_BIN} --action launch --model_type {model_name} --model_path {model_path} --model_instance_id {model_instance_id} --gateway_host {Config.GATEWAY_ADVERTISED_HOST} --gateway_port {Config.GATEWAY_PORT}"""
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


def generate(
    host: str, generation_id: int, prompts: List[str], generation_config: Dict
) -> Dict:
    
    # current_app.logger.info(f"Sending generation request to http://{host}/generate")

    # body = {"prompt": prompts, **generation_config}

    # current_app.logger.info(f"Generation request body: {body}")

    # response = requests.post(f"http://{host}/generate", json=body)

    # response_body = response.json()
    # return response_body

    # Only for GPT-J
    MODEl_GPTJ_FASTERTRANSFORMER = "ensemble" 
    OUTPUT_LEN = 128
    BATCH_SIZE = 2
    BEAM_WIDTH = 1
    TOP_K = 1
    TOP_P = 0.0
    start_id = 220
    end_id = 50256

    # Inference hyperparameters
    def prepare_tensor(name, input):
        tensor = httpclient.InferInput(
            name, input.shape, np_to_triton_dtype(input.dtype))
        tensor.set_data_from_numpy(input)
        return tensor

    # explanation
    def prepare_inputs(input0):
        bad_words_list = np.array([[""]]*(len(input0)), dtype=object)
        stop_words_list = np.array([[""]]*(len(input0)), dtype=object)
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
        runtime_top_k = (TOP_K * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        runtime_top_p = TOP_P * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.int32)
        is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
        beam_width = (BEAM_WIDTH * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

        inputs = [
            prepare_tensor("INPUT_0", input0_data),
            prepare_tensor("INPUT_1", output0_len),
            prepare_tensor("INPUT_2", bad_words_list),
            prepare_tensor("INPUT_3", stop_words_list),
            prepare_tensor("runtime_top_k", runtime_top_k),
            prepare_tensor("runtime_top_p", runtime_top_p),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
            prepare_tensor("temperature", temperature),
            prepare_tensor("len_penalty", len_penalty),
            prepare_tensor("repetition_penalty", repetition_penalty),
            prepare_tensor("random_seed", random_seed),
            prepare_tensor("is_return_log_probs", is_return_log_probs),
            prepare_tensor("beam_width", beam_width),
            prepare_tensor("start_id", start_ids),
            prepare_tensor("end_id", end_ids),
        ]
        return inputs
    
    client = httpclient.InferenceServerClient(host,
                                              concurrency=1,
                                              verbose=False)
    
    input0 = [[elm] for elm in prompts]
    inputs = prepare_inputs(input0)

    result = client.infer(MODEl_GPTJ_FASTERTRANSFORMER, inputs)
    output0 = result.as_numpy("OUTPUT_0")
    print(output0.shape)
    print(output0)


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
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} {Config.JOB_SCHEDULER_BIN} --action get_status --model_instance_id {model_instance_id}"
        #print(f"Get job health SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        #print(f"SSH get job health output: [{ssh_output}]")

        # If we didn't get any output from SSH, the job doesn't exist
        if not ssh_output.strip(' \n'):
            current_app.logger.info("No output from ssh, the model doesn't exist")
            return False

        # For now, assume that any output means the job is healthy
        print("The model is healthy")
        return True
    except Exception as err:
        print(f"Failed to issue SSH command to job manager: {err}")
        return False


def get_model_metadata() -> None:
    try:
        ssh_command = f"ssh {Config.JOB_SCHEDULER_USER}@{Config.JOB_SCHEDULER_HOST} python3 {Config.JOB_SCHEDULER_BIN} --action get_model_metadata --model_instance_id 0"
        print(f"Get model metadata SSH command: {ssh_command}")
        ssh_output = subprocess.check_output(ssh_command, shell=True).decode("utf-8")
        print(f"Get model metadata SSH output: [{ssh_output}]")
    except Exception as err:
        print(f"Failed to issue SSH command to job manager: {err}")
    return
