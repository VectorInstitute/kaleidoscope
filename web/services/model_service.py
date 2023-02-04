from __future__ import annotations
import subprocess
from typing import Dict

import requests

import models
from config import Config


def launch(model_instance: models.ModelInstance) -> None:
    try:
        ssh_output = subprocess.check_output(
            f"ssh {Config.JOB_SCHEUDLER_HOST} python3 ~/lingua/model_service/job_runner.py --model_type {model_instance.model_name}",
            shell=True,
        ).decode("utf-8")
        print(f"Sent SSH request to job runner: {ssh_output}")
    except Exception as err:
        print(f"Failed to issue SSH command to job runner: {err}")


def generate(host, generation_id: int, prompt: str, generation_args: Dict) -> Dict:

    body = {
        "id": generation_id,
        'prompt': prompt,
        'generation_args': generation_args
    }

    response = requests.post(
        f"http://{host}:5000/generate",
        body=body
    )

    response_body = response.json()
    
    response_body = {
        'generation': 'hello world'
    }

    return response_body