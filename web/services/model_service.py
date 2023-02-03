import subprocess

from celery import shared_task

from config import Config
from models import ModelInstance, ModelInstanceGeneration


@shared_task
def launch(model_instance: ModelInstance) -> None:
    try:
        ssh_output = subprocess.check_output(
            f"ssh {Config.JOB_SCHEUDLER_HOST} python3 ~/lingua/model_service/job_runner.py --model_type {model_instance.model_name}",
            shell=True,
        ).decode("utf-8")
        print(f"Sent SSH request to job runner: {ssh_output}")
        success = True
    except Exception as err:
        print(f"Failed to issue SSH command to job runner: {err}")
    return success

@shared_task
def shutdown(model_instance: ModelInstance) -> None:
    pass
    

def generate(model_instance: ModelInstance, generate_request: ModelInstanceGeneration):
    