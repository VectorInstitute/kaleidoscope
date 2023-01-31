import subprocess
from config import Config
from models import ModelInstanceState

class ModelService():
    
        def __init__(self, model_instance_id, model_name, model_host = None):
            self.model_instance_id = model_instance_id
            self.model_name = model_name
            self.model_host = model_host

        def launch(self):
            try:
                ssh_output = subprocess.check_output(
                    f"ssh {Config.JOB_SCHEUDLER_HOST} python3 ~/lingua/model_service/job_runner.py --model_type {self.model_name}",
                    shell=True,
                ).decode("utf-8")
                print(f"Sent SSH request to job runner: {ssh_output}")
                success = True
            except Exception as err:
                print(f"Failed to issue SSH command to job runner: {err}")
            return success

        def shutdown(self):
            pass

        def verify_model_health(self, model_state):

            if model_state == ModelInstanceState.LAUNCHING:
                return False
            else if model_state == ModelInstanceState.LOADING:
                return True
            else if model_state == ModelInstanceState.ACTIVE:
                return True
            else:
                return False
        