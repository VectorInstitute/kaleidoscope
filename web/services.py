import subprocess
from config import Config

class ModelService():
    
        def __init__(self, model_instance_id, model_type):
            self.model_instance_id = model_instance_id
            self.model_type = model_type

        def launch(self):
            try:
                ssh_output = subprocess.check_output(
                    f"ssh {Config.JOB_SCHEUDLER_HOST} python3 ~/lingua/model_service/job_runner.py --model_type {self.model_type}",
                    shell=True,
                ).decode("utf-8")
                print(f"Sent SSH request to job runner: {ssh_output}")
                success = True
            except Exception as err:
                print(f"Failed to issue SSH command to job runner: {err}")
            return success

        def shutdown(self):
            pass

        def verify_model_health(self):
            pass