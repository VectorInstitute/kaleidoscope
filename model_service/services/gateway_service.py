import logging
import requests

logger = logging.getLogger(__name__)

class GatewayServiceClient():

    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port

    def register_model_instance(self, model_instance_id, model_host, model_port):
        """Registers model instance with gateway service

        Args:
            model_instance_id (str): Unique identifier for model instance
            model_host (str): Hostname for model service
            model_port (int): Port for model service
        """
        model_instance_registration_url =  f"http://{self.host}:{self.port}/models/instances/{model_instance_id}/register"

        registration_data = {
            "host": f"{model_host}:{model_port}"
        }

        logger.info(
            f"Sending model registration request to {model_instance_registration_url} with data: {registration_data}"
        )

        response = requests.post(model_instance_registration_url, json=registration_data)
        logger.info(f"Model registration response: {response.content.decode('utf-8')}")
        # HTTP error codes between 450 and 500 are custom to the kaleidoscope gateway
        if int(response.status_code) >= 450 and int(response.status_code) < 500:
            raise requests.HTTPError(response.content.decode("utf-8"))
        
    def activate_model_instance(self, model_instance_id):
        """Activates model instance with gateway service

        Args:
            model_instance_id (str): Unique identifier for model instance
        """

        model_instance_activation_url = f"http://{self.host}:{self.port}/models/instances/{model_instance_id}/activate"

        logger.info(
            f"Sending model activation request to {model_instance_activation_url}"
        )
        response = requests.post(model_instance_activation_url)

        logger.info(f"Model activation response: {response.content.decode('utf-8')}")
        # HTTP error codes between 450 and 500 are custom to the kaleidoscope gateway
        if int(response.status_code) >= 450 and int(response.status_code) < 500:
            raise requests.HTTPError(response.content.decode("utf-8"))