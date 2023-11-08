"""Module for the model service API routes"""
import argparse
import logging
import importlib
from pathlib import Path
import random
import string
import time

# from pytriton.triton import Triton, TritonConfig
import ray
from ray import serve
from ray.serve.config import HTTPOptions

from services.gateway_service import GatewayServiceClient

logger = logging.getLogger("kaleidoscope.model_service")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

def initialize_model(model_type, model_variant):
    """Initializes model based on model type
    Args:
        model_type (str): Type of model to load
    """
    return importlib.import_module(f"models.{model_type}.model").Model(model_type, model_variant)

# Using Ray: Create model deployment class and add the serve.deployment decorator
@serve.deployment
class ModelDeployment():

    def __init__(self, model_type, model_variant, model_path):
        self.model = initialize_model(model_type, model_variant)
        self.model.load(model_path)

    def __call__(self, request):
        logger.debug(f"Type of request.query_params: {type(request.query_params)}")
        try:
            output = self.model.infer(**request.query_params)
        except Exception as err:
            output = {"Error": err, "Input": request.query_params}
        return output


class ModelService():
    ''' Model service is responsible for loading and serving a model.
    '''

    def __init__(self, model_instance_id: str, model_type: str, model_variant: str, model_path: str, gateway_host: str, gateway_port: int, master_host: str, master_port: int) -> None:
        """
        Args:
            model_instance_id (str): Unique identifier for model instance
            model_type (str): Type of model to load
            model_variant (str): Variant of model to load
            model_path (str): Path to pre-trained model
            gateway_host (str): Hostname for gateway service
            gateway_port (int): Port for gateway service
            master_host (str): Hostname for master service
            master_port (int): Port for master service
        """
        self.model_instance_id = model_instance_id
        self.model_type = model_type
        self.model_variant = model_variant if model_variant != "None" else ""
        self.model_path = model_path

        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
    
        self.master_host = master_host
        self.master_port = master_port

    def run(self):
        """Loads model and starts serving requests
        """

        # gateway_service = GatewayServiceClient(self.gateway_host, self.gateway_port)

        # model = initialize_model(self.model_type, self.model_variant)
        # model.load(self.model_path)

        # if model.rank == 0:
        #     logger.info(f"Starting model service for {self.model_type} on rank {model.rank}")

        #     #Placeholder static triton config for now
        #     triton_config = TritonConfig(http_address="0.0.0.0", http_port=self.master_port, log_verbose=4)
        #     triton_workspace = Path("/tmp") / Path("pytriton") / Path("".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=16)))
        #     with Triton(config=triton_config, workspace=triton_workspace) as triton:
        #         triton = model.bind(triton)
        #         triton.serve()

        # Using Ray: Deploy the Deployment class
        model_app = ModelDeployment.bind(
            model_type=self.model_type, 
            model_variant=self.model_variant, 
            model_path=self.model_path
            )
        serve.start(
            http_options=HTTPOptions(host="0.0.0.0", port=self.master_port)
            )
        handle = serve.run(model_app)
        # Hack - Keep server active
        while True:
            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", required=True, type=str, help="Type of model to load (ie. opt, gpt2)"
    )
    parser.add_argument(
        "--model_variant", required=True, type=str, help="Variant of model to load (ie. 6.7b)"
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Path to pre-trained model"
    )
    parser.add_argument(
        "--model_instance_id", required=True, type=str
    )
    parser.add_argument(
        "--master_host", required=True, type=str, help="Hostname for device communication"
    )
    parser.add_argument(
        "--master_port", required=True, type=int, help="Port for device communication"
    )
    parser.add_argument(
        "--gateway_host", required=False, type=str, help="Hostname of gateway service", default=None
    )
    parser.add_argument(
        "--gateway_port", required=False, type=int, help="Port of gateway service", default=None
    )
    args = parser.parse_args()

    # ToDo: Better init of model service
    model_service = ModelService(**args.__dict__)
    model_service.run()


if __name__ == "__main__":
    main()
