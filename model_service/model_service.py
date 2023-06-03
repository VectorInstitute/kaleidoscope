import argparse
import logging
import importlib
import os

from pytriton.triton import Triton, TritonConfig

from services.gateway_service import GatewayServiceClient

logger = logging.getLogger("kaleidoscope.model_service")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

def initialize_model(model_name, model_type):
    """Initializes model based on model type
    Args:
        model_type (str): Type of model to load
    """
    return importlib.import_module(f"models.{model_type}.model").Model(model_name)

class ModelService():
    ''' Model service is responsible for loading and serving a model.
    '''

    def __init__(self, model_instance_id: str, model_name: str, model_type: str, model_path: str, gateway_host: str, gateway_port: int, master_host: str, master_port: int) -> None:
        """
        Args:
            model_instance_id (str): Unique identifier for model instance
            model_type (str): Type of model to load
            model_path (str): Path to pre-trained model
            gateway_host (str): Hostname for gateway service
            gateway_port (int): Port for gateway service
            master_host (str): Hostname for master service
            master_port (int): Port for master service
        """
        self.model_instance_id = model_instance_id
        self.model_name= model_name
        self.model_type = model_type
        self.model_path = model_path

        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
    
        self.master_host = master_host
        self.master_port = master_port

    def run(self):
        """Loads model and starts serving requests
        """

        gateway_service = GatewayServiceClient(self.gateway_host, self.gateway_port)

        model = initialize_model(self.model_name, self.model_type)
        model.load(self.model_path)

        if model.rank == 0:
            logger.info(f"Starting model service for {self.model_type} on rank {model.rank}")
            gateway_service.register_model_instance(self.model_instance_id, self.master_host, self.master_port)
            gateway_service.activate_model_instance(self.model_instance_id)

            #Placeholder static triton config for now
            triton_config = TritonConfig(http_address="0.0.0.0", http_port=self.master_port, log_verbose=4)
            with Triton(config=triton_config) as triton:
                triton = model.bind(triton)
                triton.serve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, type=str, help="Name of model to load"
    )
    parser.add_argument(
        "--model_type", required=True, type=str, help="Type of model to load"
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
