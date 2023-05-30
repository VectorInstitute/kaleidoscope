import argparse
import logging
import importlib
import requests

from pytriton.triton import Triton, TritonConfig


logger = logging.getLogger("kaleidoscope.model_service")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def initialize_model(model_type):
    #try:
    return importlib.import_module(f"models.{model_type}.model").Model(model_type)
    # except:
    #     logger.error(f"Could not import model {model_type}")

class ModelService():

    def __init__(self, model_type, model_path, gateway_host, gateway_port, master_port) -> None:
        self.model_type = model_type
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.master_port = master_port
        self.model_path = model_path

    # def register_model_instance(self, model_instance_id):
    #     register_url = f"http://{self.gateway_host}/models/instances"
    #     print(f"Sending model registration request to {register_url}")
    #     try:
    #         response = requests.post(
    #             register_url,
    #             json={
    #                 "model_instance_id": model_instance_id,
    #                 "model_type": self.model_type,
    #                 "model_host": model_host,
    #             },
    #         )
    #         response.raise_for_status()
    #     except requests.HTTPError as e:
    #         print(e)
    #     except requests.ConnectionError as e:
    #         print(f"Connection error: {e}")
    #     except:
    #         print(f"Unknown error contacting gateway service at {self.gateway_host}")

    def run(self):

        # How do we know rank here?
        # self.register_model_instance()

        model = initialize_model(self.model_type)
        model.load(self.model_path) # Must be a blocking call

        logger.info(f"Starting model service for {self.model_type} on rank {model.rank}")
        if model.rank == 0:
            logger.info(f"Starting model service for {self.model_type} on rank {model.rank}")
            triton_config = TritonConfig(http_address="0.0.0.0", http_port=8003, log_verbose=4)
            with Triton(config=triton_config) as triton:
                triton = model.bind(triton)
                triton.serve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Path to pre-trained model"
    )
    parser.add_argument("--model_instance_id", required=True, type=str)
    parser.add_argument(
        "--gateway_host", required=False, type=str, help="Hostname of gateway service", default="llm.cluster.local"
    )
    parser.add_argument(
        "--gateway_port", required=False, type=int, help="Port of gateway service", default=3001
    )
    parser.add_argument(
        "--master_port", required=False, type=int, help="Port for device communication", default=29400
    )
    args = parser.parse_args()

    model_service = ModelService(args.model_type, args.model_path, args.gateway_host, args.gateway_port, args.master_port)
    model_service.run()


if __name__ == "__main__":
    main()
