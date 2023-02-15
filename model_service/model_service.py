import argparse
import flask
import requests
import signal
import socket
import sys
import torch
import os

from flask import Flask, request, jsonify

import config


# Globals

AVAILABLE_MODELS = ["OPT-175B", "OPT-6.7B"]


# Start the Flask service that will hand off requests to the model libraries

service = Flask(__name__)


@service.route("/health", methods=["GET"])
def health():
    return {"msg": "Still Alive"}, 200


@service.route("/module_names", methods=["GET"])
def module_names():
    result = model.module_names()
    return result


@service.route("/generate", methods=["POST"])
def generate_text():
    result = model.generate(request)
    return result


@service.route("/get_activations", methods=["POST"])
def get_activations():
    print(request)
    print(request.json)
    result = model.get_activations(request)
    return result


# We only want to load the model library that's being requested, not all of them
# TODO: Is there a way to make this happen automatically, without separate entries?


def initialize_model(model_type):
    if model_type == "OPT-175B" or model_type == "OPT-6.7B":
        from models import OPT
        return OPT.OPT()


# Signal handler to send a remove request to the gateway, if this service is killed by the system


def signal_handler(sig, frame):
    global model_type
    send_remove_request(model_type)
    sys.exit(0)


def send_remove_request(model_type):
    remove_url = f"http://{config.GATEWAY_HOST}/models/{model_type}/remove"
    try:
        response = requests.delete(remove_url)
    except requests.ConnectionError as e:
        print(f"Connection error: {e}")
    except:
        print(f"Unknown error contacting gateway service at {config.GATEWAY_HOST}")


def register_model_instance(model_instance_id):

    master_addr = os.environ['MASTER_ADDR']
    MODEL_HOST = f'{master_addr}:8888'
    print(f"Preparing model registration request")
    register_url = f"http://{config.GATEWAY_HOST}/models/instances/{model_instance_id}/register"
    register_data = {"host": MODEL_HOST}
    print(f"Registering model with url={register_url}, data={register_data}")
    try:
        response = requests.post(register_url, json=register_data)
        # HTTP error codes between 450 and 500 are custom to the lingua gateway
        if int(response.status_code) >= 450 and int(response.status_code) < 500:
            raise requests.HTTPError(response.content.decode("utf-8"))
    # If we fail to contact the gateway service, print an error but continue running anyway
    # TODO: HTTPError probably isn't the best way to catch custom errors
    except requests.HTTPError as e:
        print(e)
    except requests.ConnectionError as e:
        print(f"Connection error: {e}")
    except:
        print(f"Unknown error contacting gateway service at {config.GATEWAY_HOST}")

def activate_model_instance(model_instance_id):
    activation_url = f"http://{config.GATEWAY_HOST}/models/instances/{model_instance_id}/activate"
    response = requests.post(activation_url)
    if not response.ok:
        print(f"Model instance activation failed with status code {response.status_code}: {response.text}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        help="Model type selected in the list: " + ", ".join(AVAILABLE_MODELS),
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Path to pre-trained model"
    )
    parser.add_argument(
        "--model_instance_id", required=True, type=str
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input arguments
    if args.model_type not in AVAILABLE_MODELS:
        print(
            f"Error: model type {args.model_type} is not supported. Please use one of the following: {', '.join(AVAILABLE_MODELS)}"
        )
        sys.exit(1)

    # Setup a global model instance
    global model, model_type

    model = initialize_model(args.model_type)
    model_instance_id = args.model_instance_id
    model_type = args.model_type

    register_model_instance(model_instance_id)

    # Load the model into GPU memory
    print(f"Loading model into device {args.device}")
    model.load(args.device, args.model_path)

    # Inform the gateway service that we are serving a new model instance by calling the /models/register endpoint
  #  print(f"Preparing model registration request")
  #  register_url = f"http://{config.GATEWAY_HOST}/models/{model_instance_id}/register"
  #  register_data = {"host": config.MODEL_HOST}
  #  print(f"Registering model with url={register_url}, data={register_data}")
  #  try:
  #      response = requests.post(register_url, json=register_data)
  #      # HTTP error codes between 450 and 500 are custom to the lingua gateway
  #      if int(response.status_code) >= 450 and int(response.status_code) < 500:
  #          raise requests.HTTPError(response.content.decode("utf-8"))
  #  # If we fail to contact the gateway service, print an error but continue running anyway
  #  # TODO: HTTPError probably isn't the best way to catch custom errors
  #  except requests.HTTPError as e:
  #      print(e)
  #  except requests.ConnectionError as e:
  #      print(f"Connection error: {e}")
  #  except:
  #      print(f"Unknown error contacting gateway service at {config.GATEWAY_HOST}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    activate_model_instance(model_instance_id)
    # Now start the service. This will block until user hits Ctrl+C or the process gets killed by the system
    print("Starting model service, press Ctrl+C to exit")
    service.run(
        host=config.MODEL_HOST.split(":")[0], port=config.MODEL_HOST.split(":")[1]
    )

    # Inform the gateway service that we are shutting down and it should remove this model
    send_remove_request(args.model_type)


if __name__ == "__main__":
    main()
