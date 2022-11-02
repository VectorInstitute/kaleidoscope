import argparse
import flask
import requests
import socket
import sys
import torch

from flask import Flask, request, jsonify

import config


# Start the Flask service that will hand off requests to the model libraries

service = Flask(__name__)

@service.route("/module_names", methods=["POST"])
def module_names():
    result = model.module_names()
    return result

@service.route("/generate_text", methods=["POST"])
def generate_text():
    result = model.generate_text(request)
    return result


# We only want to load the model library that's being requested, not all of them
# TODO: Is there a way to make this happen automatically, without separate entries?

AVAILABLE_MODELS = ["gpt2", "opt_125m"]

def initialize_model(model_type):
    if model_type == "gpt2":
        from models import gpt2
        return gpt2.GPT2()
    if model_type == "opt_125m":
        from models import opt_125m
        return opt_125m.OPT_125M()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, help="Model type selected in the list: " + ", ".join(AVAILABLE_MODELS))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input arguments
    if args.model_type not in AVAILABLE_MODELS:
        print(f"Error: model type {args.model_type} is not supported. Please use one of the following: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(1)

    # Setup a global model instance
    global model
    model = initialize_model(args.model_type)

    # Load the model into GPU memory
    model.load(args.device)

    # Inform the gateway service that we are serving a new model instance by calling the /register_model endpoint
    register_url = f"http://{config.GATEWAY_HOST}/register_model"
    register_data = {
        "model_host": config.MODEL_HOST,
        "model_type": args.model_type
    }
    try:
        response = requests.get(register_url, json=register_data)
        # HTTP error codes between 450 and 500 are custom to the lingua gateway
        if int(response.status_code) >= 450 and int(response.status_code) < 500:
            raise requests.HTTPError(response.content.decode('utf-8'))
    # If we fail to contact the gateway service, print an error but continue running anyway
    # TODO: HTTPError probably isn't the best way to catch custom errors
    except requests.HTTPError as e:
        print(e)
    except requests.ConnectionError as e:
        print(f"Connection error: {e}")
    except:
        print(f"Unknown error contacting gateway service at {config.GATEWAY_HOST}")

    # Now start the service. This will block until user hits Ctrl+C or the process gets killed by the system
    print("Starting model service, press Ctrl+C to exit")
    service.run(host=config.MODEL_HOST.split(':')[0], port=config.MODEL_HOST.split(':')[1])


if __name__ == "__main__":
    main()

