import argparse
import flask
import sys
import torch

from flask import Flask, request, jsonify
from models import *

MODEL_CLASSES = { 
    "gpt2": gpt2.GPT2(),
    "opt_175b": opt_175b.OPT_175B()
}


service = Flask(__name__)

@service.route("/module_names", methods=["POST"])
def module_names():
    result = model.module_names()
    return result

@service.route("/generate_text", methods=["POST"])
def generate_text():

    prompt = request.json["prompt"]
    args = {}
    args["length"] = int(request.json["length"]) if "length" in request.json else 128
    args["temperature"] = float(request.json["temperature"]) if "temperature" in request.json else 1.0
    args["k"] = float(request.json["k"]) if "k" in request.json else 0
    args["p"] = float(request.json["p"]) if "p" in request.json else 0.9
    args["device"] = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args["num_return_sequences"] = int(request.json["num_return_sequences"]) if "num_return_sequences" in request.json else 1
    args["repetition_penalty"] = float(request.json["repetition_penalty"]) if "repetition_penalty" in request.json else 1.0

    result = model.generate_text(prompt, args)

    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    args = parser.parse_args()

    # Validate input arguments
    if args.model_type not in MODEL_CLASSES.keys():
        print(f"Error: model type {args.model_type} is not supported. Please use one of the following: {', '.join(MODEL_CLASSES.keys())}")
        sys.exit(1)

    # Setup a global model instance
    global model
    model = MODEL_CLASSES[args.model_type]

    # Load the model into GPU memory
    model.load()
    
    # Start the Flask service and loop endlessly until exiting
    print("Starting model service, press Ctrl+C to exit")
    service.run(host="0.0.0.0", port=8888, threaded=False)
    while True:
        pass


if __name__ == "__main__":
    main()

