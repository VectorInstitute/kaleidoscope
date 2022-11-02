#!/usr/bin/env python3

from flask import Flask, render_template, request

from models import ALL_MODELS

# from utils import server_parse, server_send


gateway = Flask(__name__)

ALL_MODEL_NAMES = set(ALL_MODELS.keys())
AVAILABLE_MODELS = {}


class ModelInstance():

    def __init__(self, type, host):
        self.type = type
        self.host = host


def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )


@gateway.route("/", methods=["GET"])
async def home():
    return render_template("home.html")


@gateway.route("/playground", methods=["GET"])
async def playground():
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", models=ALL_MODEL_NAMES)


@gateway.route("/all_models", methods=["GET"])
async def all_models():
    return list(ALL_MODEL_NAMES)


@gateway.route("/available_models", methods=["GET"])
async def available_models():
    return list(AVAILABLE_MODELS.keys())


@gateway.route("/register_model", methods=["GET"])
async def register_model():
    # If a model of this type has already been registered, return an error
    if request.json['model_type'] in AVAILABLE_MODELS.keys():
        result = f"ERROR: Model type {request.json['model_type']} has already been registered"
        return result, 450

    # Register model and return success
    new_model = ModelInstance(request.json['model_type'], request.json['model_host'])
    AVAILABLE_MODELS[request.json['model_type']] = new_model
    result = {"result": f"Successfully registered model {request.json['model_type']}"}
    return result, 200


# @gateway.route("/<model_name>/module_names/", methods=["GET"])
# async def module_names(model_name: str):
#     verify_request(model_name)
#     return server_send(ALL_MODELS[model_name].module_names)


# @gateway.route("/<model_name>/parameter_names/", methods=['GET'])
# async def parameter_names(model_name: str):
#     verify_request(model_name)
#     return server_send(ALL_MODELS[model_name].parameter_names)


# @gateway.route("/<model_name>/probe_points/", methods=['POST'])
# async def probe_points(model_name: str):
#     verify_request(model_name)
#     return server_send(ALL_MODELS[model_name].probe_points)


# @gateway.route("/<model_name>/get_parameters", methods=['POST'])
# async def get_parameters(model_name: str):
#     verify_request(model_name)
#     param_names = server_parse(data)
#     return server_send(ALL_MODELS[model_name].get_parameters(*param_names))


@gateway.route("/<model_name>/generate_text", methods=["POST"])
async def generate_text(model_name: str):
    verify_request(model_name)
    data = request.form.copy()
    print(data)
    prompts = data["prompt"]
    del data["prompt"]
    # client_input = server_parse(obj)
    generated_text = ALL_MODELS[model_name].generate_text(model_name, prompts, **data)
    if isinstance(generated_text, dict):
        text_output = generated_text["choices"][0]["text"].strip()
    else:
        text_output = generated_text.strip()
    return text_output
    # return render_template(
    #     "index.html", models=ALL_MODEL_NAMES, text_output=text_output
    # )


if __name__ == "__main__":
    gateway.run(host="0.0.0.0", port=3001, debug=True)
