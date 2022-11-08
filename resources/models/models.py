from flask import Blueprint, request, current_app

from . import MODEL_INSTANCES
from models import ALL_MODELS
ALL_MODEL_NAMES = set(ALL_MODELS.keys())

models_bp = Blueprint("models", __name__)

# Model Instances represents the set of models that are currently active and able to service requests

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


@models_bp.route("/", methods=["GET"])
async def all_models():
    return list(ALL_MODEL_NAMES)


@models_bp.route("/instances", methods=["GET"])
async def model_instances():
    return list(MODEL_INSTANCES.keys())


@models_bp.route("/register", methods=["GET"])
async def register_model():
    # If a model of this type has already been registered, return an error
    if request.json['model_type'] in MODEL_INSTANCES.keys():
        result = f"ERROR: Model type {request.json['model_type']} has already been registered"
        return result, 450

    # Register model and return success
    new_model = ModelInstance(request.json['model_type'], request.json['model_host'])
    MODEL_INSTANCES[request.json['model_type']] = new_model
    result = {"result": f"Successfully registered model {request.json['model_type']}"}
    return result, 200


@models_bp.route("/<model_name>/remove", methods=["GET"])
async def remove_model(model_name: str):
    # Make sure this model has actually been registered before trying to remove it
    if model_name in MODEL_INSTANCES.keys():
        del MODEL_INSTANCES[model_name]

    result = {"result": f"Successfully removed model {model_name}"}
    return result, 200


@models_bp.route("/<model_name>/generate_text", methods=["POST"])
async def generate_text(model_name: str):
    #verify_request(model_name)
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
