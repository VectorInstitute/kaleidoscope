from flask import Blueprint, request, current_app
# import sqlalchemy as sa

from . import MODEL_INSTANCES
from models import ALL_MODELS
from db import db
import sys

ALL_MODEL_NAMES = set(ALL_MODELS.keys())

models_bp = Blueprint("models", __name__)

# Model Instances represents the set of models that are currently active and able to service requests

def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )

class ModelInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String)
    host = db.Column(db.String)

    def __init__(self, type, host):
        self.type = type
        self.host = host

@models_bp.route("/", methods=["GET"])
async def get_all_models():
    return list(ALL_MODEL_NAMES), 200


def get_active_models():
    model_instance_query = db.select(ModelInstance)
    model_instances = db.session.execute(model_instance_query).all()
    active_models= {model[0].type : "http://"+model[0].host for model in model_instances}
    return active_models

def get_current_model(model_name):
    active_models= get_active_models()
    selected_model = ALL_MODELS[model_name](active_models[model_name])
    return selected_model

@models_bp.route("/instances", methods=["GET"])
async def model_instances():
    model_instances= get_active_models()
    instances= {model : "Inactive" for model in ALL_MODELS}
    for model in model_instances.keys(): 
        instances[model]= "Active"
    return instances, 200


@models_bp.route("/<model_name>/module_names", methods=["GET"])
async def get_module_names(model_name: str):
    verify_request(model_name)
    selected_model= get_current_model(model_name)
    module_names = selected_model.get_module_names()
    return module_names, 200


@models_bp.route("/register", methods=["POST"])
async def register_model():
    # If a model of this type has already been registered, return an error
    model_type = request.json['model_type']
    model_host = request.json['model_host']

    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()

    current_app.logger.info(model_instance)

    if model_instance is not None:
        result = f"ERROR: Model type {model_type} has already been registered"
        return result, 403

    # Register model and return success
    new_model_instance = ModelInstance(model_type, model_host)

    db.session.add(new_model_instance)
    db.session.commit()
    print(f"{model_type} and {model_host} registered", file=sys.stderr)
    ALL_MODELS[model_type].url= model_host

    result = {"result": f"Successfully registered model {request.json['model_type']}"}
    return result, 200

@models_bp.route("/<model_type>/remove", methods=["DELETE"])
async def remove_model(model_type: str):

    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()

    current_app.logger.info(model_instance)

    if model_instance is None:
        result = {"result": f"Model not found."}
        return result, 404

    db.session.delete(model_instance[0])
    db.session.commit()

    result = {"result": f"Model removed."}
    return result, 200

@models_bp.route("/<model_name>/generate_text", methods=["POST"])
async def generate_text(model_name: str):
    verify_request(model_name)
    data = request.form.copy()
    prompts = data["prompt"]
    del data["prompt"]
    selected_model= get_current_model(model_name)
    generated_text = selected_model.generate_text(prompts, **data)
    return generated_text, 200
