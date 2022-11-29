from flask import Blueprint, request, current_app
# import sqlalchemy as sa

from . import MODEL_INSTANCES
from models import ALL_MODELS
from db import db

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
    return list(ALL_MODEL_NAMES)


@models_bp.route("/instances", methods=["GET"])
async def model_instances():
    model_instance_query = db.select(ModelInstance)
    model_instances = db.session.execute(model_instance_query).all()
    return list(model_instances), 200


@models_bp.route("/<model_name>/module_names", methods=["GET"])
async def get_module_names(model_name: str):
    verify_request(model_name)
    module_names= ALL_MODELS[model_name].get_module_names(model_name)
    return module_names


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
    generated_text = ALL_MODELS[model_name].generate_text(model_name, prompts, **data)
    return generated_text
