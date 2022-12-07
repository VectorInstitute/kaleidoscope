from db import db
from flask import Blueprint, request, current_app
import re
import requests
import sys
from werkzeug.exceptions import HTTPException

from . import *

# Model Instances represent models that are currently active and able to service requests
class ModelInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String)
    host = db.Column(db.String)

    def __init__(self, type, host):
        self.type = type
        self.host = host


def verify_request(model_name):
    if model_name not in ALL_MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )


models_bp = Blueprint("models", __name__)


@models_bp.route("/", methods=["GET"])
async def get_all_models():
    return list(ALL_MODEL_NAMES), 200


@models_bp.route("/instances", methods=["GET"])
async def model_instances():
    model_instance_query = db.select(ModelInstance)
    model_instances = db.session.execute(model_instance_query).all()

    instances = {model : "Inactive" for model in ALL_MODEL_NAMES}
    for model in model_instances:
        instances[model[0].type] = "Active"
    return instances, 200


@models_bp.route("/<model_type>/module_names", methods=["GET"])
async def get_module_names(model_type: str):
    verify_request(model_type)

    # Retrieve the host url for the requested model
    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()
    current_app.logger.info(f"Requesting module names for {model_type}")
    model_host = model_instance[0].host

    # Retrieve module names and return
    response = requests.get("http://" + model_host + "/module_names")
    return response.json(), 200


@models_bp.route("/register", methods=["POST"])
async def register_model():

    model_type = request.json['model_type']
    model_host = request.json['model_host']

    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()

    # If a model of this type has already been registered, return an error
    if model_instance is not None:
        result = f"ERROR: Model type {model_type} has already been registered"
        return result, 403

    # Register model and return success
    new_model_instance = ModelInstance(model_type, model_host)
    db.session.add(new_model_instance)
    db.session.commit()
    current_app.logger.info(f"Registered new model instance {model_type} ({model_host})")

    result = {"result": f"Successfully registered model {request.json['model_type']}"}
    return result, 200


@models_bp.route("/<model_type>/remove", methods=["DELETE"])
async def remove_model(model_type: str):

    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()

    current_app.logger.info(f"Removing model instance of {model_type}")

    if model_instance is None:
        result = {"result": f"Model not found."}
        return result, 404

    db.session.delete(model_instance[0])
    db.session.commit()

    result = {"result": f"Model removed."}
    return result, 200


@models_bp.route("/<model_type>/generate_text", methods=["POST"])
async def generate_text(model_type: str):
    verify_request(model_type)

    model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    model_instance = db.session.execute(model_instance_query).first()

    data = request.form.copy()
    prompt = data["prompt"]
    del data["prompt"]

    result = requests.post(
        "http://" + model_instance[0].host + "/generate_text", json={"prompt": prompt, **data}
    ).json()
    current_app.logger.info(f"Generate text result: {result}")
    return result, 200
