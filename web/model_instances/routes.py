import requests
from flask import Blueprint, request, current_app
from flask_jwt_extended import jwt_required

import tasks
from db import db
from models import ModelInstance, ModelInstanceState
from . import ALL_MODEL_NAMES

model_instances_bp = Blueprint("models", __name__)


@model_instances_bp.route("/", methods=["GET"])
async def get_models():
    return list(ALL_MODEL_NAMES), 200

@model_instances_bp.route("/instances", methods=["GET"])
async def get_active_model_instances():
    model_instances = ModelInstance.get_current_instances()
    return model_instances, 200

@model_instances_bp.route("/instances", methods=["POST"])
@jwt_required
async def create_model_instance():

    model_type = request.json["type"]

    # ToDo move this into the DB
    current_model_instances = ModelInstance.get_current_instances()
    model_instance = next((mi for mi in current_model_instances if mi.type == model_type), None)

    if model_instance is None:
        model_instance = ModelInstance.create(model_type)
        tasks.launch_model_instance.delay(model_instance.id)

    return model_instance, 201

@model_instances_bp.route("instances/<model_instance_id>", methods=["GET"])
@jwt_required
async def get_model_instance(model_instance_id: int):
    model_instance = ModelInstance.get_by_id(model_instance_id)
    return model_instance, 200

@model_instances_bp.route("/instances/<model_instance_id>", methods=["DELETE"])
@jwt_required
async def remove_model_instance(model_instance_id: int):
    model_instance = ModelInstance.get_by_id(model_instance_id)
    tasks.shutdown_model_instance(model_instance.id)
    return model_instance, 200

@model_instances_bp.route("/instances/<model_instance_id>/status", methods=["PATCH"])
@jwt_required
async def update_model_instance_state(model_instance_id: int):
    model_instance = ModelInstance.get_by_id(model_instance_id)
    tasks.shutdown_model_instance.delay(model_instance.id)
    return model_instance, 200

@model_instances_bp.route("instances/<model_instance_id>/generate", methods=["POST"])
@jwt_required
async def model_instance_generate(model_instance_id: int):

    model_instance = ModelInstance.get_by_id(model_instance_id)

    if model_instance.state != ModelInstanceState.ACTIVE:
        return {"msg": "Model instance is not active"}, 400

    # ToDo restructure generation kwargs and validate
    data = request.form.copy()
    prompt = data["prompt"]
    del data["prompt"]

    model_instance.generate(prompt, data)

    data = request.form.copy()
    prompt = data["prompt"]
    del data["prompt"]

    result = requests.post(
        "http://" + model_instance[0].host + "/generate_text",
        json={"prompt": prompt, **data},
    ).json()
    current_app.logger.info(f"Generate text result: {result}")
    return result, 200


# @models_bp.route("/", methods=["GET"])
# async def get_all_models():
#     return list(ALL_MODEL_NAMES), 200


# @models_bp.route("/instances", methods=["GET"])
# async def model_instances():
#     model_instance_query = db.select(ModelInstance)
#     model_instances = db.session.execute(model_instance_query).all()

#     instances = {model: "Inactive" for model in ALL_MODEL_NAMES}
#     for model in model_instances:
#         instances[model[0].type] = "Active"
#     return instances, 200


# @models_bp.route("/<model_type>/module_names", methods=["GET"])
# async def get_module_names(model_type: str):
#     verify_request(model_type)

#     if is_model_active(model_type):
#         # Retrieve the host url for the requested model
#         model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
#         model_instance = db.session.execute(model_instance_query).first()
#         current_app.logger.info(f"Requesting module names for {model_type}")
#         model_host = model_instance[0].host

#         # Retrieve module names and return
#         response = requests.get("http://" + model_host + "/module_names")
#         return response.json(), 200
#     else:
#         return {}, 200


# @models_bp.route("/register", methods=["POST"])
# async def register_model():

#     model_type = request.json["model_type"]
#     model_host = request.json["model_host"]

#     model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
#     model_instance = db.session.execute(model_instance_query).first()

#     # If a model of this type has already been registered, return an error
#     if model_instance is not None:
#         result = f"ERROR: Model type {model_type} has already been registered"
#         return result, 403

#     # Register model and return success
#     new_model_instance = ModelInstance(model_type, model_host)
#     db.session.add(new_model_instance)
#     db.session.commit()
#     current_app.logger.info(
#         f"Registered new model instance {model_type} ({model_host})"
#     )

#     result = {"result": f"Successfully registered model {request.json['model_type']}"}
#     return result, 200


# @models_bp.route("/<model_type>/launch", methods=["POST"])
# @jwt_required()
# async def launch_model(model_type: str):
#     result = run_model_job(model_type)
#     return {}, 200


# @models_bp.route("/<model_type>/remove", methods=["DELETE"])
# async def remove_model(model_type: str):

#     model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
#     model_instance = db.session.execute(model_instance_query).first()

#     current_app.logger.info(f"Removing model instance of {model_type}")

#     if model_instance is None:
#         result = {"result": f"Model not found."}
#         return result, 404

#     db.session.delete(model_instance[0])
#     db.session.commit()

#     result = {"result": f"Model removed."}
#     return result, 200


# @models_bp.route("/<model_type>/generate_text", methods=["POST"])
# @jwt_required()
# async def generate_text(model_type: str):
#     verify_request(model_type)
#     if not is_model_active(model_type):
#         run_model_job(model_type)

#     model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
#     model_instance = db.session.execute(model_instance_query).first()

#     data = request.form.copy()
#     prompt = data["prompt"]
#     del data["prompt"]

#     result = requests.post(
#         "http://" + model_instance[0].host + "/generate_text",
#         json={"prompt": prompt, **data},
#     ).json()
#     current_app.logger.info(f"Generate text result: {result}")
#     return result, 200