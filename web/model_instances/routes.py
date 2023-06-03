import requests
from flask import Blueprint, request, current_app, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from config import Config
from db import db
import tasks
from models import MODEL_CONFIG, ModelInstance

model_instances_bp = Blueprint("models", __name__)

@model_instances_bp.route("/", methods=["GET"])
async def get_models():
    current_app.logger.info(f"Received model list request: {request}")
    models = [key for key in MODEL_CONFIG.keys()]
    return models, 200


@model_instances_bp.route("/instances", methods=["GET"])
async def get_current_model_instances():
    model_instances = ModelInstance.find_current_instances()
    response = jsonify(
        [model_instance.serialize() for model_instance in model_instances]
    )

    return response, 200


@model_instances_bp.route("/instances", methods=["POST"])
@jwt_required()
async def create_model_instance():
    current_app.logger.info(f"Received model instance creation request: {request}")
    model_name = request.json["name"]
    model_instance = ModelInstance.find_current_instance_by_name(name=model_name)
    if model_instance is None:
        model_instance = ModelInstance.create(name=model_name)
        model_instance.launch()

    return jsonify(model_instance.serialize()), 201


@model_instances_bp.route("instances/<model_instance_id>", methods=["GET"])
@jwt_required()
async def get_model_instance(model_instance_id: str):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("/instances/<model_instance_id>", methods=["DELETE"])
@jwt_required()
async def remove_model_instance(model_instance_id: str):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.shutdown()

    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("/instances/<model_instance_id>/register", methods=["POST"])
async def register_model_instance(model_instance_id: str):

    current_app.logger.info(
        f"Received model registration for ID {model_instance_id}, request: {request}"
    )
    model_instance_host = request.json["host"]

    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.register(host=model_instance_host)

    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("/instances/<model_instance_id>/activate", methods=["POST"])
async def activate_model_instance(model_instance_id: str):
    current_app.logger.info(
        f"Received model activation for ID {model_instance_id}"
    )
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.activate()

    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("instances/<model_instance_id>/generate", methods=["POST"])
@jwt_required()
async def model_instance_generate(model_instance_id: str):

    username = get_jwt_identity()
    current_app.logger.info(f"Sending generate request for {username}: {request.json}")

    prompts = request.json["prompts"]
    generation_config = request.json["generation_config"]
    
    if len(prompts) > int(Config.BATCH_REQUEST_LIMIT):
        return jsonify(msg=f"Request batch size of {len(prompts)} exceeds prescribed limit of {Config.BATCH_REQUEST_LIMIT}"), 400
    else:
        model_instance = ModelInstance.find_by_id(model_instance_id)
        inputs = {
            "prompts": prompts,
            **generation_config
        }
        generation = model_instance.generate(username, inputs)

        return jsonify(generation.serialize()), 200


@model_instances_bp.route("instances/<model_instance_id>/module_names", methods=["GET"])
@jwt_required()
async def get_module_names(model_instance_id: str):

    model_instance = ModelInstance.find_by_id(model_instance_id)
    module_names = model_instance.get_module_names()

    return jsonify(module_names), 200


@model_instances_bp.route(
    "/instances/<model_instance_id>/generate_activations", methods=["POST"]
)
@jwt_required()
async def get_activations(model_instance_id: str):

    username = get_jwt_identity()
    prompts = request.json["prompts"]
    module_names = request.json["module_names"]
    generation_config = request.json["generation_config"]

    if len(prompts) > Config.BATCH_REQUEST_LIMIT:
        return jsonify(msg=f"Request batch size of {len(prompts)} exceeds prescribed limit of {Config.BATCH_REQUEST_LIMIT}"), 400
    else:
        model_instance = ModelInstance.find_by_id(model_instance_id)
        activations = model_instance.generate_activations(
            username, prompts, module_names, generation_config
        )

        return jsonify(activations), 200

    # model_instance_query = db.select(ModelInstance).filter_by(type=model_type)
    # model_instance = db.session.execute(model_instance_query).first()

    # data = request.json
    # result = requests.post(
    #     "http://" + model_instance[0].host + "/get_activations",
    #     json=data
    # ).json()

    # return result, 200
