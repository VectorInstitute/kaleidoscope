import requests
from flask import Blueprint, request, current_app, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from db import db
import tasks
from models import MODEL_CONFIG, ModelInstance

model_instances_bp = Blueprint("models", __name__)

@model_instances_bp.route("/", methods=["GET"])
async def get_models():
    return list(MODEL_CONFIG.keys()), 200

@model_instances_bp.route("/instances", methods=["GET"])
async def get_current_model_instances():
    model_instances = ModelInstance.find_current_instances()
    response = jsonify([{ "id": model_instance.id, "name": model_instance.name, "state": model_instance.state.name } for model_instance in model_instances])
    return response, 200

@model_instances_bp.route("/instances", methods=["POST"])
@jwt_required()
async def create_model_instance():

    model_name = request.json["name"]

    model_instance = ModelInstance.find_current_instance_by_name(name=model_name)
    current_app.logger.info(f"Found model instance: {model_instance}")
    if model_instance is None:
        model_instance = ModelInstance.create(name=model_name)
        current_app.logger.info(f"Created new model instance: {model_instance}")
        tasks.launch_model_instance.delay(model_instance.id)

    response = jsonify({
        "id": model_instance.id,
        "name": model_instance.name,
        "state": model_instance.state.name,
    })

    return response, 201

@model_instances_bp.route("instances/<model_instance_id>", methods=["GET"])
@jwt_required()
async def get_model_instance(model_instance_id: int):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    return model_instance, 200

@model_instances_bp.route("/instances/<model_instance_id>", methods=["DELETE"])
@jwt_required()
async def remove_model_instance(model_instance_id: int):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.shutdown()

    return model_instance, 200

@model_instances_bp.route("/instances/<model_instance_id>/register", methods=["POST"])
@jwt_required()
async def register_model_instance(model_instance_id: int):

    model_instance_host = request.json["host"]
    current_app.logger.info(f"Registering model instance {model_instance_id} with host {model_instance_host}")
    model_instance = ModelInstance.find_by_id(model_instance_id)
    current_app.logger.info(f"Found model instance: {model_instance}")
    model_instance.register(host=model_instance_host)

    return jsonify({"id": model_instance.id, "name": model_instance.name, "state": model_instance.state.name }), 200

@model_instances_bp.route("/instances/<model_instance_id>/activate", methods=["POST"])
@jwt_required()
async def activate_model_instance(model_instance_id: int):

    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.activate()

    return jsonify({ "id": model_instance.id, "name": model_instance.name, "state": model_instance.state.name }), 200


@model_instances_bp.route("instances/<model_instance_id>/generate", methods=["POST"])
@jwt_required()
async def model_instance_generate(model_instance_id: int):
    
    username = get_jwt_identity()
    prompt = request.json["prompt"]
    
    model_instance = ModelInstance.find_by_id(model_instance_id)
    generation = model_instance.generate(username, prompt)

    return jsonify(generation), 200