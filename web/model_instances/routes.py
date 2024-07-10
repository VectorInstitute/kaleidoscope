"""Module to represent model instance API routes"""
from flask import Blueprint, request, current_app, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from config import Config
from db import db
import tasks
from models import ModelInstance, AVAIALBLE_MODELS
from errors import InvalidStateError


model_instances_bp = Blueprint("models", __name__)

@model_instances_bp.route("/", methods=["GET"])
async def get_models():
    current_app.logger.info(f"Available models: {AVAIALBLE_MODELS}")
    return AVAIALBLE_MODELS, 200


@model_instances_bp.route("/instances", methods=["GET"])
async def get_current_model_instances():
    """Retrieve current model instances"""
    model_instances = ModelInstance.find_current_instances()
    response = jsonify([model_instance.serialize() for model_instance in model_instances])

    return response, 200


@model_instances_bp.route("/instances", methods=["POST"])
@jwt_required()
async def create_model_instance():
    """Launch a model instance if not active"""
    current_app.logger.info(f"Received model instance creation request: {request}")
    model_name = request.json["name"]
    if model_name not in AVAIALBLE_MODELS:
        return (
            jsonify(
                msg=f"Model name {model_name} not found in model list {AVAIALBLE_MODELS}"
            ),
            400,
        )
    model_instance = ModelInstance.find_current_instance_by_name(name=model_name)
    if model_instance is None:
        model_instance = ModelInstance.create(name=model_name)
        model_instance.launch()

    return jsonify(model_instance.serialize()), 201


@model_instances_bp.route("instances/<model_instance_id>", methods=["GET"])
@jwt_required()
async def get_model_instance(model_instance_id: str):
    """Get model instance by ID"""
    model_instance = ModelInstance.find_by_id(model_instance_id)
    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("/instances/<model_instance_id>", methods=["DELETE"])
@jwt_required()
async def remove_model_instance(model_instance_id: str):
    """Remove a model instance by ID"""
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.shutdown()

    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("/instances/<model_instance_id>/register", methods=["POST"])
async def register_model_instance(model_instance_id: str):
    """Register a model instance by ID"""
    current_app.logger.info(
        f"Received model registration for ID {model_instance_id}, request: {request}"
    )
    model_instance_host = request.json["host"]

    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.register(host=model_instance_host)

    return jsonify(model_instance.serialize()), 200


@model_instances_bp.route("instances/<model_instance_id>/generate", methods=["POST"])
@jwt_required()
async def model_instance_generate(model_instance_id: str):
    """Retrieve generation for a model instance"""
    username = get_jwt_identity()
    prompts = request.json["prompts"]
    generation_config = request.json["generation_config"]

    if len(prompts) > int(Config.BATCH_REQUEST_LIMIT):
        return (
            jsonify(
                msg=f"Request batch size of {len(prompts)} exceeds prescribed \
        limit of {Config.BATCH_REQUEST_LIMIT}"
            ),
            400,
        )
    else:
        model_instance = ModelInstance.find_by_id(model_instance_id)
        inputs = {
            "prompts": prompts,
            **generation_config
        }
        try:
            generation = model_instance.generate(username, inputs)
        except InvalidStateError as err:
            return jsonify(msg=f"Generation failed: {err}"), 400

        if isinstance(generation.generation, tuple):
            err, input = generation.generation
            return jsonify(msg=f"Generation failed: {err}, Error Source: {input}"), 400
        if isinstance(generation.generation, Exception):
            return jsonify(msg=f"Generation failed: {generation.generation}"), 500

        return jsonify(generation.serialize()), 200
