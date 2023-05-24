"""Module to define the API routes for the web application"""
from flask import Blueprint, render_template

import models

home_bp = Blueprint("home", __name__, template_folder="templates")


@home_bp.route("/health")
async def heartbeat():
    """Retrieves heartbeat status upon call"""
    return "Still Alive"


@home_bp.route("/", methods=["GET"])
async def home():
    """Retrieves the homepage"""
    return render_template("home.html")


@home_bp.route("/login", methods=["GET"])
async def login():
    """Retrieves the login page"""
    return render_template("login.html")


@home_bp.route("/reference", methods=["GET"])
async def reference():
    """Retrieves the API reference page"""
    return render_template("reference.html")


@home_bp.route("/playground", methods=["GET"])
async def playground():
    """Retrieves the playground page"""
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", all_models=models.MODEL_CONFIG.keys())
