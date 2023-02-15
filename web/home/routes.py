from flask import Blueprint, render_template

import json
import requests

import models

home_bp = Blueprint("home", __name__, template_folder="templates")


@home_bp.route("/health")
async def heartbeat():
    return "Still Alive"


@home_bp.route("/", methods=["GET"])
async def home():
    return render_template("home.html")


@home_bp.route("/login", methods=["GET"])
async def login():
    return render_template("login.html")


@home_bp.route("/reference", methods=["GET"])
async def reference():
    return render_template("reference.html")


@home_bp.route("/playground", methods=["GET"])
async def playground():
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", all_models=models.MODEL_CONFIG.keys())
