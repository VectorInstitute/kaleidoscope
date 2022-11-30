from flask import Blueprint, redirect, render_template, request
from flask_jwt_extended import jwt_required
import requests

from models import ALL_MODELS
from resources.models import MODEL_INSTANCES

ALL_MODEL_NAMES = set(ALL_MODELS.keys())

home_bp = Blueprint("home", __name__, template_folder='templates')

def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )

@home_bp.route("/health")
@jwt_required()
async def heartbeat():
    return "Still Alive"

@home_bp.route("/", methods=["GET"])
async def home():
    return render_template("home.html")

@home_bp.route("/login", methods=["GET", "POST"])
async def login():
    # GET requests display the login form
    if request.method == "GET":
        return render_template("login.html")

    # POST requests handle the login
    auth_url = request.url_root + "authenticate"
    result = requests.post(auth_url, auth=(request.form["username"], request.form["password"]))
    if result.status_code == 401:
        return render_template("login.html", result=result)

    return redirect(request.url_root, 200)

@home_bp.route("/reference", methods=["GET"])
async def reference():
    return render_template("reference.html")

@home_bp.route("/playground", methods=["GET"])
async def playground():
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", all_models=ALL_MODEL_NAMES, active_models=MODEL_INSTANCES)
