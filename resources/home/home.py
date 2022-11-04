from flask import Blueprint, render_template
from flask_jwt_extended import jwt_required

from models import ALL_MODELS

ALL_MODEL_NAMES = set(ALL_MODELS.keys())
MODEL_INSTANCES= {'OPT'} # Hard coded available models, will be udpated dynamically

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

@home_bp.route("/playground", methods=["GET"])
async def playground():
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", all_models=ALL_MODEL_NAMES, active_models=MODEL_INSTANCES)