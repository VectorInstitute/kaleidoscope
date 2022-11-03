from flask import Blueprint, render_template
from flask_jwt_extended import jwt_required

from models import ALL_MODELS

ALL_MODEL_NAMES = set(ALL_MODELS.keys())
gateway = Blueprint("gateway", __name__)

def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )

@gateway.route("/heartbeat")
@jwt_required()
async def heartbeat():
    return "Still Alive"



@gateway.route("/", methods=["GET"])
async def home():
    return render_template("home.html")

@gateway.route("/playground", methods=["GET"])
async def playground():
    #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
    return render_template("playground.html", models=ALL_MODEL_NAMES)

@gateway.route("/all_models", methods=["GET"])
async def all_models():
    return list(ALL_MODEL_NAMES)

@gateway.route("/<model_name>/generate_text", methods=["POST"])
async def generate_text(model_name: str):
    verify_request(model_name)
    data = request.form.copy()
    print(data)
    prompts = data["prompt"]
    del data["prompt"]
    # client_input = server_parse(obj)
    generated_text = ALL_MODELS[model_name].generate_text(model_name, prompts, **data)
    if isinstance(generated_text, dict):
        text_output = generated_text["choices"][0]["text"].strip()
    else:
        text_output = generated_text.strip()
    return text_output
    # return render_template(
    #     "index.html", models=ALL_MODEL_NAMES, text_output=text_output
    # )