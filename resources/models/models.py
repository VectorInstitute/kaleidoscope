from flask import Blueprint, request, current_app

from models import ALL_MODELS
ALL_MODEL_NAMES = set(ALL_MODELS.keys())

models_bp = Blueprint("models", __name__)

def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
        )

@models_bp.route("/", methods=["GET"])
async def all_models():
    return list(ALL_MODEL_NAMES)

@models_bp.route("/<model_name>/generate_text", methods=["POST"])
async def generate_text(model_name: str):
    #verify_request(model_name)
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