import requests
from flask import Blueprint, Flask, Response, request

openai_proxy_bp = Blueprint("v1", __name__)

# Map model name to a list of host names
EXAMPLE_MODELS: dict[str, list[str]] = {
    "/model-weights/Mistral-7B-Instruct-v0.2": ["http://localhost:19132/"],
    "/model-weights/Meta-Llama-3-8B-Instruct": [],
}


def launch_model(model_name: str) -> None:
    """
    Trigger a model launch job.

    TODO: Async? In the background?
    """
    print(f"Launching model in the background: {model_name}")


@openai_proxy_bp.route("/v1", defaults={"_path": ""})
@openai_proxy_bp.route("/v1/<path:_path>", methods=["GET", "POST"])
def reverse_proxy(_path):
    api_key = request.headers.get("Authorization", "").split(" ")[-1]
    if api_key != "EMPTY":
        return {"error": "Invalid API key"}, 403

    request_json = request.json
    assert request_json is not None
    model_name = request_json["model"]

    # Either find model instance URLs or launch model
    upstream_urls = EXAMPLE_MODELS.get(model_name, [])
    if len(upstream_urls) == 0:
        return {"message": f"The requested model `{model_name}` is starting up."}, 503

    # stream request response (e.g., one token at a time) from backend to user
    response: requests.Response = requests.request(
        request.method,
        url=request.url.replace(request.host_url, upstream_urls[0]),
        data=request.get_data(),
        cookies=request.cookies,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        stream=True,
    )

    return Response(
        response.iter_content(),  # type: ignore[see readme]
        mimetype="application/octet-stream",
        direct_passthrough=True,
    )


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(openai_proxy_bp)
    app.run(host="0.0.0.0", port=25765)
