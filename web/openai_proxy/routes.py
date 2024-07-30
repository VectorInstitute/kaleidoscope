import json
import re
from threading import Lock
from typing import Any, Generator, TypeVar

import requests
from flask import Blueprint, Flask, Response, jsonify, request

openai_proxy_bp = Blueprint("v1", __name__)

# Map model name to a list of host names
EXAMPLE_MODELS: dict[str, list[str]] = {
    "/model-weights/Mistral-7B-Instruct-v0.2": ["http://localhost:19132/"],
    "/model-weights/Meta-Llama-3-8B-Instruct": [],
}
STREAMING_DATA_PATTERN = re.compile(r"(data: )?(?P<json_data>.+)\n*")


def launch_model(model_name: str) -> None:
    """
    Trigger a model launch job.

    TODO: Async? In the background?
    """
    print(f"Launching model in the background: {model_name}")


example_response_log: list[str] = []
response_log_lock = Lock()
YieldType = TypeVar("YieldType")


def parse_openai_response(response_row: str) -> Any | None:
    """Parse a line of response from the OpenAI API HTTP.

    Params:
        response_row: str, an entire line of response,
            an entire line from e.g., split-line, not just a character.

    Returns:
        json-parsed row if parsable.
        Returns None otherwise.
    """
    log_data_match = STREAMING_DATA_PATTERN.match(response_row)
    if log_data_match:
        try:
            log_data = log_data_match.groupdict()["json_data"]
            return json.loads(log_data)

        except json.JSONDecodeError:
            return


def log_openai_response_stream(
    generator: Generator[YieldType, None, None],
    logging_destination: list[Any],
) -> Generator[YieldType, None, None]:
    """Log a copy of data that the generator yields.

    This method is thread-safe.

    Params:
        generator: Generator that yields but does not accept send.
            Return value from the generator is not supported.
            Also see stackoverflow.com/a/34073559
            regarding the generator return value.
        logging_destination: A pointer to a list to log into.
            The list would be updated in-place.

    Yields:
        Same as the given generator.
    """
    response_buffer = ""
    for value in generator:
        assert isinstance(value, bytes)

        response_char = value.decode("utf8")
        if response_char == "\n":
            parsed_data = parse_openai_response(response_buffer)
            if parsed_data is not None:
                with response_log_lock:
                    logging_destination.append(parsed_data)

            response_buffer = ""
        else:
            response_buffer += response_char

        yield value

    parsed_data = parse_openai_response(response_buffer)
    if parsed_data is not None:
        with response_log_lock:
            logging_destination.append(parsed_data)


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
        log_openai_response_stream(
            response.iter_content(),
            example_response_log,
        ),  # type: ignore[see readme]
        mimetype="application/octet-stream",
        direct_passthrough=True,
    )


@openai_proxy_bp.route("/response_log")
def response_log():
    return jsonify(example_response_log)


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(openai_proxy_bp)
    app.run(host="0.0.0.0", port=25765)
