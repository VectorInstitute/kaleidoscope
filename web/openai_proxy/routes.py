import json
import logging
import os
import re
from datetime import timedelta
from threading import Lock, Thread
from time import sleep
from typing import Any, Generator, TypeVar

import requests
from flask import Blueprint, Flask, Response, jsonify, request

from ..auto_scaling import AutoScalingManager

openai_proxy_bp = Blueprint("v1", __name__)
scaling_manager = AutoScalingManager(
    min_update_interval=timedelta(seconds=10),
    max_num_historic_records=10,
    slurm_qos=os.environ.get("WORKER_SLURM_QOS"),
)

logging.basicConfig(level=logging.INFO)

STREAMING_DATA_PATTERN = re.compile(r"(data: )?(?P<json_data>.+)\n*")

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

    # Delete leading model weight folder names if any.
    model_name = model_name.split("/")[-1]

    # Either find model instance URLs or launch model
    model_backend = scaling_manager.get_llm_backend(model_name)
    backend_base_url = model_backend.base_url if model_backend else None
    if (
        (model_backend is None)
        or (not model_backend.is_ready)
        or (backend_base_url is None)
    ):
        return {
            "message": f"The requested model `{model_name}` is starting up.",
            "vector_inference_status": (
                model_backend.status.raw_status_data
                if model_backend is not None
                else None
            ),
        }, 503

    # stream request response (e.g., one token at a time) from backend to user
    response: requests.Response = requests.request(
        request.method,
        url=request.url.replace(request.host_url.rstrip("/") + "/v1", backend_base_url),
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


def auto_scaling_manager_thread_fn(scaling_manager: AutoScalingManager):
    while True:
        scaling_manager.check()
        sleep(10)


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(openai_proxy_bp)
    if "TELEMETRY_CALLBACK_URL" not in os.environ:
        raise EnvironmentError(
            "TELEMETRY_CALLBACK_URL must be set to collect callback from workers",
        )

    auto_scaling_manager_thread = Thread(
        target=auto_scaling_manager_thread_fn,
        args=(scaling_manager,),
    )
    auto_scaling_manager_thread.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PROXY_PORT", 25567)))
