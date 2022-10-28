"""NOTE: WE NEED TO RETURN THE PRE RNG TO THE USER OTHERWISE NO WAY THEY CAN REGEN AGAIN
- actually this doesn't even work, because how will the user pipe it back now? the best way might be to record the prompt and the initial RNG in a log file somewhere
- i guess we just flush after it hits a certain RAM size?

THE ASSUMPTION IS THAT EACH SERVER MODEL HAS A .model attribute

NEED TO try and catch to get better error messages and stack trace
    - probably do this via try and catch and dump out the exception into
    - an HTTPexception

#TODO: how to deal with no_grad?
"""
import traceback
from io import StringIO

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException
from transformers import set_seed
import socket

from models import ALL_MODELS
from utils import server_parse, server_send

"""when we send, make sure not to send big objects
that might be coupled
"""
DEVICE_FOR_INPUT = "cuda:0"

# ALL_MODELS = {k: v(DEVICE_FOR_INPUT) for k, v in ALL_MODELS.items()}
ALL_MODEL_NAMES = set(ALL_MODELS.keys())

app = FastAPI()


def call_with_error_handling(f, *args, **kwargs):
    """I guess we can auto hook this onto whatever
    calls the f, and return the error in HTTP request
    """

    try:
        return f(*args, **kwargs)

    except BaseException:
        buf = StringIO()
        traceback.print_exc(file=buf)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=buf.getvalue())


def server_grad_func_call(f, use_grad, /, *args, **kwargs):
    """use this function to call client_inputs that pipe to the server
    model and might require grad
    """

    with torch.set_grad_enabled(use_grad):
        output = call_with_error_handling(f, *args, **kwargs)

    return output


def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name {model_name} not found, "
            "only {ALL_MODEL_NAMES} supported",
        )


@app.get("/")
async def servers():
    return f"sample inference server for models: {set(ALL_MODELS.keys())}"


@app.get("/all_models")
async def all_models():
    return server_send(ALL_MODEL_NAMES)


@app.get("/{model_name}/module_names/")
async def module_names(model_name: str):
    verify_request(model_name)

    return server_send(ALL_MODELS[model_name].module_names)


@app.get("/{model_name}/parameter_names/")
async def parameter_names(model_name: str):
    verify_request(model_name)

    return server_send(ALL_MODELS[model_name].parameter_names)


@app.get("/{model_name}/probe_points/")
async def probe_points(model_name: str):
    verify_request(model_name)

    return server_send(ALL_MODELS[model_name].probe_points)


@app.post("/{model_name}/get_parameters")
async def get_parameters(model_name: str, data: bytes = File()):
    verify_request(model_name)

    param_names = server_parse(data)

    return server_send(ALL_MODELS[model_name].get_parameters(*param_names))


# the suffix here should all match remote models'
@app.post("/{model_name}/generate_text")
# async def inference(model_name: str, data: bytes = File()):
async def generate_text(model_name: str, obj: dict):
    # verify_request(model_name)
    print(obj)

    # client_input = server_parse(obj)
    # print(client_input)
    generated_text= ALL_MODELS[model_name].generate_text(model_name, obj)
    # generated_text = server_grad_func_call(
    #     ALL_MODELS[model_name].generate_text,
    #     client_input["use_grad"],
    #     client_input["prompts"],
    #     **client_input["gen_kwargs"],
    # )
    print(generated_text.text)
    return server_send(generated_text)


@app.post("/{model_name}/encode")
async def encode(model_name: str, data: bytes = File()):
    verify_request(model_name)

    client_input = server_parse(data)

    tokens = ALL_MODELS[model_name].encode(
        client_input["prompts"], **client_input["tokenizer_kwargs"]
    )

    return server_send(tokens)


@app.post("/{model_name}/call")
async def call(model_name: str, data: bytes = File()):
    verify_request(model_name)

    client_input = server_parse(data)

    output = server_grad_func_call(
        ALL_MODELS[model_name].__call__,
        client_input["use_grad"],
        client_input["probe_dict"],
        *client_input["args"],
        **client_input["kwargs"],
    )

    return server_send(output)


if __name__ == "__main__":
    hostname = socket.gethostname()
    ipaddr = socket.gethostbyname(hostname)
    print(ipaddr)
    set_seed(6)
    uvicorn.run("fastapi_server:app", host=str(ipaddr), port=8000)



