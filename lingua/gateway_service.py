#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify, redirect, url_for
import asyncio
import json

from models import ALL_MODELS
from utils import server_parse, server_send


app = Flask(__name__)


"""prevent sending big objects that might be coupled"""
# DEVICE_FOR_INPUT = "cuda:0"

# ALL_MODELS = {k: v(DEVICE_FOR_INPUT) for k, v in ALL_MODELS.items()}
ALL_MODEL_NAMES = set(ALL_MODELS.keys())

# def call_with_error_handling(f, *args, **kwargs):
#     """I guess we can auto hook this onto whatever
#     calls the f, and return the error in HTTP request
#     """

#     try:
#         return f(*args, **kwargs)
        
#     except BaseException:
#         buf = StringIO()
#         traceback.print_exc(file=buf)
#         traceback.print_exc()
#         raise HTTPException(status_code=400, detail=buf.getvalue())

# def server_grad_func_call(f, use_grad, /, *args, **kwargs):
#     """use this function to call client_inputs that pipe to the server
#     model and might require grad
#     """
#     with torch.set_grad_enabled(use_grad):
#         output = call_with_error_handling(f, *args, **kwargs)
#     return output

def verify_request(model_name):
    if model_name not in ALL_MODELS.keys():
        raise HTTPException(
            status_code=422,
            detail=f"model_name <model_name> not found, "
            "only {ALL_MODEL_NAMES} supported",
       )


@app.route("/", methods=['GET'])
async def home():
   #  return f"sample inference server for models: {set(ALL_MODELS.keys())}"
   return render_template('index.html', models=ALL_MODEL_NAMES)


@app.route("/all_models", methods=['GET'])
async def all_models():
    return server_send(ALL_MODEL_NAMES)


@app.route("/<model_name>/module_names/", methods=['GET'])
async def module_names(model_name: str):
    verify_request(model_name)
    return server_send(ALL_MODELS[model_name].module_names)


# @app.route("/<model_name>/parameter_names/", methods=['GET'])
# async def parameter_names(model_name: str):
#     verify_request(model_name)
#     return server_send(ALL_MODELS[model_name].parameter_names)


# @app.route("/<model_name>/probe_points/", methods=['POST'])
# async def probe_points(model_name: str):
#     verify_request(model_name)
#     return server_send(ALL_MODELS[model_name].probe_points)


# @app.route("/<model_name>/get_parameters", methods=['POST'])
# async def get_parameters(model_name: str):
#     verify_request(model_name)
#     param_names = server_parse(data)
#     return server_send(ALL_MODELS[model_name].get_parameters(*param_names))


# the suffix here should all match remote models'
@app.route("/<model_name>/generate_text", methods=['POST'])
# async def inference(model_name: str, data: bytes = File()):
async def generate_text(model_name: str):
    # verify_request(model_name)
    data= request.form.copy() 
    prompts= data['prompt']
    del data['prompt']
    # client_input = server_parse(obj)
    generated_text= ALL_MODELS[model_name].generate_text(model_name, prompts, **data)
    text_output= generated_text['choices'][0]['text']
    print(text_output)
    # generated_text = server_grad_func_call(
    #     ALL_MODELS[model_name].generate_text,
    #     client_input["use_grad"],
    #     client_input["prompts"],
    #     **client_input["gen_kwargs"],
    # )
    return render_template('index.html', models=ALL_MODEL_NAMES, text_output= text_output.lstrip())


# @app.route("/<model_name>/encode", methods=['POST'])
# async def encode(model_name: str):
#     verify_request(model_name)

#     client_input = server_parse(data)

#     tokens = ALL_MODELS[model_name].encode(
#         client_input["prompts"], **client_input["tokenizer_kwargs"]
#     )
#     return server_send(tokens)


# @app.route("/<model_name>/call", methods=['POST'])
# async def call(model_name: str):
#     verify_request(model_name)

#     client_input = server_parse(data)

#     output = server_grad_func_call(
#         ALL_MODELS[model_name].__call__,
#         client_input["use_grad"],
#         client_input["probe_dict"],
#         *client_input["args"],
#         **client_input["kwargs"],
#     )
#     return server_send(output)



if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True)
