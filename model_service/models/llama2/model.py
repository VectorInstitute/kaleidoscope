"""Module for llama2 LLM configurations"""
import cloudpickle
import codecs
from collections import defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import os
import pathlib
import pickle
import pprint
import psutil
import queue
import requests
import socket
import sys
import threading
import time
import torch
from typing import Dict, Callable

from ..abstract_model import AbstractModel, Task
from pytriton.decorators import batch, group_by_values
from pytriton.model_config import ModelConfig, Tensor

# Need to add the models/llama directory to Python system path
cwd = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(cwd)

from llama import ModelArgs, Transformer, Tokenizer, Llama
import distributed_utils
from hosting_utils import (
    RequestObject,
    ResponseObject,
    build_host_logger,
    setup_model_parallel,
    load_llama,
)
from hook_utils import get_activation_capture_hook_dict, apply_forward_hook
from activation_utils import ActivationPayload


def encode_obj(obj):
    return codecs.encode(cloudpickle.dumps(obj), "base64").decode("utf-8")


def decode_str(obj_in_str):
    return pickle.loads(codecs.decode(obj_in_str.encode("utf-8"), "base64"))


def get_my_ip():
    """
    returns ip / hostname of current host
    """
    return socket.gethostbyname(socket.gethostname())


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


# global state (mutable!)
MAX_REQUESTS = None
GENERATOR = None
PORT = get_free_port()

logger = build_host_logger()
logger = logging.getLogger("kaleidoscope.model_service.llama2")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):
    """Class to represent llama ML model"""

    def __init__(self, model_type, model_variant):
        self.model_type = model_type
        self.model_variant = model_variant
        cwd = str(pathlib.Path(__file__).parent.resolve())
        self.config_path = f"{cwd}/config.json"
        self.generation_args = {}


    def load(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        global GENERATOR
        rank, world_size = setup_model_parallel()

        GENERATOR = load_llama(
            local_rank=rank,
            world_size=world_size,
            max_batch_size=1,
            max_seq_len=512,
            ckpt_dir=f"{self.model_path}",
            tokenizer_path=f"{self.model_path}/tokenizer.model",
        )

        # Non rank-0 workers continuously wait for work
        if rank != 0:
            while True:
                try:
                    request_object = distributed_utils.broadcast_object(
                        None,
                        src_rank=0,
                        group=distributed_utils.get_global_group(),
                    )

                    encoded_activation_payload = request_object.encoded_activation_payload
                    act_retrieval_aux = request_object._aux
                    logger.info(f"Rank{torch.distributed.get_rank()}: worker generating on args {request_object}")
                    if encoded_activation_payload is not None:
                        hook_dict, _ = get_activation_capture_hook_dict(
                            GENERATOR.model,
                            encoded_activation_payload,
                            aux=act_retrieval_aux,
                        )

                        with apply_forward_hook(GENERATOR.model, hook_dict):
                            _, _ = GENERATOR.generate(
                                request_object.prompts,
                                request_object.max_gen_len,
                                request_object.temperature,
                                request_object.top_p,
                                logprobs=True
                            )
                    else:
                        _, _ = GENERATOR.generate(
                            request_object.prompts,
                            request_object.max_gen_len,
                            request_object.temperature,
                            request_object.top_p,
                            logprobs=True
                        )

                except Exception as err:
                    logger.info(f"Rank{torch.distributed.get_rank()} caught exception: {err}")


    def load_default_args(self, task_name):
        """Load model config"""
        logger.info(f"Loading default args from self.config_path: {self.config_path}")
        self.generation_args = {}
        try:
            with open(self.config_path) as file:
                json_data = file.read()
            default_args = json.loads(json_data)["parameters"]
            logger.info(pprint.pformat(default_args))
            self.generation_args = {k: v["default"][task_name] for k, v in default_args.items() if v["default"][task_name] is not None}
        except Exception as err:
            logger.error(f"Failed to load model {task_name} default configuration: {err}")


    def bind(self, triton):
        triton.bind(
            model_name=f"{self.model_type}-{self.model_variant}",
            infer_func=self.infer,
            inputs=[
                Tensor(name="task", dtype=np.int64, shape=(1,)),
                Tensor(name="prompts", dtype=bytes, shape=(1,)),
                Tensor(name="modules", dtype=bytes, shape=(1,), optional=True),
                Tensor(name='max_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='min_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='temperature', dtype=np.float64, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='top_k', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='repetition_penalty', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='encoded_activation_payload', dtype=bytes, shape=(1,), optional=True),
                Tensor(name='echo', dtype=np.bool_, shape=(1,), optional=True)
            ],
            outputs=[
                Tensor(name="activations", dtype=np.bytes_, shape=(-1,)),
                Tensor(name="sequences", dtype=object, shape=(-1,)),
                Tensor(name="tokens", dtype=object, shape=(-1,)),
                Tensor(name="logprobs", dtype=object, shape=(-1,)),
            ],
            config=ModelConfig(batching=False),
        )

        return triton


    @property
    def rank(self):
        return torch.distributed.get_rank()


    @batch
    @group_by_values("task")
    def infer(self, **inputs):
        """Dispatch request to a handler function based on the task"""
        start_time = datetime.now()
        self.load_default_args("generate")

        task = Task(inputs['task'][0])
        if task == Task.GET_ACTIVATIONS:
            response = self.get_activations(inputs)
        elif task == Task.EDIT_ACTIVATIONS:
            response = self.edit_activations(inputs)
        else:
            response = self.generate(inputs)

        end_time = datetime.now()
        function_time = end_time - start_time
        print(f"PROFILER Infer function ran in {function_time}")

        return response


    def get_activations(self, inputs):
        """Retrieve activations for a list of prompts and list of module names"""
        self.load_default_args("activations")
        response = {}

        # If the modules are base-64 encoded, this is a manipulation request
        try:
            module_names = np.char.decode(inputs["modules"][0], encoding="utf-8")
            inputs["encoded_activation_payload"] = ActivationPayload(
                module_names_activation_retrieval=[module_names.tolist()],
            )
            logger.info(f"Created ActivationPayload={ActivationPayload}, module_names={module_names}, inputs={inputs}")
            response = self.generate(inputs)

        # Handle all other errors
        except Exception as err:
            response["activations"] = torch.empty(0, dtype=torch.float32)
            response["error"] = f"Error with activations request: {err}"

        return response


    def edit_activations(self, inputs):
        """Edit activations for a list of prompts and list of modules"""
        self.load_default_args("activations")

        # If the modules are base-64 encoded, this is a manipulation request
        try:
            # Extract modules + editing functions from encoded request
            encoded_modules = np.char.decode(inputs["modules"][0], encoding="utf-8")
            # TODO: This only works for a single module name. Add code to handle multiple modules.
            decoded_modules = decode_str(str(encoded_modules))
            editing_fns: Dict[str, Callable] = {}
            for module_name, edit_fn in decoded_modules.items():
                if edit_fn is not None:
                    editing_fns[module_name] = edit_fn

            # Define activation payload
            inputs["encoded_activation_payload"] = encode_obj(
                ActivationPayload(
                    module_names_activation_retrieval=list(decoded_modules.keys()),
                    module_editing_fn_pairs=editing_fns,
                )
            )
            response = self.generate(inputs)

        # Handle all other errors
        except Exception as err:
            response = {}
            response["activations"] = torch.empty(0, dtype=torch.float32)
            response["error"] = f"Error with activations request: {err}"

        return response


    def generate(self, request):
        """Generate sequences from a prompt"""
        logger.info(f"Rank{torch.distributed.get_rank()} Generate function called with request: {request}")
        global GENERATOR

        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        logger.info(f"PROFILER Start of generate function, memory usage: {mem_usage}")

        prompts = [
            p.decode("utf-8") for p in request["prompts"]
        ]

        # Tokenize the prompts (needs to be done manually now)
        prompt_tokens = [GENERATOR.tokenizer.encode(s=prompt, bos=True, eos=False) for prompt in prompts]

        # Recv request and enqueue
        request_object = RequestObject(
            prompts=prompt_tokens,
            max_gen_len=int(request["max_tokens"][0]) if "max_tokens" in request else int(self.generation_args["max_tokens"]),
            temperature=float(request["temperature"][0]) if "temperature" in request else float(self.generation_args["temperature"]),
            top_p=float(request["top_p"][0]) if "top_p" in request else float(self.generation_args["top_p"]),
            encoded_activation_payload=request["encoded_activation_payload"] if "encoded_activation_payload" in request else None
        )
        request_object._aux = (len(request_object.prompts),)

        logger.info(f"Rank{torch.distributed.get_rank()}: generation RequestObject: {request_object}")

        if torch.distributed.get_rank() == 0:
            distributed_utils.broadcast_object(
                request_object,
                src_rank=0,
                group=distributed_utils.get_global_group(),
            )

        activation_dict = {}
        encoded_activation_payload = request_object.encoded_activation_payload
        act_retrieval_aux = request_object._aux

        start_time = time.time()
        logger.info(f"Checking for activations...")
        if encoded_activation_payload is not None:
            logger.info(f"Activations is not None! GENERATOR.model={GENERATOR.model}, encoded_activation_payload={encoded_activation_payload}, act_retrieval_aux={act_retrieval_aux}")
            activation_start_time = datetime.now()
            hook_dict, activation_dict = get_activation_capture_hook_dict(
                GENERATOR.model,
                encoded_activation_payload,
                aux=act_retrieval_aux,
            )
            logger.info(f"Calling apply_forward_hook with hook_dict={hook_dict}")
            with apply_forward_hook(GENERATOR.model, hook_dict):
                generation, logprobs = GENERATOR.generate(
                    request_object.prompts,
                    request_object.max_gen_len,
                    request_object.temperature,
                    request_object.top_p,
                    logprobs=True
                )
            activation_end_time = datetime.now()
            activation_time = end_time - start_time
            print(f"PROFILER Activation retrieval ran in {activation_time}")
        else:
            generation, logprobs = GENERATOR.generate(
                request_object.prompts,
                request_object.max_gen_len,
                request_object.temperature,
                request_object.top_p,
                logprobs=True
            )

        logger.info(f"Rank{torch.distributed.get_rank()}: Generation took "
                    f"{time.time() - start_time} seconds")

        ret_dict = {}
        for k, v in activation_dict.items():
            logger.info(f"Rank{torch.distributed.get_rank()}: Module "
                        f"{k} activation shape: {v.shape}")
            ret_dict[k] = codecs.encode(
                pickle.dumps(v.clone()),
                "base64",
            ).decode("utf-8")

        del activation_dict

        response_object = ResponseObject(
            generations=generation,
            logprobs=logprobs,
            activations=ret_dict,
        )

        logger.info(f"Rank{torch.distributed.get_rank()}: generation ResponseObject: {response_object}")

        results = response_object.json()

        # Compile the results into a structure consistent with other kaleidoscope models
        activations = results["choices"][0]["activations"]
        generated_sequences = GENERATOR.tokenizer.decode(results["choices"][0]["text"])
        tokens = []
        for sequence in results["choices"][0]["text"]:
            tokens.append([GENERATOR.tokenizer.decode(token) for token in sequence])
        logprobs = results["choices"][0]["logprobs"]

        return_val = {
            "activations": np.array(activations, dtype=np.bytes_),
            "sequences": np.array(generated_sequences, dtype=object),
            "tokens": np.array(tokens, dtype=object),
            "logprobs": np.array(logprobs, dtype=object)
        }

        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        logger.info(f"PROFILER End of generate function, memory usage: {mem_usage}")

        return return_val
