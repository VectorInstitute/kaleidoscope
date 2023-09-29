"""Module for llama2 LLM configurations"""
import cloudpickle
import codecs
from collections import defaultdict
import json
import logging
import numpy as np
import pathlib
import pickle
import pprint
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
REQUEST_QUEUE = None
RESPONSE_QUEUE = None
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
        self.worker_main()


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
            config=ModelConfig(max_batch_size=128),
        )

        return triton


    @property
    def rank(self):
        return torch.distributed.get_rank()


    @batch
    @group_by_values("task")
    def infer(self, **inputs):
        """Dispatch request to a handler function based on the task"""
        self.load_default_args("generate")

        task = Task(inputs['task'][0][0])
        if task == Task.GET_ACTIVATIONS:
            response = self.get_activations(inputs)
        elif task == Task.EDIT_ACTIVATIONS:
            response = self.edit_activations(inputs)
        else:
            response = self.generate(inputs)

        return response


    def get_activations(self, inputs):
        """Retrieve activations for a list of prompts and list of module names"""
        self.load_default_args("activations")
        reponse = {}

        # If the modules are base-64 encoded, this is a manipulation request
        try:
            module_names = np.char.decode(inputs["modules"][0][0], encoding="utf-8")
            inputs["encoded_activation_payload"] = ActivationPayload(
                module_names_activation_retrieval=[module_names.tolist()],
            )
            response = self.generate(inputs)

        # Handle all other errors
        except Exception as err:
            response["activations"] = torch.empty(0)
            response["error"] = f"Error with activations request: {err}"

        return response


    def edit_activations(self, inputs):
        """Edit activations for a list of prompts and list of modules"""
        self.load_default_args("activations")

        # If the modules are base-64 encoded, this is a manipulation request
        try:
            # Extract modules + editing functions from encoded request
            encoded_modules = np.char.decode(inputs["modules"][0][0], encoding="utf-8")
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
            response["activations"] = torch.empty(0)
            response["error"] = f"Error with activations request: {err}"

        return response


    def generate(self, request):
        """Generate sequences from a prompt"""
        logger.info(f"Generate function called with request: {request}")
        logger.info(f"Self generation args: {self.generation_args}")
        global GENERATOR

        prompts = [
            p[0].decode("utf-8") for p in request["prompts"]
        ]

        logger.info(f"Rank{torch.distributed.get_rank()}: completions")

        # Tokenize the prompts (needs to be done manually now)
        prompt_tokens = [GENERATOR.tokenizer.encode(s=prompt, bos=True, eos=False) for prompt in prompts]
        
        # Recv request and enqueue
        request_object = RequestObject(
            prompts=prompt_tokens,
            max_gen_len=int(request["max_tokens"]) if "max_tokens" in request else int(self.generation_args["max_tokens"]),
            temperature=float(request["temperature"]) if "temperature" in request else float(self.generation_args["temperature"]),
            top_p=float(request["top_p"]) if "top_p" in request else float(self.generation_args["top_p"]),
            encoded_activation_payload=request["encoded_activation_payload"] if "encoded_activation_payload" in request else None
        )

        logger.info(f"Rank{torch.distributed.get_rank()}: completions - made "
                    f"RequestObject: {request_object}")

        REQUEST_QUEUE.put(request_object)
        logger.info(f"Rank{torch.distributed.get_rank()}: completions - "
                    f"RequestObject enqueued")

        # Recv response and parse
        response_object = RESPONSE_QUEUE.get()

        logger.info(f"Rank{torch.distributed.get_rank()}: completions - response "
                    f"recv")

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
        return return_val


    def worker_main(self):
        """
        Hosted version of the web UI for generation.
        """
        global REQUEST_QUEUE
        global RESPONSE_QUEUE
        global GENERATOR

        rank, world_size = setup_model_parallel()

        load_fn = load_llama

        start_time = time.time()
        GENERATOR = load_fn(
            local_rank=rank,
            world_size=world_size,
            max_seq_len=512,
            max_batch_size=32,
            ckpt_dir=f"{self.model_path}",
            tokenizer_path=f"{self.model_path}/tokenizer.model",
        )

        logger.info(f"Rank {torch.distributed.get_rank()} loaded in "
                    f"{time.time() - start_time:.2f} seconds")

        if torch.distributed.is_initialized():
            request_object = distributed_utils.broadcast_object(
                None, src_rank=0, group=distributed_utils.get_global_group(),
            )
        else:
            raise Exception("Please initialize torch distributed.")

        # Rank0 launches server on new thread
        if torch.distributed.get_rank() == 0:
            REQUEST_QUEUE = queue.Queue()
            RESPONSE_QUEUE = queue.Queue()
            logger.info(f"Worker engaged! {get_my_ip()}:{PORT}")
            thread = threading.Thread(
                target=self.batching_loop, args=(GENERATOR,), daemon=True,
            )
            thread.start()
            #app.run(host="0.0.0.0", port=PORT, threaded=True)

        # Other ranks continuously wait for work
        else:
            logger.info(
                f"Rank{torch.distributed.get_rank()} Looping engaged! "
                f"{get_my_ip()}:{PORT}"
            )
            while True:
                try:
                    request_object = distributed_utils.broadcast_object(
                        None,
                        src_rank=0,
                        group=distributed_utils.get_global_group(),
                    )


                    encoded_activation_payload = request_object.encoded_activation_payload
                    act_retrieval_aux = request_object._aux
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

                    logger.info(f"Rank{torch.distributed.get_rank()}: Batching "
                                f"loop - generating on args {request_object}")
                    _, _ = GENERATOR.generate(
                        request_object.prompts,
                        request_object.max_gen_len,
                        request_object.temperature,
                        request_object.top_p,
                        logprobs=True
                    )
                except Exception as err:
                    logger.info(f"Worker main caught exception: {err}")


    def batching_loop(self, generator):
        """
        Until forever, execute generations once we reach the max threshold of
        request objects. This runs only on the head node rank0. LLaMA works on
        batched prompt inputs, so we just implement batching naiively here.
        """
        logger.info(f"Rank{torch.distributed.get_rank()}: Batching loop")
        while True:
            request_object = REQUEST_QUEUE.get()
            logger.info(f"Rank{torch.distributed.get_rank()}: Batching loop - "
                        f"got RequestObject")

            # aux data needed for act retrieval
            # TODO: Surely a better way to impl this?
            request_object._aux = (len(request_object.prompts),)


            distributed_utils.broadcast_object(
                request_object,
                src_rank=0,
                group=distributed_utils.get_global_group(),
            )
            logger.info(f"Rank{torch.distributed.get_rank()}: Batching loop - "
                        f"broadcasted RequestObject")

            logger.info(f"Rank{torch.distributed.get_rank()}: Batching "
                        f"loop - generating on args {request_object}")

            activation_dict = {}
            encoded_activation_payload = request_object.encoded_activation_payload
            act_retrieval_aux = request_object._aux

            if encoded_activation_payload is not None:
                hook_dict, activation_dict = get_activation_capture_hook_dict(
                    generator.model,
                    encoded_activation_payload,
                    aux=act_retrieval_aux,
                )
                start_time = time.time()
                with apply_forward_hook(generator.model, hook_dict):
                    generation, logprobs = generator.generate(
                        request_object.prompts,
                        request_object.max_gen_len,
                        request_object.temperature,
                        request_object.top_p,
                        logprobs=True
                    )

            else:
                start_time = time.time()
                generation, logprobs = generator.generate(
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

            ret_obj = ResponseObject(
                generations=generation,
                logprobs=logprobs,
                activations=ret_dict,
            )

            RESPONSE_QUEUE.put(ret_obj)
            logger.info(f"Rank{torch.distributed.get_rank()}: Batching loop - "
                        f"send response")
