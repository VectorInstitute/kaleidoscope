"""Module for llama LLM configurations"""
import cloudpickle
import codecs
from collections import defaultdict
import json
import logging
import numpy as np
import pathlib
import pickle
import queue
import requests
import socket
import sys
import threading
import time
import torch
from typing import Dict, Callable

from ..abstract_model import AbstractModel
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor

# Need to add the models/llama directory to Python system path
cwd = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(cwd)

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
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
logger = logging.getLogger("kaleidoscope.model_service.llama")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):
    """Class to represent llama ML model"""

    def __init__(self, model_type, model_variant):
        self.model_type = model_type
        self.model_variant = model_variant
        self.load_default_args("config.json")


    def load(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = self.model_class.from_pretrained(model_path)
        self.model_path = model_path
        self.worker_main()
        #self.model.to(self.device)


    def load_default_args(self, config_file):
        """Load model config"""
        try:
            with json.loads(config_file) as config:
                default_args = config["parameters"]
            logger.info(default_args)
            self.default_args = {k: v["default"] for k, v in default_args.items() if v}
        except Exception as err:
            logger.error(f"Failed to load model default configuration: {err}")


    def bind(self, triton):
        triton.bind(
            model_name=f"{self.model_type}-{self.model_variant}",
            infer_func=self.infer,
            inputs=[
                Tensor(name="task", dtype=bytes, shape=(1,)),
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
                #Tensor(name="activations", dtype=np.bytes_, shape=(-1,)),
                Tensor(name="sequences", dtype=object, shape=(-1,)),
                #Tensor(name="tokens", dtype=object, shape=(-1,)),
                #Tensor(name="logprobs", dtype=object, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        """
        triton.bind(
            model_name=f"{self.model_type}-{self.model_variant}_activations",
            infer_func=self.get_activations,
            inputs=[
                Tensor(name="prompts", dtype=bytes, shape=(1,)),
                Tensor(name='max_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='min_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='temperature', dtype=np.float64, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='top_k', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='repetition_penalty', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='encoded_activation_payload', dtype=bytes, shape=(1,), optional=True),
                Tensor(name='echo', dtype=np.bool_, shape=(1,), optional=True),
                Tensor(name='module_names', dtype=bytes,shape=(1,)),
            ],
            outputs=[
                Tensor(name="activations", dtype=np.float64, shape=(-1,)),
                Tensor(name="sequences", dtype=object, shape=(-1,)),
                Tensor(name="tokens", dtype=object, shape=(-1,)),
                Tensor(name="logprobs", dtype=np.float64, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        """
        return triton


    @property
    def rank(self):
        return torch.distributed.get_rank()


    @batch
    def infer(self, **inputs):
        """Generate sequences from a prompt"""
        response = self.generate(inputs)
        logger.info(f"Infer function returning response: {response}")
        return response


    """
    @batch
    def get_activations(self, **inputs):
        activation_payload = ActivationPayload(
            module_names_activation_retrieval = inputs["module_names"][0][0],
        )
        inputs["encoded_activation_payload"][:] = activation_payload
        inputs["echo"][:] = True
        inputs["max_tokens"][:] = 0
        response = self.generate(inputs)
        return response
    """
    def get_activations(self, *args, **kwargs):
        raise NotImplementedError


    def generate(self, request):
        prompts = [
            p[0].decode("utf-8") for p in request["prompts"]
        ]
        # TODO: Read in default parameter values from config
        max_gen_len = 64
        if "max_tokens" in request:
            max_gen_len = int(request["max_tokens"].max())

        logger.info(f"Rank{torch.distributed.get_rank()}: completions")

        # Recv request and enqueue
        request_object = RequestObject(
            prompts=prompts,
            max_gen_len=max_gen_len,
            #temperature=request["temperature"],
            #top_p=request["top_p"],
            #encoded_activation_payload=request["encoded_activation_payload"]
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
        activations = []
        generated_sequences = []
        tokens = []
        logprobs = []
        logger.info(f"{results}")
        for result in results["choices"][0]["text"]:
            #activations.append(result["activations"])
            #activations.append(torch.empty(0))
            generated_sequences.append(result)
            #tokens.append(torch.empty(0))
            #logprobs.append(torch.empty(0, dtype=bytes))

        return_val = {
            #"activations": np.array(activations, dtype=np.bytes_),
            "sequences": np.array(generated_sequences, dtype=object),
            #"tokens": np.array(tokens, dtype=object),
            #"logprobs": np.array(logprobs, dtype=object)
        }
        logger.info(f"Generate returning return_val: {return_val}")
        return return_val

    def edit_activations(self, request):
        # Extract modules + editing functions from encoded request
        decoded_modules = decode_str(request.json['modules'])
        editing_fns: Dict[str, Callable] = {}
        for module_name, edit_fn in decoded_modules.items():
            if edit_fn is not None:
                logger.info(f"Adding module name {module_name} with edit function {edit_fn}")
                editing_fns[module_name] = edit_fn

        # Define activation payload
        activation_payload = ActivationPayload(
            module_names_activation_retrieval=list(decoded_modules.keys()),
            module_editing_fn_pairs=editing_fns,
        )

        request.json["encoded_activation_payload"] = encode_obj(activation_payload)
        request.json["echo"] = True
        request.json["max_tokens"] = 0

        response = self.generate(request)

        return response

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
            tokenizer_path=f"{self.model_path}/../tokenizer.model",
        )
        print(GENERATOR.model)

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
                            _ = GENERATOR.generate(
                                request_object.prompts,
                                request_object.max_gen_len,
                                request_object.temperature,
                                request_object.top_p,
                            )
                    else:
                        _ = GENERATOR.generate(
                            request_object.prompts,
                            request_object.max_gen_len,
                            request_object.temperature,
                            request_object.top_p,
                        )

                    logger.info(f"Rank{torch.distributed.get_rank()}: Batching "
                                f"loop - generating on args {request_object}")
                    _ = GENERATOR.generate(
                        request_object.prompts,
                        request_object.max_gen_len,
                        request_object.temperature,
                        request_object.top_p,
                    )
                except Exception:
                    pass


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
                    generation = generator.generate(
                        request_object.prompts,
                        request_object.max_gen_len,
                        request_object.temperature,
                        request_object.top_p,
                    )

            else:
                start_time = time.time()
                generation = generator.generate(
                    request_object.prompts,
                    request_object.max_gen_len,
                    request_object.temperature,
                    request_object.top_p,
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
                activations=ret_dict,
            )

            RESPONSE_QUEUE.put(ret_obj)
            logger.info(f"Rank{torch.distributed.get_rank()}: Batching loop - "
                        f"send response")
