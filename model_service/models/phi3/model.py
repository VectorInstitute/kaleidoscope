"""Module for phi3 LLM configurations"""
import json
import logging
import numpy as np
import pathlib
import requests
import socket
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..abstract_model import AbstractModel, Task
from pytriton.decorators import batch, group_by_values
from pytriton.model_config import ModelConfig, Tensor


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
model = None
tokenizer = None

logger = logging.getLogger("kaleidoscope.model_service.phi3")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):
    """Class to represent phi3 ML model"""

    def __init__(self, model_type, model_variant):
        self.model_type = model_type
        self.model_variant = model_variant
        cwd = str(pathlib.Path(__file__).parent.resolve())
        self.config_path = f"{cwd}/config.json"
        self.generation_args = {}


    def load(self, model_path):
        global model
        global tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        print(f"Loading model at path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Finished loading model")


    def load_default_args(self, task_name):
        """Load model config"""
        logger.info(f"Loading default args from self.config_path: {self.config_path}")
        self.generation_args = {}
        try:
            with open(self.config_path) as file:
                json_data = file.read()
            default_args = json.loads(json_data)["parameters"]
            logger.info(f"Default args: {default_args}")
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
                Tensor(name='temperature', dtype=np.float64, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=np.float32, shape=(1,), optional=True),
            ],
            outputs=[
                Tensor(name="sequences", dtype=object, shape=(-1,))
                #Tensor(name="tokens", dtype=object, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )

        return triton


    @property
    def rank(self):
        return 0


    @batch
    @group_by_values("task")
    def infer(self, **inputs):
        """Dispatch request to a handler function based on the task"""
        self.load_default_args("generate")
        response = self.generate(inputs)
        return response


    def generate(self, request):
        """Generate sequences from a prompt"""
        logger.info(f"Generate function called with request: {request}")
        global model
        global tokenizer

        prompts = [prompt[0].decode() for prompt in request['prompts']]
        tokenized_prompts = tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        generation_args = {}
        generation_args['max_new_tokens'] = int(request["max_tokens"][0][0]) if "max_tokens" in request else int(self.generation_args["max_tokens"])
        generation_args['temperature'] = float(request["temperature"][0][0]) if "temperature" in request else float(self.generation_args["temperature"])
        generation_args['top_p'] = float(request["top_p"][0][0]) if "top_p" in request else float(self.generation_args["top_p"])
        generation_args['eos_token_id'] = tokenizer.eos_token_id
        generation_args['output_logits'] = True
        generation_args['output_scores'] = True

        # TODO: Should we include the attention masks here? Generate complains when we don't send them, but it fails silently when we do
        try:
            logger.info(f"Sending generation request with tokenized_prompts: {tokenized_prompts}")
            outputs = model.generate(
                tokenized_prompts['input_ids'],
                do_sample=True,
                **generation_args
            )
        except Exception as err:
            logger.info(f"Generation request failed: {err}")

        logger.info(f"Generation request returned outputs: {outputs}")

        try:
            decoded_sequences = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in outputs]
        except Exception as err:
            logger.info(f"Decoding outputs failed: {err}")

        return_val = {
            "sequences": np.array(decoded_sequences, dtype=str)
        }
        return return_val
