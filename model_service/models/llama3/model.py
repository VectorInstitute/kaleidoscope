"""Module for llama3 LLM configurations"""
import cloudpickle
import codecs
from collections import defaultdict
import json
import logging
import numpy as np
import pathlib
import queue
import requests
import socket
import sys
import threading
import time
import torch
from typing import Dict, Callable

from llama import Llama

from ..abstract_model import AbstractModel, Task
from pytriton.decorators import batch, group_by_values
from pytriton.model_config import ModelConfig, Tensor

logger = logging.getLogger("kaleidoscope.model_service.llama3")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

model = None


class Model(AbstractModel):
    """Class to represent llama ML model"""

    def __init__(self, model_type, model_variant):
        self.model_type = model_type
        self.model_variant = model_variant
        cwd = str(pathlib.Path(__file__).parent.resolve())
        self.config_path = f"{cwd}/config.json"
        self.generation_args = {}


    def load(self, model_path):
        global model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        print(f"Loading model at path: {model_path}")
        model = Llama.build(
            ckpt_dir=self.model_path,
            tokenizer_path=f"{self.model_path}/tokenizer.model",
            max_seq_len=512,
            max_batch_size=64
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
                Tensor(name='max_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='temperature', dtype=np.float64, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='echo', dtype=np.bool_, shape=(1,), optional=True)
            ],
            outputs=[
                Tensor(name="sequences", dtype=bytes, shape=(-1,)),
                Tensor(name="tokens", dtype=bytes, shape=(-1,)),
                Tensor(name="logprobs", dtype=np.float64, shape=(-1,))
            ],
            config=ModelConfig(max_batch_size=64),
        )

        return triton


    @property
    def rank(self):
        return 0


    @batch
    def infer(self, **inputs):
        """Dispatch request to a handler function based on the task"""
        self.load_default_args("generate")
        response = self.generate(inputs)
        logger.info(f"Infer function returning response: {response}")
        return response


    def generate(self, request):
        """Generate sequences from a prompt"""
        logger.info(f"Generate function called with request: {request}")
        global model
        global tokenizer

        prompts = [
            p[0].decode("utf-8") for p in request["prompts"]
        ]
        max_gen_len = int(request["max_tokens"][0][0]) if "max_tokens" in request else int(self.generation_args["max_tokens"])
        temperature = float(request["temperature"][0][0]) if "temperature" in request else float(self.generation_args["temperature"])
        top_p = float(request["top_p"][0][0]) if "top_p" in request else float(self.generation_args["top_p"])
        echo = bool(request["echo"][0][0]) if "echo" in request else float(self.generation_args["echo"])

        try:
            results = model.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,
                echo=echo
            )
        except Exception as err:
            logger.info(f"ERROR: Generation request failed: {err}")
       
        # Now compile the results into data structures we'll return
        generated_sequences = [result['generation'].encode("utf-8") for result in results]
        tokens = [result['tokens'] for result in results]
        for index, value in enumerate(tokens):
            tokens[index] = [this_token.encode('utf-8') for this_token in value]
        logprobs = [result['logprobs'] for result in results]

        return_val = {
            "sequences": np.array(generated_sequences, dtype=bytes),
            "tokens": np.array(tokens, dtype=bytes),
            "logprobs": np.array(logprobs, dtype=np.float64)
        }
        return return_val
