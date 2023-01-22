from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Dict
from urllib.parse import urljoin
from collections import namedtuple

import torch

from .hooks import TestForwardHook
from .utils import get, post


@dataclass
class RModel:

    host: str
    port: int
    model_name: str
    auth_key: str

    probe_dict: Dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.base_addr = f"http://{self.host}:{self.port}/"
        self.create_addr = partial(urljoin, self.base_addr)
        # TODO: can cache this
        all_model_names = get(self.create_addr("models"))
        model_instances = get(self.create_addr("models/instances"))
        active_model_instances = [
            models for models in model_instances if model_instances[models] == "Active"
        ]
        print(
            f"Available models: {all_model_names} \nActive models: {active_model_instances}"
        )
        if self.model_name not in all_model_names:
            raise ValueError(
                "asked for model {} but server only supports model "
                "names {}".format(self.model_name, all_model_names)
            )

        self.model_base_addr = (
            f"http://{self.host}:{self.port}/models/{self.model_name}/"
        )
        self.model_create_addr = partial(urljoin, self.model_base_addr)

    def generate_text(self, prompt, /, **gen_kwargs):
        """TODO: should support batching"""
        model_generate_addr = urljoin(self.model_base_addr, "generate_text")
        generate_configs = {}
        generate_configs["prompt"] = prompt
        generate_configs.update(gen_kwargs)
        generate_configs["use_grad"] = torch.is_grad_enabled()

        parameters = gen_kwargs.keys()

        generate_configs["max-tokens"] = (
            generate_configs.pop("max_tokens") if "max_tokens" in parameters else None
        )
        generate_configs["top-k"] = (
            generate_configs.pop("top_k") if "top_k" in parameters else None
        )
        generate_configs["top-p"] = (
            generate_configs.pop("top_p") if "top_p" in parameters else None
        )
        generate_configs["num_return_sequences"] = (
            generate_configs.pop("num_sequences")
            if "num_sequences" in parameters
            else None
        )
        generate_configs["repetition_penalty"] = (
            generate_configs.pop("rep_penalty") if "rep_penalty" in parameters else None
        )

        print(f"Submission: {generate_configs}")
        generation = post(model_generate_addr, generate_configs, self.auth_key)
        GenerationObj = namedtuple("GenObj", generation.keys())
        results = GenerationObj(**generation)
        print(f"Success:\n{prompt} {results.text}")
        return results

    @cached_property
    def module_names(self):
        return get(self.model_create_addr("module_names"))

    @cached_property
    def parameter_names(self):
        return get(self.model_create_addr("parameter_names"))

    @cached_property
    def probe_points(self):
        return get(self.model_create_addr("probe_points"))

    def get_models(self):
        model_instance_names = get(self.create_addr("models/instances"))
        return model_instance_names

    def get_parameters(self, *names):
        return post(self.model_create_addr("get_parameters"), names)

    def encode(self, prompts, /, return_tensors="pt", **tokenizer_kwargs):
        tokenizer_kwargs.setdefault("return_tensors", return_tensors)
        tokenizer_kwargs.setdefault("padding", True)

        return post(
            self.model_create_addr("encode"),
            {"prompts": prompts, "tokenizer_kwargs": tokenizer_kwargs,},
        )

    def __call__(self, *args, **kwargs):
        model_output, new_probe_dict = post(
            self.model_create_addr("call"),
            {
                "probe_dict": self.probe_dict,
                "args": args,
                "kwargs": kwargs,
                "use_grad": torch.is_grad_enabled(),
            },
        )

        return model_output, new_probe_dict
