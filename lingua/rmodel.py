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

    probe_dict: Dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.base_addr = f"http://{self.host}:{self.port}/"
        self.create_addr = partial(urljoin, self.base_addr)
        # TODO: can cache this
        all_model_names = get(self.create_addr("models"))
        model_instance_names = get(self.create_addr("models/instances"))
        print(f"Available models: {all_model_names} \nActive models: {model_instance_names}")
        if self.model_name not in all_model_names:
            raise ValueError(
                "asked for model {} but server only supports model "
                "names {}".format(self.model_name, all_model_names)
            )

        self.model_base_addr = f"http://{self.host}:{self.port}/models/{self.model_name}/"
        self.model_create_addr = partial(urljoin, self.model_base_addr)

    def get_models(self):
        model_instance_names = get(self.create_addr("models/instances"))
        return model_instance_names

    def generate_text(self, prompt, /, **gen_kwargs):
        """TODO: should support batching
        """
        model_generate_addr = urljoin(self.model_base_addr, "generate_text")
        generate_configs = {}
        generate_configs['prompt']= prompt
        generate_configs.update(gen_kwargs)
        generate_configs['use_grad'] = torch.is_grad_enabled()
        print(f"Submission: {generate_configs}")
        generation = post(model_generate_addr, generate_configs)
        GenerationObj = namedtuple('GenObj', generation.keys())
        results = GenerationObj(**generation)
        print(f"Success:\n{results.text}")
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

