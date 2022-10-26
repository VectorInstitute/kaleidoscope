from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Dict
from urllib.parse import urljoin
import argparse
import json

import torch

from hooks import TestForwardHook
from utils import get, post


@dataclass
class RModel:
    """rmodels should just take in model_names from server
    and then call things based on model name, but the interface for all these generation models should be similar
    """

    host: str
    port: int
    # must be supported by the server
    model_name: str

    probe_dict: Dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.base_addr = f"http://{self.host}:{self.port}/"
        self.create_addr = partial(urljoin, self.base_addr)

        # TODO: can cache this
        # all_model_names = get(self.create_addr("all_models"))

        # if self.model_name not in all_model_names:
        #     raise ValueError(
        #         "asked for model {} but server only supports model "
        #         "names {}".format(self.model_name, all_model_names)
        #     )

        # model based stuff should use this
        self.model_base_addr = f"http://{self.host}:{self.port}/{self.model_name}"
        self.model_create_addr = partial(urljoin, self.model_base_addr)

    def generate_text(self, prompts, /, **gen_kwargs):
        """TODO: should have some interface so the server knows how to parse this
        should support batching
        """
        print(self.model_base_addr)
        return post(
            self.model_base_addr,
            {
                "prompt": prompts,
                "gen_kwargs": gen_kwargs,
                "use_grad": torch.is_grad_enabled(),
            },
        )

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
            {
                "prompts": prompts,
                "tokenizer_kwargs": tokenizer_kwargs,
            },
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


if __name__ == "__main__":
    # Sample text generation using default hyperparameters
    parser = argparse.ArgumentParser(
        description="remote model test with live mini opt server"
    )
    parser.add_argument("--host", action="store", required=True)
    parser.add_argument("--port", action="store", required=True)
    parser.add_argument("--model_name", action="store")
    args = parser.parse_args()
    host = str(args.host)
    port = int(args.port)
    model_name = str(args.model_name)
    rmodel = RModel(host, port, model_name)
    text_input = str(input("\nEnter text for generation: "))
    print("Generating...")
    response = rmodel.generate_text(text_input)
    try:
        print(response.json()["choices"][0]["text"])
    except:
        print("Accomodating for different output response structure: ")
        print(response.text)

    # rmodel = RModel("localhost", 8000, "GPT2XL")
    # rmodel.probe_dict.update({"transformer.wte.post_activation": [TestForwardHook()]})
    # x = ["what is the meaning of wife", "hello welcome"]

    # encoding = rmodel.encode(x)

    # # mimics it onto the host
    # with torch.no_grad():
    #     output = rmodel(**encoding)
    # breakpoint()

    # out = rmodel.generate_text(
    # ["what is the meaning of life?"],
    # do_sample=True,
    # temperature=0.9,
    # max_length=300,
    # )
    # out = rmodel.get_parameters(
    # "transformer.h.27.mlp.fc_in.weight", "transformer.h.20.attn.k_proj.weight"
    # )
    # tokens = rmodel.encode("hello", padding=True)

    # with torch.no_grad() should work
    # probe_points = rmodel.probe_points
