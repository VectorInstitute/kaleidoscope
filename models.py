from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, List
import requests
import re

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

ALL_MODELS = {}

HOOK_DICT = {
    "pre_activation": nn.Module.register_forward_pre_hook,
    "post_activation": nn.Module.register_forward_hook,
    "backward": nn.Module.register_full_backward_hook,
}


def get_module(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m

    raise LookupError(name)


class ProbeContext(AbstractContextManager):
    def __init__(self, model, probe_dict: Dict[str, List[Any]]):

        self.probe_handles = []

        for probe_name, probes in probe_dict.items():

            module_name, hook_type = probe_name.rsplit(".", 1)

            mod = get_module(model, module_name)

            # install the probes at the mod
            # for the given hook type
            for p in probes:
                hook = HOOK_DICT[hook_type](mod, p)
                self.probe_handles.append(hook)

    def __exit__(self, *args):
        for handle in self.probe_handles:
            handle.remove()


class _ServerModel:
    """a class registery that also defines the interface the models should implement
    NOTE and TODO: so far I'm thinking this should only implement "interfaces" i.e no implementation inheritance, just needs to implement specified interfaces to keep things consistent, maybe Protocols will be handy
    """

    # this is just a class registery
    def __init_subclass__(cls, /, model_name=None):
        super().__init_subclass__()

        ALL_MODELS[model_name or cls.__name__] = cls


@dataclass
class OPT(_ServerModel):

    device_for_input: str

    hf_model_name: str = r"facebook/opt-125m"

    tokenizer: PreTrainedTokenizerBase = field(
        default=None, init=False, repr=False, compare=False
    )

    model: PreTrainedModel = field(
        default=None, init=False, repr=False, compare=False
    )

    url: str = "http://172.17.8.51:8000"

    def __post_init__(self):
        self.lazy_init()

    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def setup_model(self, torch_dtype=None):
        return AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch_dtype,
            revision="float16",
            low_cpu_mem_usage=True,
            # TODO: this device occupy is too greedy
            device_map="auto",
        )

    def free(self):
        self.tokenizer = None
        self.model = None

    def lazy_init(self):
        if self.tokenizer is None:
            self.tokenizer = self.setup_tokenizer()
        if self.model is None:
            self.model = self.setup_model()

    def generate_text(self, prompts, /, **gen_kwargs):
        max_tokens= gen_kwargs['max-tokens']
        top_p= gen_kwargs['top-p']
        del gen_kwargs['max-tokens']
        del gen_kwargs['top-p']
        result = requests.post(
            OPT.url + "/generate_text", json={"prompt": prompts, "max_tokens": max_tokens, "top_p": top_p, **gen_kwargs}
        )
        result_output= result.json()
        print(f"{result_output}")
        response = {}
        response['text'] = result_output['choices'][0]['text']
        response['text_tokens'] = result_output['choices'][0]['all_tokens_text']
        response['tokens'] = result_output['choices'][0]['logprobs']['tokens']
        response['logprobs'] = result_output['choices'][0]['logprobs']['token_logprobs']
        response['activations'] = result_output['choices'][0]['activations']
        return response

    def generate(self, encoding, probes=None, /, **gen_kwargs):
        """encoding must be the batched encodings"""
        encoding = encoding.to(self.device_for_input)
        generated_ids = self.model.generate(**encoding, **gen_kwargs)
        return generated_ids

    def encode(self, prompts, /, **tokenizer_kwargs):
        """tokenizes the prompts"""
        return self.tokenizer(prompts, **tokenizer_kwargs)

    def get_module_names(self):
        response = requests.get(
            OPT.url + "/module_names")
        return response.json()

    @property
    def parameter_names(self):
        return tuple(n for n, _ in self.model.named_parameters())

    def get_parameters(self, *names):
        return {
            n: p.cpu()
            for n, p in self.model.named_parameters()
            if n in set(names)
        }

    @property
    def probe_points(self):
        return tuple(
            probe
            for module_name in self.module_names
            for probe in (
                f"{module_name}.pre_activation",
                f"{module_name}.post_activation",
                f"{module_name}.backward",
            )
        )

    def __call__(self, probe_dict, /, *args, **kwargs):
        with ProbeContext(self.model, probe_dict) as p:
            return self.model(*args, **kwargs), probe_dict


@dataclass
class GPT2(_ServerModel):

    device_for_input: str

    hf_model_name: str = r"huggingface/gpt2"

    tokenizer: PreTrainedTokenizerBase = field(
        default=None, init=False, repr=False, compare=False
    )

    model: PreTrainedModel = field(
        default=None, init=False, repr=False, compare=False
    )

    url: str = "http://172.17.8.50:8000"

    def __post_init__(self):
        self.lazy_init()

    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def setup_model(self, torch_dtype=None):
        return AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch_dtype,
            revision="float16",
            low_cpu_mem_usage=True,
            # TODO: this device occupy is too greedy
            device_map="auto",
        )

    def free(self):
        self.tokenizer = None
        self.model = None

    def lazy_init(self):
        if self.tokenizer is None:
            self.tokenizer = self.setup_tokenizer()
        if self.model is None:
            self.model = self.setup_model()

    def generate_text(self, prompts, /, **gen_kwargs):
        max_tokens= gen_kwargs['max-tokens']
        top_p= gen_kwargs['top-p']
        top_k= gen_kwargs['top-k']
        del gen_kwargs['max-tokens']
        del gen_kwargs['top-p']
        del gen_kwargs['top-k']
        result = requests.post(
            GPT2.url + "/generate_text", json={"prompt": prompts, "length": max_tokens, "p": top_p, "k": top_k, **gen_kwargs}
        )
        response= result.json()
        tokenized_text = response['text']
        response['tokens'] = re.split("(\s+)", tokenized_text)

        return response

    def generate(self, encoding, probes=None, /, **gen_kwargs):
        """encoding must be the batched encodings"""
        encoding = encoding.to(self.device_for_input)
        generated_ids = self.model.generate(**encoding, **gen_kwargs)
        return generated_ids

    def encode(self, prompts, /, **tokenizer_kwargs):
        """tokenizes the prompts"""
        return self.tokenizer(prompts, **tokenizer_kwargs)

    def get_module_names(self):
        response = requests.get(
            GPT2.url + "/module_names")
        return response.json()

    @property
    def parameter_names(self):
        return tuple(n for n, _ in self.model.named_parameters())

    def get_parameters(self, *names):
        return {
            n: p.cpu()
            for n, p in self.model.named_parameters()
            if n in set(names)
        }

    @property
    def probe_points(self):
        return tuple(
            probe
            for module_name in self.module_names
            for probe in (
                f"{module_name}.pre_activation",
                f"{module_name}.post_activation",
                f"{module_name}.backward",
            )
        )

    def __call__(self, probe_dict, /, *args, **kwargs):
        with ProbeContext(self.model, probe_dict) as p:
            return self.model(*args, **kwargs), probe_dict
