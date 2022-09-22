import codecs
import random
import itertools
import pickle
from dataclasses import dataclass
from functools import cached_property

import requests


def check_response(resp):
    assert (
        resp.status_code == 200
    ), f"error in request with code {resp.status_code} resp {resp.json()}"


def decode_str(obj_in_str):
    return pickle.loads(codecs.decode(obj_in_str.encode("utf-8"), "base64"))


@dataclass
class Client:
    host: str
    port: int

    def __post_init__(self):
        self.addr = f"http://{self.host}:{self.port}/completions"
        self.encode_addr = f"http://{self.host}:{self.port}/encode"
        self.module_name_addr = f"http://{self.host}:{self.port}/module_names"
        self.weight_addr = f"http://{self.host}:{self.port}/weight"

    def _generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        echo=False,
        logprobs=0,
        desired_module_activations=(),
    ):
        prompt_dict = {
            "prompt": prompts,
            # can tune these parameters
            "temperature": temperature,
            "max_tokens": response_length,
            "top_p": top_p,
            # this arg same as the semantics of
            # https://github.com/facebookresearch/metaseq/blob/689fb79b53a2441bf815ae30e64b9438dac027bd/metaseq/hub_utils.py#L568
            "echo": echo,
            "desired_module_activations": desired_module_activations,
            "logprobs": logprobs,
        }

        resp = requests.post(self.addr, json=prompt_dict)

        check_response(resp)

        result = resp.json()

        return result

    def generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        echo=False,
    ):
        return self._generate(prompts, temperature, response_length, top_p, echo)

    @cached_property
    def module_names(self):
        resp = requests.get(self.module_name_addr)
        check_response(resp)

        return resp.json()["module_names"]


    def weight(self, module_name):
        """
        Helper function that gives some flexibility to pinging various model
        states. This only retrieves a single rank's weights however, so do not
        use outside of debugging.
        """
        resp = requests.get(self.weight_addr, json={"module_name": module_name})

        check_response(resp)

        ret_string = resp.json()["weight"]

        return decode_str(ret_string).cpu()

    def tokenize(
        self,
        list_of_strs,
    ):
        prompt_dict = {
            "prompt": list_of_strs,
        }

        resp = requests.post(self.encode_addr, json=prompt_dict)
        check_response(resp)

        return resp.json()["tok"]

    def score(self, input_list, target_list):
        """can think of context as the input_list and the token logprobs we want as the target_list"""
        all_toks = self.tokenize([p for p in itertools.chain(input_list, target_list)])

        all_inputs, input_tok_lens, target_tok_lens = [], [], []

        # concatenate the input and target tokens
        for input_tok, target_tok in zip(
            all_toks[: len(input_list)], all_toks[len(input_list) :]
        ):
            all_inputs.append(input_tok + target_tok)
            # track the length of the tokens
            target_tok_lens.append(len(target_tok))

        result = self._generate(
            prompts=all_inputs,
            temperature=1.0,
            response_length=0,
            top_p=1.0,
            echo=True,
        )

        tok_log_probs = [c["logprobs"]["token_logprobs"] for c in result["choices"]]

        output = []

        for target_len, tok_probs in zip(target_tok_lens, tok_log_probs):
            output.append(sum(tok_probs[-target_len:]))

        return output

    def get_activations(self, prompts, desired_module_activations):
        result = self._generate(
            prompts=prompts,
            temperature=1.0,
            response_length=0,
            top_p=1.0,
            echo=False,
            desired_module_activations=desired_module_activations,
        )

        activations = [
            {k: decode_str(v) for k, v in c["activations"].items()}
            for c in result["choices"]
        ]

        return activations
