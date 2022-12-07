import argparse
import json
import logging
import numpy as np
import random
import re
import time
import torch

from .abstract_model import AbstractModel
from werkzeug.exceptions import HTTPException

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


class GPT2(AbstractModel):

    def __init__(self):
        self.model_class = GPT2LMHeadModel
        self.tokenizer_class = GPT2Tokenizer
        self.model = None
        self.device = None


    def load(self, device):
        self.device = device
        self.model = self.model_class.from_pretrained("/h/coatsworth/scratch/models/gpt2")
        self.model.to(device)


    def module_names(self):
        return {
            "module_names": tuple(
                module[0] for module in self.model.base_model.named_modules() if module[0] != ""
            )
        }


    def generate_text(self, request):

        prompt = request.json['prompt']
        length = int(request.json['max-tokens']) if 'max-tokens' in request.json else 128
        temperature = float(request.json['temperature']) if 'temperature' in request.json else 1.0
        top_k = float(request.json['top-k']) if 'top-k' in request.json else 0
        top_p = float(request.json['top-p']) if 'top-p' in request.json else 0.9
        num_return_sequences = int(request.json['num_return_sequences']) if 'num_return_sequences' in request.json else 1
        repetition_penalty = float(request.json['repetition_penalty']) if 'repetition_penalty' in request.json else 1.0

        tokenizer = self.tokenizer_class.from_pretrained("/h/coatsworth/scratch/models/gpt2")
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []
        random_logprobs = []
        random_tokens= []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(request['stop_token']) if 'stop_token' in request.json else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)

            # TODO: Add the real text tokens
            random_tokens.extend(re.split("(\s+)", total_sequence))

            # TODO: Add the real logprobs
            for i in range(len(random_tokens):
                random_logprobs.append(random.uniform(-3, -0.001))


        generated_text = "".join(str(x) for x in total_sequence)

        response = {}
        response['text'] = generated_text
        response['tokens'] = random_tokens
        response['logprobs'] = random_logprobs
        response['activations'] = {}

        return json.dumps(response)
