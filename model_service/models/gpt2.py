import argparse
import logging
import numpy as np
import time
import torch

from model_service import AbstractModel
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


    def load(self):
        self.model = self.model_class.from_pretrained("/scratch/models/gpt2")
        self.model.to("cpu")


    def module_names(self):
        print("Called GPT2.module_names()")


    def generate_text(self, prompt, args):
        tokenizer = self.tokenizer_class.from_pretrained("/scratch/models/gpt2")
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args["device"])

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=args["length"] + len(encoded_prompt[0]),
            temperature=args["temperature"],
            top_k=args["k"],
            top_p=args["p"],
            repetition_penalty=args["repetition_penalty"],
            do_sample=True,
            num_return_sequences=args["num_return_sequences"],
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args["stop_token"]) if "stop_token" in args else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)

        generated_text = "".join(str(x) for x in total_sequence)

        return generated_text