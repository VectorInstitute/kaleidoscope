"""Module for GPT2 LLM configurations"""
import json
import random
import re
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .abstract_model import AbstractModel


class GPT2(AbstractModel):
    """Class to represent GPT2 ML model"""

    def __init__(self):
        """Initialize GPT2 model"""
        self.model_class = GPT2LMHeadModel
        self.model_path = None
        self.tokenizer_class = GPT2Tokenizer
        self.model = None
        self.device = None

    def load(self, device, model_path):
        """load GPT2 model"""
        self.device = device
        self.model = self.model_class.from_pretrained(model_path)
        self.model_path = model_path
        self.model.to(device)

    def module_names(self):
        """Retrieve module names for GPT2"""
        return {
            "module_names": tuple(
                module[0] for module in self.model.base_model.named_modules() if module[0] != ""
            )
        }

    def generate(self, request):
        """Generate text from GPT2 model"""
        prompt = request.json["prompt"]
        length = int(request.json["max-tokens"]) if "max-tokens" in request.json else 128
        temperature = float(request.json["temperature"]) if "temperature" in request.json else 1.0
        top_k = int(request.json["top-k"]) if "top-k" in request.json else 0
        top_p = float(request.json["top-p"]) if "top-p" in request.json else 0.9
        num_return_sequences = (
            int(request.json["num_return_sequences"])
            if "num_return_sequences" in request.json
            else 1
        )
        repetition_penalty = (
            float(request.json["repetition_penalty"])
            if "repetition_penalty" in request.json
            else 1.0
        )
        stop_sequence = None
        if "stop_token" in request.json:
            stripped_sequence = str(request.json["stop_token"]).strip()
            if len(stripped_sequence) != 0:
                stop_sequence = request.json["stop_token"]

        tokenizer = self.tokenizer_class.from_pretrained(self.model_path)
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
        random_tokens = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(stop_sequence) if stop_sequence else None]

            # Add the prompt at the beginning of the sequence.
            # Remove the excess text that was used for pre-processing
            total_sequence = text[
                len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :
            ]

            generated_sequences.append(total_sequence)
            print(total_sequence)

            # TODO: Add the real text tokens
            random_tokens.extend(re.split(r"(\s+)", total_sequence))

            # TODO: Add the real logprobs
            for _ in range(len(random_tokens)):
                random_logprobs.append(random.uniform(-3, -0.001))

        generated_text = "".join(str(x) for x in total_sequence)

        response = {}
        response["text"] = generated_text
        response["tokens"] = random_tokens
        response["logprobs"] = random_logprobs
        response["activations"] = {}

        return json.dumps(response)

    def get_activations(self, request):
        """Retrieve intermediate activations from GPT2 model"""
        response = self.generate(request)
        response["activations"] = torch.empty(0)
        response["error"] = "Activation retrival not implemented for GPT2 model."
        return response
