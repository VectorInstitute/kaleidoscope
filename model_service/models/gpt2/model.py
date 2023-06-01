import logging
import numpy as np
import random
import re
import torch

from ..abstract_model import AbstractModel

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logger = logging.getLogger("kaleidoscope.model_service.gpt2")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):
    def __init__(self, model_type, model_variant):
        self.model_class = GPT2LMHeadModel
        self.model_path = None
        self.model_type = model_type
        self.model_variant = model_variant
        self.tokenizer_class = GPT2Tokenizer
        self.model = None
        self.device = None

    def load(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_class.from_pretrained(model_path)
        self.model_path = model_path
        self.model.to(self.device)

    def bind(self, triton):
        triton.bind(
            model_name=f"{self.model_type}{self.model_variant}",
            infer_func=self.infer,
            inputs=[
                Tensor(name="prompts", dtype=bytes, shape=(1,)),
                Tensor(name='max_tokens', dtype=int, shape=(1,), optional=True),
                Tensor(name='min_tokens', dtype=int, shape=(1,), optional=True),
                Tensor(name='temperature', dtype=float, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=float, shape=(1,), optional=True),
                Tensor(name='top_k', dtype=int, shape=(1,), optional=True),
            ],
            outputs=[
                Tensor(name="sequences", dtype=bytes, shape=(-1,)),
                # Tensor(name="text", dtype=bytes, shape=(-1,)),
                # Tensor(name="tokens", dtype=bytes, shape=(-1,)),
                # Tensor(name="logprobs", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        return triton

    @property
    def rank(self):
        return 0

    @batch
    def infer(self, **inputs):
        """Generate sequences from a prompt"""
        self.generate(inputs)

    @batch
    def generate(self, **inputs):
        logger.info(f"Starting generation on GPT2 model")
        logger.info(f"inputs = {inputs}")
        """
        prompt = request.json["prompt"]

        length = (
            int(request.json["max-tokens"]) if "max-tokens" in request.json else 128
        )
        temperature = (
            float(request.json["temperature"]) if "temperature" in request.json else 1.0
        )
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
        """
        tokenizer = self.tokenizer_class.from_pretrained(self.model_path)
        prompts = np.char.decode(inputs.pop("prompts").astype("bytes"), encoding="utf-8")
        prompts = np.squeeze(prompts, axis=-1).tolist()
        logger.info(f"Prompts: {prompts}")
        encoded_prompt = tokenizer.encode(
            prompts, add_special_tokens=False, return_tensors="pt"
        )
        logger.info(f"Encoded prompt: {encoded_prompt}")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=128 + len(encoded_prompt[0]),
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
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
            text = tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )

            # Remove all text after the stop token
            #text = text[: text.find(stop_sequence) if stop_sequence else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = text[
                len(
                    tokenizer.decode(
                        encoded_prompt[0], clean_up_tokenization_spaces=True
                    )
                ) :
            ]

            generated_sequences.append(total_sequence)
            print(total_sequence)

            # TODO: Add the real text tokens
            random_tokens.extend(re.split("(\s+)", total_sequence))

            # TODO: Add the real logprobs
            for i in range(len(random_tokens)):
                random_logprobs.append(random.uniform(-3, -0.001))

        generated_text = "".join(str(x) for x in total_sequence)

        return {"sequences": np.array(generated_sequences)}


    @batch
    def get_activations(self, request):
        response = self.generate(request)
        response["activations"] = torch.empty(0)
        response["error"] = "Activation retrival not implemented for GPT2 model."
        return response
