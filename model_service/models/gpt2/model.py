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
                Tensor(name='max_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='min_tokens', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='temperature', dtype=np.float32, shape=(1,), optional=True),
                Tensor(name='top_p', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='top_k', dtype=np.int64, shape=(1,), optional=True),
                Tensor(name='repitition_penalty', dtype=np.float32, shape=(1,), optional=True)
            ],
            outputs=[
                Tensor(name="sequences", dtype=object, shape=(-1,)),
                Tensor(name="tokens", dtype=object, shape=(-1,)),
                Tensor(name="logprobs", dtype=np.float32, shape=(-1,)),
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
        return self.generate(inputs)


    def generate(self, inputs):

        # Check the input parameters, and set default values if not present
        max_tokens = inputs["max_tokens"][0][0] if "max_tokens" in inputs else 128
        temperature = inputs["temperature"][0][0] if "temperature" in inputs else 1.0
        top_p = inputs["top_p"][0][0] if "top_p" in inputs else 0.9
        top_k = inputs["top_k"][0][0] if "top_k" in inputs else 0
        repetition_penalty = inputs["repetition_penalty"][0][0] if "repetition_penalty" in inputs else 1.0

        # Load the tokenizer and encode prompts
        tokenizer = self.tokenizer_class.from_pretrained(self.model_path)
        prompts = np.char.decode(inputs.pop("prompts").astype("bytes"), encoding="utf-8")
        prompts = np.squeeze(prompts, axis=-1).tolist()
        encoded_prompt = tokenizer.encode(
            prompts, add_special_tokens=False, return_tensors="pt"
        )
        encoded_prompt = encoded_prompt.to(self.device)

        # Run the generation
        input_ids = encoded_prompt if encoded_prompt.size()[-1] != 0 else None
        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=max_tokens + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=len(prompts),
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []
        random_logprobs = []
        random_tokens = []

        logger.info(f"About to loop over output sequences...")
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            logger.info(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )

            # Remove all text after the stop token
            #text = text[: text.find(stop_sequence) if stop_sequence else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            logger.info
            total_sequence = text[
                len(
                    tokenizer.decode(
                        encoded_prompt[0], clean_up_tokenization_spaces=True
                    )
                ) :
            ]
            generated_sequences.append(total_sequence)

            # TODO: Add the real text tokens
            logger.info("Extending random tokens")
            random_tokens.extend(re.split("(\s+)", total_sequence))

            # TODO: Add the real logprobs
            logger.info("Extending random logprobs")
            for i in range(len(random_tokens)):
                random_logprobs.append(random.uniform(-3, -0.001))

        return {
            "sequences": np.array(generated_sequences, dtype="S"),
            "logprobs": np.array(random_logprobs, dtype="f"),
            "tokens": np.array(random_tokens, dtype="S")
        }


    @batch
    def get_activations(self, request):
        response = self.generate(request)
        response["activations"] = torch.empty(0)
        response["error"] = "Activation retrival not implemented for GPT2 model."
        return response
