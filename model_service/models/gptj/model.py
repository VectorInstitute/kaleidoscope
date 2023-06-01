import argparse
import configparser
import logging
import numpy as np
from pathlib import Path
import random
import re
import torch
import os

from .abstract_model import AbstractModel

from .gpt import GptInitModelParameters
from .parallel_gpt import ParallelGPT
from pytriton.decorators import batch
from .utils import patch_gpt_model_if_needed


logger = logging.getLogger("kaleidoscope.model_service.GPT_J")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):

    def __init__(self):
        self.model_path = None
        self.model = None
        self.device = None


    def load(self, device, model_path):
        self.device = device

        args = argparse.Namespace(
            model_name="gptj",
            pipeline_para_size=2,
            int8_mode=False,
            data_type="fp32",
            sparse=False,
        )

        config_path = Path(model_path) / "config.ini"
        config_reader = configparser.ConfigParser()
        config_reader.read(config_path)
        init_parameters = GptInitModelParameters.from_args(args, config_reader)

        lib_path = Path("/workspace/FasterTransformer/build/lib/libth_transformer.so")
        gpt_params = init_parameters.gpt_init_kwargs()

        # set RANK and WORLD_SIZE for nccl backend - TODO: configure them later
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        print(os.environ["MASTER_ADDR"])
        print(os.environ["MASTER_PORT"])
        # print(os.environ["NCCL_SOCKET_IFNAME"])
        # print(os.environ["NCCL_IB_DISABLE"])
        # print(os.environ["NCCL_IBEXT_DISABLE"])
        
        self.model = ParallelGPT(**gpt_params, lib_path=lib_path)

        # TODO - patch model required?
        patch_gpt_model_if_needed(self.model, config_reader.getint("gptj", "inter_size"), 
                                  tp=config_reader.getint("gptj", "tensor_para_size"))

        if not self.model.load(ckpt_path=Path(model_path).as_posix()):
            raise RuntimeError(f"Could not load {model_path} checkpoint")
        assert 1 == 0 # REMOVE

        if init_parameters.sparse:
            self.model.sparse()

        # eval model
        self.model.eval()
        # set to device
        self.model.to(device)
        
        self.model_path = model_path


    def module_names(self):
        return {
            "module_names": tuple(
                module[0]
                for module in self.model.base_model.named_modules()
                if module[0] != ""
            )
        }
    
    def bind(self, triton):
        


    @batch
    def generate(self, **inputs):
        logger.info(f"Starting generation on GPT_J model")
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
        response["error"] = "Activation retrival not implemented for GPT_J model."
        return response
