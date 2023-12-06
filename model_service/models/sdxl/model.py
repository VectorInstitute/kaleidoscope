"""Module for Stable Diffusion SDXL configurations"""
import logging
import numpy as np
from omegaconf import OmegaConf
import random
import re
import sys
import torch

from ..abstract_model import AbstractModel

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor


logger = logging.getLogger("kaleidoscope.model_service.sdxl")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Model(AbstractModel):

    def __init__(self, model_type, model_variant):
        self.model_config = None
        self.model_path = None
        self.model_type = model_type
        self.model_variant = model_variant
        self.model = None
        self.device = None


    def load(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model, path: {model_path}")

        # How does SDXL do these from_pretrained calls?
        self.model_config = OmegaConf.load(f"v1-inference.yaml")
        cpkt_path = f"{model_path}/v2-1_768-ema-pruned.ckpt"
        logger.info(f"Model checkpoint path: {cpkt_path}")
        self.model = self.load_model_from_config(self.model_config, cpkt_path)

        self.model_path = model_path
        self.model.to(self.device)


    # Code from https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
    def load_model_from_config(config, ckpt, verbose=False):
        logger.info(f"Called load_model_from_config: config={config}, ckpt={ckpt}")
        logger.info(f"Loading model from {ckpt}")
        #pl_sd = torch.load(ckpt, map_location="cpu")
        pl_sd = torch.load(ckpt, map_location="cuda")
        logger.info(f"Now loading config and state dict")
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            logger.info("missing keys:")
            logger.info(m)
        if len(u) > 0 and verbose:
            logger.info("unexpected keys:")
            logger.info(u)

        model.cuda()
        model.eval()
        return model


    def bind(self, triton):
        triton.bind(
            model_name=f"{self.model_type}{self.model_variant}",
            infer_func=self.infer,
            inputs=[
                Tensor(name="task", dtype=np.int64, shape=(1,)),
                Tensor(name="prompts", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="images", dtype=np.bytes_, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        return triton


    @property
    def rank(self):
        return 0


    @batch
    def infer(self, **inputs):
        """Generate images from a prompt"""
        return self.generate(inputs)


    def generate(self, inputs):

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
        output_images = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            num_return_images=len(prompts),
        )

        # Remove the batch dimension when returning multiple images
        if len(output_images.shape) > 2:
            output_images.squeeze_()

        generated_images = []

        logger.info(f"About to loop over output images...")
        for generated_image_idx, generated_image in enumerate(output_images):
            logger.info(f"=== GENERATED IMAGE {generated_image_idx + 1} ===")
            generated_image = generated_image.tolist()

            # Decode text
            text = tokenizer.decode(generated_image, clean_up_tokenization_spaces=True)

            total_image = text[
                len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :
            ]
            generated_images.append(total_image.encode('utf-8').strip())

        return {
            "images": np.array(generated_images, dtype=np.bytes_),
        }


    @batch
    def get_activations(self, request):
        """Retrieve intermediate activations from SDXL model"""
        response = self.generate(request)
        response["activations"] = torch.empty(0)
        response["error"] = "Activation retrieval not implemented for SDXL model."
        return response


    @batch
    def edit_activations(self, request):
        """Edit intermediate activations from SDXL model"""
        response = self.generate(request)
        response["activations"] = torch.empty(0)
        response["error"] = "Activation editing not implemented for SDXL model."
        return response
