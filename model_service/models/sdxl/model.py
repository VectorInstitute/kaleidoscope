"""Module for Stable Diffusion SDXL configurations"""
import base64
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import random
import re
import sys
import string
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
        logger.info(f"Loading model from path: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model to device: {self.device}")
        #self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)
        self.pipeline.to(self.device)


    def bind(self, triton):
        triton.bind(
            model_name=f"{self.model_type}-{self.model_variant}",
            infer_func=self.infer,
            inputs=[
                Tensor(name="task", dtype=np.int64, shape=(1,)),
                Tensor(name="prompts", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="sequences", dtype=np.bytes_, shape=(-1,)),
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
        logger.info(f"Received inference request with inputs: {inputs}")
        return self.generate(inputs)


    def generate(self, inputs):

        logger.info(f"Running generation with inputs: {inputs}")
        prompts = np.char.decode(inputs.pop("prompts").astype("bytes"), encoding="utf-8")
        prompts = np.squeeze(prompts, axis=-1).tolist()

        # Iterate over the prompts and generate an image for each one
        random_characters = string.ascii_letters + string.digits
        generated_images = []
        for prompt in prompts:
            image = self.pipeline(prompt).images[0]

            # Save the image to a temporary file, then store it as base64 encoding
            random_filename = ''.join(random.choice(random_characters) for _ in range(16))
            tmp_file = Path(f"/tmp/{random_filename}.png")
            image = image.save(tmp_file)
            with open(tmp_file, "rb") as f:
                encoded_image = base64.b64encode(f.read())
                generated_images.append(encoded_image)

            # Now delete the temp image file
            tmp_file.unlink()

        return {
            "sequences": np.array(generated_images, dtype=np.bytes_),
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
