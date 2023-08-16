from dataclasses import dataclass
import logging
import json
import os
import sys
import time
from typing import List, Any, Dict, Tuple
import uuid
from pathlib import Path

import transformers
import torch
from torch import Tensor
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, Llama


@dataclass
class RequestObject:
    """Request object for generation."""
    prompts: List[str]
    max_gen_len: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    encoded_activation_payload: str = None   # TODO: Typehint
    _aux: Tuple[Any] = None


@dataclass
class ResponseObject:
    """OpenAI API response-like object."""
    generations: List[str]
    activations: Dict[str, Tensor]

    def __post_init__(self):
        self._response_id = str(uuid.uuid4())
        self._created = int(time.time())

    def json(self):
        """Return the result in a json dict format."""
        return {
            "id": self._response_id,
            "object": "text_completion",
            "created": self._created,
            "model": "llama2",
            "choices": [
                # TODO: Impl rest of keys under logprobs
                {
                    "text": self.generations,
                    "logprobs": None,
                    "activations": self.activations,
                }
            ]
        }


def build_host_logger():
    """Build logger."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    _logger = logging.getLogger("llama2.host_model")
    return _logger


def setup_model_parallel() -> Tuple[int, int]:
    """Parse distributed state and initialize model parallel."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    torch.distributed.init_process_group("nccl")
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()
    initialize_model_parallel(global_world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return global_rank, global_world_size


def load_llama(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> Llama:
    logger = build_host_logger()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    if torch.distributed.is_initialized():
        assert torch.distributed.get_world_size() == len(
            checkpoints
        ), (f"Loading a checkpoint for MP={len(checkpoints)} but world size is "
            f"{world_size}")
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    logger.info(f"Hosting utils tokenizer: {tokenizer}")
    logger.info(f"Hosting utils dir(tokenizer): {dir(tokenizer)}")
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    logger.info(f"Hosting utils model_args: {model_args}")
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = Llama(model, tokenizer)
    return generator
