import argparse
import logging
import numpy as np
import os
import sys
import time
import torch

from .abstract_model import AbstractModel
from werkzeug.exceptions import HTTPException


from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.workers import WorkItem
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    MAX_BEAM,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
    UNBATCHED_ARG_DICT,
)
from metaseq.service.utils import get_my_ip, encode_fn, build_logger
from metaseq.service.responses import OAIResponse

#from utils.hook_utils import get_activation_capture_hook_dict, apply_forward_hook


class OPT_175B(AbstractModel):

    def __init__(self):
        self.model = None
        self.device = None
        self.generator = None
        self.MODE = None

    def load(self, device):
        self.device = device

        # dumb defaults overriding
        parser = options.get_generation_parser()
        parser.set_defaults(lr_scheduler=None, criterion=None)
        flat_launch_args = []
        for s in LAUNCH_ARGS:
            flat_launch_args += s.split()

        args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
        args.data = os.path.dirname(args.path)  # hardcode the data arg

        cfg = convert_namespace_to_omegaconf(args)
        cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

        distributed_utils.call_main(cfg, self.worker_main, namespace_args=args)


    def module_names(self):
        print("Called OPT_175B.module_names()")
        return "Placeholder return text for OPT_175B.module_names()"


    def generate_text(self, prompt, args):
        print("Called OPT_175B.generate_text()")
        return "Placeholder return text for OPT_175B.generate_text()"


    def worker_main(self, cfg1: MetaseqConfig, namespace_args=None):
        # disable multithreading in tokenizers and torch, as different Flask threads
        # may then fight for resources.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(1)
        global generator
        global MODE

        # make sure generations are stochastic since we have many workers
        torch.manual_seed(6 + torch.distributed.get_rank())
        torch.cuda.manual_seed(6 + torch.distributed.get_rank())
        MODE = "worker"
        cfg = cfg1

        generator = GeneratorInterface(cfg)
        models = generator.load_model()  # noqa: F841
