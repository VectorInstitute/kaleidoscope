import argparse
import ast
import copy
import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Callable
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from metaseq import checkpoint_utils, tasks
from metaseq import models as metaseq_models
from metaseq.logging import metrics
from metaseq import utils
from metaseq.data import encoders
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.file_io import PathManager
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from metaseq.logging import meters, metrics, progress_bar
from metaseq.model_parallel.megatron_trainer import MegatronTrainer
from metaseq.trainer import Trainer
from metaseq.distributed.utils import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

logger = logging.getLogger(__name__)

def build_reference_model(task, ckpt_states, ):
    ref_model_list = []
    for c in ckpt_states:
      ref_model = task.build_model(c["cfg"].model).half().cuda()
      #ref_model.make_generation_fast_()
      ref_model = fsdp_wrap(ref_model)
      ref_model.load_state_dict(c["model"], 
                                strict=True, model_cfg=c["cfg"].model)
      ref_model_list.append(ref_model)
    return ref_model_list

def load_ckpt_states_and_models(cfg, reference_model=True):
    utils.import_user_module(cfg.common)

    # Fix seed for stochastic decoding
    task = tasks.setup_task(cfg.task)
    logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
    cfg.common_eval.path, 
    cfg.checkpoint.checkpoint_suffix,
    cfg.checkpoint.checkpoint_shard_count,
    ))
    logger.info("task: {}".format(task.__class__.__name__))

    if cfg.distributed_training.ddp_backend == "fully_sharded":
        if cfg.distributed_training.distributed_world_size == 1:
            data_parallel_rank = 0
        else: 
            data_parallel_rank = distributed_utils.get_data_parallel_rank()
        suffix = cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(
                    data_parallel_rank
                )
    else:
        suffix = None

    default_restore_file = "checkpoint_last.pt"
    checkpoint_path_to_load = checkpoint_utils.find_checkpoint_path_to_load(
                                  cfg.checkpoint, suffix, 
                                  default_restore_file
                              )

    bexists = PathManager.isfile(checkpoint_path_to_load)
    if bexists:
        logger.info(f"Preparing to load checkpoint {checkpoint_path_to_load}")
        ckpt_state = checkpoint_utils.load_checkpoint_to_cpu(
            checkpoint_path_to_load,
            load_on_all_ranks=True,
        )
        logger.info(f"Loaded state for {checkpoint_path_to_load}")
    return [ckpt_state, ], None

    #utils.import_user_module(cfg.common)
    ## Fix seed for stochastic decoding
    #if (
    #    cfg.common.seed is not None
    #    and not cfg.generation.no_seed_provided
    #):
    #    np.random.seed(cfg.common.seed)
    #    utils.set_torch_seed(cfg.common.seed)

    ## Setup task, e.g., translation
    #task = tasks.setup_task(cfg.task)

    ## Load the model
    #overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    #logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
    #cfg.common_eval.path, 
    #cfg.checkpoint.checkpoint_suffix,
    #cfg.checkpoint.checkpoint_shard_count,
    #))
    #with fsdp_enable_wrap(
    #    cfg.distributed_training,
    #    use_sharded_state=cfg.distributed_training.use_sharded_state,
    #):
    #    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
    #    ckpt_states = load_checkpoint_state_ensemble_and_task(
    #        utils.split_paths(cfg.common_eval.path),
    #        arg_overrides=overrides,
    #        task=task,
    #        suffix=cfg.checkpoint.checkpoint_suffix,
    #        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
    #        num_shards=cfg.checkpoint.checkpoint_shard_count,
    #    )
    #    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
    #    if reference_model:
    #        ref_models = build_reference_model(task, ckpt_states)
    #    else:
    #        ref_models = None
    #    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    #return ckpt_states, ref_models


class GeneratorInterface:
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, cfg: MetaseqConfig):
        self.cfg = cfg
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

    def load_checkpoints(self, ckpt_states):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        def _build_model(cfg, task):
            model = task.build_model(cfg.model).half().cuda()
            model.make_generation_fast_()
            return fsdp_wrap(model)
        
        def _make_model_generation(model):
            model.make_generation_fast_()
        # Load the model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
        self.cfg.common_eval.path, 
        self.cfg.checkpoint.checkpoint_suffix,
        self.cfg.checkpoint.checkpoint_shard_count,
        ))
        models = []
        _task = task
        with fsdp_enable_wrap(
            self.cfg.distributed_training,
            use_sharded_state=self.cfg.distributed_training.use_sharded_state,
        ):
            for c in ckpt_states: 
              if _task is None:
                  _task = tasks.setup_task(c['cfg'].task)
              _model = _build_model(c['cfg'], _task)
              models.append(_model)

        #logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
        #import pdb; pdb.set_trace()
        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        return models

    def copy_model_from_trainer(self, trainer_interface, models):
        # assume len(self.models) == 1
        assert len(models) == 1
        for model in models: 
            _ = model.load_state_dict(
                    trainer_interface.trainer.model.state_dict(), 
                    strict = True,
                )
        self.models = models
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())


    def copy_model(self, 
                   src_models, 
                   tgt_models, 
                   #ckpt_states,
                   ):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        #with fsdp_enable_wrap(
        #    self.cfg.distributed_training,
        #    use_sharded_state=self.cfg.distributed_training.use_sharded_state,
        #):
        #    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
        for tgt_model, src_model in zip(
                tgt_models, 
                src_models, #ckpt_states,
                ): 
            _ = tgt_model.load_state_dict(
                    src_model.state_dict(), 
                    strict = True,
                    #model_cfg= ckpt_state['cfg'].model
                )
        self.models = tgt_models
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    def load_model(self):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        def _build_model(cfg, task):
            model = task.build_model(cfg.model).half().cuda()
            model.make_generation_fast_()
            return fsdp_wrap(model)
        
        def _make_model_generation(model):
            model.make_generation_fast_()
        # Load the model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
        self.cfg.common_eval.path, 
        self.cfg.checkpoint.checkpoint_suffix,
        self.cfg.checkpoint.checkpoint_shard_count,
        ))
        with fsdp_enable_wrap(
            self.cfg.distributed_training,
            use_sharded_state=self.cfg.distributed_training.use_sharded_state,
        ):
            logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            ckpt_states = load_checkpoint_state_ensemble_and_task(
                utils.split_paths(self.cfg.common_eval.path),
                arg_overrides=overrides,
                task=task,
                suffix=self.cfg.checkpoint.checkpoint_suffix,
                strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
                num_shards=self.cfg.checkpoint.checkpoint_shard_count,
            )
            logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

            models, _task = list(zip(*[build_models_tasks_from_states(
                               s['cfg'], 
                               task=task, 
                               build_model_hook=_build_model
                               ) for s in ckpt_states]))

            _ = [m.load_state_dict(
                s['model'], 
                strict = (s['cfg'].checkpoint.checkpoint_shard_count == 1),
                model_cfg=s['cfg'].model
                    )for m,s in zip(models, ckpt_states)]

            logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            #import pdb; pdb.set_trace()
            #models, _model_args, _task = load_model_ensemble_and_task(
            #    utils.split_paths(self.cfg.common_eval.path),
            #    arg_overrides=overrides,
            #    task=task,
            #    suffix=self.cfg.checkpoint.checkpoint_suffix,
            #    strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            #    num_shards=self.cfg.checkpoint.checkpoint_shard_count,
            #    build_model_hook=_build_model,
            #)
            #logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        return models

    def generate(
        self,
        models,
        inputs: List[List[int]],
        min_tokens: List[int] = None,
        max_tokens: List[int] = None,
        temperature: float = 1.0,
        top_p: float = -1.0,
        logprobs: int = 0,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[int]] = None,
        seed: Optional[int] = None,
        use_cuda: bool = True,
    ):
        """
        Generate from sequences.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create
        inputs: a list of models
        inputs: a list of pre-tokenized prompts
        min_tokens: blocks EOS until at least this many tokens is provided
        max_tokens: forces EOS after this many tokens
        temperature: softmax temperature
        top_p: nucleus probability
        log_probs: return this cutoff of the probability distribution
        n: beam size
        best_of: number of beams to return. must be <= n
        echo: if true, returned text/tokens/scores includes the prompt.
            This is useful for getting PPL evaluations.
        stop: a list of terminating tokens
        seed: an integer if desired
        use_cuda: should we use GPUs.
        """

        ############
        ## set eval
        models = [m.eval() for m in models]
        ############
        if seed:
            utils.set_torch_seed(seed)
        start_time = time.time()
        total_generation_time = 0

        # Initialize generator
        if not best_of:
            best_of = n
        assert best_of >= n
        self.cfg.generation.sampling_topp = top_p if top_p > 0 else -1
        self.cfg.generation.sampling = top_p > 0.0
        self.cfg.generation.beam = best_of
        if temperature > 0:
            self.cfg.generation.temperature = temperature
        else:
            self.cfg.generation.temperature = 1.0

        MAX_SEQ_LEN = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in models]
        )

        # TODO(roller): simplify
        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        batches = self.task.get_generator_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
        ).next_epoch_itr(shuffle=False)
        for batch in batches:
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            batchsize = src_tokens.size(0)

            # set generation args
            # prevent us from ever generating past our max sequence length
            if max_tokens is None:
                max_tokens = [MAX_SEQ_LEN] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(
                MAX_SEQ_LEN, max(max_tokens) + src_lengths.max().item()
            )
            total_min_tokens = max(min_tokens) + src_lengths.max().item()
            self.cfg.generation.min_len = total_min_tokens
            self.cfg.generation.max_len_b = total_max_tokens
            self.cfg.generation.max_len_a = 0

            logger.info(f"Preparing generator with settings {self.cfg.generation}")
            generator = self.task.build_generator(
                models, self.cfg.generation, extra_gen_cls_kwargs={"stop": stop}
            )

            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            translate_start_time = time.time()
            translations = self.task.inference_step(generator, models, batch)

            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            # possibly cut off any bsz padding we did
            translations = translations[: len(inputs)]
            # actually turn everything into strings
            for i in range(len(translations)):
                decoding = translations[i]
                beams = []
                for beam in decoding:
                    # first beam is always the highest scoring
                    tokens = beam["tokens"].tolist()  # implicit move to cpu
                    scores = beam["positional_scores"].tolist()
                    if logprobs > 0:
                        distributions = beam["distributions"].cpu()
                    else:
                        distributions = None

                    tokens, scores, distributions = GeneratorInterface._filter_special(
                        tokens, scores, distributions
                    )
                    prompt_len = src_lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[: prompt_len + max_tokens[i] - 1]
                        scores = scores[: prompt_len + max_tokens[i] - 1]
                        if logprobs > 0:
                            distributions = distributions[
                                : prompt_len + max_tokens[i] - 1
                            ]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len - 1 :][: max_tokens[i]]
                        scores = scores[prompt_len - 1 :][: max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[prompt_len - 1 :][
                                : max_tokens[i]
                            ]
                    # turn it into a string
                    text = self.bpe.bpe.decode(tokens)
                    # re-encode it so we get offsets
                    token_offsets = [s for s, e in self.bpe.bpe.encode(text).offsets]

                    result = {
                        "text": self.bpe.bpe.decode(tokens),
                        "tokens": [self.bpe.bpe.decode([t]) for t in tokens],
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        "text_offset": token_offsets,
                        "token_scores": scores,
                    }
                    if logprobs > 0:
                        # final result is a List[Dict[str, float]]
                        # where each item in the list corresponds to a token in the
                        # sequence, and the dict provides the probabilities of the
                        # top-k tokens at that timestep.
                        out_logprobs = []
                        all_top_toks, all_top_scores = distributions.topk(
                            k=logprobs, dim=-1
                        )
                        for top_scores, top_toks in zip(all_top_toks, all_top_scores):
                            lp = {
                                self.bpe.bpe.decode([t.item()]): s.item()
                                for t, s in zip(top_toks, top_scores)
                            }
                            out_logprobs.append(lp)
                        result["top_logprobs"] = out_logprobs
                    else:
                        result["top_logprobs"] = None

                    beams.append(result)
                retval.append(beams)

        logger.info(
            "Total time: {:.3f} seconds; generation time: {:.3f}".format(
                time.time() - start_time, total_generation_time
            )
        )
        return retval

    @staticmethod
    def _filter_special(
        tokens: List[int],
        scores: List[float],
        distributions,
        pad_token: int = 1,
    ):
        """
        Cut off tokens after finding a special tokens.
        """

        # tokens is a 1D list of token IDs of length seqlen
        # scores is a 1D list of log-probability scores for those tokens (length seqlen)
        # distributions (optional) is a seqlen x vocab_size tensor corresponding to
        # the full distribution of predictions at each timestep

        output = []
        mask = []
        for t, s in zip(tokens, scores):
            if t == pad_token:
                # simply skip pads
                mask.append(False)
                continue
            if t <= 3:
                # and other special tokens should end things
                break
            mask.append(True)
            output.append((t, s))
        new_tokens, new_scores = zip(*output)

        # cut off at stop and drop pads
        if distributions is not None:
            distributions = distributions[: len(mask)][mask]
            distributions = distributions[: len(output)]
        return new_tokens, new_scores, distributions

####################################################
class GeneratorInterfaceSharded(GeneratorInterface):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def load_model(self):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        def _build_model(cfg, task):
            model = task.build_model(cfg.model).half().cuda()
            model.make_generation_fast_()
            return fsdp_wrap(model)
        
        # Load the model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        extra = {
            "use_sharded_state": self.cfg.distributed_training.use_sharded_state,
        }
        
        logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
        self.cfg.common_eval.path, 
        self.cfg.checkpoint.checkpoint_suffix,
        self.cfg.checkpoint.checkpoint_shard_count,
        ))
        with fsdp_enable_wrap(self.cfg.distributed_training, **extra):
            model = task.build_model(self.cfg.model)
            model.make_generation_fast_()
            model = fsdp_wrap(
                model,
                process_group=distributed_utils.get_data_parallel_group(),
            )
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
        logger.info(model)
        logger.info("task: {}".format(task.__class__.__name__))
        logger.info("model: {}".format(model.__class__.__name__))

        suffix = self.checkpoint_suffix
        default_restore_file = "checkpoint_last.pt"
        checkpoint_path_to_load = checkpoint_utils.find_checkpoint_path_to_load(
                                      self.cfg.checkpoint, suffix, 
                                      default_restore_file
                                  )

        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(checkpoint_path_to_load)
        if bexists:
            logger.info(f"Preparing to load checkpoint {checkpoint_path_to_load}")
            load_on_all_ranks = (
                self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # FSDP requires loading checkpoint shards on all ranks
                or self.is_fsdp
            )

            if load_on_all_ranks or self.is_data_parallel_master:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    checkpoint_path_to_load,
                    load_on_all_ranks=load_on_all_ranks,
                )
                logger.info(f"Loaded state for {checkpoint_path_to_load}")
                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory exploding.
            else:
                state = None

            # load model parameters
            try:
                model.load_state_dict(
                    state["model"], strict=True, model_cfg=self.cfg.model
                )
                # save memory for later steps
                del state["model"]
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(checkpoint_path_to_load)
                )

        model = metaseq_models.DistributedModel(
                    self.cfg.distributed_training,
                    model,
                    process_group=self.data_parallel_process_group,
                    device=torch.device("cuda"),
                )

        models = [model, ]
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        return models


    def generate(
        self,
        models,
        inputs: List[List[int]],
        min_tokens: List[int] = None,
        max_tokens: List[int] = None,
        temperature: float = 1.0,
        top_p: float = -1.0,
        logprobs: int = 0,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[int]] = None,
        seed: Optional[int] = None,
        use_cuda: bool = True,
    ):
        """
        Generate from sequences.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create
        inputs: a list of models
        inputs: a list of pre-tokenized prompts
        min_tokens: blocks EOS until at least this many tokens is provided
        max_tokens: forces EOS after this many tokens
        temperature: softmax temperature
        top_p: nucleus probability
        log_probs: return this cutoff of the probability distribution
        n: beam size
        best_of: number of beams to return. must be <= n
        echo: if true, returned text/tokens/scores includes the prompt.
            This is useful for getting PPL evaluations.
        stop: a list of terminating tokens
        seed: an integer if desired
        use_cuda: should we use GPUs.
        """

        ############
        ## set eval
        models = [m.eval() for m in models]
        ############
        if seed:
            utils.set_torch_seed(seed)
        start_time = time.time()
        total_generation_time = 0

        # Initialize generator
        if not best_of:
            best_of = n
        assert best_of >= n
        self.cfg.generation.sampling_topp = top_p if top_p > 0 else -1
        self.cfg.generation.sampling = top_p > 0.0
        self.cfg.generation.beam = best_of
        if temperature > 0:
            self.cfg.generation.temperature = temperature
        else:
            self.cfg.generation.temperature = 1.0

        MAX_SEQ_LEN = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in models]
        )

        # TODO(roller): simplify
        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        #batches = self.task.get_batch_iterator(
        #        dataset=self.task.build_dataset_for_online_training(tokens, lengths),
        #        max_tokens=None,
        #        #max_sentences=8, ## this is a dummy variable? sentences are set by batch_size
        #        max_sentences=self.cfg.dataset.batch_size,
        #        max_positions=None,
        #        ignore_invalid_inputs=False,
        #).next_epoch_itr(shuffle=False)
        batches = self.task.get_generator_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
        ).next_epoch_itr(shuffle=False)
        for batch in batches:
            #import pdb; pdb.set_trace()
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            batchsize = src_tokens.size(0)

            # set generation args
            # prevent us from ever generating past our max sequence length
            if max_tokens is None:
                max_tokens = [MAX_SEQ_LEN] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(
                MAX_SEQ_LEN, max(max_tokens) + src_lengths.max().item()
            )
            total_min_tokens = max(min_tokens) + src_lengths.max().item()
            self.cfg.generation.min_len = total_min_tokens
            self.cfg.generation.max_len_b = total_max_tokens
            self.cfg.generation.max_len_a = 0

            logger.info(f"Preparing generator with settings {self.cfg.generation}")
            generator = self.task.build_generator(
                models, self.cfg.generation, extra_gen_cls_kwargs={"stop": stop}
            )

            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            ##
            #model_out = models[0].decoder(
            #               batch["net_input"]["src_tokens"],
            #               #incremental_state=incremental_states,
            #)
            #import pdb; pdb.set_trace()
            ##
            translate_start_time = time.time()
            translations = self.task.inference_step(generator, models, batch)

            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            # possibly cut off any bsz padding we did
            translations = translations[: len(inputs)]
            # actually turn everything into strings
            for i in range(len(translations)):
                decoding = translations[i]
                beams = []
                for beam in decoding:
                    # first beam is always the highest scoring
                    tokens = beam["tokens"].tolist()  # implicit move to cpu
                    scores = beam["positional_scores"].tolist()
                    if logprobs > 0:
                        distributions = beam["distributions"].cpu()
                    else:
                        distributions = None

                    tokens, scores, distributions = GeneratorInterface._filter_special(
                        tokens, scores, distributions
                    )
                    prompt_len = src_lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[: prompt_len + max_tokens[i] - 1]
                        scores = scores[: prompt_len + max_tokens[i] - 1]
                        if logprobs > 0:
                            distributions = distributions[
                                : prompt_len + max_tokens[i] - 1
                            ]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len - 1 :][: max_tokens[i]]
                        scores = scores[prompt_len - 1 :][: max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[prompt_len - 1 :][
                                : max_tokens[i]
                            ]
                    # turn it into a string
                    text = self.bpe.bpe.decode(tokens)
                    # re-encode it so we get offsets
                    token_offsets = [s for s, e in self.bpe.bpe.encode(text).offsets]

                    result = {
                        "text": self.bpe.bpe.decode(tokens),
                        "tokens": [self.bpe.bpe.decode([t]) for t in tokens],
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        "text_offset": token_offsets,
                        "token_scores": scores,
                    }
                    if logprobs > 0:
                        # final result is a List[Dict[str, float]]
                        # where each item in the list corresponds to a token in the
                        # sequence, and the dict provides the probabilities of the
                        # top-k tokens at that timestep.
                        out_logprobs = []
                        all_top_toks, all_top_scores = distributions.topk(
                            k=logprobs, dim=-1
                        )
                        for top_scores, top_toks in zip(all_top_toks, all_top_scores):
                            lp = {
                                self.bpe.bpe.decode([t.item()]): s.item()
                                for t, s in zip(top_toks, top_scores)
                            }
                            out_logprobs.append(lp)
                        result["top_logprobs"] = out_logprobs
                    else:
                        result["top_logprobs"] = None

                    beams.append(result)
                retval.append(beams)

        logger.info(
            "Total time: {:.3f} seconds; generation time: {:.3f}".format(
                time.time() - start_time, total_generation_time
            )
        )
        return retval


    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return (self.data_parallel_world_size > 1) or (
            self.is_fsdp and self.cfg.distributed_training.cpu_offload
        )

    @property
    def is_fsdp(self):
        return self.cfg.distributed_training.ddp_backend == "fully_sharded"

    @property
    def use_sharded_state(self):
        return self.cfg.distributed_training.use_sharded_state

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        if not self.use_sharded_state:
            return self.cfg.checkpoint.checkpoint_suffix
        elif self.is_fsdp:
            return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(
                self.data_parallel_rank
            )
        else:
            return self.cfg.checkpoint.checkpoint_suffix or ""



####################################################
class TrainerInterface:

    def __init__(self, cfg: MetaseqConfig):
        self.cfg = cfg
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

    def load_model(self,):
        utils.import_user_module(self.cfg.common)

        if (
            distributed_utils.is_master(self.cfg.distributed_training)
            and "job_logging_cfg" in self.cfg
        ):
            # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
            logging.config.dictConfig(OmegaConf.to_container(self.cfg.job_logging_cfg))

        assert (
            self.cfg.dataset.max_tokens is not None or self.cfg.dataset.batch_size is not None
        ), "Must specify batch size either with --max-tokens or --batch-size"
        metrics.reset()

        if self.cfg.common.log_file is not None:
            handler = logging.FileHandler(filename=self.cfg.common.log_file)
            logger.addHandler(handler)

        np.random.seed(self.cfg.common.seed)
        utils.set_torch_seed(self.cfg.common.seed)
        
        checkpoint_utils.verify_checkpoint_directory(self.cfg.checkpoint.save_dir)

        # Print nvidia smi stats
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

        # Print args
        logger.info(self.cfg)

        if self.cfg.checkpoint.write_checkpoints_asynchronously:
            try:
                import iopath  # noqa: F401
            except ImportError:
                logging.exception(
                    "Asynchronous checkpoint writing is specified but iopath is "
                    "not installed: `pip install iopath`"
                )
                return

        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(self.cfg.task)

        assert self.cfg.criterion, "Please specify criterion to train a model"

        # Build model and criterion
        if self.cfg.distributed_training.ddp_backend == "fully_sharded":
            extra = {
                "use_sharded_state": self.cfg.distributed_training.use_sharded_state,
            }
            
            #import pdb; pdb.set_trace()
            logger.info("loading model(s) from {}, checkpoint suffix {}, checkpoint_shard_count {}".format(
            self.cfg.common_eval.path, 
            self.cfg.checkpoint.checkpoint_suffix,
            self.cfg.checkpoint.checkpoint_shard_count,
            ))
            with fsdp_enable_wrap(self.cfg.distributed_training, **extra):
                model = task.build_model(self.cfg.model)
                model = fsdp_wrap(
                    model,
                    process_group=distributed_utils.get_data_parallel_group(),
                )
        else:
            model = task.build_model(self.cfg.model)
        criterion = task.build_criterion(self.cfg.criterion)
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
        #import pdb; pdb.set_trace()
        #exit()

        logger.info(model)
        logger.info("task: {}".format(task.__class__.__name__))
        logger.info("model: {}".format(model.__class__.__name__))
        logger.info("criterion: {}".format(criterion.__class__.__name__))
        logger.info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
                sum(
                    getattr(p, "_orig_size", p).numel()
                    for p in model.parameters()
                    if p.requires_grad
                ),
            )
        )
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str()) 


        ## Load valid dataset (we load training data below, based on the latest checkpoint)
        ## We load the valid dataset AFTER building the model
        #data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
        #if cfg.dataset.combine_valid_subsets:
        #    task.load_dataset("valid", combine=True, epoch=1)
        #else:
        #    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        #        task.load_dataset(valid_sub_split, combine=False, epoch=1)

        # Build trainer
        if self.cfg.common.model_parallel_size == 1:
            logger.info("################ Built local trainer")
            trainer = Trainer(self.cfg, task, model, criterion)
        else:
            logger.info("################ Built megatron trainer")
            trainer = MegatronTrainer(self.cfg, task, model, criterion)
        logger.info(
            "training on {} devices (GPUs/TPUs)".format(
                self.cfg.distributed_training.distributed_world_size
            )
        )
        logger.info(
            "max tokens per GPU = {} and batch size per GPU = {}".format(
                self.cfg.dataset.max_tokens,
                self.cfg.dataset.batch_size,
            )
        )
        logger.info("Built trainer")
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
        
        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        #extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        #    cfg.checkpoint,
        #    trainer,
        #    # don't cache epoch iterators for sharded datasets
        #    disable_iterator_cache=True,
        #)
        extra_state = checkpoint_utils.load_checkpoint(
            self.cfg.checkpoint,
            trainer,
            # don't cache epoch iterators for sharded datasets
            ignore_dataloader=True,
            disable_iterator_cache=True,
        )
        logger.info("loaded checkpoint with trainer")
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())


        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        self.task = task
        self.bpe = bpe
        self.extra_state = extra_state
        self.trainer = trainer

        return trainer

    def reset_optim_from_checkpoint(self, models, ckpt_states):
        assert len(ckpt_states) == 1

        ckpt_state = ckpt_states[0]
        last_optim_state = ckpt_state.get("last_optimizer_state", None)
        load_on_all_ranks = (
                self.trainer.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # FSDP requires loading checkpoint shards on all ranks
                or self.trainer.is_fsdp
            )
        is_distributed = self.trainer.data_parallel_world_size > 1

        reset_lr_scheduler = self.cfg.checkpoint.reset_lr_scheduler
        reset_optimizer = self.cfg.checkpoint.reset_optimizer

        assert reset_optimizer == False, "reset_optimizer has to be false because we finetuning"  
        assert reset_lr_scheduler == False, "reset_lr_scheduler has to be false because we finetuning"  

        if last_optim_state is not None and not reset_optimizer:
            self.trainer._optim_history = ckpt_state["optimizer_history"]
            # rebuild optimizer after loading model, since params may have changed
            self.trainer._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self.trainer._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.trainer.get_criterion().__class__.__name__
            ), (
                f"Criterion does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['criterion_name']} vs "
                f"{self.trainer.get_criterion().__class__.__name__}"
            )
            assert last_optim["optimizer_name"] == self.trainer.optimizer.__class__.__name__, (
                f"Optimizer does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['optimizer_name']} vs "
                f"{self.trainer.optimizer.__class__.__name__}"
            )

            if not reset_lr_scheduler:
                self.trainer.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if not load_on_all_ranks and is_distributed:
                last_optim_state = self.trainer.optimizer.broadcast_global_state_dict(
                    last_optim_state
                )
            elif self.trainer.is_fsdp and not self.trainer.use_sharded_state:
                last_optim_state = self.trainer.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
                logger.info(f"FSDP got shard from optim_state")

            self.trainer.optimizer.load_state_dict(last_optim_state, optimizer_overrides)
            logger.info(f"reset optim_state from checkpoint")
            self.trainer.set_num_updates(last_optim["num_updates"])


 
    def train(
        self, 
        models, 
        inputs: List[List[int]],
        min_tokens: List[int] = None,
        max_tokens: List[int] = None,
        temperature: float = 1.0,
        top_p: float = -1.0,
        logprobs: int = 0,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[int]] = None,
        seed: Optional[int] = None,
        use_cuda: bool = True,
    ):
        
        max_epoch = self.cfg.optimization.max_epoch or 1
        train_meter = meters.StopwatchMeter()
        train_meter.start()

        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        epoch_itr = self.task.get_batch_iterator(
                dataset=self.task.build_dataset_for_online_training(tokens, lengths),
                max_tokens=None,
                #max_sentences=8, ## this is a dummy variable? sentences are set by batch_size
                max_sentences=self.cfg.dataset.batch_size,
                max_positions=None,
                ignore_invalid_inputs=False,
        )
        logger.info(
              f"training max epoch: {max_epoch}"
        )
        ####################################
        while epoch_itr.next_epoch_idx <= max_epoch:
            # train for one epoch
            valid_losses, should_stop = train(self.cfg, self.trainer, self.task, epoch_itr)
            if should_stop:
                break

            #print("here!")
            #exit()

            # only use first validation loss to update the learning rate
            self.trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            ## do not have to update the epoch_itr, self update
            #epoch_itr = trainer.get_train_iterator(
            #    epoch_itr.next_epoch_idx,
            #    # don't cache epoch iterators for sharded datasets
            #    disable_iterator_cache=True,
            #)
        train_meter.stop()
        logger.info("done training in {:.1f} seconds".format(train_meter.sum))

        # ioPath implementation to wait for all asynchronous file writes to complete.
        if self.cfg.checkpoint.write_checkpoints_asynchronously:
            logger.info(
                "ioPath PathManager waiting for all asynchronous checkpoint "
                "writes to finish."
            )
            PathManager.async_close()
            logger.info("ioPath PathManager finished waiting.")



def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.BaseTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=True,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    if update_freq > 1:
        itr = iterators.GroupedIterator(
            itr,
            update_freq,
            skip_remainder_batch=(
                not cfg.optimization.train_with_epoch_remainder_batch
            ),
        )

    progress = progress_bar.get_progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)
    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    def train(
        i,
        samples,
    ):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            if update_freq == 1:
                samples = [samples]
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg,
            trainer,
            task,
            epoch_itr,
            valid_subsets,
            end_of_epoch,
            log_output is not None,
        )

        return valid_losses, should_stop

    for i, samples in enumerate(progress):
        
        logger.info(f"training batch size: {samples['target'].shape}, #tokens: {samples['ntokens']}")
        #import pdb; pdb.set_trace()

        if (
            distributed_utils.get_global_rank() == 0
            and cfg.common.new_profiler
            and i == 5
        ):
            logger.info("STARTING PROFILER")
            with profiler.profile() as prof:
                valid_losses, should_stop = train(i, samples)
            torch.cuda.synchronize()
            prof.export_chrome_trace(
                os.path.join(cfg.checkpoint.save_dir, "profiler_trace.json")
            )
        else:
            valid_losses, should_stop = train(i, samples)
        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.BaseTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    was_successful_step: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # was_successful_step is necessary since we don't increment step counters
    # on OOM or overflow. Thus if we get multiple bad steps right after
    # loading a checkpoint (when step counter is exactly when we would step)
    # then we will start overwriting! omg!

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
            and was_successful_step
        )
    )
    do_validate = (
        (
            not end_of_epoch and do_save and not cfg.checkpoint.no_best_checkpoints
        )  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
            and was_successful_step
        )
    ) and not cfg.dataset.disable_validation
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint,
            trainer,
            epoch_itr,
            valid_losses[0],
            training_finished=should_stop,
            async_callback_fn=functools.partial(post_checkpoint_callback, cfg)
            if cfg.checkpoint.cloud_upload_path
            else None,
        )

    trainer.reset_dummy_batch(epoch_itr.first_batch)
    return valid_losses, should_stop


def post_checkpoint_callback(cfg, filename):
    if cfg.checkpoint.cloud_upload_path is not None:
        if "blob.core.windows.net" in cfg.checkpoint.cloud_upload_path:
            azcopy_logs = filename + "_azcopy_logs"
            os.environ["AZCOPY_CONCURRENCY_VALUE"] = "10"
            os.environ["AZCOPY_LOG_LOCATION"] = azcopy_logs
            os.makedirs(azcopy_logs, exist_ok=True)
            logger.info(
                f"preparing to azcopy {filename} to {cfg.checkpoint.cloud_upload_path}; logs in {azcopy_logs}"
            )
            cmd = [
                "azcopy",  # TODO(susanz): require azcopy to be installed.
                "copy",
                "--cap-mbps",
                "96.0",
                filename,
                cfg.checkpoint.cloud_upload_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print("Error: {}, azcopy failed".format(res.returncode))
                print("Azcopy stdout = {}".format(res.stdout))
                sys.exit(1)
            # Delete original checkpoint on local storage
            # TODO make this configurable
            logger.info(
                f"Successfully copied {filename} to {cfg.checkpoint.cloud_upload_path}"
            )
            os.remove(filename)
        else:
            try:
                # PathManager only supports writing to S3, but this function call
                # can be replaced with other APIs for copying checkpoints.
                PathManager.copy_from_local(
                    filename,
                    os.path.join(
                        cfg.checkpoint.cloud_upload_path, os.path.basename(filename)
                    ),
                    overwrite=True,
                )
            except (FileNotFoundError, AssertionError) as e:
                logger.info(f"could not upload {filename}: {e}")


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.BaseTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    with metrics.aggregate(new_root=True) as combined_agg:
        for subset in subsets:
            logger.info(
                'begin validation on "{}" subset on rank {}'.format(
                    subset, distributed_utils.get_global_rank()
                )
            )

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )

            logger.info(
                'got valid iterator on "{}" subset on rank {}'.format(
                    subset, distributed_utils.get_global_rank()
                )
            )

            progress = progress_bar.get_progress_bar(
                itr,
                log_format=cfg.common.log_format,
                log_interval=cfg.common.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                wandb_project=(
                    cfg.common.wandb_project
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
                ),
            )

            logger.info(
                'Begin looping over validation "{}" subset with length "{}"'.format(
                    subset, len(progress)
                )
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate() as agg:
                for i, sample in enumerate(progress):
                    if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                    ):
                        break
                    trainer.valid_step(sample)
            # log validation stats
            stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    stats = get_valid_stats(cfg, trainer, combined_agg.get_smoothed_values())
    progress.print(stats, tag="valid/combined", step=trainer.get_num_updates())
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats




####################################################

def load_checkpoint_state_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    assert state is None or len(filenames) == 1

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None

    print("#######################")
    print("load ensemble states")
    print(filenames)
    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
                #filename = filename.replace(".pt", suffix + f"-shard{shard_idx}" + ".pt")
                #filename = orig_filename.replace(".pt", suffix + f"-shard{shard_idx}" + ".pt")
            logger.info(f"#shards:{num_shards}, suffix:{suffix}, orig_filename:{orig_filename}, filename:{filename}") 

            if state is None:
                state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
            
            if "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"!!! cfg does not exist in state keys = {state.keys()} !!!"
                )
            print("#############checkpoint loading##################")
            print("checkpoint arch: {}".format(getattr(cfg.model, "arch", None)))
            print("checkpoint mp size: {}".format(cfg.common.model_parallel_size))
            print("task_state in state: {}".format("task_state" in state))
            if "task_state" in state:
                print(state["task_state"])
            else:
                state["task_state"] = {}
            print("strict: {}".format(strict))
            print("model_cfg:")
            print(cfg.model)
            print("state_dict:")
            print(state["model"])
            print("###############################")
            # Load 175B model trained on megatron (model parallel) branch
            # "cfg.common.model_parallel_size == 1" checks if model parallel is
            # enabled at load time. If it's not, fall back to non-MP
            # transformer code path.
            if (
                getattr(cfg.model, "arch", None) == "transformer_lm_megatron"
                and cfg.common.model_parallel_size == 1
            ):
                print("########### load megatron on single gpu")
                logger.info("load transformer_lm_megatron and mp == 1")
                cfg.model.arch = "transformer_lm_gpt"
                cfg.model._name = "transformer_lm_gpt"
                oproj_key = "decoder.output_projection.weight"
                emb_key = "decoder.embed_tokens.weight"
                if emb_key in state["model"] and oproj_key not in state["model"]:
                    state["model"][oproj_key] = state["model"][emb_key]

            #if task is None:
            #    task = tasks.setup_task(cfg.task)

            #if "task_state" in state:
            #    task.load_state_dict(state["task_state"])

            #if build_model_hook is not None:
            #    model = build_model_hook(cfg, task)
            #else:
            #    # build model for ensemble
            #    model = task.build_model(cfg.model)

            #model.load_state_dict(state["model"], strict=strict, model_cfg=cfg.model)
            logger.info("Done loading state dict")
            # reset state so it gets loaded for the next model in ensemble
            #state = None

        ensemble.append(state)
    return ensemble


def build_models_tasks_from_states(
    cfg: MetaseqConfig,
    task=None,
    build_model_hook=None
):
    if task is None:
        task = tasks.setup_task(cfg.task)

    #if "task_state" in state:
    #    task.load_state_dict(state["task_state"])

    if build_model_hook is not None:
        model = build_model_hook(cfg, task)
    else:
        # build model for ensemble
        model = task.build_model(cfg.model)
    return model, task




def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
    build_model_hook=None,
):
    assert state is None or len(filenames) == 1

    from metaseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None

    print("#######################")
    print("load model ensemble")
    print(filenames)
    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
                #filename = filename.replace(".pt", suffix + f"-shard{shard_idx}" + ".pt")
                #filename = orig_filename.replace(".pt", suffix + f"-shard{shard_idx}" + ".pt")
            print("getting filenames")
            print(num_shards)
            print(suffix)
            print(orig_filename)
            print(filename)
            print("#######################")

            if state is None:
                state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
            
            if "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"!!! cfg does not exist in state keys = {state.keys()} !!!"
                )
            print("#############checkpoint loading##################")
            print("checkpoint arch: {}".format(getattr(cfg.model, "arch", None)))
            print("checkpoint mp size: {}".format(cfg.common.model_parallel_size))
            print("task_state in state: {}".format("task_state" in state))
            if "task_state" in state:
                print(state["task_state"])
            else:
                state["task_state"] = {}
            print("build_model_hook is not Non: {}".format(build_model_hook is not None))
            print("strict: {}".format(strict))
            print("model_cfg:")
            print(cfg.model)
            print("state_dict:")
            print(state["model"])
            print("###############################")
            # Load 175B model trained on megatron (model parallel) branch
            # "cfg.common.model_parallel_size == 1" checks if model parallel is
            # enabled at load time. If it's not, fall back to non-MP
            # transformer code path.
            if (
                getattr(cfg.model, "arch", None) == "transformer_lm_megatron"
                and cfg.common.model_parallel_size == 1
            ):
                logger.info("load transformer_lm_megatron and mp == 1")
                cfg.model.arch = "transformer_lm_gpt"
                cfg.model._name = "transformer_lm_gpt"
                oproj_key = "decoder.output_projection.weight"
                emb_key = "decoder.embed_tokens.weight"
                if emb_key in state["model"] and oproj_key not in state["model"]:
                    state["model"][oproj_key] = state["model"][emb_key]

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if build_model_hook is not None:
                model = build_model_hook(cfg, task)
            else:
                # build model for ensemble
                model = task.build_model(cfg.model)

            model.load_state_dict(state["model"], strict=strict, model_cfg=cfg.model)
            logger.info("Done loading state dict")
            # reset state so it gets loaded for the next model in ensemble
            state = None

        ensemble.append(model)
    return ensemble, cfg, task


