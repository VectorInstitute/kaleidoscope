#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import queue
import pkg_resources
from collections import defaultdict
import random
import threading
import traceback
import pickle
import codecs
import functools

import torch
from flask import Flask, request, jsonify
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

from hook_utils import get_activation_capture_hook_dict, apply_forward_hook


app = Flask(__name__)

# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()


def batching_loop(timeout=100, max_tokens=MAX_BATCH_TOKENS):
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via BATCH_QUEUE)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.

    param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    global BATCH_QUEUE

    batch_dict = defaultdict(list)
    target_queue = None
    while True:
        try:
            assert len(batch_dict) <= 1

            b_key, bs_list = (
                next(iter(batch_dict.items())) if batch_dict else (None, [])
            )

            # for now, we only have 1 worker, so can always index to shard 0
            if target_queue is None:
                # TODO: try to process a request with the same arg
                target_queue = BATCH_QUEUE.queue_shards[0].get_largest_queue(b_key)
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            # accumulate the batch until it gets too big
            longest = max([item] + bs_list).cost
            batch_cost = longest * (len(bs_list) + 1)
            # overflow corresponds to whether max(prompt_len) + gen_len will
            # fit the max sequence length
            max_prompt_len = max(x.prompt_len for x in [item] + bs_list)
            max_gen_len = max(x.gen_len for x in [item] + bs_list)

            # changed this to enable batching
            overflow = max_prompt_len + max_gen_len > MAX_SEQ_LEN

            # naive impl, as soon as the args are different fire
            # the previous batch
            if bs_list and (
                batch_cost > max_tokens
                or overflow
                # if the item is not the same arg, flush out the batch
                # and put back the item that doesn't match
                or (b_key is not None and b_key != item.queue_key())
            ):
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                batch_dict[item.queue_key()].append(item)

        except queue.Empty:
            target_queue = None
            if batch_dict:
                # because we always flush out the batch as soon as we have a different key
                # the length of the batch_dict should always be 1

                assert len(batch_dict) == 1
                (batch,) = batch_dict.values()

                request_object = {
                    "inputs": [],
                    "min_tokens": [],
                    "max_tokens": [],
                }

                # use this to check for correctness
                unique_dict = {}

                logger.info("length of batch is {}".format(len(batch)))

                for work_item in batch:
                    ro = work_item.data
                    request_object["inputs"].append(ro["input"])
                    request_object["min_tokens"].append(ro.get("min_tokens", 0))
                    request_object["max_tokens"].append(
                        ro.get("max_tokens", MAX_SEQ_LEN)
                    )

                    for key in UNBATCHED_ARG_DICT:
                        if key in unique_dict and unique_dict[key] != ro.get(
                            key, unique_dict[key]
                        ):
                            raise ValueError(
                                "the remaining args are not the same, currently {}, but want {} with key {}".format(
                                    unique_dict,
                                    ro[key],
                                    key,
                                )
                            )

                        if key in ro:
                            request_object[key] = ro[key]
                            unique_dict[key] = ro[key]

                        else:
                            # if key not in ro then it should take default value
                            unique_dict[key] = UNBATCHED_ARG_DICT[key]

                # WARNING: seed will not be deterministic when we batch
                # TODO: do we include the seed or not? we can't guarantee the correctness of this parameter anyway
                # if "seed" not in request_object:
                request_object["seed"] = random.randint(0, 20000)

                # NOTE: aux is a tuple containing any necessary aux data for
                #       activation retrieval (only batch size right now)
                # TODO: Is there a better way to do this? Heavily constrained
                #       by broadcasting to ranks. Can't nest dicts
                request_object["_aux"] = (len(batch),)

                if torch.distributed.get_rank() == 0:
                    logger.info("request object {}".format(request_object))

                if torch.distributed.is_initialized():
                    distributed_utils.broadcast_object(
                        request_object,
                        src_rank=0,
                        group=distributed_utils.get_global_group(),
                    )

                activation_dict = {}

                try:
                    desired_module_activations = request_object.pop(
                        "desired_module_activations", None
                    )

                    act_retrieval_aux = request_object.pop("_aux", None)

                    if desired_module_activations:
                        hook_dict, activation_dict = get_activation_capture_hook_dict(
                            generator.models[0],
                            desired_module_activations,
                            aux=act_retrieval_aux,
                        )

                        with apply_forward_hook(generator.models[0], hook_dict):
                            generations = generator.generate(**request_object)

                    else:
                        generations = generator.generate(**request_object)

                except RuntimeError:
                    # Probably cuda died. Unfortunately, we need to hard crash
                    # here to kick in our self-healing mechanisms.
                    raise
                except BaseException as e:
                    # propagate any exceptions to the response so we can report it
                    generations = [e] * len(batch)

                # broadcast them back
                for i, (work_item, gen) in enumerate(zip(batch, generations)):

                    if not isinstance(gen, Exception):
                        assert len(gen) == 1
                        assert "activations" not in gen
                        num_real_tokens = len(gen[0]["all_tokens"])

                        # need the clone because otherwise the views are shared
                        # and will need to be transferred
                        # this is padded RIGHT

                        # activations should come in as B x S x D
                        ret_dict = {}

                        for k, v in activation_dict.items():
                            # attention map
                            if "self_attn.dropout_module" in k:
                                val = v[
                                    i,
                                    :,
                                    1 : num_real_tokens + 1,
                                    1 : num_real_tokens + 1,
                                ].clone()
                            else:
                                # cut off the starting token because metaseq
                                # adds. It should take out the pad to reduce bandwidth
                                val = v[i, 1 : num_real_tokens + 1].clone()

                            ret_dict[k] = codecs.encode(
                                pickle.dumps(val),
                                "base64",
                            ).decode("utf-8")

                        gen[0]["activations"] = ret_dict

                    work_item.return_queue.put((work_item.uid, gen))

                activation_dict.clear()
                batch_dict.clear()
            else:
                # back to the loop
                continue


def worker_main(cfg1: MetaseqConfig, namespace_args=None):
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

    if torch.distributed.get_rank() == 0:
        print(models[0])    # Cleaner to print
        logger.info("Model training: {}".format(models[0].training))

    assert len(models) == 1

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = distributed_utils.broadcast_object(
        None, src_rank=0, group=distributed_utils.get_global_group()
    )
    if torch.distributed.get_rank() == 0:
        logger.info(f"Worker engaged! {get_my_ip()}:{port}")
        thread = threading.Thread(target=batching_loop, daemon=True)
        thread.start()
        # app.run(host="0.0.0.0", port=port, threaded=True)
        app.run(host="0.0.0.0", port=port, threaded=False)
    else:
        # useful in FSDP setting
        logger.info(f"Looping engaged! {get_my_ip()}:{port}")
        while True:
            try:
                request_object = distributed_utils.broadcast_object(
                    None, src_rank=0, group=distributed_utils.get_global_group()
                )

                desired_module_activations = request_object.pop(
                    "desired_module_activations", None
                )
                act_retrieval_aux = request_object.pop("_aux", None)

                if desired_module_activations:
                    hook_dict, _ = get_activation_capture_hook_dict(
                        generator.models[0],
                        desired_module_activations,
                        aux=act_retrieval_aux,
                    )

                    with apply_forward_hook(generator.models[0], hook_dict):
                        _ = generator.generate(**request_object)
                else:
                    _ = generator.generate(**request_object)

            except Exception:
                # continue looping for the next generation so we don't lock up
                pass


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    http_code = 400 if isinstance(e, ValueError) else 500
    return _create_error_response(
        str(e), http_code, stacktrace=traceback.format_tb(e.__traceback__)
    )


def _validate_key(key):
    # denylist a few placeholders various people have used
    if key == "":
        return False
    if "YOUR_NAME_HERE" in key:
        return False
    if "$USER" in key:
        return False
    if "your-key-here" in key:
        return False
    return True


def _create_error_response(msg, http_code, **others):
    error_dict = {
        "message": msg,
        "type": "invalid_request_error",
        "param": None,
        "code": None,
        **others,
    }
    print(msg)

    response = jsonify({"error": error_dict})
    response.status = http_code
    return response


@app.route("/encode", methods=["POST"])
def encode():
    # MUST BE a list of strings!
    prompts = request.json["prompt"]

    return {"tok": [encode_fn(generator, p) for p in prompts]}


@app.route("/module_names", methods=["GET"])
def module_names():
    return {
        "module_names": tuple(
            n for n, _ in generator.models[0].named_modules() if n != ""
        )
    }


@app.route("/weight", methods=["GET"])
def weight():
    module_name = request.json["module_name"]

    def _rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [obj] + attr.split("."))

    module_weight = _rgetattr(generator.models[0], module_name)

    ret = codecs.encode(
        pickle.dumps(module_weight),
        "base64",
    ).decode("utf-8")

    return {
        "weight": ret
    }


@app.route("/completions", methods=["POST"])
@app.route("/v1/engines/<engine>/completions", methods=["POST"])
@app.route("/v2/engines/<engine>/completions", methods=["POST"])
@app.route("/engines/<engine>/completions", methods=["POST"])
def completions(engine=None):
    # before anything else, check that we've got a valid API key

    # TODO: reenable
    # if not _validate_key(request.headers.get("authorization", "")):
    # return _create_error_response("Invalid API key or API key missing.", 401)

    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of ints. Pretokenized. Return one generation
    # - list of str. Multiple generations, one per prompt
    # - list of list of ints. Pretokenized multiple generations.

    # our approach is to turn everything into the last case

    prompts = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if isinstance(prompts, str):
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [encode_fn(generator, prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [encode_fn(generator, p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    assert len(prompts[0]) > 0
    print(prompts)

    if "min_tokens" in generation_args:
        generation_args["min_tokens"] = int(generation_args["min_tokens"])
    if "max_tokens" in generation_args:
        generation_args["max_tokens"] = int(generation_args["max_tokens"])
    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [encode_fn(generator, stop)[0]]
        else:
            stop = [encode_fn(generator, s)[0] for s in stop]
        generation_args["stop"] = stop
    if "temperature" in generation_args:
        generation_args["temperature"] = round(float(generation_args["temperature"]), 1)
    else:
        generation_args["temperature"] = UNBATCHED_ARG_DICT["temperature"]
    if "top_p" in generation_args:
        generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
    else:
        generation_args["top_p"] = UNBATCHED_ARG_DICT["top_p"]
    # beam search top n
    if "n" in generation_args:
        generation_args["n"] = min(MAX_BEAM, max(1, int(generation_args["n"])))
    else:
        generation_args["n"] = UNBATCHED_ARG_DICT["n"]

    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        gen_len = generation_args.get("max_tokens", 0)
        if gen_len + len(prompt) + 1 > MAX_SEQ_LEN:
            # cut off the prompt to always fit with number of generations we need
            # +1 to always have the EOS token
            prompt = prompt[-(MAX_SEQ_LEN - gen_len - 1) :]
        request_object = {"input": prompt, **generation_args}
        BATCH_QUEUE.put(
            WorkItem(
                cost=len(prompt) + gen_len,
                uid=i,
                return_queue=ret_queue,
                data=request_object,
                prompt_len=len(prompt),
                gen_len=gen_len,
            )
        )
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for prompt, (_, generations) in zip(prompts, reordered):
        if isinstance(generations, Exception):
            raise generations
        results += generations

    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    fn = pkg_resources.resource_filename("metaseq", "service/index.html")
    with open(fn) as f:
        return f.read()


def cli_main():
    """
    Hosted version of the web UI for generation.
    """

    global port, MODE, cfg
    parser = options.get_generation_parser()

    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()
