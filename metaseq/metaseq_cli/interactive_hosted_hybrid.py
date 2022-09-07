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
import random
import shutil
import threading
import traceback
import logging
import sys
import time
import psutil
from psutil._common import bytes2human

import torch

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

from metaseq_cli.interactive_hosted_utils import (
    GeneratorInterface, 
    TrainerInterface, 
    load_ckpt_states_and_models,
)

from metaseq import options
from metaseq import checkpoint_utils, tasks
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from metaseq.logging import meters, metrics, progress_bar
#from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.workers import WorkItem
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    CHECKPOINT_LOCAL,
    CHECKPOINT_FOLDER,
    LAUNCH_ARGS,
    MAX_BEAM,
)
from metaseq.service.utils import get_my_ip, encode_fn
from metaseq.service.responses import OAIResponse


app = Flask(__name__)

# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("metaseq_cli.hybrid")


def parse_request_json_dict(json_dict):
    prompts = json_dict["prompt"]
    del json_dict["prompt"]
    generation_args = json_dict

    class AttributeDict(dict):
        def __getattr__(self, attr):
            return self[attr]
        def __setattr__(self, attr, value):
            self[attr] = value

    generator['bpe'] = bpe
    generator = AttributeDict(generator)

    print(f"prompts is {prompts}")
    print(f"generation_args is {generation_args}")

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
        generation_args["temperature"] = 1.0
    if "top_p" in generation_args:
        generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
    else:
        generation_args["top_p"] = 1.0
    # beam search top n
    if "n" in generation_args:
        generation_args["n"] = int(generation_args["n"])
    else:
        generation_args["n"] = 1

    ################# generation / train mode
    if "mode" in generation_args:
        generation_args["mode"] = generation_args["mode"] if generation_args["mode"] in ['train', 'generation', 'reset_to_checkpoint'] else 'generation'
    else:
        generation_args["mode"] = 'generation'

    return prompts, generation_args


def parse_queue_request_batch(batch, seed=None):
    #print(len(batch))
    request_object = {
        "inputs": [],
        "min_tokens": [],
        "max_tokens": [],
    }
    for work_item in batch:
        if isinstance(work_item, dict):
            ro = work_item
        else:
            ro = work_item.data
        request_object["inputs"].append(ro["input"])
        request_object["min_tokens"].append(ro.get("min_tokens", 0))
        request_object["max_tokens"].append(
            ro.get("max_tokens", MAX_SEQ_LEN)
        )
        # assumption: everyone has the same remaining args
        for key in [
            "temperature",
            "top_p",
            "n",
            "best_of",
            "echo",
            "logprobs",
            "stop",
            "mode", # new flag
        ]:
            if key in ro:
                request_object[key] = ro[key]
    
    # give the same seed to the whole batch? TODO: how to fix?
    if seed is None:
        request_object["seed"] = random.randint(1, 20000)
    else:
        request_object["seed"] = 2022
        
    return request_object



def get_request_object(request_dict, bpe):
    prompts, generation_args = parse_request_json_dict(request_dict)

    work_batch = [{"input":prompt, **generation_args} for prompt in prompts]
    request_object = parse_queue_request_batch(work_batch, seed=2022)

    return request_object


def batching_queue_loop(timeout=100, max_tokens=MAX_BATCH_TOKENS):
    """
    modified from
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

    :param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    global BATCH_QUEUE

    batch = []
    target_queue = None
    while True:
        try:
            # for now, we only have 1 worker, so can always index to shard 0
            if target_queue is None:
                target_queue = BATCH_QUEUE.queue_shards[0].get_largest_queue()
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            logger.info("target_queue found and got items")
            
            #print("#########building queue############")
            logger.info(f"building queue: current batch size {len(batch)}, new item {item}")
            # accumulate the batch until it gets too big
            longest = max([item] + batch).cost
            batch_cost = longest * (len(batch) + 1)

            # overflow to detect max seq length
            max_prompt_len = max(x.prompt_len for x in [item] + batch)
            max_gen_len = max(x.gen_len for x in [item] + batch)
            overflow = max_prompt_len + max_gen_len > MAX_SEQ_LEN

            if batch and (batch_cost > max_tokens or overflow):
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                logger.info("worker putting item to batch")
                # batch is empty or under budget
                batch.append(item)
        except queue.Empty:
            if batch:
                logger.info(f"queue.Empty caught so we finished building batch, size {len(batch)}")
                #print("######### queue is empty in except############")
                request_object = parse_queue_request_batch(batch)

                queue_ret_hook = batch
                return request_object, queue_ret_hook
            else:
                # back to the loop
                continue

def return_queue_results(queue_ret_hook, results):
    if torch.distributed.get_rank() == 0:
        #if queue_ret_hook is None:
        #    print("got empty return queue hook, assume this is headless test")
        #    exit()
        # broadcast them back
        for work_item, gen in zip(queue_ret_hook, results):
            work_item.return_queue.put((work_item.uid, gen))
    else:
        pass


def get_headless_test_data():
    request_dict = {
        "prompt": [
            "hello vector",
            "what is the meaning of life?",
            "input: 111+179\noutput: 290\ninput: 345+132\noutput: 577\ninput: 234+233\noutput:",
        ],
        # can tune these parameters
        "temperature": 0.7,
        "max_tokens": 32,
        "top_p": 0.9,
        "echo": True,

        # added in seed to specify the seed to use for generation
        # NOTE: this only works if we modify interactive_hosted.py
        # "seed": 6,
    }
    request_object = get_request_object(request_dict, bpe)
    return request_object, None


def get_headless_test_data2():
    request_dict = {
        "prompt": [
            "hello vector, this is fine-tuning magic.",
            "what is the meaning of life? this is total gibrish and see if the model reproduce this.",
            ##
            "A dog walking along a frosty path.",
            "A dog walks in a foggy place.",
            "A white dog walking in the snow.",
            "A white dog walks along a path which has a thin covering of snow flakes.",
            "A man and a woman wearing a head covering turn the video camera on themselves.",
            "A man and woman pose for the video camera that they are holding.",
            "A man in a brown sweater and a woman smile for their video camera.",
            "A smiling couple with a camcorder.",
            "Woman standing with man holding a camcorder and smiling into lens.",
            "Three bicyclists racing up a dirt path through the forest.",
            "An older couple with Goucho Marx masks and fake cigars.",
            "An older couple with joke glasses and cigars.",
            "Man and woman wearing false eyebrows and noses.",
            "The man and woman are posing with fake glasses and moustaches.",
            "A dog stands on a bench in the snow.",
            "A dog stands on a bench while snow is falling.",
            "A Jack Russell Terrier stands on a bench in the snow.",
            "a white and brown dog stands on a stone wall at night whilst snow is falling.",
            "The dog stands on something while the snow falls around him.",
            "A man performs a skateboard trick in a parking lot.",
            "A person on a skateboard during a high jump.",
            "a skateboarder is airborne in front of some low brick buildings.",
            "A skateboarder is flying through the air on a red skateboard.",
            "A skateboarding soaring through the air in front of a blue building.",
            "A group of dancers wearing black spandex are dancing in a studio.",
            "A group of men and women in a dancing studio , their arms reaching to the sky .",
            "Men and women in a dance class.",
            "Men and women in dark shorts and tops are stretching their hands to the ceiling.",
            "Several women and men are lifting their arms straight up above their heads. a couple each wearing a silly mask and holding a fake cigar. A man and a woman are dressed as Groucho Marx with fake cigars. a man and a woman dress up A man and a woman wear funny masks and pretend to smoke large , fake cigars. A woman and a man pose with Groucho Marx disguises."
            ##
        ],
        # can tune these parameters
        "max_tokens": 32,
        #"top_p": 0.9,
        #"temperature": 0.7,
        "top_p": 1.,
        "temperature": 0.4,
        "echo": True,

        # added in seed to specify the seed to use for generation
        # NOTE: this only works if we modify interactive_hosted.py
        # "seed": 6,
        "mode": "train",
    }
    request_object = get_request_object(request_dict, bpe)
    return request_object, None


def get_request_object_from_queue_test_finetune():
    if torch.distributed.get_rank() == 0:
        #request_object, queue_ret_hook = batching_queue_loop()
        request_object, queue_ret_hook = get_headless_test_data2()

        print(f"{get_my_ip()}:{port} broadcast request_objects to all ranks")
        dist_utils.broadcast_object(
            request_object, src_rank=0, group=dist_utils.get_global_group()
        )
        return request_object, queue_ret_hook
    else:
        request_object = dist_utils.broadcast_object(
            None, src_rank=0, group=dist_utils.get_global_group()
        )
        print(f"{get_my_ip()}:{port} get {request_object}")
        return request_object, None

def get_request_object_from_queue_test():
    if torch.distributed.get_rank() == 0:
        #request_object, queue_ret_hook = batching_queue_loop()
        request_object, queue_ret_hook = get_headless_test_data()

        print(f"{get_my_ip()}:{port} broadcast request_objects to all ranks")
        dist_utils.broadcast_object(
            request_object, src_rank=0, group=dist_utils.get_global_group()
        )
        return request_object, queue_ret_hook
    else:
        request_object = dist_utils.broadcast_object(
            None, src_rank=0, group=dist_utils.get_global_group()
        )
        print(f"{get_my_ip()}:{port} get {request_object}")
        return request_object, None


def get_request_object_from_queue():
    if torch.distributed.get_rank() == 0:
        request_object, queue_ret_hook = batching_queue_loop()
        #request_object, queue_ret_hook = get_headless_test_data()

        print(f"{get_my_ip()}:{port} broadcast request_objects to all ranks")
        dist_utils.broadcast_object(
            request_object, src_rank=0, group=dist_utils.get_global_group()
        )
        return request_object, queue_ret_hook
    else:
        request_object = dist_utils.broadcast_object(
            None, src_rank=0, group=dist_utils.get_global_group()
        )
        print(f"{get_my_ip()}:{port} get {request_object}")
        return request_object, None


def worker_main(cfg1: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    global bpe
    global MODE
    
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    MODE = "worker"
    cfg = cfg1
   
    #########################
    ###### load from a checkpoint #######
    #ckpt_states, ref_models = load_ckpt_states_and_models(cfg)
    #generator = GeneratorInterface(cfg)
    #models = generator.load_checkpoints(ckpt_states)
    #generator.copy_model(ref_models, models, ckpt_states)

    ckpt_states, _ = load_ckpt_states_and_models(cfg, reference_model=False)
    del ckpt_states[0]["model"] # save memory

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841
    trainerIF = TrainerInterface(cfg)
    trainerIF.load_model()

    #models = [trainerIF.trainer.model, ]
    #import pdb; pdb.set_trace()

    #########################    

    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
    logger.info(cpu_memory_stats_str())
    task = tasks.setup_task(cfg.task)
    bpe = task.build_bpe(cfg.bpe)

    logger.info(models)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(models.__class__.__name__))

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = dist_utils.broadcast_object(
        None, src_rank=0, group=dist_utils.get_global_group()
    )
    #print(request_object)
    #################################################
    ## start flask server in a nonblocking thread on rank 0
    if torch.distributed.get_rank() == 0:
        thread = threading.Thread(target=lambda:app.run(host="0.0.0.0", port=port, threaded=True), daemon=True)
        thread.start()
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
    logger.info(cpu_memory_stats_str())
    #################################################
    ########### main loop ###############
    while True:
        #request_object, queue_ret_hook = get_request_object_from_queue()
        request_object, queue_ret_hook = get_request_object_from_queue_test_finetune()

        mode = request_object.pop('mode')
        if mode == 'train':
            #logger.info("call training")
            #train_ret = trainerIF.train(models, **request_object)
            #logger.info("copy trainer weights to the generator")
            #generator.copy_model_from_trainer(trainerIF, models)
            #logger.info("trainer reset optimizer")
            #trainerIF.reset_optim_from_checkpoint(models, ckpt_states)
            #train_ret = generator.generate(models, **request_object)
            #logger.info(f"fine-tune results: {train_ret}")
            #logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            #logger.info(cpu_memory_stats_str())
            #return_queue_results(queue_ret_hook, train_ret)
            
            ############## test debug ###############
            request_object, queue_ret_hook = get_request_object_from_queue_test()
            mode = request_object.pop('mode')
            generations = generator.generate(models, **request_object)
            logger.info(f"initial results: {generations}")
            request_object, queue_ret_hook = get_request_object_from_queue_test_finetune()
            mode = request_object.pop('mode')
            logger.info("call training")
            for i in range(2): 
                train_ret = trainerIF.train(models, **request_object)
                logger.info("copy trainer weights to the generator")
                generator.copy_model_from_trainer(trainerIF, models)
                logger.info("trainer reset optimizer")
                trainerIF.reset_optim_from_checkpoint(models, ckpt_states)
                train_ret = generator.generate(models, **request_object)
                logger.info(f"fine-tune results: {train_ret}")
                logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

            request_object, queue_ret_hook = get_request_object_from_queue_test()
            mode = request_object.pop('mode')
            generations = generator.generate(models, **request_object)
            logger.info(f" after fine-tune results: {generations}")

            #logger.info("reset back to checkpoint")
            #trainerIF.load_model()
            #generator.copy_model_from_trainer(trainerIF, models)
            #logger.info("reset everything")
            #generations = generator.generate(models, **request_object)
            #logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            #logger.info(f"results: {generations}")
            return_queue_results(queue_ret_hook, generations)
            #######################################
        elif mode == 'generation':
            logger.info("call generation")
            generations = generator.generate(models, **request_object)
            logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            logger.info(cpu_memory_stats_str())
            logger.info(f"results: {generations}")
            return_queue_results(queue_ret_hook, generations)
        elif mode == 'reset_to_checkpoint':
            logger.info("reset back to checkpoint")
            trainerIF.load_model()
            generator.copy_model_from_trainer(trainerIF, models)
            logger.info("reset everything")
            generations = generator.generate(models, **request_object)
            logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())
            logger.info(cpu_memory_stats_str())
            logger.info(f"results: {generations}")
            return_queue_results(queue_ret_hook, generations)


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
    response = jsonify({"error": error_dict})
    response.status = http_code
    return response


@app.route("/completions", methods=["POST"])
@app.route("/v1/engines/<engine>/completions", methods=["POST"])
@app.route("/v2/engines/<engine>/completions", methods=["POST"])
@app.route("/engines/<engine>/completions", methods=["POST"])
def completions(engine=None):
    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of ints. Pretokenized. Return one generation
    # - list of str. Multiple generations, one per prompt
    # - list of list of ints. Pretokenized multiple generations.

    # our approach is to turn everything into the last case
    prompts, generation_args = parse_request_json_dict(request.json)

    logger.info("put prompts into BATCH_QUEUE")
    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args.get("max_tokens", 0)
        BATCH_QUEUE.put(WorkItem(len(prompt) + max_len, i, ret_queue, request_object))

    logger.info("waiting to get results from ret_queue")
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for prompt, (_, generations) in zip(prompts, reordered):
        results += generations
    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    fn = pkg_resources.resource_filename("metaseq", "service/index.html")
    with open(fn) as f:
        return f.read()


def _copy_checkpoint_cache():
    if CHECKPOINT_LOCAL == CHECKPOINT_FOLDER:
        # user didn't have a local SSD
        return
    if os.path.exists(os.path.dirname(CHECKPOINT_LOCAL)):
        logger.info("Local checkpoint copy already exists, skipping copy")
    else:
        logger.info(
            f"Making a local copy of the checkpoint. {CHECKPOINT_FOLDER} -> {CHECKPOINT_LOCAL}"
        )
        shutil.copytree(CHECKPOINT_FOLDER, os.path.dirname(CHECKPOINT_LOCAL))


def cpu_memory_stats_str():
    sys_mem = psutil.virtual_memory()
    proc_mem = psutil.Process().memory_info()
    return f"psutil stats: mem_used {bytes2human(proc_mem.vms)}, free_mem {bytes2human(sys_mem.available)}, mem_pct {sys_mem.percent}, mem_used_rss {bytes2human(proc_mem.rss)}"


def cli_main():
    global port, MODE, cfg
    #"""
    #Hosted version of the web UI for generation.
    #"""
    #LAUNCH_ARGS[1] = '--model-parallel-size 2'
    #LAUNCH_ARGS[2] = '--distributed-world-size 2'
    #LAUNCH_ARGS[9] = '--path /scratch/ssd001/home/jba/OPT/1B/reshard_no_os/reshard.pt'

    ##_copy_checkpoint_cache()

    #parser = options.get_generation_parser()

    ## dumb defaults overriding
    #parser.set_defaults(lr_scheduler=None, criterion=None)
    #flat_launch_args = []
    #for s in LAUNCH_ARGS:
    #    flat_launch_args += s.split()
    #args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    #args.data = os.path.dirname(args.path)  # hardcode the data arg
    #port = DEFAULT_PORT
    #cfg = convert_namespace_to_omegaconf(args)
    ##print(cfg.checkpoint)
    ##cfg.checkpoint.checkpoint_shard_count = 3
    ##cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE
    ##dist_utils.call_main(cfg, worker_main, namespace_args=args)

    parser = options.get_training_parser()
    #args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    ##########################
    #SYSTEM_ARGS = [
    #    " /scratch/ssd001/home/jba/OPT/30B-mp8-shard2/reshard_no_os/reshard.pt",
    #    "--num-shards 2",
    #    "--num-workers 8",
    #    #"--num-workers-valid 0", 
    #    "--model-parallel-size 8 --distributed-world-size 16 --distributed-port 13000",
    #    "--merges-filename /scratch/ssd001/home/jba/OPT/gpt2-merges.txt",
    #    "--vocab-filename /scratch/ssd001/home/jba/OPT/gpt2-vocab.json",
    #    "--finetune-from-model /scratch/ssd001/home/jba/OPT/30B-mp8-shard2/reshard_no_os/reshard.pt",
    #    #"--reset-dataloader", 
    #    #"reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False",
    #]
    #MODEL_ARGS = [
    #    "--task streaming_online_language_modeling",
    #    "--arch transformer_lm_megatron",
    #    "--activation-fn relu",
    #    "--memory-efficient-fp16 --fp16-init-scale 4",
    #    "--ddp-backend fully_sharded --no-reshard-after-forward",
    #    "--use-sharded-state",
    #    "--criterion vocab_parallel_cross_entropy",
    #    "--share-decoder-input-output-embed",
    #    "--decoder-layers 48 --decoder-embed-dim 7168",
    #    "--decoder-ffn-embed-dim 28672 --decoder-attention-heads 56",
    #    "--decoder-learned-pos --no-scale-embedding",
    #    "--tokens-per-sample 2048",
    #    "--sample-break-mode none", #one sample per sequence?
    #]
    SYSTEM_ARGS = [
        " /scratch/ssd001/home/jba/OPT/6B-mp4/reshard_no_os/reshard.pt",
        "--num-workers 4",
        #"--num-workers-valid 0", 
        "--model-parallel-size 4 --distributed-world-size 4 --distributed-port 13000",
        "--merges-filename /scratch/ssd001/home/jba/OPT/gpt2-merges.txt",
        "--vocab-filename /scratch/ssd001/home/jba/OPT/gpt2-vocab.json",
        "--finetune-from-model /scratch/ssd001/home/jba/OPT/6B-mp4/reshard_no_os/reshard.pt",
        #"--reset-dataloader", 
        #"reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False",
    ]
    MODEL_ARGS = [
        "--task streaming_online_language_modeling",
        "--arch transformer_lm_megatron",
        "--activation-fn relu",
        "--memory-efficient-fp16 --fp16-init-scale 4",
        "--ddp-backend fully_sharded --no-reshard-after-forward",
        "--use-sharded-state",
        "--criterion vocab_parallel_cross_entropy",
        "--share-decoder-input-output-embed",
        "--decoder-layers 32 --decoder-embed-dim 4096",
        "--decoder-ffn-embed-dim 16384 --decoder-attention-heads 32",
        "--decoder-learned-pos --no-scale-embedding",
        "--tokens-per-sample 2048",
        "--sample-break-mode none", #one sample per sequence?
    ]
    #SYSTEM_ARGS = [
    #    " /scratch/ssd001/home/jba/OPT/1B/reshard_no_os/reshard.pt",
    #    "--num-workers 2",
    #    #"--num-workers-valid 0", 
    #    "--model-parallel-size 2 --distributed-world-size 2 --distributed-port 13000",
    #    "--merges-filename /scratch/ssd001/home/jba/OPT/gpt2-merges.txt",
    #    "--vocab-filename /scratch/ssd001/home/jba/OPT/gpt2-vocab.json",
    #    "--finetune-from-model /scratch/ssd001/home/jba/OPT/1B/reshard_no_os/reshard.pt",
    #    #"--reset-dataloader", 
    #    #"reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False",
    #]
    #MODEL_ARGS = [
    #    "--task streaming_online_language_modeling",
    #    "--arch transformer_lm_megatron",
    #    "--activation-fn relu",
    #    "--memory-efficient-fp16 --fp16-init-scale 4",
    #    "--ddp-backend fully_sharded --no-reshard-after-forward",
    #    "--use-sharded-state",
    #    "--criterion vocab_parallel_cross_entropy",
    #    "--share-decoder-input-output-embed",
    #    "--decoder-layers 24 --decoder-embed-dim 2048",
    #    "--decoder-ffn-embed-dim 8192 --decoder-attention-heads 32",
    #    "--decoder-learned-pos --no-scale-embedding",
    #    "--tokens-per-sample 2048",
    #    "--sample-break-mode none", #one sample per sequence?
    #]
    CHECKPOINT_ARGS = [
        "--checkpoint-activations", # off-load to cpu or something
        "--distribute-checkpointed-activations",
        "--tensor-parallel-init-model-on-gpu",
        "--full-megatron-init --megatron-init-sigma 0.006",
        "--save-interval-updates 2000",
        "--save-interval 5", ## if we set no last checkpoints, never save the model
        "--no-epoch-checkpoints --no-last-checkpoints --no-best-checkpoints",
        "--save-dir /h/jba/tmp/test_v0.bm_none.gpt2.me_fp16",
    ]
    TRAINING_ARGS = [
        '--max-epoch 3',
        #'--max-epoch 2',
        '--optimizer adam --adam-betas (0.9,0.95)',
        "--adam-eps 1e-08",
        "--clip-norm 1.0 --clip-norm-type l2",
        "--lr-scheduler polynomial_decay ",
        "--lr 0.00006 --end-learning-rate 5.9999999999999995e-05", 
        #"--warmup-updates 715 ",
        "--warmup-updates 0",
        "--total-num-update 572204 ",
        "--dropout 0.1 --attention-dropout 0.1 ",
        "--no-emb-dropout --weight-decay 0.1 ",
        "--batch-size 16", 
        #"--batch-size 64", 
        "--update-freq 1 --max-update 572204 --seed 1",
        "--log-format json --log-interval 1 ",
        "--required-batch-size-multiple 1",
    ]
    #TRAINING_ARGS = [
    #    '--max-epoch 2',
    #    '--optimizer adam --adam-betas (0.9,0.95)',
    #    "--adam-eps 1e-08",
    #    "--clip-norm 1.0 --clip-norm-type l2",
    #    "--lr-scheduler polynomial_decay ",
    #    "--lr 0.00006 --end-learning-rate 5.9999999999999995e-05", 
    #    #"--warmup-updates 715 ",
    #    "--warmup-updates 0",
    #    "--total-num-update 572204 ",
    #    "--dropout 0.1 --attention-dropout 0.1 ",
    #    "--no-emb-dropout --weight-decay 0.1 ",
    #    "--batch-size 16", 
    #    #"--batch-size 64", 
    #    "--update-freq 1 --max-update 572204 --seed 1",
    #    "--log-format json --log-interval 1 ",
    #    "--required-batch-size-multiple 1",
    #]
    LAUNCH_ARGS = SYSTEM_ARGS + MODEL_ARGS + CHECKPOINT_ARGS + TRAINING_ARGS
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
        #flat_launch_args += s.split()
    # args = options.parse_args_and_arch(parser, modify_parser=None)
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    ##########################

    # arg parsing to get metaseq cfg
    cfg = convert_namespace_to_omegaconf(args)

    # hack: setup eval and bpe encoder/decoder flags manually. TO-DO how to add?
    # maybe maybe options.get_training_parser ??
    cfg.common_eval = {
              '_name': None, 
              'path': cfg.checkpoint.finetune_from_model, 
              'quiet': False, 'model_overrides': '{}', 'results_path': None}
    cfg.bpe = {'_name': 'hf_byte_bpe', 
           'bpe_merges': cfg.model.merges_filename, 
           'bpe_vocab': cfg.model.vocab_filename, 
           'bpe_add_prefix_space': False}
    
    dist_utils.call_main(cfg, worker_main)


if __name__ == "__main__":
    cli_main()
