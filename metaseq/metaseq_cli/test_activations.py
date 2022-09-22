#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from functools import partial
import logging

from accelerate import Accelerator
from einops import rearrange
from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer,
    OPTAttention,
    OPTLearnedPositionalEmbedding,
)
import torch

from opt_client import Client
from hook_utils import get_activation_capture_hook_dict, apply_forward_hook

logger = logging.getLogger(__name__)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def get_hf_logits(client, hf_model, prompts):
    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))

    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list]).cuda()
    with torch.no_grad():
        # slice out the start of seq
        logits_hf = hf_model(input_ids)[0]

    logits_hf_list = [
        logits_hf[i, 1 : len(toks), :] for i, toks in enumerate(input_ids_list)
    ]
    return logits_hf_list


def get_hf_activations(client, hf_model, prompts):
    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))

    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list]).cuda()

    act = hf_model(
        input_ids,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=True,
    )

    return act


def init_opt_hf_mappings(num_layers):
    # mappings is a dict of equivalent layer types as keys, where the values
    # are 1. A list of (sub)module names for forward hooks, or 2. A list
    # containing "custom" as the first entry, and a function which formats the
    # collection of output activations
    # NOTE: Activation function outputs cannot be compared, since OPT uses
    #       FusedLayerNorm layers
    opt_mappings = {
        "logits": ["decoder"],
        "embed_tokens": ["decoder.embed_tokens"],
        "embed_positions": ["decoder.embed_positions"],
        "transformer_layers": [f"decoder.layers.{i}" for i in range(num_layers - 1)] + ["decoder.layer_norm"],
        "attention_maps": [f"decoder.layers.{i}.self_attn.dropout_module" for i in range(num_layers)],
        "self_attention": [f"decoder.layers.{i}.self_attn" for i in range(num_layers)],
        "q_proj": [f"decoder.layers.{i}.self_attn.q_proj" for i in range(num_layers)],
        "k_proj": [f"decoder.layers.{i}.self_attn.k_proj" for i in range(num_layers)],
        "v_proj": [f"decoder.layers.{i}.self_attn.v_proj" for i in range(num_layers)],
        "self_attention_layer_norm": [f"decoder.layers.{i}.self_attn_layer_norm" for i in range(num_layers)],
        "fc1": [f"decoder.layers.{i}.fc1" for i in range(num_layers)],
        "fc2": [f"decoder.layers.{i}.fc2" for i in range(num_layers)],
        "decoder_layer_norm": [f"decoder.layers.{i}.final_layer_norm" for i in range(num_layers)],
        "output_layer_norm": ["decoder.layer_norm"],
    }
    hf_mappings = {
        "logits": [
            "custom",
            get_hf_logits,
            lambda output: [output]],
        "embed_tokens": ["model.decoder.embed_tokens"],
        "embed_positions": ["model.decoder.embed_positions"],
        "transformer_layers": [
            "custom",
            get_hf_activations,
            lambda output: output["hidden_states"][1:]],
        "attention_maps": [
            "custom",
            get_hf_activations,
            lambda output: output["attentions"]],
        "self_attention": [f"model.decoder.layers.{i}.self_attn" for i in range(num_layers)],
        "q_proj": [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(num_layers)],
        "k_proj": [f"model.decoder.layers.{i}.self_attn.k_proj" for i in range(num_layers)],
        "v_proj": [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(num_layers)],
        "self_attention_layer_norm": [f"model.decoder.layers.{i}.self_attn_layer_norm" for i in range(num_layers)],
        "fc1": [f"model.decoder.layers.{i}.fc1" for i in range(num_layers)],
        "fc2": [f"model.decoder.layers.{i}.fc2" for i in range(num_layers)],
        "decoder_layer_norm": [f"model.decoder.layers.{i}.final_layer_norm" for i in range(num_layers)],
        "output_layer_norm": [f"model.decoder.final_layer_norm"],
    }
    return opt_mappings, hf_mappings


def retrieve_opt_activations(mapping, client, prompts):
    """
    Run OPT model to get activations, given by the module names in
    mapping. Format the resulting activations to match the HF activation
    colletion.
    """
    if "custom" in mapping:
        raise NotImplementedError
    else:
        module_names = mapping

    acts = client.get_activations(prompts, module_names)

    def _format_results(activations_batched):
        result = {n: [] for n in module_names}
        for acts in activations_batched:
            for name in module_names:
                result[name].append(acts[name])

        # Can't torch.stack along new batch axis since prompt length is dynamic
        return result

    acts = _format_results(acts)
    return acts


def retrieve_hf_activations(mapping, client, model, prompts, aux):
    """
    Get the activations from a HF model. HF models can output logits, 
    attention maps, and the outputs of decoder transformer layers without
    forward hooks. However, if there are module names provided
    in the mapping, use forward hooks to retrieve the activations. Format the
    return value as specified in the mapping.
    """
    # Helper functions for hooked activations
    def _retrieve_hooked_acts(client, model, prompts, module_names):
        hook_dict, acts = get_activation_capture_hook_dict(
            model,
            module_names,
            aux=aux,
            model_type="hf",
        )

        with apply_forward_hook(model, hook_dict):
            results = get_hf_logits(client, model, prompts)

        return acts

    def _format_hooked_acts(hf_results, module_names):
        """Standard formatter for forward hooks."""
        results = []
        for name in module_names:
            results.append(hf_results[name])

        return results

    # Case 1: Use provided retrieval and formatting functions
    if "custom" in mapping:
        module_names = None
        retrieval_fn = mapping[1]
        format_fn = mapping[2]

    # Case 2: Use forward hooks with standard formatter
    else:
        module_names = mapping
        retrieval_fn = _retrieve_hooked_acts
        format_fn = _format_hooked_acts

    if not module_names:
        # HF has built-in retrieval
        acts = retrieval_fn(client, model, prompts)
        acts = format_fn(acts)

    elif module_names:
        # Need to hook onto HF model for retrieval
        acts = retrieval_fn(client, model, prompts, module_names)
        acts = format_fn(acts, module_names)

    else:
        raise Exception("No valid configuration for HF activation retrieval")

    return acts


def assert_activations_correctness(hf_results, opt_results, act_type="transformer_layer", crash_on_false=True):
    """
    Helper function taking HF and OPT activation collections, and
    makes sure they're allclose within some range
    """
    def _assert_allclose_and_get_summed_diff(hf_acts, opt_acts):
        def _get_diff(x, y):
            return (x - y).sum() / x.numel()

        hf_acts = hf_acts.cpu().float()
        opt_acts = opt_acts.cpu().float()

        diff = _get_diff(hf_acts, opt_acts)

        allclose = torch.allclose(hf_acts, opt_acts, atol=1e-1, rtol=1e-2)

        if allclose:
            return diff, allclose

        elif not allclose and crash_on_false:
            raise Exception("Large diff in {}: {}".format(act_type, diff))

        elif not allclose and not crash_on_false:
            return diff, allclose

        else:
            raise Exception("Assertion resulted in invalid args")

    total_diff = 0.0
    allclose_fails = []
    # Of shape (layer_idx, batch_idx)
    for layer_idx, ((module_name, opt_acts), hf_acts) in enumerate(zip(opt_results.items(), hf_results)):
        for batch_idx, (opt_act, hf_act) in enumerate(zip(opt_acts, hf_acts)):

            opt_act = opt_act.detach().cpu().float()
            hf_act = hf_act.detach().cpu().float()

            bound = slice(1, opt_act.shape[-2] + 1)  # Trim start token and padding

            # NOTE: Need to dynamically trim HF returned activations per
            #       example rather than batch. HF pads until max sequence length.
            if act_type in [
                    "embed_tokens",
                    "embed_positions",
                    "transformer_layers",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "self_attention",
                    "self_attention_layer_norm",
                    "fc1",
                    "fc2",
                    "decoder_layer_norm",
                    "output_layer_norm",
            ]:
                hf_act = hf_act[bound]

            elif act_type == "attention_maps":
                hf_act = hf_act[:, bound, bound]

            # TODO: This is not comparable to HF q, k, v activations. qkv need
            #       to be split in OPT
            elif act_type == "qkv_proj":
                raise NotImplementedError

            elif act_type == "logits":  # Logits has a custom retrieval
                pass

            else:
                raise NotImplementedError

            diff, allclose = _assert_allclose_and_get_summed_diff(hf_act, opt_act)

            if not allclose:
                allclose_fails.append(f"{act_type} | layer {layer_idx} | batch {batch_idx}")

            total_diff += diff

    print("{} has average diff: {}".format(act_type, total_diff))

    return total_diff, allclose_fails


def main(args):
    # Make client connection
    client = Client(args.host, args.port)

    # Create huggingface model
    accelerator = Accelerator()
    hf_model, output_loading_info = OPTForCausalLM.from_pretrained(
        "facebook/opt-6.7b",
        cache_dir="/checkpoint/opt_test/original/OPT-6.7B",
        output_loading_info=True,
        low_cpu_mem_usage=True, # Prevents random init of params before load
        #torch_dtype=torch.float32, # float32 gives better acc, but T4 can't load
    )
    hf_model.eval()
    assert not sum(list(output_loading_info.values()), []) # No keys randomly init
    hf_model = accelerator.prepare(hf_model)

    prompts = [
        "vector matrix",
        "nice working with you all :)",
        "Today is a beautiful day and I want to",
        "what is the meaning of life?",
    ]
    batch_size = len(prompts)

    # Initialize OPT-HF layer mappings
    opt_mappings, hf_mappings = init_opt_hf_mappings(len(hf_model.model.decoder.layers))

    # For each mapping, retrieve + format activations and compare using allclose
    diff = {}
    fails = []
    for (opt_type, opt_map), (hf_type, hf_map) in zip(opt_mappings.items(), hf_mappings.items()):
        assert opt_type == hf_type

        opt_acts = retrieve_opt_activations(opt_map, client, prompts)
        hf_acts = retrieve_hf_activations(hf_map, client, hf_model, prompts, aux=(batch_size,))

        diff[opt_type], allclose_fails = assert_activations_correctness(hf_acts, opt_acts, act_type=opt_type, crash_on_false=False)
        fails.append(allclose_fails)

    for mapping_allclose_fails in fails:
        if mapping_allclose_fails:
            print(mapping_allclose_fails)


if __name__ == "__main__":
    main(prepare_args())
