from contextlib import contextmanager
from functools import partial
import logging

from einops import rearrange
import torch

from megatron.mpu.mappings import gather_from_tensor_model_parallel_region
from megatron.mpu import ColumnParallelLinear, RowParallelLinear
from metaseq.modules.layer_norm import FusedLayerNorm
from metaseq.modules.dropout import Dropout
from metaseq.distributed import utils as distributed_utils
from metaseq.model_parallel.modules.transformer_layer import (
    ModelParallelTransformerDecoderLayer,
)
from metaseq.model_parallel.modules.multihead_attention import (
    ModelParallelMultiheadAttention,
)
from metaseq.model_parallel.models.transformer import (
    ModelParallelTransformerDecoder,
)
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer,
    OPTAttention,
    OPTLearnedPositionalEmbedding,
)


logger = logging.getLogger(__name__)


@contextmanager
def apply_forward_hook(model, hook_dict):
    """
    Hook dict should be names of modules keyed by functions all hooks must
    have the actual signature as the register forward hook in pytorch
    """
    all_hooks = []

    for n, m in model.named_modules():
        if n in hook_dict:
            all_hooks.append(m.register_forward_hook(hook_dict[n]))

    try:
        yield

    finally:
        for h in all_hooks:
            h.remove()

        all_hooks.clear()


def get_activation_capture_hook_dict(model, desired_module_activations, aux=None, model_type="opt"):
    """
    Attach the specified hook forward-pass hook functions onto the given
    model. The model types are one of [opt, hf]
    """
    activation_dict, hook_dict = {}, {}

    desired_module_activations = set(desired_module_activations)

    for n, m in model.named_modules():
        if n in desired_module_activations:
            if model_type == "opt":
                hook_dict[n] = partial(opt_forward_hook_fn, n, activation_dict, aux)

            elif model_type == "hf":
                hook_dict[n] = partial(hf_forward_hook_fn, n, activation_dict, aux)

    return hook_dict, activation_dict


def opt_forward_hook_fn(registered_name, save_dict, aux, m, _, outputs):
    # NOTE: Don't consider inputs, since there can be arbitrary code between
    #       module calls
    type_m = type(m)

    # every rank needs to do this
    if type_m == ColumnParallelLinear:
        if not m.gather_output:
            outputs = (
                gather_from_tensor_model_parallel_region(outputs[0]),
                *outputs[1:],
            )
    elif type_m == Dropout:
        if "self_attn" in registered_name:
            # NOTE: Newest megatron supports both first and last dim
            # megatron only gathers the last dim, but we want to
            # gather on the first dim so we permute it to the end
            # and then permute it back
            if not aux:
                logger.info(
                    ("Rank {}: Required auxillary data for self-attention maps"
                    "is not present").format(torch.distributed.get_rank()))

            outputs = gather_from_tensor_model_parallel_region(
                rearrange(outputs, "(b k) s1 s2 -> s1 s2 b k", b=aux[0]))

    elif type_m == torch.nn.Linear:
        outputs = gather_from_tensor_model_parallel_region(outputs)

    # only rank 0 needs to do the rest
    if torch.distributed.get_rank() != 0:
        return

    # too scared to do isinstance checks
    if type_m == ColumnParallelLinear:

        output = outputs[0]

        if m.skip_bias_add:
            output = output + outputs[1]

        layer_type = registered_name.split(".")[-1]

        # not always S x B x D, it's only when it's used in qkv_proj
        # NOTE: OPT's qkv_proj combined projection is hard to compare with HF
        if layer_type == "qkv_proj":
            output = rearrange(output, "s b d -> b s d")

        elif layer_type in ["q_proj", "k_proj", "v_proj"]:
            output = rearrange(output, "s b d -> b s d")

        elif "fc" in layer_type:
            output = rearrange(output, "(s b) d -> b s d", b=aux[0])

    elif type_m == RowParallelLinear:

        output = outputs[0]

        if m.skip_bias_add:
            output = output + outputs[1]

        layer_type = registered_name.split(".")[-1]

        # not always S x B x D, it's only when it's used in outproj
        if layer_type == "out_proj":
            output = rearrange(output, "s b d -> b s d")

        elif "fc" in layer_type:
            output = rearrange(output, "(s b) d -> b s d", b=aux[0])

    elif type_m in (
        ModelParallelTransformerDecoder,
        ModelParallelTransformerDecoderLayer,
    ):
        # the rest are aux info
        output = outputs[0]

    elif type_m == ModelParallelMultiheadAttention:
        # the other param is just None
        output, attn_bias = outputs[0]

        if attn_bias is not None:
            output = output + attn_bias

    elif type_m == Dropout:
        output = outputs

        if "self_attn" not in registered_name:
            output = rearrange(output, "s b d -> b s d")

        else:
            output = rearrange(output, "s1 s2 b k -> b k s1 s2")

    else:
        # VocabParallelEmbedding and final output_projection case
        output = outputs

    # some layers are always in S x B x D, permute them back
    if type_m in (
        ModelParallelTransformerDecoderLayer,
        FusedLayerNorm,
        ModelParallelMultiheadAttention,
    ):
        output = rearrange(output, "s b d -> b s d")

    save_dict[registered_name] = output.detach().cpu()


def hf_forward_hook_fn(registered_name, save_dict, aux, m, _, outputs):
    """
    Forward hook function for HuggingFace OPT model, conditioning on
    (sub)module types. This function should not be used outside of testing.
    """
    type_m = type(m)
    layer_type = registered_name.split(".")[-1] # In the case of duplicate types

    if type_m == torch.nn.Embedding or type_m == OPTLearnedPositionalEmbedding:
        output = outputs

    elif type_m == OPTAttention:
        output = outputs[0] # (attn_out, attn_weights_reshaped, past_key_values)

    elif type_m == torch.nn.LayerNorm:
        # Having "layers" in the name means m is a per-module layer norm
        if layer_type == "final_layer_norm" and "layers" in registered_name:
            output = rearrange(outputs, "(b s) h -> b s h", b=aux[0])

        else:
            output = outputs

    elif type_m == torch.nn.Linear:
        if layer_type in ["q_proj", "k_proj", "v_proj"]:
            output = outputs

        else:
            output = rearrange(outputs, "(b s) h -> b s h", b=aux[0])

    save_dict[registered_name] = output.detach().cpu()
