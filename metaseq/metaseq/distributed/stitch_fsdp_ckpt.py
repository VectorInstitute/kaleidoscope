# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import re
import time
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
from typing import(
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
from tqdm import tqdm

from metaseq.distributed.fully_sharded_data_parallel import FSDP as FSDP
from metaseq.file_io import load_and_pop_last_optimizer_state

logger = logging.getLogger(__name__)


def _get_shard_number(x) -> int:
    match = re.search(r"shard(\d+).pt", x)
    if match is None:
        #raise AssertionError(f"{x} did not match shard(\\d+).pt")
        print(f"{x} did not match shard(\\d+).pt")
        return 0
    else:
        return int(match.groups()[0])


def consolidate_fsdp_shards(
    pth_prefix: str,
    save_prefix=None,
    strict=False,
    new_arch_name=None,
    no_stitch_megatron=False,
    megatron_part=None,
    new_model_part_count=None,
    consolidate_partial_parts=False,
) -> str:
    logger.info(f"Entering consolidate_fsdp_shards with params: {locals()}")
    if pth_prefix.endswith(".pt"):
        pth_prefix = pth_prefix[:-3]
    if save_prefix is None:
        save_prefix = pth_prefix + "_consolidated"  # .pt'

    all_ckpt_files = list(
        sorted(glob(f"{pth_prefix}*shard*.pt"), key=_get_shard_number)
    )
    if not all_ckpt_files:
        all_ckpt_files = list(
            sorted(glob(f"{pth_prefix}*part*.pt"), key=_get_shard_number)
        )


    if megatron_part is not None:
        no_stitch_megatron = True
        all_ckpt_files = [
            x for x in all_ckpt_files if (
                f"model_part-{megatron_part}" in x
                or f"model_part-{megatron_part + 1}" in x
            )
        ]

    ## NEW ##
    saved_paths = []
    parts_index_base = 0
    parts_all_ckpt_files = [all_ckpt_files,]
    if consolidate_partial_parts:
        parts_index_base = new_model_part_count // len(all_ckpt_files)
        new_model_part_count = parts_index_base
        parts_all_ckpt_files = [[i, ] for i in all_ckpt_files]
    for all_ckpt_files in parts_all_ckpt_files:
        min_part_num = find_min_num_parts(all_ckpt_files)
        print(all_ckpt_files)
        assert all_ckpt_files, f"no paths matched {pth_prefix}*shard*.pt"
        print("################## arguments to this function ###############")
        #no_stitch_megatron = True
        print(f"save_prefix: {save_prefix}, strict: {strict}, no_stitch_megatron: {no_stitch_megatron}, megatron_part: {megatron_part}")
        print("################## loading up custom metadata ###############")
        import pickle, os
        metadata_filename = os.path.dirname(pth_prefix) + '/shard_metadata.pkl'
        with open(metadata_filename, 'rb') as f:
            custom_metadata = pickle.load(f)
        print(custom_metadata)
        print("################## done loading ###############")
        weights = []
        metadata = []
        expert_paths = []
        expert_dest_paths = []
        expert_ranks = []
        names = []
        dense = True
        t0 = time.time()
        logger.info(f"loading checkpoints: {all_ckpt_files}")
        for p in tqdm(all_ckpt_files):
            names.append(Path(p).name)
            if re.search(r"rank-(\d+)", os.path.basename(p)):  # expert checkpoint
                expert_paths.append(p)
                r = re.search(r"rank-(\d+)", os.path.basename(p)).groups()[0]
                assert r not in expert_ranks
                expert_ranks.append(r)
                expert_dest_paths.append(f"{save_prefix}-rank-{r}.pt")
            else:
                ckpt = load_and_pop_last_optimizer_state(p)
                ckpt["shard_metadata"] = custom_metadata
                logger.info("model_name: {}".format(ckpt["cfg"]["model"]._name))
                weights.append(ckpt["model"])
                metadata.append(ckpt["shard_metadata"])
        assert weights, f"all files were considered experts: {all_ckpt_files}"
        logger.info(f"done loading checkpoints")

        do_consolidate = True
        if "decoder.embed_tokens.weight" in weights[0].keys():
            shape = weights[0]["decoder.embed_tokens.weight"].shape
            logger.info(
                f"This ckpt does not seem sharded. I see unflat params! like "
                f"decoder.embed_tokens.weight shaped {shape}. Will just copy files "
                f"and remove optim_state."
            )
            do_consolidate = False

        logger.info(f"do_consolidate: {do_consolidate}")
        print(weights[0])

        if do_consolidate:
            num_parts = find_num_parts(names)
            if num_parts:
                if new_model_part_count is None:
                    new_model_part_count = 1
                if consolidate_partial_parts:
                    names = [re.sub(r'\d+', "0", n) for n in names]
                    logger.info(f"rename part {min_part_num} to {names}")
                #import pdb; pdb.set_trace()
                logger.info(f"consolidate_model_parallel from {num_parts} parts to {new_model_part_count} parts.")
                #import pdb; pdb.set_trace()
                consolidated_weights, new_metadata_param_dict = consolidate_model_parallel(
                    metadata,
                    names,
                    strict,
                    weights,
                    parts=num_parts,
                    no_stitch_megatron=no_stitch_megatron,
                    new_model_part_count=new_model_part_count,
                )
            else:
                logger.info("FSDP.consolidate_shard_weights")
                consolidated_weights = FSDP.consolidate_shard_weights(
                    shard_weights=weights, shard_metadata=metadata, strict=strict
                )
            del weights, metadata
            gc.collect()
            done_consolidate = time.time()
            logger.info(f"Done consolidating after {done_consolidate-t0//60} minutes")
        else:
            consolidated_weights = weights[0]

        #consolidated_weights = defaultdict(list)
        #consolidated_weights[0] = weights[0]
        #consolidated_weights[1] = weights[1]
        logger.info("################ consolidated weights #############")
        logger.info(consolidated_weights)
        for part_id, part_consolidated_weights in consolidated_weights.items():
            print(part_id)
        #exit(1)


        if new_arch_name is not None:
            ckpt["cfg"]["model"]._name = new_arch_name

        if not os.path.exists(os.path.dirname(save_prefix)):
            logger.info(f"make directory {os.path.dirname(save_prefix)}")
            os.mkdir(os.path.dirname(save_prefix))

        #import pdb; pdb.set_trace()
        custom_metadata["param_metadata"][0]['params']['flat_param_0'] = new_metadata_param_dict
        with open(os.path.dirname(save_prefix)+'/shard_metadata.pkl', 'wb') as f:
            pickle.dump(custom_metadata, f)

        if dense:
            logger.info("dense")

            def save_checkpoint(weights_to_save, prefix):
                ckpt_consolidated = dict(
                    model=weights_to_save,
                    cfg=ckpt["cfg"],
                    extra_state=ckpt["extra_state"],
                    optimizer_history=ckpt["optimizer_history"],
                    args=ckpt.get("args"),
                )
                save_path = f"{prefix}.pt"
                logger.info(f"Saving to {save_path} ...")
                torch.save(ckpt_consolidated, save_path)
                logger.info(f"Done after {time.time()-t0//60} minutes")
                return save_path

            if no_stitch_megatron:
                for part_id, part_consolidated_weights in consolidated_weights.items():
                    real_part_id = part_id + str(parts_index_base * min_part_num)
                    saved_paths.append(
                        save_checkpoint(
                            part_consolidated_weights, f"{save_prefix}-model_part-{real_part_id}"
                        )
                    )
                #return saved_paths
            else:
                #return save_checkpoint(consolidated_weights, save_prefix)
                saved_paths.append(save_checkpoint(consolidated_weights, save_prefix))

        del consolidated_weights
        gc.collect()

    if dense:
        return saved_paths

    ckpt_shared = dict(
        model=consolidated_weights,
        cfg=ckpt["cfg"],
        extra_state=ckpt["extra_state"],
        optimizer_history=ckpt["optimizer_history"],
        args=ckpt["args"],
    )
    logger.info("saving..")
    torch.save(ckpt_shared, f"{save_prefix}-shared.pt")
    logger.info(f"Done saving. Total time: {time.time()-t0//60} minutes,  ")
    # Process experts
    for src, dst in tqdm(
        list(zip(expert_paths, expert_dest_paths)), desc="expert files"
    ):
        ckpt = load_and_pop_last_optimizer_state(src)
        if do_consolidate:
            expert_wt = FSDP.consolidate_shard_weights(
                shard_weights=[ckpt["model"]],
                shard_metadata=[ckpt["shard_metadata"]],
                strict=False,
            )
            ckpt = dict(
                model=expert_wt,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt["args"],
            )

        torch.save(ckpt, dst)
    logger.info(f"saved consolidated MoE with prefix {save_prefix}.pt")
    return f"{save_prefix}.pt"


def consolidate_model_parallel(
    metadata, names, strict, weights, parts=2, no_stitch_megatron=False, new_model_part_count=None,
):
    model_parts = defaultdict(list)
    metadata_parts = defaultdict(list)
    for i, n in enumerate(names):
        for p in range(parts):
            if f"part-{p}" in n:
                model_parts[p].append(weights[i])
                metadata_parts[p].append(metadata[i])

    all_parts_consolidated = defaultdict(list)
    for k, v in model_parts.items():
        part_weights = FSDP.consolidate_shard_weights(
            shard_weights=v, shard_metadata=metadata_parts[k], strict=strict
        )
        all_parts_consolidated[k] = part_weights

    #if no_stitch_megatron:
    #    return all_parts_consolidated
    # glue to be a single megatron mdoel part
    models = reshard_megatron_parts(all_parts_consolidated, new_model_part_count=new_model_part_count)
    #import pdb; pdb.set_trace()
    new_metadata_param_dict = {}
    new_metadata_param_dict['names'] = metadata[0]["param_metadata"][0]['params']['flat_param_0']['names']
    new_metadata_param_dict['shapes'] = [models[0][n].shape for n in metadata[0]["param_metadata"][0]['params']['flat_param_0']['names']]
    new_metadata_param_dict['numels'] = [models[0][n].numel() for n in metadata[0]["param_metadata"][0]['params']['flat_param_0']['names']]
    new_metadata_param_dict['padding'] = metadata[0]["param_metadata"][0]['params']['flat_param_0']['padding']

    new_all_parts_consolidated = defaultdict(list)
    if new_model_part_count == 1:
        print(models)
        logger.info("############### glue megatron parts into an unflatten model ####################")
        model = models[0]
        return model, new_metadata_param_dict
    else:
        for k, v in enumerate(models):
            part_weights = flatten_unshard_weights(
                consolidated_weights=[v,], shard_metadata=metadata_parts[0], strict=strict
            )
            new_all_parts_consolidated[k] = part_weights
        logger.info("############### reshard megatron parts ####################")
        logger.info("reshard models length: {}".format(len(models)))
        #import pdb; pdb.set_trace()
        return new_all_parts_consolidated, new_metadata_param_dict


def flatten_unshard_weights(
    consolidated_weights: List[Dict[str, torch.Tensor]],
    shard_metadata: List[Dict[str, Any]],
    with_module_buffers: bool = True,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Given a list of consolidated (non-sharded) weights and meta data associated to the original N shards,
    flatten the weights so FSDP can load back up.
    WARNING: this is a very hacky script based on consolidate_shard_weights in fairsacale.
             this script pretty much just an inverse function of consolidate_shard_weights.
             we are currently ignoring
       -- padding
       -- shared_param_info
       -- and we do not update the metadata to the new shapes
    Args:
        consolidated_weights (List[Dict[str, torch.Tensor]]):
            List of dictionaries that contains sharded weights from
            each rank.
        shard_metadata (List[Dict[str, Any]]):
            List of dictionaries that contains metadata from each shard.
            See `local_metadata_dict` above.
        with_module_buffers (bool):
            If shard 0's buffer should be returned in the consolidated
            weight dict.
            Default: True.
        strict (bool):
            allow incomplete shard weights. if True, every key in the metadata must be present in the weights.
    """
    if len(consolidated_weights) != len(shard_metadata) or not len(consolidated_weights):
        raise ValueError("Require metadata for each shard and non-empty shards")

    print("############### inside FSDP consolidate #################")
    print(len(consolidated_weights))
    #print(shard_metadata[0]["param_metadata"])
    #exit(1)

    flatten_weights = {}
    original_world_size = len(consolidated_weights)

    total_sum = 0
    # For every FSDP instance.
    for fsdp_obj_idx, metadata in enumerate(shard_metadata[0]["param_metadata"]):
        fsdp_path = metadata["fsdp_path"]
        params = metadata["params"]
        # For every this-FSDP-owned param, flattened or not.
        for backing_param_name, v in params.items():
            in_state_dict_key = ".".join([fsdp_path, backing_param_name]) if fsdp_path else backing_param_name
            print(in_state_dict_key)
            #print("key not in shard_weights: {}".format(in_state_dict_key not in shard_weights[0]))
            ## Get full param back with pad removed.
            #if in_state_dict_key not in shard_weights[0] and (not strict):
            #    continue
            flatten_param_list = []
            ###### we ignore padding for now... ######
            #shards = []
            #for rank in range(original_world_size):
            #    shard = shard_weights[rank][in_state_dict_key]
            #    print("shard weights size: {}".format(shard.shape))
            #    pad = shard_metadata[rank]["param_metadata"][fsdp_obj_idx]["params"][backing_param_name]["padding"]
            #    print("param padding size: {}".format(pad))
            #    shards.append(_unpad(shard, pad))
            #    if metadata["no_broadcast_optim_state"]:
            #        break
            #full_param = torch.cat(shards, dim=0)
            #print("full_param shape: {}".format(full_param.shape))
            # (Potentially), split the full param and create original params.
            names, shapes, numels, _ = v.values()
            print("sum(numels): {}".format(sum(numels)))
            total_sum += sum(numels)
            for n, s in zip(names, shapes):
                out_state_dict_key = ".".join([fsdp_path, n]) if fsdp_path else n
                print("unflattening param name: {}, shape:{}, metadata_shape: {}".format(out_state_dict_key, consolidated_weights[0][out_state_dict_key].shape, s))
                flatten_param_list.append(consolidated_weights[0][out_state_dict_key].view(-1))

            full_param = torch.cat(flatten_param_list, dim=0)
            if sum(numels) == full_param.size(0):
                print("unflatten the same size as metadata")
            else:
                print("WARNING! unflatten size is {}, but metadata size is {}.".format(full_param.size(0), sum(numels)))
            flatten_weights[in_state_dict_key] = full_param

    print(f"total param sum: {total_sum}")
    print(metadata["shared_param_info"])
    print(shard_metadata[0]["buffer_names"])
    # copy shared parameters
    for src_path, dest_path in metadata["shared_param_info"]:
        consolidated_weights[0][dest_path] = consolidated_weights[0][src_path]

    # Deal with the buffers, which are not parameters and are not sharded by FSDP
    # and therefore are replicated among the different shards.
    # We take the values of the first shard (this assumes that there is some form
    # of synchronization between shards or that all shards buffers are equivalent).
    if with_module_buffers:
        for buffer_name in shard_metadata[0]["buffer_names"]:
            if buffer_name not in consolidated_weights[0] and (not strict):
                continue
            flatten_weights[buffer_name] = consolidated_weights[0][buffer_name]

    print('flatten_weights[buffer_name]={}'.format(flatten_weights[buffer_name]))
    return flatten_weights


def handle_qkv_proj(model_parts, key, new_model_part_count):
    parts = [model_parts[part_id][key] for part_id in range(len(model_parts))]
    ks, vs, qs = [], [], []
    for p in parts:
        k, v, q = torch.split(p, p.shape[0] // 3)
        ks.append(k)
        vs.append(v)
        qs.append(q)
    resharded_ks = torch.chunk(torch.cat(ks, dim=0), new_model_part_count)
    resharded_vs = torch.chunk(torch.cat(vs, dim=0), new_model_part_count)
    resharded_qs = torch.chunk(torch.cat(qs, dim=0), new_model_part_count)
    return resharded_ks, resharded_vs, resharded_qs


def _handle_one(parts, is_weight):
    """Make it look like a normal LayerNorm"""
    n_parts = len(parts)
    err_msg = f"Redundant ModelParallelFusedLayerNorm params have been updated."
    if is_weight:
        init = 1.0
        assert not torch.logical_and(parts[0].ne(1), parts[1].ne(1)).any(), err_msg

    else:
        init = 0.0
        assert not torch.logical_and(parts[0].ne(0), parts[1].ne(0)).any(), err_msg
    ret_val = torch.cat([p.unsqueeze(-1) for p in parts], dim=1).sum(1) - (
        init * (n_parts - 1)
    )
    return ret_val


def handle_legacy_ln(glued_model, n_parts):
    """Consolidate ffn_layernorm.lns.weight.{part_id} -> ffn_layernorm.weight"""
    if "decoder.layers.0.ffn_layernorm.lns.0.weight" not in glued_model:
        return
    n_layers = get_n_layers(glued_model)
    for i in range(n_layers):
        layer_weights = [
            glued_model.pop(f"decoder.layers.{i}.ffn_layernorm.lns.{p}.weight")
            for p in range(n_parts)
        ]
        layer_biases = [
            glued_model.pop(f"decoder.layers.{i}.ffn_layernorm.lns.{p}.bias")
            for p in range(n_parts)
        ]
        glued_model[f"decoder.layers.{i}.ffn_layernorm.weight"] = _handle_one(
            layer_weights, True
        )
        glued_model[f"decoder.layers.{i}.ffn_layernorm.bias"] = _handle_one(
            layer_biases, False
        )


def get_n_layers(glued_model):
    n_layers = 0
    while True:
        if f"decoder.layers.{n_layers}.fc1.weight" in glued_model:
            n_layers += 1
        else:
            assert (
                n_layers > 0
            ), f"found 0 layers bc no keys matching decoder.layers.0.fc1.weight"
            return n_layers


def reshard_megatron_parts(model_parts, new_model_part_count=1):
    """
    Reshard to different number of model parts.
    When new_model_part_count=1 return glued model
    """
    new_model_parts = [OrderedDict() for _ in range(new_model_part_count)]

    def assert_all_close(key):
        for part_id in range(len(model_parts)):
            if not torch.allclose(model_parts[part_id][key], model_parts[0][key]):
                err = (
                    (model_parts[part_id][key] - model_parts[0][key])
                    .float()
                    .abs()
                    .max()
                    .item()
                )
                logger.info(f"max discrepancy {key}: {err}")

    def _consolidate_and_reshard(key, dim):
        consolidated_tensor = torch.cat(
            [model_parts[part_id][key] for part_id in range(len(model_parts))],
            dim=dim,
        )
        assert consolidated_tensor.size(dim) % new_model_part_count == 0
        newly_resharded_tensors = torch.chunk(
            consolidated_tensor,
            new_model_part_count,
            dim=dim,
        )
        for i in range(new_model_part_count):
            new_model_parts[i][key] = newly_resharded_tensors[i].clone()

    def _copy_key_to_all_parts(key):
        for new_model_part in new_model_parts:
            new_model_part[key] = model_parts[0][key].clone()

    for key in model_parts[0]:
        if "qkv" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                resharded_ks, resharded_vs, resharded_qs = handle_qkv_proj(
                    model_parts, key, new_model_part_count
                )
            else:
                assert key.endswith("weight")
                resharded_ks, resharded_vs, resharded_qs = handle_qkv_proj(
                    model_parts, key, new_model_part_count
                )

            for i in range(new_model_part_count):
                new_model_parts[i][key] = torch.cat(
                     (resharded_ks[i], resharded_vs[i], resharded_qs[i]), dim=0
                )

            # Handle the special case when new_model_part_count = 1 (converting to a singleton checkpoint)
            if new_model_part_count == 1:
                new_model_parts[0][key.replace("qkv", "k")] = resharded_ks[0]
                new_model_parts[0][key.replace("qkv", "v")] = resharded_vs[0]
                new_model_parts[0][key.replace("qkv", "q")] = resharded_qs[0]
            else:
                for i in range(new_model_part_count):
                    new_model_parts[i][key] = torch.cat(
                        (resharded_ks[i], resharded_vs[i], resharded_qs[i]), dim=0
                    )

        elif "ffn_layernorm" in key:
            _consolidate_and_reshard(key, dim=0)

        elif "layer_norm" in key:
            assert_all_close(key)
            _copy_key_to_all_parts(key)

        elif "fc1" in key or "k_proj" in key or "q_proj" in key or "v_proj" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                _consolidate_and_reshard(key, dim=0)
            # weights of CP gets concatenated along dim 0
            else:
                assert key.endswith("weight")
                _consolidate_and_reshard(key, dim=0)
                # FC1 is CP

        # FC2 is RP
        elif "fc2" in key or "out_proj" in key:
            # Bias of RP gets replicated
            if key.endswith("bias"):
                assert_all_close(key)
                _copy_key_to_all_parts(key)
            # weights of RP gets concatenated along dim 1
            else:
                assert key.endswith("weight")
                _consolidate_and_reshard(key, dim=1)

        elif "embed_tokens.weight" in key:
            _consolidate_and_reshard(key, dim=0)

        elif "embed_positions" in key:
            if "_float_tensor" in key:
                # Assume embed positions are non learned ie.e sinusoidal
                for new_model_part in new_model_parts:
                    new_model_part[key] = torch.zeros([1])
            else:
                assert_all_close(key)
                _copy_key_to_all_parts(key)

        elif "version" in key:
            _copy_key_to_all_parts(key)

        else:
            assert_all_close(key)
            _copy_key_to_all_parts(key)

    for new_model_part in new_model_parts:
        assert len(new_model_part.keys()) >= len(model_parts[0].keys())
        assert "decoder.layers.0.ffn_layernorm.lns.0.weight" not in new_model_part

    return new_model_parts


def find_num_parts(names) -> int:
    parts = []
    for n in names:
        part = re.search(r"-model_part-(\d+)", n)
        if part is not None:
            parts.append(int(part.groups()[0]))
    if parts:
        return max(parts) + 1
    else:
        return 0


def find_min_num_parts(names) -> int:
    parts = []
    for n in names:
        part = re.search(r"-model_part-(\d+)", n)
        if part is not None:
            parts.append(int(part.groups()[0]))
    if parts:
        return min(parts)
    else:
        return 0
