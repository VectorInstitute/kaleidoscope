
#!/usr/bin/env python

## usecase:
# Same script as consolidate_fsdp_shards.py
# python -m metaseq.scripts.reshard_model_parts  --pth_prefix /checkpoint/opt_test/original/OPT-125M/reshard --save_prefix /checkpoint/opt_test/original/OPT-125M/test_reshard --new_model_part_count 4 --consolidate_partial_parts
## python -m metaseq.scripts.reshard_model_parts --pth_prefix ~/OPT/175B/reshard_no_os/reshard --save_prefix /scratch/ssd002/projects/opt_test/OPT-175B-mp32/reshard --new_model_part_count 32 --consolidate_partial_parts

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from metaseq.distributed.stitch_fsdp_ckpt import consolidate_fsdp_shards
import fire
import logging, os, sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

if __name__ == "__main__":
    # This is expected to be used before evaluation, not during training.
    fire.Fire(consolidate_fsdp_shards)
