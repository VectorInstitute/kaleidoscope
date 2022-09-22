# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict

# args that cannot be batched unless the value is the same,
# otherwise it wrong
# TODO: if the key doesn't exist it defaults to None
# some stuff like echo should default to False
UNBATCHED_ARG_DICT = OrderedDict([
    ["temperature", 1.0],
    ["top_p", 1.0],
    ["n", 1],
    ["best_of", None],
    # how many of the top logprobs do we want
    ["logprobs", 0],
    ["stop", None],
    ["echo", False],
    # tuple/list of things
    ["desired_module_activations", tuple()],
])

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
MAX_BEAM = 16

DEFAULT_PORT = 6969


# CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the
# checkpoints from S3 are stored. As an example:
# CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
# $ ls /example/175B/reshard_no_os
# reshard-model_part-0.pt
# reshard-model_part-1.pt
# reshard-model_part-2.pt
# reshard-model_part-3.pt
# reshard-model_part-4.pt
# reshard-model_part-5.pt
# reshard-model_part-6.pt
# reshard-model_part-7.pt

# TODO: change this for each model
MODEL_PARALLEL = 2
TOTAL_WORLD_SIZE = 2

CHECKPOINT_FOLDER = "/checkpoint/opt_test/original/OPT-6.7B"
#CHECKPOINT_FOLDER = "/checkpoint/opt_test/original/OPT-125M"

###

# tokenizer files
BPE_MERGES = "/scratch/ssd002/projects/opt_test/gpt2-merges.txt"
BPE_VOCAB = "/scratch/ssd002/projects/opt_test/gpt2-vocab.json"
# MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")

# MEGATRON stuff
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "megatronreshard.pt")


LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {CHECKPOINT_FOLDER}/megatronreshard.pt",
    "--beam 1 --nbest 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
