#!/usr/bin/env python3
import os
from metaseq import checkpoint_utils, tasks, utils
from transformers import OPTForCausalLM
from packaging import version
import torch
import unittest
import torch.nn.functional as F
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.distributed import utils as distributed_utils
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.hub_utils import tensorize_input, get_next_token, setup_vocab_and_merges


prompts = [
    "Today is a beautiful day and I want to",
    "In the city of",
    "Paris is the capital of France and",
    "Computers and mobile phones have taken",
]


def load_mp_model_and_run_eval(cfg: MetaseqConfig, **kwargs):
    vocab_file, merges_file, tokenizer = setup_vocab_and_merges(kwargs["model_path"])
    orig_dims = []

    prompt_ids = []
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        # Pad sequence to length 32 to avoid Megatron assertion errors
        orig_dims.append(input_ids.shape[1])
        input_ids = F.pad(
            input=input_ids, pad=(0, 32 - input_ids.shape[1], 0, 0), value=1
        )
        prompt_ids.append(input_ids)

    prompt_ids = torch.cat(prompt_ids).cuda()

    task = tasks.setup_task(cfg.task)

    def _build_model(cfg, task):
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).cuda()
        return model

    models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=None,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=True,
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        build_model_hook=_build_model,
    )
    model = models[0]

    model = model.eval()

    with torch.no_grad():
        logits = model(prompt_ids)[0]

    gathered_logits = [
        torch.zeros_like(logits)
        for _ in range(distributed_utils.get_model_parallel_world_size())
    ]
    torch.distributed.all_gather(
        gathered_logits, logits, group=distributed_utils.get_global_group()
    )
    gathered_logits = torch.cat(gathered_logits, dim=2)

    # Unwrap gathered logits into separate components for each prompt, and
    # trim them to match orig_dims
    trimmed_logits = [
        logits[:orig_dim].unsqueeze(0)
        for logits, orig_dim in zip(gathered_logits, orig_dims)
    ]
    return trimmed_logits


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
@unittest.skipIf(
    version.parse(torch.__version__) < version.parse("1.9.1"),
    "test requires a pytorch version of at least 1.9.1",
)
class TestMegatronCompatability(unittest.TestCase):
    def test_model_parallel_metaseq_hf_compatibility(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")
        vocab_file, merges_file, tokenizer = setup_vocab_and_merges(model_path)

        cfg = create_generation_config_with_defaults(model_path, megatron=True)
        mp_logits_list = distributed_utils.call_main(
            cfg, load_mp_model_and_run_eval, model_path=model_path
        )

        hf_model = OPTForCausalLM.from_pretrained(model_path)
        hf_logits_list = []

        for prompt in prompts:
            input_ids = tensorize_input(tokenizer, prompt)
            with torch.no_grad():
                logits_hf = hf_model(input_ids)[0]

            hf_logits_list.append(logits_hf)

        self.assertTrue(len(hf_logits_list) == len(mp_logits_list))

        for hf_logit, mp_logit in zip(hf_logits_list, mp_logits_list):
            metaseq_next_token = get_next_token(mp_logit, tokenizer)
            mp_next_token = get_next_token(hf_logit, tokenizer)

            self.assertEqual(metaseq_next_token, mp_next_token)

            self.assertTrue(
                torch.allclose(
                    hf_logit.cpu().float(), mp_logit.cpu().float(), atol=1e-1
                )
            )


if __name__ == "__main__":
    unittest.main()
