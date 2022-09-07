# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in plaintext and performs
on-the-fly tokenization.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import II

from metaseq.data import (
    Dictionary,
    BaseDataset,
    JsonlDataset,
    PartitionedStreamingDataset,
    ResamplingDataset,
    StreamingShuffleDataset,
    StreamingTokenBlockDataset,
    StreamingSrcTgtDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    iterators,
)
from metaseq.dataclass import MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


logger = logging.getLogger(__name__)

DEFAULT_MULTICORPUS_MAX = -1


@dataclass
class StreamingOnlineLanguageModelingConfig(MetaseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory with JSONL files"}
    )
    vocab_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-vocab.json"}
    )
    merges_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-merges.txt"}
    )
    end_of_document_symbol: Optional[str] = field(
        default="</s>", metadata={"help": "symbol indicating an end-of-document"}
    )
    sample_break_mode: Optional[str] = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )
    multicorpus_sampling_alpha: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "smoothing alpha for sample rations across multiple datasets"
        },
    )
    multicorpus_sampling_maximum: Optional[float] = field(
        default=DEFAULT_MULTICORPUS_MAX,
        metadata={"help": "Maximum size for example proportional sampling"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    data_buffer_size: int = II("dataset.data_buffer_size")
    update_freq: List[int] = II("optimization.update_freq")


@register_task("streaming_online_language_modeling", dataclass=StreamingOnlineLanguageModelingConfig)
class StreamingOnlineLanguageModelingTask(LegacyTask):
    """
    Train a language model on a stream of data. Currently we assume the stream
    is in JSONL format and we tokenize inputs on-the-fly.
    Note that we append an end-of-document symbol to the end of each document.
    Args:
        tokenizer (tokenizers.ByteLevelBPETokenizer): the BPE tokenizer to use
    """

    def __init__(self, args):
        super().__init__(args)

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        self.tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename, args.merges_filename
        )

        if max(args.update_freq) > 1:
            raise NotImplementedError(
                "--update-freq is not compatible with StreamingOnlineLanguageModelingTask"
            )

        self.eod = self.tokenizer.token_to_id(args.end_of_document_symbol)
        if self.eod is None:
            # This will be executed for old models that do not have the args.end_of_document_symbol explicitly set
            # and do not use <s/> (the default) but <EOS>
            self.eod = self.tokenizer.token_to_id("<EOS>")

        assert (
            self.eod is not None
        ), "Cannot find end-of-document symbol ({}) in tokenizer".format(
            args.end_of_document_symbol
        )

        # construct a dummy metaseq Dictionary corresponding to the given tokenizer
        self.dictionary = Dictionary()
        tok_vocab_size = self.tokenizer.get_vocab_size()

        for id in range(self.dictionary.nspecial, tok_vocab_size):
            self.dictionary.add_symbol(self.tokenizer.id_to_token(id))
        final_vocab_size = args.final_vocab_size
        # final_vocab_size = 51200 for roberta dictionary
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            self.dictionary.pad_to_multiple_(final_vocab_size)
        else:
            self.dictionary.pad_to_multiple_(8)

        # confirm that metaseq dictionary and BPE have matching special symbols
        assert self.dictionary.bos_index == 0
        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.dictionary.pad_index == 1
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 2
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 3
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def _tokenize_one_json(self, json):
        text = json["text"]
        return torch.LongTensor(
            # append an end-of-document symbol after each document
            self.tokenizer.encode(text.rstrip()).ids
            + [self.eod]
        )

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by corpus. This helps small corpus by upsampling them.
        """
        if self.args.multicorpus_sampling_maximum == DEFAULT_MULTICORPUS_MAX:
            prob = dataset_lens / dataset_lens.sum()
            smoothed_prob = prob**self.args.multicorpus_sampling_alpha
            smoothed_prob = smoothed_prob / smoothed_prob.sum()
        else:
            dataset_lens = [
                min(l, self.args.multicorpus_sampling_maximum) for l in dataset_lens
            ]
            smoothed_prob = dataset_lens / sum(dataset_lens)
        return smoothed_prob

    def _alpha_sampling(self, datasets, corpora, epoch=1):
        """
        Up or down sample corpora with alpha sampling.
        """
        dataset_lengths = np.array(
            [len(d) for d in datasets],
            dtype=float,
        )
        logger.info(f"loaded total {dataset_lengths.sum()} blocks for all corpora")
        sample_probs = self._get_sample_prob(dataset_lengths)

        logger.info(
            "Sample probability by corpus: %s",
            {
                corpus: "{0:.4f}".format(sample_probs[id])
                for id, corpus in enumerate(corpora)
            },
        )
        size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
        # TODO: add an option for shrinking all size ratios to below 1
        # if self.args.multicorpus_sampling_alpha != 1:
        #   size_ratio /= size_ratio.max()

        # Fix numeric errors in size ratio computation
        #   0.999999999999999999 -> 1
        #   1.000000000000000002 -> 1
        for i in range(len(size_ratio)):
            size_ratio[i] = round(size_ratio[i], 8)

        logger.info(
            "Up/Down Sampling ratio by corpus: %s",
            {
                corpus: "{0:.2f}".format(size_ratio[id])
                for id, corpus in enumerate(corpora)
            },
        )
        logger.info(
            "Actual dataset size by corpus: %s",
            {
                corpus: "{0:.2f}".format(len(datasets[id]))
                for id, corpus in enumerate(corpora)
            },
        )
        resampled_datasets = [
            ResamplingDataset(
                datasets[i],
                size_ratio=size_ratio[i],
                seed=self.args.seed,
                epoch=epoch,
                replace=size_ratio[i] > 1.0,
            )
            for i, d in enumerate(datasets)
        ]
        # TODO: estimate the actual steps or tokens seen in training before launching experiments.
        logger.info(
            "Resampled dataset size by corpus: %s",
            {
                corpus: "{0:.2f}".format(len(resampled_datasets[id]))
                for id, corpus in enumerate(corpora)
            },
        )
        return resampled_datasets

    def get_shard_str(self, epoch, split):
        shards = {}
        for shard_id in os.listdir(os.path.join(self.args.data, split)):
            assert (
                int(shard_id) not in shards
            ), f"shard id: {shard_id} not in shards: {shards}"
            shards[int(shard_id)] = shard_id
        assert min(shards.keys()) == 0
        assert max(shards.keys()) == len(shards) - 1

        cur_shard_str = shards[(epoch - 1) % len(shards)]
        return cur_shard_str

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        The folder structure is assumed to look like:
            /path/to/data/train/00/foo.jsonl
            /path/to/data/train/00/bar.jsonl
            /path/to/data/train/01/foo.jsonl
            /path/to/data/train/01/bar.jsonl
            /path/to/data/valid/00/foo.jsonl
            /path/to/data/valid/00/bar.jsonl
        In this example, we have two "shards" of training data, which will be
        iterated over in epochs 1 and 2, respectively. Subsequent epochs will
        cycle back over the same data. We also have two different data sources
        in each shard (foo and bar), which will be combined and shuffled.
        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split
        cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        for file in sorted(
            os.listdir(os.path.join(self.args.data, split, cur_shard_str))
        ):
            if not file.endswith(".jsonl"):
                continue
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    tokenizer=self._tokenize_one_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        # chunk into blocks of tokens
        self.datasets[split] = StreamingTokenBlockDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=self.args.sample_break_mode,
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            # 1284 is a randomly-generated offset to decouple the seed used here
            # from the seed used above in StreamingShuffleDataset
            seed=1284 + self.args.seed,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def build_dataset_for_online_training(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)
        dataset = StreamingTokenBlockDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode="eos_pad_8",
            # we drop the remainder block during training
            drop_last=False,
            padding_idx=self.source_dictionary.pad(),
            # 1284 is a randomly-generated offset to decouple the seed used here
            # from the seed used above in StreamingShuffleDataset
            seed=1284 + self.args.seed,
        )

        logger.info(f"build dataset for online training seed:{self.args.seed} block_size:{self.args.tokens_per_sample}")

        self.datasets['train'] = dataset
        self.datasets['valid'] = dataset

        return dataset
        #src_dataset = PrependTokenDataset(
        #    dataset,
        #    token=(
        #        self.source_dictionary.bos()
        #        if getattr(self.args, "add_bos_token", False)
        #        else self.source_dictionary.eos()
        #    ),
        #)
        #tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        #src_input = PadDataset(
        #                src_dataset,
        #                pad_idx=self.source_dictionary.pad(),
        #                left_pad=False,
        #            )
        #target = PadDataset(
        #            tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
        #        )
        #return NestedDictionaryDataset(
        #    {
        #        "id": IdDataset(),
        #        "net_input": {
        #            "src_tokens": src_input,
        #            "src_lengths": NumelDataset(src_dataset, reduce=False),
        #        },
        #        "target": target,
        #        
        #    },
        #    sizes=[np.array(src_lengths)],
        #)

    def _collate_fn_TokenBlock(self, items: List[Dict[str, Any]]):
        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}

        tokens = data_utils.collate_tokens(
            [x["block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        # generate inputs and targets
        input = tokens[:, :-1].contiguous()
        target = tokens[:, 1:].contiguous()

        ids = torch.cat([x["ids"] for x in items if x is not None])
        if ids.numel() != torch.unique(ids).numel():
            n_duplicate = ids.numel() - torch.unique(ids).numel()
            logger.error(
                f"found {n_duplicate}/{ids.numel()} duplicate document IDs in the same batch!"
            )

        # metaseq expects batches to have the following structure
        return {
            "id": ids,
            "net_input": {
                "src_tokens": input,
                "src_lengths": input.ne(self.dictionary.pad()).sum(-1)
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
        }
    def _collate_fn_SrcTgt(self, items: List[Dict[str, Any]]):
        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}

        src_tokens = data_utils.collate_tokens(
            [x["src_block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        tgt_tokens = data_utils.collate_tokens(
            [x["tgt_block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )

        # generate inputs and targets
        input = src_tokens[:, :-1].contiguous()
        target = tgt_tokens[:, 1:].contiguous()

        ids = torch.cat([x["ids"] for x in items if x is not None])
        if ids.numel() != torch.unique(ids).numel():
            n_duplicate = ids.numel() - torch.unique(ids).numel()
            logger.error(
                f"found {n_duplicate}/{ids.numel()} duplicate document IDs in the same batch!"
            )

        # metaseq expects batches to have the following structure
        return {
            "id": ids,
            "net_input": {
                "src_tokens": input,
                "src_lengths": input.ne(self.dictionary.pad()).sum(-1)
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
            "ntokens_target": target.ne(self.dictionary.pad()).sum(),
        }

    def dataset(self, split):
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        batch_by_size=True,
        skip_remainder_batch=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (torch.utils.data.Dataset): dataset to batch
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator
                (default: False).
            batch_by_size (bool, optional):
                batch sequences of similar length together to reduce padding.
                If false, each batch will be of size max_sentences.
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller
                than local_batch_size * distributed_word_size (default: ``True``).
        Returns:
            ~metaseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert max_tokens is None

        # Up to this point, we have shuffled documents, flattened them into a 1D
        # tensor, then chunked into token blocks. But if documents are long, then
        # adjacent blocks may be from a single document, and naively distributed
        # sequential blocks to GPUs may cause entire updates to be dominated by a
        # handful of unique documents. Instead we have a readahead buffer that
        # reads in 10 full batches of data and shuffles sequences across them,
        # thus increasing randomness. This assumes that no single document spans
        # 10 full batches, which is reasonable when batch sizes are in the
        # millions and documents are on average much smaller.
        assert isinstance(dataset, StreamingTokenBlockDataset) or isinstance(
            dataset, StreamingSrcTgtDataset
        )
        shuffle_buffer_size = 10 * max_sentences * num_shards
        logger.info(f"setting shuffle buffer size to {shuffle_buffer_size}")
        dataset.set_shuffle_buffer_size(shuffle_buffer_size)

        if type(dataset) == StreamingSrcTgtDataset:
            collate_fn = self._collate_fn_SrcTgt
        elif type(dataset) == StreamingTokenBlockDataset:
            collate_fn = self._collate_fn_TokenBlock

        # partition dataset across data parallel workers
        dataset = PartitionedStreamingDataset(
            dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            drop_last=skip_remainder_batch,
        )
        # create a stateful/checkpointable iterator for the current data
        # parallel worker
        return iterators.StreamingEpochBatchIterator(
            dataset=dataset,
            batch_size=max_sentences,
            collate_fn=collate_fn,
            drop_last=skip_remainder_batch,
            num_workers=num_workers,
            epoch=epoch,
        )

    def get_generator_batch_iterator(
        self,
        dataset: BaseDataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        batch_by_size=True,
        skip_remainder_batch=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~metaseq.data.BaseDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator
                (default: False).
            batch_by_size (bool, optional):
                batch sequences of similar length together to reduce padding.
                If false, each batch will be of size max_sentences.
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
        Returns:
            ~metaseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        if not disable_iterator_cache and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, BaseDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        if batch_by_size:
            # create mini-batches with given size constraints
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            assert (
                max_sentences is not None
            ), "If batch_by_size=False, max_sentences must be passed. Got None"
            starts = indices[::max_sentences]
            batch_sampler = [indices[s : s + max_sentences] for s in starts]
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter



    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
