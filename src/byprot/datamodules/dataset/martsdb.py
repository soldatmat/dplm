
# Copyright (c) 2023 Microsoft Corporation
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Xinyou Wang on Jul 21, 2024
#
# Original file was released under MIT, with the full license text
# available at https://github.com/microsoft/evodiff/blob/main/LICENSE
#
# This modified file is released under the same license.


import json
import os
import pickle as pkl
from typing import Union, TypeVar, Sequence
import torch.distributed as dist
import numpy as np
from transformers import EsmTokenizer
import math
from typing import Iterable
import torch
import numpy as np
from torch.utils.data import BatchSampler, Dataset, Sampler, DataLoader
from byprot import utils

log = utils.get_logger(__name__)

T_co = TypeVar('T_co', covariant=True)

class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(
        self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0
    ):
        if dist.is_available():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        drop_last=False,
        batch_size=None,
        max_len=512
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.batches = self._build_batches()
        # TODO delete
        # print(len(self.batches))
        # print(len(self.batches[0]))
        # print(len(self.batches[1]))
        # print(len(self.batches[2]))
        # print(len(self.batches[-1]))
        # exit()
        
    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print('Current batch is empty! idx is ', idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)
            
        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(f'==============Local Rank {self.sampler.rank} Num Samples {num_samples}==============')
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(f'==============All Reduce Num Samples {num_samples}==============')
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(f'==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}==============')
        return batches
            
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
            
import pandas as pd
from byprot import utils

log = utils.get_logger(__name__)

class MartsDBDataset(Dataset):
    """
    Dataset that loads sequences from a CSV file.

    - 'split': "train", "valid", or "test"

    This class was written by Matouš Soldát on 2025-04-29.

    Args:
        csv_file (str): Path to the CSV file containing the sequences.
        split (str, optional): The split of the dataset to use. All data are used if split=None. Defaults to None.
        max_len (int, optional): Maximum length of the sequences. Defaults to 2048.
        sequence_column (str, optional): Column name for the sequences. Defaults to "Aminoacid_sequence".
        split_column (str, optional): Column name for the split. Defaults to "split".
    """

    def __init__(
        self,
        csv_file: str,
        split: str = None,
        max_len=2048,
        sequence_column: str = "Aminoacid_sequence",
        split_column: str = "split",
    ):
        self.csv_file = csv_file
        self.max_len = max_len
        self.sequence_column = sequence_column
        self.split_column = split_column
        
        data = pd.read_csv(self.csv_file)
        if split is not None:
            data = data[data[split_column] == split]
        data = data[[sequence_column]]
        self.data = data

        self.sequence_lens = self.data[sequence_column].str.len().tolist()

        log.info(f"Dataset size: {len(self.data)}")
        

    def __len__(self):
        return len(self.data)

    def get_metadata_lens(self):
        return self.sequence_lens

    def __getitem__(self, idx):
        consensus = self.data.iloc[idx][self.sequence_column]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return consensus

class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class DPLMCollater(object):
    def __init__(self, tokenizer_path=None):
        # by default we use the EsmTokenizer and the esm vocab. 
        # if you want to use the different vocab, 
        # please set the vocab path to the tokenizer_path
        if tokenizer_path is None:
            self.alphabet = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        else:
            self.alphabet = EsmTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, sequences):
        if len(list(zip(*sequences))) == 0:
            print("list idx error!")
            print(sequences)
        input_data = sequences
        batch = self.alphabet.batch_encode_plus(input_data,
                                                add_special_tokens=True,
                                                padding="longest",
                                                return_tensors='pt')

        batch = {
                'input_ids':  batch['input_ids'],
                'input_mask': batch['attention_mask'].bool(),
                'targets':    batch['input_ids'].clone()
            }
        return batch


def setup_dataloader(
        ds: MartsDBDataset,  
        max_tokens=6000, bucket_size=1000, 
        max_batch_size=800,  num_workers=8,
        rank=0, world_size=1,
        mini_run=False,
        max_len=512,
    ) -> DataLoader:
    collater = DPLMCollater()
    if mini_run:
        dl = DataLoader(dataset=ds, shuffle=True, batch_size=1, num_workers=4, collate_fn=collater)
    else:
        lens = ds.get_metadata_lens()
        train_sortish_sampler = SortishSampler(lens, bucket_size, num_replicas=world_size, rank=rank)
        train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, lens,
                                           max_len=max_len)
        dl = DataLoader(
            dataset=ds, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collater
        )
    return dl
