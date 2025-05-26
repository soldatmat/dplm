# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from byprot import utils
from byprot.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from byprot.datamodules.dataset.martsdb import (
    MartsDBDataset,
    setup_dataloader,
    Subset,
)

log = utils.get_logger(__name__)


@register_datamodule("martsdb")
class MartsDBDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str, # = "data/tape",
        max_tokens: int = 6000,
        max_len: int = 2048,
        collater: str = "esm",
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        mini_run: bool = False,
        num_seqs: int = 40,  # used for testing
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == "fit":
            # TODO add correct args to MartsDBDataset initialization
            # TODO then modify the remaining two occurrences of MartsDBDataset according to this one
            self.train_dataset = MartsDBDataset(
                csv_file=self.hparams.data_path,
                split=None,
                max_len=self.hparams.max_len,
            )
            # TODO add correct args to MartsDBDataset initialization
            # TODO ? remove valid_dataset (it is the same as train_dataset, might be redundant if metrics are calculated on train)
            self.valid_dataset = MartsDBDataset(
                csv_file=self.hparams.data_path,
                split=None,
                max_len=self.hparams.max_len,
            )
            if self.hparams.mini_run:
                mini_size = 100
                # subsample trainset
                tindices = np.arange(0, 1000)
                train_indices = np.sort(
                    np.random.choice(tindices, mini_size, replace=False)
                )
                self.train_dataset = Subset(self.train_dataset, train_indices)
                # subsample validset
                vindices = np.arange(1, 80000, 1)
                valid_indices = np.random.choice(vindices, mini_size)
                self.valid_dataset = Subset(self.valid_dataset, valid_indices)
        elif stage == "test" or stage == "predict":
            # TODO add correct args to MartsDBDataset initialization
            self.test_dataset = MartsDBDataset(
                csv_file=self.hparams.data_path,
                split="valid",
                max_len=self.hparams.max_len,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}.")
        self.stage = stage

    def train_dataloader(self):
        return setup_dataloader(
            self.train_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            mini_run=self.hparams.mini_run,
            max_len=self.hparams.max_len,
            max_batch_size=(
                1 if self.stage == "test" or self.stage == "predict" else 800
            ),
        )

    def val_dataloader(self):
        return setup_dataloader(
            self.valid_dataset,
            # collater=self.hparams.collater,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            mini_run=self.hparams.mini_run,
            max_len=self.hparams.max_len,
        )

    def test_dataloader(self):
        return setup_dataloader(
            self.test_dataset,
            # collater=self.hparams.collater,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
            bucket_size=self.hparams.num_seqs,
            max_batch_size=self.hparams.num_seqs,
        )
