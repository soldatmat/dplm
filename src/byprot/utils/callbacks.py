# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import importlib
import operator
import os
from pathlib import Path
from types import SimpleNamespace
from time import time
from uuid import uuid4
from shutil import rmtree
import pickle

# from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pkg_resources
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from packaging.version import Version
from pkg_resources import DistributionNotFound
from pytorch_lightning import callbacks
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from rich import reconfigure
from torch import Tensor
import gdown # type: ignore
import logging

from byprot import utils
from byprot.utils.generation import generate

from enzymeexplorer.src.screening.tps_predict_fasta import get_embedding_extractor, predict_tps
from enzymeexplorer.src.screening.gather_detections_to_csv import main as enzymeexplorer_gather_detections_to_csv

logger = utils.get_logger(__name__)


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _compare_version(
    package: str, op: Callable, version: str, use_base_version: bool = False
) -> bool:
    """Compare package version with some requirements.

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    >>> _compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(
                pkg_resources.get_distribution(package).version
            )
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


_RICH_AVAILABLE = _package_available("rich") and _compare_version(
    "rich", operator.ge, "10.2.2"
)


if _RICH_AVAILABLE:
    from pytorch_lightning.callbacks.progress.rich_progress import (
        CustomProgress,
        MetricsTextColumn,
        RichProgressBar,
    )
    from rich import get_console, reconfigure
    from rich.text import Text

    # NOTE[zzx]: modify here to display float in e-format when lower than 1e-3
    def float_fmt(float_value):
        if float_value.is_integer():
            return round(float_value)
        elif float_value < 1e-3:
            return f"{float_value:.2e}"
        else:
            return round(float_value, 3)

    class BetterMetricsTextColumn(MetricsTextColumn):
        """A column containing text."""

        def render(self, task) -> Text:
            if (
                self._trainer.state.fn != "fit"
                or self._trainer.sanity_checking
                or self._trainer.progress_bar_callback.train_progress_bar_id
                != task.id
            ):
                return Text()
            if self._trainer.training and task.id not in self._tasks:
                self._tasks[task.id] = "None"
                if self._renderable_cache:
                    self._tasks[
                        self._current_task_id
                    ] = self._renderable_cache[self._current_task_id][1]
                self._current_task_id = task.id
            if self._trainer.training and task.id != self._current_task_id:
                return self._tasks[task.id]

            text = ""

            for k, v in self._metrics.items():
                text += f"{k}: {float_fmt(v) if isinstance(v, float) else v} "
            return Text(text, justify="left", style=self._style)

    class BetterRichProgressBar(RichProgressBar):
        def _init_progress(self, trainer):
            if self.is_enabled and (
                self.progress is None or self._progress_stopped
            ):
                self._reset_progress_bar_ids()
                reconfigure(**self._console_kwargs)
                self._console = get_console()
                self._console.clear_live()
                self._metric_component = BetterMetricsTextColumn(
                    trainer,
                    self.theme.metrics,
                    text_delimiter=",",
                    metrics_format=".2f",
                )
                self.progress = CustomProgress(
                    *self.configure_columns(trainer),
                    self._metric_component,
                    auto_refresh=False,
                    disable=self.is_disabled,
                    console=self._console,
                )
                self.progress.start()
                # progress has started
                self._progress_stopped = False


class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if (
            trainer.global_step % self.every_n_step == 0
            and trainer.global_step != 0
        ):
            trainer.validate()


class CheckpointEveryNSteps(pl.Callback):
    """Save a checkpoint every N steps, instead of Lightning's default that
    checkpoints based on validation loss."""

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    # def on_batch_end(self, trainer: pl.Trainer, _):
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
        """Check if we should save a checkpoint after every train batch."""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = (
                    f"{self.prefix}_epoch={epoch}_step={global_step}.ckpt"
                )
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            trainer.save_checkpoint(ckpt_path)


class ModelCheckpoint(callbacks.ModelCheckpoint):

    CHECKPOINT_NAME_BEST = "best"

    # @classmethod
    def _format_checkpoint_name(
        self,
        filename,
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        filename = super()._format_checkpoint_name(
            filename, metrics, prefix, auto_insert_metric_name
        )
        filename = filename.replace(
            "/", "_"
        )  # avoid '/' in filename unexpectedly creates folder
        return filename

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_start(trainer, pl_module)
        trainer.callback_metrics[self.monitor] = self.best_model_score

    def _update_best_and_save(
        self,
        current: Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, Tensor],
    ) -> None:
        k = (
            len(self.best_k_models) + 1
            if self.save_top_k == -1
            else self.save_top_k
        )

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"),
                device=current.device,
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        # update best checkpoint
        if self.best_model_path == filepath:
            self._save_checkpoint(
                trainer,
                self.format_checkpoint_name(
                    monitor_candidates, self.CHECKPOINT_NAME_BEST
                ),
            )

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and previous != filepath:
            trainer.strategy.remove_checkpoint(previous)


class TrackNorms(pl.Callback):

    # TODO do callbacks happen before or after the method in the main LightningModule?
    # @rank_zero_only # needed?
    def on_after_training_step(
        self,
        batch,
        batch_idx,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        # Log extra metrics
        metrics = {}

        if hasattr(pl_module, "_grad_norms"):
            metrics.update(pl_module._grad_norms)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # example to inspect gradient information in tensorboard
        if OmegaConf.select(
            trainer.hparams, "trainer.track_grad_norms"
        ):  # TODO dot notation should work with omegaconf?
            norms = {}
            for name, p in pl_module.named_parameters():
                if p.grad is None:
                    continue

                # param_norm = float(p.grad.data.norm(norm_type))
                param_norm = torch.mean(p.grad.data**2)
                norms[f"grad_norm.{name}"] = param_norm
            pl_module._grad_norms = norms


class ValidateWithEnzymeExplorer(pl.Callback):
    @staticmethod
    def _normalize_positive_int_list(
        value: Union[int, List[int]], field_name: str
    ) -> List[int]:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an int or a list of ints")

        if isinstance(value, int):
            normalized = [value]
        elif isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"{field_name} cannot be an empty list")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
                raise TypeError(
                    f"{field_name} must be an int or a list of ints"
                )
            normalized = value
        else:
            raise TypeError(f"{field_name} must be an int or a list of ints")

        if any(v <= 0 for v in normalized):
            raise ValueError(f"{field_name} values must be > 0")
        return normalized

    def __init__(
        self,
        enzymeexplorer_checkpoint_dir: str,
        template_sequences_file: Optional[str] = None,
        sequence_column_name: str = "sequence",
        class_id_column_name: str = None,
        num_seqs: Optional[Union[int, List[int]]] = None,
        seq_lens: Optional[Union[int, List[int]]] = None,
        max_iter: int = 500,
        sampling_strategy: str = "gumbel_argmax",
        temperature: float = 1.0,
        saveto: str = "./dplm_generated",
        generation_batch_size: int = 256,
        generation_batch_lens_together: bool = True,
        enzymeexplorer_detection_threshold: float = 0.0,
        enzymeexplorer_detect_precursor_synthases: bool = True,
        enzymeexplorer_model: str = "esm-1v-finetuned-subseq",
        every_n_train_steps: int = 1000,
    ):
        if not isinstance(generation_batch_lens_together, bool):
            raise TypeError(
                "generation_batch_lens_together must be a boolean"
            )
        self.generation_batch_lens_together = generation_batch_lens_together

        if template_sequences_file is not None and (
            num_seqs is not None or seq_lens is not None
        ):
            raise ValueError(
                "Provide either template_sequences_file or both num_seqs and seq_lens, not both"
            )

        if template_sequences_file is None and (
            num_seqs is None or seq_lens is None
        ):
            raise ValueError(
                "Provide template_sequences_file, or provide both num_seqs and seq_lens"
            )

        if template_sequences_file is not None:
            logger.info(f"template_sequences_file: {template_sequences_file}")
            data = pd.read_csv(template_sequences_file)
            seq_lens_raw = data[sequence_column_name].apply(len).tolist()
            if class_id_column_name is not None:
                class_ids_raw = data[class_id_column_name].tolist()
            else:
                class_ids_raw = None

            if self.generation_batch_lens_together:
                self.seq_lens = seq_lens_raw
                self.num_seqs = [1 for _ in self.seq_lens]
                self.class_ids = class_ids_raw
            else:
                # In non-together mode, generate() iterates over length buckets and
                # expects matching num_seqs counts for each bucket.
                bucket_counts = {}
                bucket_order = []
                for i, seq_len in enumerate(seq_lens_raw):
                    class_id = class_ids_raw[i] if class_ids_raw is not None else None
                    key = (seq_len, class_id)
                    if key not in bucket_counts:
                        bucket_counts[key] = 0
                        bucket_order.append(key)
                    bucket_counts[key] += 1

                self.seq_lens = [seq_len for seq_len, _ in bucket_order]
                self.num_seqs = [bucket_counts[key] for key in bucket_order]
                if class_ids_raw is not None:
                    self.class_ids = [class_id for _, class_id in bucket_order]
                else:
                    self.class_ids = None
        else:
            self.num_seqs = self._normalize_positive_int_list(num_seqs, "num_seqs")  # type: ignore[arg-type]
            self.seq_lens = self._normalize_positive_int_list(seq_lens, "seq_lens")  # type: ignore[arg-type]
            if len(self.num_seqs) not in {1, len(self.seq_lens)}:
                raise ValueError(
                    "num_seqs must have length 1 or match seq_lens length"
                )
            self.class_ids = None
            if class_id_column_name is not None:
                logger.warning(
                    "class_id_column_name is ignored when template_sequences_file is not provided"
                )
            logger.info(
                "ValidateWithEnzymeExplorer explicit generation plan: num_seqs=%s, seq_lens=%s",
                self.num_seqs,
                self.seq_lens,
            )

        logger.info(
            "ValidateWithEnzymeExplorer generation plan: batch_lens_together=%s, buckets=%d",
            self.generation_batch_lens_together,
            len(self.seq_lens),
        )

        self.max_iter = max_iter
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.saveto = saveto
        self.generation_batch_size = generation_batch_size

        self.enzymeexplorer_detection_threshold = enzymeexplorer_detection_threshold
        self.enzymeexplorer_detect_precursor_synthases = enzymeexplorer_detect_precursor_synthases
        self.enzymeexplorer_model = enzymeexplorer_model
        logger.info(f"enzymeexplorer_checkpoint_dir: {enzymeexplorer_checkpoint_dir}")
        self.enzymeexplorer_checkpoint_dir = enzymeexplorer_checkpoint_dir
        self.enzymeexplorer_classifier_checkpoint_path = self._prepare_plm_checkpoint()
        logger.info(f"enzymeexplorer_classifier_checkpoint_path: {self.enzymeexplorer_classifier_checkpoint_path}")
        self.enzymeexplorer_max_len = 1022

        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be > 0")
        self.every_n_train_steps = every_n_train_steps
        self._last_eval_global_step: Optional[int] = None

        # args: model, plm_checkpoint_dir, max_len
        args = SimpleNamespace()
        args.model = self.enzymeexplorer_model
        args.plm_checkpoint_dir = self.enzymeexplorer_plm_checkpoint_dir
        args.max_len = self.enzymeexplorer_max_len
        self.compute_embeddings_partial = get_embedding_extractor(args)

        with open(self.enzymeexplorer_classifier_checkpoint_path, "rb") as file:
            self.all_classifiers = pickle.load(file)
        logger.info("ValidateWithEnzymeExplorer init finished")

    def _prepare_plm_checkpoint(self):
        self.enzymeexplorer_plm_checkpoint_dir = Path(self.enzymeexplorer_checkpoint_dir) / "plm_checkpoints"
        if not self.enzymeexplorer_plm_checkpoint_dir.exists():
            self.enzymeexplorer_plm_checkpoint_dir.mkdir(parents=True)
        assert self.enzymeexplorer_model in {
            "esm-1v",
            "esm-1v-finetuned-subseq",
            "ankh_tps",
            "ankh_base",
        }, f"Model {self.enzymeexplorer_model} is not supported. Choose between esm-1v, esm-1v-finetuned-subseq, ankh_base, and ankh_tps"
        plm_path = self.enzymeexplorer_plm_checkpoint_dir / ("checkpoint-tps-esm1v-t33-subseq.ckpt" if self.enzymeexplorer_model == "esm-1v-finetuned-subseq" else "tps_ankh_lr=5e-05_bs=32.pth")
        if not plm_path.exists():
            logger.info("Downloading TPS language model checkpoint..")
            url = "https://drive.google.com/uc?id=1jU76oUl0-CmiB9m3XhaKmI2HorFhyxC7"
            gdown.download(url, str(plm_path), quiet=False)
        clf_chkpt_path = Path(self.enzymeexplorer_checkpoint_dir) / "classifier_plm_checkpoints.pkl"
        if not clf_chkpt_path.exists():
            logger.info("Downloading classifier checkpoints..")
            url = "https://drive.google.com/uc?id=15_OFrrVUy9r9Urj-R2CjTRj_DHcazdAl"
            gdown.download(url, str(clf_chkpt_path), quiet=False)
        
        return clf_chkpt_path
    
    def _validate_with_enzymeexplorer(self, input_fasta_path: str, output_csv_path: str):
        run_id = f"{int(time() * 1000)}-{uuid4()}"
        intermediate_outputs_root = f"_temp_dir_{run_id}"
        if not Path(intermediate_outputs_root).exists():
            Path(intermediate_outputs_root).mkdir(parents=True)
        
        args = SimpleNamespace()
        args.batch_size = 4
        args.clf_batch_size = 4096
        args.max_len = self.enzymeexplorer_max_len
        args.fasta_path = input_fasta_path
        args.starting_i = 0
        args.end_i = 700_000
        args.output_id = ""
        args.verbose = False
        args.output_root = intermediate_outputs_root
        args.detection_threshold = self.enzymeexplorer_detection_threshold
        args.detect_precursor_synthases = self.enzymeexplorer_detect_precursor_synthases
        args.gpu="0"
        predict_tps(args, self.compute_embeddings_partial, self.all_classifiers)

        args = SimpleNamespace()
        args.delete_individual_files = False
        args.screening_results_root = f"{intermediate_outputs_root}/detections_plm"
        args.output_path = output_csv_path
        enzymeexplorer_gather_detections_to_csv(args)

        rmtree(intermediate_outputs_root)

    def _run_evaluation(self, trainer, pl_module, run_label: str):
        model = pl_module.model
        tokenizer = pl_module.model.net.tokenizer if hasattr(pl_module.model, 'net') else pl_module.model.decoder.net.tokenizer

        # Generate sequences
        args = SimpleNamespace()
        args.seed = None
        args.architecture = type(model).__name__
        args.num_seqs = self.num_seqs
        args.seq_lens = self.seq_lens
        args.class_ids = self.class_ids
        args.saveto = os.path.join(self.saveto, run_label)
        args.temperature = self.temperature
        args.sampling_strategy = self.sampling_strategy
        args.max_iter = self.max_iter
        args.batch_lens_together = self.generation_batch_lens_together
        args.batch_size = self.generation_batch_size
        args.cond_seq = None # TODO add cond_seq support
        args.cache_dir = None
        fasta_path = generate(args, model, tokenizer, verbose=False)

        # Evaluate generated sequences with EnzymeExplorer
        output_csv_path = fasta_path.replace(".fasta", "_enzyme_explorer_sequence_only.csv")
        logger.info("Validating generated sequences with EnzymeExplorer")
        self._validate_with_enzymeexplorer(fasta_path, output_csv_path)

    def on_sanity_check_end(self, trainer, pl_module):
        logger.info("ValidateWithEnzymeExplorer before training")
        self._run_evaluation(trainer, pl_module, "sanity_check_step_0")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step <= 0:
            return
        if global_step == self._last_eval_global_step:
            return
        if global_step % self.every_n_train_steps != 0:
            return

        self._last_eval_global_step = global_step
        logger.info(
            "ValidateWithEnzymeExplorer after global training step %s",
            global_step,
        )
        self._run_evaluation(trainer, pl_module, f"step_{global_step}")
