#!/usr/bin/env python
"""
Evaluate a trained checkpoint by generating sequences and validating with EnzymeExplorer.

Usage:
    python evaluate_checkpoint.py checkpoint_path=path/to/ckpt.ckpt
"""

import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from time import time
from uuid import uuid4
from shutil import rmtree
import pickle

import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
import torch

from byprot.utils.generation import generate
from enzymeexplorer.src.screening.tps_predict_fasta import (
    get_embedding_extractor,
    predict_tps,
)
from enzymeexplorer.src.screening.gather_detections_to_csv import (
    main as enzymeexplorer_gather_detections_to_csv,
)


def load_model_from_checkpoint(checkpoint_path: str, cfg: DictConfig):
    """Load a trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Instantiate the configured task/model stack and restore checkpoint weights.
    from byprot import utils as byprot_utils

    task_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model = byprot_utils.instantiate_from_config(
        cfg=task_cfg,
        group="task",
        model=model_cfg,
    )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def _resolve_under_datadir(path_value: str, datadir: Path) -> Path:
    """Resolve possibly-relative path under datadir without double-prefixing.

    Handles legacy values like "dplm/foo" by stripping the leading datadir
    basename before joining.
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj

    parts = path_obj.parts
    if parts and parts[0] == datadir.name:
        path_obj = Path(*parts[1:]) if len(parts) > 1 else Path(".")

    return datadir / path_obj


def _build_generation_plan_from_template(
    template_sequences_file: Path,
    sequence_column_name: str,
    length_column_name: str,
    count_column_name: str,
):
    data = pd.read_csv(template_sequences_file)
    if sequence_column_name in data.columns:
        seq_lens = data[sequence_column_name].astype(str).apply(len).tolist()
        if not seq_lens:
            raise ValueError(f"No sequences found in template file: {template_sequences_file}")
        num_seqs = [1 for _ in seq_lens]
        return num_seqs, seq_lens, "sequence"

    if length_column_name in data.columns and count_column_name in data.columns:
        seq_lens = data[length_column_name].tolist()
        num_seqs = data[count_column_name].tolist()
        if not seq_lens:
            raise ValueError(f"No rows found in template file: {template_sequences_file}")

        seq_lens = [int(v) for v in seq_lens]
        num_seqs = [int(v) for v in num_seqs]
        if any(v <= 0 for v in seq_lens):
            raise ValueError("All sequence lengths in template must be positive")
        if any(v <= 0 for v in num_seqs):
            raise ValueError("All sequence counts in template must be positive")
        return num_seqs, seq_lens, "length_count"

    raise ValueError(
        "Template format not recognized. Expected either sequence column "
        f"'{sequence_column_name}' or tuple columns '{length_column_name}' and "
        f"'{count_column_name}' in: {template_sequences_file}"
    )


def _parse_int_list(value, field_name: str):
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"No values provided for {field_name}")
        return [int(v) for v in parts]
    raise TypeError(f"Unsupported type for {field_name}: {type(value)}")


def _parse_bool(value, field_name: str):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise ValueError(
            f"Invalid integer boolean value for {field_name}: {value!r}. "
            "Expected 0 or 1"
        )
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
        raise ValueError(
            f"Invalid boolean value for {field_name}: {value!r}. "
            "Expected one of: true/false/1/0"
        )
    raise TypeError(
        f"Unsupported type for {field_name}: {type(value)}"
    )


def validate_with_enzyme_explorer(
    input_fasta_path: str,
    output_csv_path: str,
    enzymeexplorer_checkpoint_dir: str,
    enzymeexplorer_max_len: int = 1022,
    detection_threshold: float = 0.0,
    detect_precursor_synthases: bool = True,
    enzymeexplorer_model: str = "esm-1v-finetuned-subseq",
):
    """Validate generated sequences with EnzymeExplorer."""
    enzymeexplorer_checkpoint_dir = Path(enzymeexplorer_checkpoint_dir)
    enzymeexplorer_plm_checkpoint_dir = (
        enzymeexplorer_checkpoint_dir / "plm_checkpoints"
    )
    if not enzymeexplorer_plm_checkpoint_dir.exists():
        enzymeexplorer_plm_checkpoint_dir.mkdir(parents=True)
    
    # Prepare PlM checkpoint
    assert enzymeexplorer_model in {
        "esm-1v",
        "esm-1v-finetuned-subseq",
        "ankh_tps",
        "ankh_base",
    }, f"Model {enzymeexplorer_model} is not supported."
    
    plm_filename = (
        "checkpoint-tps-esm1v-t33-subseq.ckpt"
        if enzymeexplorer_model == "esm-1v-finetuned-subseq"
        else "tps_ankh_lr=5e-05_bs=32.pth"
    )
    plm_path = enzymeexplorer_plm_checkpoint_dir / plm_filename
    if not plm_path.exists():
        print("Downloading TPS language model checkpoint..")
        import gdown
        url = "https://drive.google.com/uc?id=1jU76oUl0-CmiB9m3XhaKmI2HorFhyxC7"
        gdown.download(url, str(plm_path), quiet=False)
    
    clf_chkpt_path = (
        enzymeexplorer_checkpoint_dir / "classifier_plm_checkpoints.pkl"
    )
    if not clf_chkpt_path.exists():
        print("Downloading classifier checkpoints..")
        import gdown
        url = "https://drive.google.com/uc?id=15_OFrrVUy9r9Urj-R2CjTRj_DHcazdAl"
        gdown.download(url, str(clf_chkpt_path), quiet=False)
    
    # Prepare embeddings and classifiers
    args = SimpleNamespace()
    args.model = enzymeexplorer_model
    args.plm_checkpoint_dir = str(enzymeexplorer_plm_checkpoint_dir)
    args.max_len = enzymeexplorer_max_len
    compute_embeddings_partial = get_embedding_extractor(args)
    
    with open(clf_chkpt_path, "rb") as file:
        all_classifiers = pickle.load(file)
    
    # Run TPS prediction
    run_id = f"{int(time() * 1000)}-{uuid4()}"
    intermediate_outputs_root = f"_temp_dir_{run_id}"
    if not Path(intermediate_outputs_root).exists():
        Path(intermediate_outputs_root).mkdir(parents=True)
    
    try:
        args = SimpleNamespace()
        args.batch_size = 4
        args.clf_batch_size = 4096
        args.max_len = enzymeexplorer_max_len
        args.fasta_path = input_fasta_path
        args.starting_i = 0
        args.end_i = 700_000
        args.output_id = ""
        args.verbose = False
        args.output_root = intermediate_outputs_root
        args.detection_threshold = detection_threshold
        args.detect_precursor_synthases = detect_precursor_synthases
        args.gpu = "0"
        predict_tps(args, compute_embeddings_partial, all_classifiers)
        
        args = SimpleNamespace()
        args.delete_individual_files = False
        args.screening_results_root = f"{intermediate_outputs_root}/detections_plm"
        args.output_path = output_csv_path
        enzymeexplorer_gather_detections_to_csv(args)
    finally:
        rmtree(intermediate_outputs_root, ignore_errors=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # Parse arguments
    checkpoint_path = cfg.get("checkpoint_path", None)
    input_fasta_path = cfg.get("input_fasta_path", None)
    eval_name = cfg.get("eval_name", "checkpoint_eval")
    datadir = cfg.get("datadir", None)

    if datadir is None:
        raise ValueError("datadir must be specified")

    datadir = Path(datadir)

    # Make optional input FASTA path absolute if relative.
    if isinstance(input_fasta_path, str):
        input_fasta_path = input_fasta_path.strip() or None
    if input_fasta_path is not None:
        input_fasta_path = _resolve_under_datadir(str(input_fasta_path), datadir)
        if not input_fasta_path.exists():
            raise FileNotFoundError(
                f"Provided input_fasta_path does not exist: {input_fasta_path}"
            )

    # Checkpoint is only required when we need to generate new sequences.
    if input_fasta_path is None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be specified when input_fasta_path is not provided"
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = datadir / checkpoint_path

    logs_dir = datadir / "logs" / eval_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generation parameters
    gen_use_template = _parse_bool(
        cfg.get("gen_use_template", True), "gen_use_template"
    )
    gen_sequence_column_name = cfg.get("gen_sequence_column_name", "Aminoacid_sequence")
    gen_template_length_column_name = cfg.get("gen_template_length_column_name", "length")
    gen_template_count_column_name = cfg.get("gen_template_count_column_name", "count")
    gen_num_seqs_cfg = cfg.get("gen_num_seqs", 1000)
    gen_seq_lens_cfg = cfg.get("gen_seq_lens", 350)
    gen_max_iter = cfg.get("gen_max_iter", 500)
    gen_batch_size = cfg.get("gen_batch_size", 256)
    gen_batch_lens_together = _parse_bool(
        cfg.get("gen_batch_lens_together", True),
        "gen_batch_lens_together",
    )
    gen_sampling_strategy = cfg.get("gen_sampling_strategy", "gumbel_argmax")
    gen_temperature = cfg.get("gen_temperature", 1.0)
    
    # EnzymeExplorer parameters
    enzyme_explorer_template_seqs = cfg.get(
        "enzyme_explorer_template_seqs", "dplm/tps_scaffolds.csv"
    )
    enzyme_explorer_checkpoint_dir = cfg.get(
        "enzyme_explorer_checkpoint_dir", "dplm/enzymeexplorer_checkpoints"
    )
    enzyme_explorer_detection_threshold = cfg.get(
        "enzyme_explorer_detection_threshold", 0.0
    )
    enzyme_explorer_detect_precursor_synthases = _parse_bool(
        cfg.get("enzyme_explorer_detect_precursor_synthases", True),
        "enzyme_explorer_detect_precursor_synthases",
    )
    
    # Make enzyme explorer paths absolute (without datadir double-prefixing).
    enzyme_explorer_template_seqs = _resolve_under_datadir(
        str(enzyme_explorer_template_seqs), datadir
    )
    enzyme_explorer_checkpoint_dir = _resolve_under_datadir(
        str(enzyme_explorer_checkpoint_dir), datadir
    )

    if input_fasta_path is None:
        if gen_use_template:
            num_seqs, seq_lens, template_mode = _build_generation_plan_from_template(
                enzyme_explorer_template_seqs,
                str(gen_sequence_column_name),
                str(gen_template_length_column_name),
                str(gen_template_count_column_name),
            )
        else:
            template_mode = "manual"
            num_seqs = _parse_int_list(gen_num_seqs_cfg, "gen_num_seqs")
            seq_lens = _parse_int_list(gen_seq_lens_cfg, "gen_seq_lens")
            if len(num_seqs) == 1 and len(seq_lens) > 1:
                num_seqs = [num_seqs[0] for _ in seq_lens]
            elif len(seq_lens) == 1 and len(num_seqs) > 1:
                seq_lens = [seq_lens[0] for _ in num_seqs]
            elif len(num_seqs) != len(seq_lens):
                raise ValueError(
                    "gen_num_seqs and gen_seq_lens must have the same number of "
                    "items, or one of them must contain exactly one item"
                )
    
    if input_fasta_path is None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(str(checkpoint_path), cfg)

        # Get tokenizer
        tokenizer = (
            model.model.net.tokenizer
            if hasattr(model.model, "net")
            else model.model.decoder.net.tokenizer
        )

        # Generate sequences
        if gen_use_template:
            if template_mode == "sequence":
                print(
                    f"Generating {len(num_seqs)} sequences from sequence template in: "
                    f"{enzyme_explorer_template_seqs}"
                )
            else:
                print(
                    "Generating from length/count template with "
                    f"{len(num_seqs)} length buckets in: {enzyme_explorer_template_seqs}"
                )
        else:
            print(
                "Generating fixed plan with "
                f"num_seqs={num_seqs}, seq_lens={seq_lens}"
            )
        gen_output_dir = logs_dir / "generated_sequences"
        gen_output_dir.mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace()
        args.seed = None
        args.architecture = type(model.model).__name__
        args.num_seqs = num_seqs
        args.seq_lens = seq_lens
        args.class_ids = None
        args.saveto = str(gen_output_dir)
        args.temperature = float(gen_temperature)
        args.sampling_strategy = gen_sampling_strategy
        args.max_iter = int(gen_max_iter)
        args.batch_lens_together = bool(gen_batch_lens_together)
        args.batch_size = int(gen_batch_size)
        args.cond_seq = None
        args.cache_dir = None

        fasta_path = generate(args, model.model, tokenizer, verbose=True)
        print(f"Generated sequences saved to: {fasta_path}")
    else:
        fasta_path = str(input_fasta_path)
        print(f"Using pre-generated sequences from FASTA: {fasta_path}")
    
    # Evaluate with EnzymeExplorer
    print("Evaluating generated sequences with EnzymeExplorer...")
    output_csv_path = fasta_path.replace(
        ".fasta", "_enzyme_explorer_sequence_only.csv"
    )
    
    validate_with_enzyme_explorer(
        input_fasta_path=fasta_path,
        output_csv_path=output_csv_path,
        enzymeexplorer_checkpoint_dir=str(enzyme_explorer_checkpoint_dir),
        detection_threshold=float(enzyme_explorer_detection_threshold),
        detect_precursor_synthases=enzyme_explorer_detect_precursor_synthases,
    )
    
    print(f"Evaluation results saved to: {output_csv_path}")
    print("Checkpoint evaluation complete!")


if __name__ == "__main__":
    main()
