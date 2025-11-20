#!/usr/bin/env python3 -u
#
# Matouš Soldát, 2025
#
# Altered script from https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
#
# ##### Original script's copyright: #####
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# (source tree at https://github.com/facebookresearch/esm)
# ########################################

import os
import argparse
import pathlib
import warnings

import torch
import numpy as np
import pandas as pd
import json

import esm
from byprot import utils
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.models.dplm.dplm_class import DPLMClass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"
    )

    # Input
    # TODO implement reading from CSV
    parser.add_argument(
        "--input_file",
        type=pathlib.Path,
        help="FASTA or CSV file on which to extract representations",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        help="column name for sequence IDs if input is a CSV file",
        default="ID",
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        help="column name for amino acid sequences if input is a CSV file",
        default="sequence",
    )

    # DPLM model
    parser.add_argument(
        "--model_name",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        default="airkingbd/dplm_150m"
    )
    parser.add_argument("--from_huggingface", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--architecture", type=str, default="DiffusionProteinLanguageModel")
    parser.add_argument("--cache_dir", type=str, default=None)

    # Embedding extraction options
    # TODO --tokens_per_batch currently do not count padding and truncation, so the actual number of tokens may differ
    parser.add_argument("--tokens_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value, special tokens are added after truncation",
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    # TODO allow extracting multiple embedding types at the same time
    # TODO implemet per-token embeddings saving
    parser.add_argument(
        "--embedding_type",
        choices=["mean", "bos", "per_token"],
        help="specify which representation to return",
        default="mean",
    )

    # Save options
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Directory to save the extracted embeddings. default: None (same directory as input file).",
        default=None,
    )
    parser.add_argument(
        "--save_as",
        choices=["torch", "numpy", "csv"],
        nargs="+",
        default=["torch"],
        help="Format(s) to save embeddings; choose one or more of: torch, numpy, csv",
    )

    args = parser.parse_args()
    return args


def main(args):
    torch.set_float32_matmul_precision("high")

    # Prepare save path
    partial_fasta_path = os.path.splitext(args.input_file)[0]
    partial_save_path = partial_fasta_path + "_embeddings_" + args.embedding_type
    if args.output_dir is not None:
        partial_fasta_path = os.path.join(
            args.output_dir, os.path.basename(partial_fasta_path)
        )

    # Save args
    args_file = partial_save_path + "_args.json"
    with open(args_file, "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    
    # Load model
    if args.architecture == "DiffusionProteinLanguageModel":
        model = DiffusionProteinLanguageModel.from_pretrained(args.model_name, net_override={"cache_dir": args.cache_dir}, from_huggingface=args.from_huggingface)
        tokenizer = model.tokenizer
    elif args.architecture == "DPLMClass":
        if args.from_huggingface == True:
            raise ValueError(
                "DPLMClass does not support from_huggingface=True."
            )
        model = DPLMClass.from_pretrained(args.model_name)
        tokenizer = model.decoder.tokenizer
    else:
        raise ValueError(
            f"Unsupported architecture: {args.architecture}."
            "Please choose either 'DiffusionProteinLanguageModel' or 'DPLMClass'."
        )
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device

    # Load input sequences
    if args.input_file.suffix.lower() == ".fasta" or args.input_file.suffix.lower() == ".fa":
        dataset = esm.FastaBatchedDataset.from_file(args.input_file)
    elif args.input_file.suffix.lower() == ".csv":
        df = pd.read_csv(args.input_file)
        ids = df[args.id_column].tolist()
        sequences = df[args.sequence_column].tolist()
        dataset = esm.FastaBatchedDataset(ids, sequences)
    else:
        raise ValueError("Input file must be a FASTA (.fasta, .fa) or CSV (.csv) file.")

    original_ids = [entry[0] for entry in dataset]
    batches = dataset.get_batch_indices(args.tokens_per_batch, extra_toks_per_seq=2)

    long_count = sum(1 for _, seq in dataset if len(seq) > args.truncation_seq_length)
    if long_count > 0:
        warnings.warn(
            f"{long_count} sequences are longer than truncation_seq_length ({args.truncation_seq_length}) and will be truncated.",
            UserWarning,
        )

    def collate_fn(samples):
        batch = tokenizer(
            samples,
            return_tensors="pt",
            padding="longest", # options: "longest", "max_length"
            truncation="longest_first", # "longest_first"
            max_length=args.truncation_seq_length + 2,
        )
        batch["labels"] = [s[0] for s in samples]
        return batch

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_sampler=batches
    )
    print(f"Read {args.input_file} with {len(dataset)} sequences")

    # Extract emebeddings
    embeddings = None
    saved_embeddings_count = 0
    embedding_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(iter(data_loader)):
            tokens = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({tokens.size(0)} sequences)"
            )

            if torch.cuda.is_available() and not args.nogpu:
                tokens = tokens.to(device="cuda", non_blocking=True)
                attention_mask = attention_mask.to(device="cuda", non_blocking=True)

            out = model(tokens, attention_mask=attention_mask, return_last_hidden_state=True)
            representations = out[1]
                
            attention_mask = attention_mask.bool()

            for i, label in enumerate(labels):
                if args.embedding_type == "mean":
                    # Exclude BOS and EOS from attention for mean pooling
                    idxs = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                    attention_mask[i, idxs[0]] = 0
                    attention_mask[i, idxs[-1]] = 0

                    embedding = representations[i, attention_mask[i, :], :].mean(0)
                elif args.embedding_type == "bos":
                    embedding = representations[i, 0, :]
                elif args.embedding_type == "per_token":
                    raise NotImplementedError("Per-token embeddings saving is not implemented yet.")

                if embeddings == None:
                    embeddings = torch.empty(
                        (len(dataset), embedding.shape[0]), dtype=embedding.dtype
                    )
                embeddings[saved_embeddings_count, :] = embedding.to(device="cpu")
                saved_embeddings_count += 1
                embedding_labels.append(label)

    # Reorder embeddings to match original FASTA order
    processed_index = {label: idx for idx, label in enumerate(embedding_labels)}
    reordered = torch.empty_like(embeddings)
    for i, oid in enumerate(original_ids):
        reordered[i, :] = embeddings[processed_index[oid], :]
    embeddings = reordered
    embedding_labels = original_ids
        
    # Save with Torch
    if "torch" in args.save_as:
        torch.save(embeddings, partial_save_path + ".pt")

    # Save with Numpy
    if "numpy" in args.save_as:
        np.save(partial_save_path + ".npy", embeddings.numpy())

    # Save as CSV
    df = pd.DataFrame({"id": embedding_labels})
    if "csv" in args.save_as:
        embedding_df = pd.DataFrame(embeddings.numpy())
        df = pd.concat([df, embedding_df], axis=1)
    df.to_csv(partial_save_path + ".csv", index=False)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
