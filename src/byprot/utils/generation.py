import argparse
import os
import json
import warnings
from pprint import pprint

import torch

from byprot import utils


def initialize_generation(
    num_seqs, length, tokenizer, device,  padded_length=None, cond_seq=None, class_id=None, args=None,
):
    seq = ["<mask>"] * length
    if cond_seq is not None:
        # Inpainting generation, conditioned on some sequence segments
        seq_segment_list, position_list = format_check(args)
        for i, (start_pos, end_pos) in enumerate(position_list):
            seq[start_pos : end_pos + 1] = [
                char for char in seq_segment_list[i]
            ]
    if padded_length is not None:
        seq += ["<pad>"] * (padded_length - length)

    seq = ["".join(seq)]
    init_seq = seq * num_seqs
    batch = tokenizer.batch_encode_plus(
        init_seq,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    batch = {
        "input_ids": batch["input_ids"],
        # "input_mask": batch["attention_mask"].bool(),
        "class_ids": torch.tensor([class_id]*num_seqs, dtype=torch.long) if class_id is not None else None,
    }
    # if cond_seq is None:
    #     batch['input_ids'], _ = _full_mask(batch['input_ids'].clone(), collater.alphabet)
    batch = utils.recursive_to(batch, device)
    # pprint(batch)

    return batch["input_ids"], batch["class_ids"]


def initialize_generation_combined_batch(
    num_seqs, lengths, tokenizer, device, cond_seqs=None, class_ids=None, args=None,
):
    for i, length in enumerate(lengths):
        cs = None if cond_seqs is None else cond_seqs[i]
        ci = None if class_ids is None else class_ids[i]
        partial_input_ids, partial_class_ids = initialize_generation(
            num_seqs[i], length, tokenizer, device, padded_length=max(lengths), cond_seq=cs, class_id=ci, args=args
        )
        if i == 0:
            input_ids = partial_input_ids
            if class_ids is not None:
                class_ids_out = partial_class_ids
        else:
            input_ids = torch.cat((input_ids, partial_input_ids), dim=0)
            if class_ids is not None:
                class_ids_out = torch.cat((class_ids_out, partial_class_ids), dim=0)
    if class_ids is None:
        class_ids_out = None
    return input_ids, class_ids_out


def initialize_generation_batched(
    num_seqs, lengths, tokenizer, device, cond_seqs=None, class_ids=None, batch_size=32, args=None,
):
    n_sequences = len(lengths)
    assert len(num_seqs) == n_sequences, "The length of num_seqs and lengths does not match."
    if cond_seqs is not None:
        assert len(cond_seqs) == n_sequences, "The length of cond_seqs and lengths does not match."
    if class_ids is not None:
        assert len(class_ids) == n_sequences, "The length of class_ids and lengths does not match."

    batch_lens = [batch_size] * (n_sequences // batch_size)
    if n_sequences % batch_size != 0:
        batch_lens.append(n_sequences % batch_size)

    batches = []
    for batch_len in batch_lens:
        num_seqs_batch = num_seqs[:batch_len]
        num_seqs = num_seqs[batch_len:]

        lengths_batch = lengths[:batch_len]
        lengths = lengths[batch_len:]

        if cond_seqs is not None:
            cond_seqs_batch = cond_seqs[:batch_len]
            cond_seqs = cond_seqs[batch_len:]
        else:
            cond_seqs_batch = None
        
        if class_ids is not None:
            class_ids_batch = class_ids[:batch_len]
            class_ids = class_ids[batch_len:]
        else:
            class_ids_batch = None

        input_ids, class_ids_out = initialize_generation_combined_batch(
            num_seqs_batch, lengths_batch, tokenizer, device,
            cond_seqs=cond_seqs_batch,
            class_ids=class_ids_batch,
            args=args
        )
        batches.append((input_ids, class_ids_out))

    return batches


def generate_iteration(args, model, tokenizer, input_tokens, seq_len, class_ids=None):
    # batch fields: class_ids, prev_tokens (masked input tokens), 

    # Generate sequences
    partial_mask = input_tokens.ne(model.mask_id)
    with torch.cuda.amp.autocast():
        if args.architecture == "DiffusionProteinLanguageModel":
            output_tokens = model.generate(
                input_tokens=input_tokens,
                tokenizer=tokenizer,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                partial_masks=partial_mask,
                temperature=args.temperature,
            )
        elif args.architecture == "DPLMClass":
            if class_ids is None:
                raise ValueError(
                    "class_ids must be provided for DPLMClass generation."
                )
            batch = {
                "prev_tokens": input_tokens,
                "class_ids": class_ids,
            }
            output_tokens, output_scores = model.generate(
                batch=batch,
                tokenizer=tokenizer,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                partial_masks=partial_mask,
                temperature=args.temperature,
            )

    # Extract generated sequences
    print("final:")
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]
    pprint(output_results)

    return output_results


def generate(args, model, tokenizer):
    if args.saveto is not None:
        os.makedirs(args.saveto, exist_ok=True)
        args_file = os.path.join(args.saveto, "args.json")
        with open(args_file, "w") as fh:
            json.dump(vars(args), fh, indent=2, default=str)

    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device

    if len(args.num_seqs) == 1:
        num_seqs = args.num_seqs * len(args.seq_lens)
    else:
        assert len(args.num_seqs) == len(
            args.seq_lens
        ), "The length of num_seqs and seq_lens does not match."
        num_seqs = args.num_seqs
    
    if args.class_ids is None:
        class_ids = None
    else:
        if len(args.class_ids) == 1:
            class_ids = args.class_ids * len(args.seq_lens)
        else:
            assert len(args.class_ids) == len(
                args.seq_lens
            ), "The length of class_ids and seq_lens does not match."
            class_ids = args.class_ids
    
    cond_seqs = args.cond_seq
    if cond_seqs is not None:
        if len(cond_seqs) == 1:
            cond_seqs = cond_seqs * len(lengths)
        else:
            assert len(cond_seqs) == len(
                lengths
            ), "The length of cond_seqs and lengths does not match."

    output_results = []
    if args.batch_lens_together:
        batches = initialize_generation_batched(
            num_seqs, args.seq_lens, tokenizer, device,
            cond_seqs=cond_seqs, class_ids=class_ids, batch_size=args.batch_size, args=args
        )
        for input_tokens, class_ids_batch in batches:
            generated_sequences = generate_iteration(args, model, tokenizer, input_tokens, max(args.seq_lens), class_ids=class_ids_batch)
            output_results.extend(generated_sequences)
    else:
        warnings.warn(
            "The --batch_size argument is not used when --batch_lens_together is False; "
            "generation will iterate per sequence length using num_seqs for each length.",
            UserWarning,
        )
        for i, seq_len in enumerate(args.seq_lens):
            input_tokens, class_ids_out = initialize_generation(
                num_seqs[i], seq_len, tokenizer, device, cond_seq=cond_seqs[i] if cond_seqs is not None else None, class_id=class_ids[i] if args.class_ids is not None else None, args=args
            )
            generated_sequences = generate_iteration(args, model, tokenizer, input_tokens, seq_len, class_ids=class_ids_out)
            output_results.extend(generated_sequences)
    
    # Save generated sequences to fasta file
    if args.saveto is not None:
        fasta_file_name = "generated_sequences.fasta"
        saveto_name = os.path.join(
            args.saveto, fasta_file_name
        )
        fp_save = open(saveto_name, "w")
        for idx, seq in enumerate(output_results):
            fp_save.write(f">SEQUENCE_{idx}\n")
            fp_save.write(f"{seq}\n")
        fp_save.close()
