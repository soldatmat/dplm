"""
Usage example:
python generate_dplm_fixed.py \
  --model_name /home2/soldat/documents/dplm/logs/TPS_dplm_150m_stage3_run_6/checkpoints/best.ckpt \
  --saveto /home2/soldat/documents/terpene_synthases/output/dplm/TPS_dplm_150m_stage3_run_6 \
  --seq_lens 100 200 300 400 500 \
  --num_seqs 40
"""

import argparse
import os
from pprint import pprint
import json

import torch

from byprot import utils
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.models.dplm.dplm_class import DPLMClass


def format_check(args):
    seq_list = args.cond_seq
    cond_position = args.cond_position
    assert len(seq_list) == len(
        cond_position
    ), "The length of cond_seq and cond_position does not match."
    position_list = []
    for pos in cond_position:
        pos = pos.split("-")
        assert (
            len(pos) == 2
        ), "The format of position is illegal, which is not correctly splited by '-'"
        start_pos, end_pos = int(pos[0]), int(pos[1])
        assert (
            end_pos >= start_pos
        ), "The end position is smaller than start position."
        position_list.append((start_pos, end_pos))
    # check if position segment has overlap
    temp_position_list = [pos for tup in position_list for pos in tup]
    for i in range(1, len(temp_position_list) - 2, 2):
        assert (
            temp_position_list[i + 1] > temp_position_list[i]
        ), "The position segment has overlap, which is not supported"
    # check if the length of each position segment and seq segment matches
    for i, (start_pos, end_pos) in enumerate(position_list):
        assert len(seq_list[i]) == (
            end_pos - start_pos + 1
        ), "The length of each position segment and seq segment does not match."
    return seq_list, position_list


def initialize_generation(
    num_seqs, length, tokenizer, device,  padded_length=None, cond_seq=None, args=None,
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
        "input_mask": batch["attention_mask"].bool(),
    }
    # if cond_seq is None:
    #     batch['input_ids'], _ = _full_mask(batch['input_ids'].clone(), collater.alphabet)
    batch = utils.recursive_to(batch, device)
    # pprint(batch)
    return batch["input_ids"]


def initialize_generation_combined_batch(
    num_seqs, lengths, tokenizer, device, cond_seqs=None, args=None
):
    for i, length in enumerate(lengths):
        cs = None if cond_seqs is None else cond_seqs[i]
        partial_input_ids = initialize_generation(
            num_seqs[i], length, tokenizer, device, padded_length=max(lengths), cond_seq=cs, args=args
        )
        if i == 0:
            input_ids = partial_input_ids
        else:
            input_ids = torch.cat((input_ids, partial_input_ids), dim=0)

    return input_ids


def generate(args):
    os.makedirs(args.saveto, exist_ok=True)
    args_file = os.path.join(args.saveto, "args.json")
    with open(args_file, "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    if args.architecture == "DiffusionProteinLanguageModel":
        model = DiffusionProteinLanguageModel.from_pretrained(
            args.model_name, net_override={"cache_dir": args.cache_dir}, from_huggingface=args.from_huggingface
        )
    elif args.architecture == "DPLMClass":
        if args.from_huggingface == True:
            raise ValueError(
                "DPLMClass does not support from_huggingface=True."
            )
        model = DPLMClass.from_pretrained(
            args.model_name,
        )
    else:
        raise ValueError(
            f"Unsupported architecture: {args.architecture}."
            "Please choose either 'DiffusionProteinLanguageModel' or 'DPLMClass'."
        )
    tokenizer = model.tokenizer
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
    
    cond_seqs = args.cond_seq
    if cond_seqs is not None:
        if len(cond_seqs) == 1:
            cond_seqs = cond_seqs * len(lengths)
        else:
            assert len(cond_seqs) == len(
                lengths
            ), "The length of cond_seqs and lengths does not match."

    if args.batch_lens_together:
        input_tokens = initialize_generation_combined_batch(
            num_seqs, args.seq_lens, tokenizer, device,
            cond_seqs=cond_seqs, args=args
        )
        generate_iteration(args, model, tokenizer, input_tokens, max(args.seq_lens))
    else:
        for i, seq_len in enumerate(args.seq_lens):
            input_tokens = initialize_generation(
                num_seqs[i], seq_len, tokenizer, device, cond_seq=cond_seqs[i] if cond_seqs is not None else None, args=args
            )
            generate_iteration(args, model, tokenizer, input_tokens, seq_len)


def generate_iteration(args, model, tokenizer, input_tokens, seq_len):
    # Generate sequences
    partial_mask = input_tokens.ne(model.mask_id)
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            input_tokens=input_tokens,
            tokenizer=tokenizer,
            max_iter=args.max_iter,
            sampling_strategy=args.sampling_strategy,
            partial_masks=partial_mask,
            temperature=args.temperature,
        )
    output_tokens = outputs

    # Extract generated sequences
    print("final:")
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]
    pprint(output_results)

    # Save generated sequences to fasta file
    fasta_file_name = f"iter_{args.max_iter}_L_{seq_len}.fasta"
    saveto_name = os.path.join(
        args.saveto, fasta_file_name
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results):
        fp_save.write(f">SEQUENCE_{idx}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm_150m"
    )
    parser.add_argument("--from_huggingface", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--architecture", type=str, default="DiffusionProteinLanguageModel")
    parser.add_argument("--num_seqs", nargs="*", type=int, default=[40])
    parser.add_argument("--seq_lens", nargs="*", type=int)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="gumbel_argmax"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_lens_together", default=False, action=argparse.BooleanOptionalAction)
    # inpainting
    # Note: the format of --cond_position and --cond_seq should split by ','
    # the number and the length of segments should match.
    # Like this:
    # --cond_position 1-4 8-10 (position starts from 0)
    # --cond_seq ALVE EME
    parser.add_argument("--cond_position", nargs="*", type=str)
    parser.add_argument("--cond_seq", nargs="*", type=str)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    generate(args)


if __name__ == "__main__":
    main()
