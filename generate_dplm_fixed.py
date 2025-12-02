"""
Usage example:
python generate_dplm_fixed.py \
  --model_name /home2/soldat/documents/dplm/logs/TPS_dplm_150m_stage3_run_6/checkpoints/best.ckpt \
  --saveto /home2/soldat/documents/terpene_synthases/output/dplm/TPS_dplm_150m_stage3_run_6 \
  --seq_lens 100 200 300 400 500 \
  --num_seqs 40
"""

import argparse

import torch

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.models.dplm.dplm_class import DPLMClass
from byprot.utils.generation import generate


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm_150m"
    )
    parser.add_argument("--from_huggingface", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--architecture", type=str, default="DiffusionProteinLanguageModel")
    parser.add_argument("--num_seqs", nargs="*", type=int, default=[40])
    parser.add_argument("--seq_lens", nargs="*", type=int)
    parser.add_argument("--class_ids", nargs="*", type=int, default=None)
    parser.add_argument("--saveto", type=str, default="./dplm_generated")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="gumbel_argmax"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_lens_together", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", type=int, default=32)
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

    return args


def load_model(args):
    if args.architecture == "DiffusionProteinLanguageModel":
        model = DiffusionProteinLanguageModel.from_pretrained(
            args.model_name, net_override={"cache_dir": args.cache_dir}, from_huggingface=args.from_huggingface
        )
        tokenizer = model.tokenizer
    elif args.architecture == "DPLMClass":
        if args.from_huggingface == True:
            raise ValueError(
                "DPLMClass does not support from_huggingface=True."
            )
        model = DPLMClass.from_pretrained(
            args.model_name,
        )
        tokenizer = model.decoder.net.tokenizer
    else:
        raise ValueError(
            f"Unsupported architecture: {args.architecture}."
            "Please choose either 'DiffusionProteinLanguageModel' or 'DPLMClass'."
        )

    return model, tokenizer


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    model, tokenizer = load_model(args)
    
    generate(args, model, tokenizer)
    



if __name__ == "__main__":
    main()
