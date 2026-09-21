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
    # A3 Option D: logit-space classifier-free guidance scale. 0 (default) =
    # off / unchanged behaviour. Only used by DPLMClass models trained with a
    # learned null embedding (cfg_dropout > 0).
    parser.add_argument("--guidance_w", type=float, default=0.0)
    # A3 same-class-neighbor conditioning source (NeighborEncoder models only).
    #   medoid  (default) -> let the encoder fall back to the per-class MEDOID
    #            embedding (cond_emb is None), the original inference behaviour.
    #   random  -> for EACH generated sequence supply a RANDOM same-class enzyme's
    #            640-d `emb` row (resampled independently per sequence) as cond_emb,
    #            sourced from the neighbor artifact filtered to the conditioned
    #            class via the MARTS-DB first-cyclization CSV.
    parser.add_argument(
        "--neighbor_cond_source",
        type=str,
        default="medoid",
        choices=["medoid", "random"],
    )
    # Path to the neighbor_conditioning_emb.pt artifact: a torch dict with
    # keys `emb` [N_enz,640], `enzyme_ids` (list aligned to emb rows),
    # `class_medoid` [22,640]. Required when --neighbor_cond_source random.
    parser.add_argument("--neighbor_artifact_path", type=str, default=None)
    # Path to the MARTS-DB first-cyclization CSV (columns Enzyme_marts_ID,
    # First_cyclization_product_id) providing per-enzyme class membership.
    # Required when --neighbor_cond_source random.
    parser.add_argument("--class_csv", type=str, default=None)
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


def build_neighbor_random_pools(args):
    """Build {class_id: Tensor[n_class_enz, 640]} of per-class neighbor `emb`
    rows for --neighbor_cond_source random.

    Filters the neighbor artifact's `emb` rows to each conditioned class via
    the MARTS-DB CSV's per-enzyme First_cyclization_product_id. Attaches the
    result to args.neighbor_random_pools so generation.py can resample a random
    same-class row per generated sequence.
    """
    import pandas as pd

    if not args.neighbor_artifact_path:
        raise ValueError(
            "--neighbor_cond_source random requires --neighbor_artifact_path."
        )
    if not args.class_csv:
        raise ValueError(
            "--neighbor_cond_source random requires --class_csv."
        )

    artifact = torch.load(args.neighbor_artifact_path, map_location="cpu")
    emb = artifact["emb"].float()  # [N_enz, 640]
    enzyme_ids = list(artifact["enzyme_ids"])  # aligned 1:1 with emb rows
    assert emb.shape[0] == len(enzyme_ids), (
        f"emb rows {emb.shape[0]} != enzyme_ids {len(enzyme_ids)}"
    )

    csv = pd.read_csv(args.class_csv)
    # One class label per enzyme (dedup multi-product rows; class is per-enzyme).
    enz_to_class = (
        csv.drop_duplicates("Enzyme_marts_ID")
        .set_index("Enzyme_marts_ID")["First_cyclization_product_id"]
        .to_dict()
    )

    # Conditioned class set (flatten args.class_ids).
    cond_classes = sorted(set(int(c) for c in (args.class_ids or [])))
    pools = {}
    for c in cond_classes:
        row_idx = [
            i
            for i, eid in enumerate(enzyme_ids)
            if enz_to_class.get(eid, None) == c
        ]
        if not row_idx:
            raise ValueError(
                f"No neighbor-artifact enzymes found for conditioned class {c}."
            )
        pools[c] = emb[torch.tensor(row_idx, dtype=torch.long)]
        print(
            f"[neighbor random] class {c}: {len(row_idx)} same-class enzyme "
            f"embeddings available for resampling"
        )
    args.neighbor_random_pools = pools


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model(args)

    args.neighbor_random_pools = None
    if args.neighbor_cond_source == "random":
        if args.architecture != "DPLMClass":
            raise ValueError(
                "--neighbor_cond_source random is only valid with "
                "--architecture DPLMClass."
            )
        build_neighbor_random_pools(args)

    generate(args, model, tokenizer)


if __name__ == "__main__":
    main()
