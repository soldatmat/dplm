#!/usr/bin/env python
"""Merge a DPLM LoRA-finetune Lightning checkpoint into its base HF model and save.

The training recipe wraps the base HF `AutoModelForMaskedLM` inside a PEFT LoRA
adapter (rank=1, alpha=2, target on V-projection). Lightning saves the state
under the prefix `model.net.base_model.model.esm...`. This script:

  1. Loads the Lightning ckpt (weights_only=False).
  2. Loads the base HF model, wraps it with an identical PEFT LoRA config.
  3. Strips the Lightning prefix and loads the state into the PEFT model.
  4. Calls `merge_and_unload()` to fold LoRA A.B back into the base weights.
  5. Saves the merged model + tokenizer to `--out-dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer

LIGHTNING_PREFIX = "model.net.base_model.model."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--base-hf", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--lora-alpha", type=int, default=2)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target",
        default=r"(esm.encoder.layer.[0-9]*.attention.self.value)",
    )
    args = parser.parse_args()

    print(f"[load] Lightning ckpt: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    lit_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    print(f"[load] base HF model: {args.base_hf}")
    base = AutoModelForMaskedLM.from_pretrained(args.base_hf)
    tokenizer = AutoTokenizer.from_pretrained(args.base_hf)

    print(
        f"[peft] LoRA rank={args.lora_rank} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} target={args.lora_target}"
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target,
        bias="none",
    )
    peft_model = get_peft_model(base, lora_config)

    # Strip Lightning prefix — keep only keys under `model.net.base_model.model.`.
    stripped: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for k, v in lit_state.items():
        if k.startswith(LIGHTNING_PREFIX):
            stripped[k[len(LIGHTNING_PREFIX) :]] = v
        else:
            dropped.append(k)
    print(f"[strip] kept {len(stripped)} keys, dropped {len(dropped)}")
    if dropped:
        print(f"[strip] example dropped keys: {dropped[:5]}")

    # PEFT-wrapped model expects the base-model params under
    # `base_model.model.<orig-key>` (with LoRA A/B modules injected). The
    # stripped state matches that layout since Lightning saved via the PEFT
    # wrapper. Load with strict=False and inspect.
    result = peft_model.base_model.model.load_state_dict(stripped, strict=False)
    print(f"[load_state] missing (in model, not in ckpt): {len(result.missing_keys)}")
    if result.missing_keys:
        print(f"  first 5: {result.missing_keys[:5]}")
    print(
        f"[load_state] unexpected (in ckpt, not in model): {len(result.unexpected_keys)}"
    )
    if result.unexpected_keys:
        print(f"  first 5: {result.unexpected_keys[:5]}")

    # Sanity: LoRA A and B params should exist and at least one should be nonzero
    # (rank-1 delta learned during fine-tuning).
    lora_a_norms: list[float] = []
    lora_b_norms: list[float] = []
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            lora_a_norms.append(param.detach().norm().item())
        elif "lora_B" in name:
            lora_b_norms.append(param.detach().norm().item())
    print(
        f"[lora] {len(lora_a_norms)} A-mats  (sum-of-norms={sum(lora_a_norms):.4f}), "
        f"{len(lora_b_norms)} B-mats (sum-of-norms={sum(lora_b_norms):.4f})"
    )

    print("[merge] merge_and_unload()")
    merged = peft_model.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] {args.out_dir}")
    merged.save_pretrained(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))

    n_params = sum(p.numel() for p in merged.parameters())
    print(f"[done] merged params: {n_params:,}")
    print("[done] files:")
    for entry in sorted(args.out_dir.iterdir()):
        print(f"  {entry.name}  ({entry.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
