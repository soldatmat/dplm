# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import logging
log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from byprot.utils import load_yaml_config

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.peft_model import PeftModel
except:
    pass


@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""


@dataclass
class LoRAConfig:
    enable: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_module: str = field(default="")
    modules_to_save: str = field(default="")


def get_net_class(dplm_type):
    from byprot.models import MODEL_REGISTRY

    net_class = MODEL_REGISTRY.get(dplm_type, None)
    if net_class is None:
        raise ValueError(f"Invalid architecture: {dplm_type}.")
    return net_class


def get_net(cfg):
    if cfg.net.arch_type == "esm":
        from byprot.models.dplm.modules.dplm_modeling_esm import EsmForDPLM

        config = AutoConfig.from_pretrained(f"{cfg.net.name}")
        net = EsmForDPLM(config, dropout=cfg.net.dropout, conditioning_mode=getattr(cfg, "encoder_conditioning_mode", None))
    # TODO: dplm will support more architectures, such as Llama
    else:
        raise NotImplementedError

    # 2-stage training (please refer to our paper for more details.)
    ## stage 1: pretrain a masked language model (MLM) from scratch
    ## stage 2: continue pretrain a diffusion language model based on the pretrained MLM
    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if is_local:
            # load your pretrained model from local
            # state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')['state_dict']
            # net.load_state_dict(state_dict, strict=True)
            pretrained_state_dict = torch.load(
                pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            from collections import OrderedDict

            new_pretrained_state_dict = OrderedDict()
            # remove the module prefix "model.net."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[10:]] = v
            net.load_state_dict(new_pretrained_state_dict, strict=True)
        else:
            # or you can load a pretrained model from huggingface
            ptrn_net = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            net.load_state_dict(ptrn_net.state_dict(), strict=True)
            del ptrn_net

    # activate lora training if possible
    if cfg.lora.enable:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(",")
        if modules_to_save == [""]:
            modules_to_save = None

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
        )
        net = get_peft_model(net, peft_config)
        trainable_params, all_param = net.get_nb_trainable_parameters()
        log.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    return net


def get_net_dplm2(cfg):
    training_stage = getattr(cfg, "training_stage", "train_from_dplm")

    # dplm2 initialize from a pretrained dplm model
    if cfg.net.arch_type == "esm":
        from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2

        config = AutoConfig.from_pretrained(f"{cfg.net.name}")

        # training_state == "train_from_dplm" means initializing from a pretrained sequence-based DPLM,
        # whose vocab_size is 33 containing the standerd amino acid and special tokens
        # (https://huggingface.co/airkingbd/dplm_650m/blob/main/vocab.txt).
        if training_stage == "train_from_dplm" and cfg.net.pretrain:
            net = EsmForDPLM2(config, dropout=cfg.net.dropout, vocab_size=33)

        # training_state == "continue_train_from_dplm2" means continue training from a pretrained DPLM-2,
        # whose vocabulary contains amino acid and struct tokens,
        # and the vocab_size should be 33 + number of struct tokens and special tokens (e.g., 33 + 8192 + 4)
        elif (
            training_stage == "continue_train_from_dplm2"
            or not cfg.net.pretrain
        ):
            net = EsmForDPLM2(
                config,
                dropout=cfg.net.dropout,
                vocab_size=getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4),
            )

        else:
            raise NotImplementedError
    # TODO: dplm2 will support more architectures, such as Llama
    else:
        raise NotImplementedError

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if training_stage == "train_from_dplm":
            from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

            pretrained_state_dict = (
                DiffusionProteinLanguageModel.from_pretrained(
                    pretrained_model_name_or_path
                ).net.state_dict()
            )
            net.load_state_dict(pretrained_state_dict, strict=True)

            # expand the embedding weights
            # # initialize the new embedding with the mean and variance of pretrained embeddings.
            net.resize_token_embeddings(
                getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4)
            )

            pretrained_bias = net.lm_head.bias
            net.lm_head.bias = nn.Parameter(
                torch.zeros(
                    getattr(cfg.tokenizer, "vocab_size", 33 + 8192 + 4)
                )
            )
            net.lm_head.bias.data[:33] = pretrained_bias.data[:33]
        elif training_stage == "continue_train_from_dplm2":
            assert is_local
            from byprot.models.dplm2.dplm2 import (
                MultimodalDiffusionProteinLanguageModel,
            )

            pretrained_net = (
                MultimodalDiffusionProteinLanguageModel.from_pretrained(
                    pretrained_model_name_or_path, from_huggingface=False
                ).net
            )
            if issubclass(type(pretrained_net), PeftModel):
                pretrained_net = pretrained_net.merge_and_unload()
            pretrained_state_dict = pretrained_net.state_dict()
            net.load_state_dict(pretrained_state_dict, strict=True)
        else:
            raise ValueError(f"Invalid training stage {training_stage}.")

        del pretrained_state_dict

    # activate lora training if possible
    if cfg.lora.enable:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=32,
            lora_dropout=cfg.lora.lora_dropout,
        )
        net = get_peft_model(net, peft_config)

    return net


def get_net_dplm2_bit(cfg):
    # dplm2 initialize from a pretrained dplm model
    if cfg.net.arch_type == "esm":
        from byprot.models.dplm2.modules.dplm2_bit_modeling_esm import (
            EsmForDPLM2Bit,
        )

        config = AutoConfig.from_pretrained(f"{cfg.net.name}")
        net = EsmForDPLM2Bit(
            config,
            dropout=cfg.net.dropout,
            codebook_embed_dim=getattr(cfg.bit, "codebook_embed_dim", 13),
        )
    # TODO: dplm2 will support more architectures, such as Llama
    else:
        raise NotImplementedError

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        from byprot.models.dplm import DiffusionProteinLanguageModel

        pretrained_state_dict = DiffusionProteinLanguageModel.from_pretrained(
            pretrained_model_name_or_path
        ).net.state_dict()
        net.load_state_dict(pretrained_state_dict, strict=False)

        del pretrained_state_dict

    # activate lora training if possible
    if cfg.lora.enable:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module.split(",")
        modules_to_save = cfg.lora.modules_to_save.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=32,
            lora_dropout=cfg.lora.lora_dropout,
        )
        net = get_peft_model(net, peft_config)

    return net


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_masking_prior(
    scores, cutoff_len, stochastic=False, temp=1.0, prior_mask=None
):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(
        dim=-1, index=cutoff_len
    )  # + torch.tensor(1e-10)
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def mask_fill_811(inputs, masked_indices, mask_id):
    prev_tokens = inputs.clone()
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.8)).bool()
        & masked_indices
    )
    prev_tokens[indices_replaced] = mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(4, 24, prev_tokens.shape).type_as(prev_tokens)
    prev_tokens[indices_random] = random_words[indices_random]

    return prev_tokens


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(
    logits=None, temperature=1.0, noise_scale=1.0
):
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
    )
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.95, filter_value=-float("Inf")
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape)
    return logits


def init_weights_from_checkpoint(
    pl_module: nn.Module,
    ckpt_path: str,
    merge_lora_mode: str = "auto",
) -> None:
    """Load only the model state_dict from a checkpoint into ``pl_module``.

    Optimizer / LR scheduler / global_step / epoch are NOT touched -- use
    ``Trainer.fit(ckpt_path=...)`` for a full resume.

    Source and target may have different LoRA topology. PEFT wraps modules
    with a 'base_model.model.' prefix and renames each wrapped Linear's
    parameter from '.weight'/'.bias' to '.base_layer.weight'/'.base_layer.bias'.
    Both decorations are stripped to a canonical key so the source values land
    in matching target slots regardless of which side has LoRA.

    ``merge_lora_mode`` controls source LoRA deltas (lora_A / lora_B):
        "auto"   - fold into base_layer.weight only when target lacks the
                   matching LoRA slot for that module (different rank or LoRA
                   disabled). When the target has the matching slot, A and B
                   load independently. (default)
        "always" - always fold A and B into base_layer.weight; target's LoRA
                   stays at fresh init even when slots match.
        "never"  - never fold; A and B load only when target has matching
                   slots, otherwise their info is dropped silently.
    Folding uses ``alpha/rank`` from the checkpoint's
    ``hyper_parameters.model.lora.lora_alpha``; a scaling=1 fallback is used
    with a warning if it is missing.
    """
    log.info(
        f"Loading model weights from <{ckpt_path}> "
        f"(weights only; optimizer / LR scheduler / global_step start fresh)"
    )
    blob = torch.load(ckpt_path, map_location="cpu")
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob

    def _norm(k: str) -> str:
        return k.replace(".base_model.model.", ".").replace(".base_layer.", ".")

    src_norm = {_norm(k): v for k, v in state_dict.items()}
    target_sd = pl_module.state_dict()
    target_canon = {_norm(k): k for k in target_sd}

    merge_mode = str(merge_lora_mode)
    lora_modules = sorted({
        ck[: -len(".lora_A.default.weight")]
        for ck in src_norm
        if ck.endswith(".lora_A.default.weight")
    })

    hp_lora = (((blob.get("hyper_parameters", {}) if isinstance(blob, dict) else {})
                .get("model", {}) or {})
               .get("lora", {}) or {})
    src_alpha = hp_lora.get("lora_alpha")

    merged_modules = []
    merged_canonical_keys = set()
    alpha_fallback_used = False
    sample_delta_norm = None
    sample_base_norm = None
    for m in lora_modules:
        cA, cB, cW = (
            f"{m}.lora_A.default.weight",
            f"{m}.lora_B.default.weight",
            f"{m}.weight",
        )
        A, B, W = src_norm.get(cA), src_norm.get(cB), src_norm.get(cW)
        if A is None or B is None or W is None:
            continue

        if merge_mode == "never":
            continue
        if merge_mode == "auto":
            tA, tB = target_canon.get(cA), target_canon.get(cB)
            target_lora_matches = (
                tA is not None and tB is not None
                and target_sd[tA].shape == A.shape
                and target_sd[tB].shape == B.shape
            )
            if target_lora_matches:
                continue  # let A/B load into target slots; don't merge

        rank = A.shape[0]
        if src_alpha is None:
            alpha = rank  # scaling = 1.0 fallback
            alpha_fallback_used = True
        else:
            alpha = src_alpha
        scaling = float(alpha) / float(rank)
        delta = torch.matmul(B.float(), A.float()) * scaling
        merged = (W.float() + delta).to(W.dtype)
        if sample_delta_norm is None:
            sample_delta_norm = float(delta.norm().item())
            sample_base_norm = float(W.float().norm().item())
        src_norm[cW] = merged
        src_norm.pop(cA, None)
        src_norm.pop(cB, None)
        merged_modules.append(m)
        merged_canonical_keys.update({cA, cB})

    new_sd = {}
    shape_mismatch = []
    for tk, tv in target_sd.items():
        sv = src_norm.get(_norm(tk))
        if sv is None:
            continue
        if sv.shape != tv.shape:
            shape_mismatch.append(tk)
            continue
        new_sd[tk] = sv

    target_norm = set(target_canon.keys())
    src_unmapped = [
        k for k in state_dict
        if _norm(k) not in target_norm and _norm(k) not in merged_canonical_keys
    ]

    missing, unexpected = pl_module.load_state_dict(new_sd, strict=False)
    log.info(
        f"Init from ckpt: matched {len(new_sd)} target slots; "
        f"merged LoRA into base for {len(merged_modules)} module(s) "
        f"(merge_mode={merge_mode}, alpha={src_alpha}); "
        f"{len(src_unmapped)} source keys dropped (no target slot); "
        f"{len(shape_mismatch)} keys had shape mismatch; "
        f"{len(missing)} target keys remain unfilled (e.g. fresh LoRA params); "
        f"{len(unexpected)} unexpected (should be 0)."
    )
    if merged_modules and sample_delta_norm is not None:
        log.info(
            f"LoRA merge sanity: first merged module {merged_modules[0]} -> "
            f"||delta||={sample_delta_norm:.4g}, ||base||={sample_base_norm:.4g}"
        )
    if alpha_fallback_used:
        log.warning(
            "lora_alpha not found in ckpt's hyper_parameters; "
            "used scaling=alpha/rank=1.0 as fallback for at least one merged module."
        )
    if shape_mismatch:
        log.warning(f"Shape mismatch (first 5): {shape_mismatch[:5]}")
    if src_unmapped:
        log.warning(f"Source keys dropped (first 5): {src_unmapped[:5]}")
    if missing:
        log.warning(f"Unfilled target keys (first 5): {missing[:5]}")
    if unexpected:
        log.warning(f"Unexpected keys (first 5): {unexpected[:5]}")


def get_struct_tokenizer(
    model_name_or_path="airkingbd/struct_tokenizer", eval_mode=True
):
    from byprot.models.structok.structok_lfq import VQModel

    if os.path.exists(model_name_or_path):
        root_path = f"{model_name_or_path}/.hydra"
    else:
        root_path = Path(snapshot_download(repo_id=model_name_or_path))
    cfg = load_yaml_config(f"{root_path}/config.yaml")
    stok = VQModel(**cfg)
    pretrained_state_dict = torch.load(
        f"{root_path}/dplm2_struct_tokenizer.ckpt",
        map_location=torch.device("cpu"),
    )
    missing, unexpected = stok.load_state_dict(
        pretrained_state_dict, strict=False
    )
    print(
        f'Restored from "{model_name_or_path}" with {len(missing)} missing and {len(unexpected)} unexpected keys'
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
    stok = stok.requires_grad_(False)
    return stok.train(not eval_mode)
