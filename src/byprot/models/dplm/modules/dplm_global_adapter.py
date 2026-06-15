# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmIntermediate,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)

from byprot import utils
from byprot.models.dplm import DiffusionProteinLanguageModel
from byprot.models.utils import NetConfig, LoRAConfig, get_net

# TODO https://github.com/bytedance/dplm/issues/47
# from byprot.models.dplm.modules.dplm_modeling_esm import (
#     ModifiedEsmAttention,
#     ModifiedEsmSelfAttention,
# )

logger = utils.get_logger(__name__)


@dataclass
class DPLMWithGlobalAdapterConfig:
    num_diffusion_timesteps: int = field(default=100)
    adapter_dropout: float = field(default=0.1)
    encoder_d_model: int = field(default=512)
    encoder_conditioning_mode: str = field(default="cross_attention")  # cross_attention, expanded_cross_attention, sum, ignore, prepend, adaln
    # adaLN-single bottleneck rank (only used when encoder_conditioning_mode == 'adaln').
    adaln_bottleneck_rank: int = field(default=64)
    adapter_intermediate_size: int = field(default=320)
    adapter_hidden_size: int = field(default=80)
    dplm_name: str = field(default="")
    from_huggingface: bool = field(default=True)
    net: NetConfig = field(default=NetConfig())
    lora: LoRAConfig = field(default=LoRAConfig())
    # When the adapter is enabled (any non-prepend mode), the freeze rule
    # 'adapter not in pname' freezes the base ESM stack. These two flags
    # opt back in to fine-tuning the post-encoder LayerNorm and the LM head.
    finetune_emb_layer_norm_after: bool = field(default=False)
    finetune_lm_head: bool = field(default=False)
    # If True (default), copy weights from the original last ESM layer into
    # the corresponding adapter_* submodules of the new GlobalAdapterLayer
    # whenever shapes match. Set False to keep adapter_* at random init.
    init_adapter_from_orig: bool = field(default=True)
    # Allowlist of substrings: when non-empty, restrict the freeze loop's
    # 'all adapter_* trainable' default by additionally freezing every
    # adapter_* parameter whose name does NOT contain any of these
    # substrings. Use to fine-tune only a subset of adapter submodules
    # without LoRA. Default empty: every adapter_* parameter trainable.
    trainable_adapter_substrings: Any = field(default=None)


class DPLMWithConditionalGlobalAdapter(nn.Module):
    _default_cfg = DPLMWithGlobalAdapterConfig()

    @classmethod
    def from_pretrained(cls, cfg):
        net_override = {"conditioning_mode": cfg.encoder_conditioning_mode}
        # adaLN-single: forward the bottleneck rank so EsmForDPLM.__init__ sets it
        # on the HF config BEFORE ModifiedEsmModel builds the modulation module
        # (which reads it via getattr(config, 'adaln_bottleneck_rank', 64)).
        if cfg.encoder_conditioning_mode == "adaln":
            net_override["adaln_bottleneck_rank"] = getattr(cfg, "adaln_bottleneck_rank", 64)
        net = DiffusionProteinLanguageModel.from_pretrained(cfg.dplm_name, net_override=net_override, from_huggingface=cfg.from_huggingface).net

        del net.esm.contact_head

        # Add conditioning adapter to the architecture
        if cfg.encoder_conditioning_mode in ("prepend", "adaln"):
            # Neither prepend nor adaln swaps the last layer or uses a
            # GlobalAdapterLayer: prepend injects a class token, adaln injects
            # per-layer modulation built inside ModifiedEsmModel.
            pass
        else:
            # change net.last_layer to GlobalAdapterLayer
            if cfg.encoder_conditioning_mode == "mini_cross_attention":
                adapter = GlobalAdapterLayerMini(cfg, deepcopy(net.config))
            else:
                adapter = GlobalAdapterLayer(cfg, deepcopy(net.config))
            net_last_layer = net.esm.encoder.layer[-1]
            # Loads the main-path slots ('attention.*', 'LayerNorm.*',
            # 'intermediate.*', 'output.*') of the new layer; adapter_*
            # slots have no counterpart here and stay at default init.
            adapter.load_state_dict(net_last_layer.state_dict(), strict=False)

            # Optionally also seed the adapter_* submodules with the same
            # original-layer weights (only where shapes match).
            if getattr(cfg, "init_adapter_from_orig", True):
                orig_state = net_last_layer.state_dict()
                adapter_target = adapter.state_dict()
                remapped = {}
                for k, v in orig_state.items():
                    if k.startswith("attention."):
                        new_k = "adapter_cross" + k  # attention.* -> adapter_crossattention.*
                    elif k.startswith(("LayerNorm.", "intermediate.", "output.")):
                        new_k = "adapter_" + k
                    else:
                        continue
                    if new_k in adapter_target and adapter_target[new_k].shape == v.shape:
                        remapped[new_k] = v
                if remapped:
                    adapter.load_state_dict(remapped, strict=False)
                    logger.info(
                        f"Initialized {len(remapped)} adapter_* tensors of "
                        f"layer[-1] from the original last-layer weights."
                    )

            net.esm.encoder.layer[-1] = adapter
            del net_last_layer

        # Initialize DPLMWithConditionalGlobalAdapter
        dplm_adapter = cls(cfg, net)

        # Freeze parameters
        if cfg.encoder_conditioning_mode == "adaln":
            # Frozen backbone: freeze EVERY decoder param, then re-enable only
            # the adaLN-single modulation module (shared MLP + per-layer offset
            # table). The per-class ClassEncoder is a SIBLING module on DPLMClass
            # (DPLMClass.encoder, not inside this adapter) and its trainability is
            # governed there by ClassEncoderConfig.frozen / null_embedding.
            trainable, frozen = 0, 0
            for pname, param in dplm_adapter.named_parameters():
                if "adaln_modulation_module" in pname:
                    param.requires_grad = True
                    trainable += param.numel()
                else:
                    param.requires_grad = False
                    frozen += param.numel()
            logger.info(
                f"adaLN-single conditioning: {trainable} trainable decoder params "
                f"(adaln_modulation_module.*), {frozen} frozen backbone params. "
                "Class encoder trainability is set separately on DPLMClass.encoder."
            )
        elif cfg.encoder_conditioning_mode == "prepend":
            pass
        else:
            for pname, param in dplm_adapter.named_parameters():
                if "adapter" not in pname:
                    param.requires_grad = False
            allowlist = getattr(cfg, "trainable_adapter_substrings", None)
            if allowlist:
                allowlist = list(allowlist)
                kept, frozen = 0, 0
                for pname, param in dplm_adapter.named_parameters():
                    if "adapter" in pname:
                        if any(s in pname for s in allowlist):
                            kept += 1
                        else:
                            param.requires_grad = False
                            frozen += 1
                logger.info(
                    f"trainable_adapter_substrings={allowlist}: kept "
                    f"{kept} adapter_* params trainable, froze {frozen}."
                )
            if getattr(cfg, "finetune_emb_layer_norm_after", False):
                dplm_adapter.net.esm.encoder.emb_layer_norm_after.requires_grad_(True)
            if getattr(cfg, "finetune_lm_head", False):
                dplm_adapter.net.lm_head.requires_grad_(True)

        # Activate LoRA with the PEFT library
        if cfg.lora.enable:
            logger.info(f"Activating LoRA training with the PEFT library for net ({type(dplm_adapter.net)}):")
            from peft import LoraConfig, TaskType, get_peft_model

            lora_target_module = cfg.lora.lora_target_module
            # Accept either comma-separated string ("a,b,c") or list/ListConfig (["a","b","c"]).
            # PEFT matches modules_to_save by suffix; '' would match every module, so drop empties.
            modules_to_save_raw = getattr(cfg.lora, "modules_to_save", "")
            if isinstance(modules_to_save_raw, str):
                modules_to_save = [m for m in modules_to_save_raw.split(",") if m]
            else:
                modules_to_save = [m for m in modules_to_save_raw if m]

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=lora_target_module,
                modules_to_save=modules_to_save or None,
                inference_mode=False,
                r=cfg.lora.lora_rank,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=getattr(cfg.lora, "lora_dropout", 0.1),
            )
            dplm_adapter.net = get_peft_model(dplm_adapter.net, peft_config)
            dplm_adapter.net.print_trainable_parameters()
        
        return dplm_adapter

    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)

        self.net = get_net(cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

    def forward(
        self,
        input_ids,
        encoder_out=None,
        tokens=None,
        loss_mask=None,
        forward_diffusion=False,
        **kwargs
    ):
        encoder_hidden_states = encoder_out["feats"]

        encoder_attention_mask = (
            encoder_out["encoder_attention_mask"]
            if "encoder_attention_mask" in encoder_out
            else input_ids.ne(self.pad_id)
        )

        outputs = self.net(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return outputs

    def compute_loss(
        self,
        batch,
        weighting="constant",
        encoder_out=None,
        tokens=None,
        label_smoothing=False,
        return_outputs=False,
    ):
        target = batch["tokens"] if tokens is None else tokens
        partial_masks = torch.zeros_like(target).bool()

        # couple
        t1, t2 = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        x_t, t, loss_mask = list(
            self.q_sample_coupled(
                target,
                t1,
                t2,
                maskable_mask=self.get_non_special_sym_mask(
                    target, partial_masks
                ),
            ).values()
        )
        target = target.repeat(2, 1)

        logits = self.forward(
            x_t,
            encoder_out=encoder_out,
            loss_mask=loss_mask,
            forward_diffusion=True,
        )["logits"]

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (
                num_timesteps - (t - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[weighting][:, None].float() / num_timesteps
        weight = weight.expand(loss_mask.size())

        return logits, batch["tokens"].repeat(2, 1), loss_mask, weight

    def _update_cfg(self, cfg):
        # if '_target_' in cfg.denoiser:
        #     cfg.denoiser.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = t1 == t2
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (
            u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (
            u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]
        ) & (maskable_mask[t1_eq_t2_mask]) # DPLM original code. I think it is wrong & should be deleted.
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0),
        }

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask


class GlobalAdapterLayer(nn.Module):
    def __init__(self, cfg, config, adapter_config=None, qdim=None):
        if adapter_config is None:
            adapter_config = deepcopy(config)

        super().__init__()
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)

        kdim = vdim = getattr(cfg, "encoder_d_model", 512)
        adapter_config.hidden_dropout_prob = getattr(cfg, "adapter_dropout", 0.0)
        self.adapter_crossattention = GlobalAdapterEsmAttention(
            adapter_config, qdim=qdim, kdim=kdim, vdim=vdim
        )
        
        self.adapter_LayerNorm = nn.LayerNorm(adapter_config.hidden_size, eps=adapter_config.layer_norm_eps)
        self.adapter_intermediate = EsmIntermediate(adapter_config)
        self.adapter_output = EsmOutput(adapter_config)

        self.conditioning_mode = getattr(cfg, "encoder_conditioning_mode", "cross_attention")
        self.conditioning_chunk = {
            "cross_attention": self.cross_attention_conditioning_chunk,
            "mini_cross_attention": self.cross_attention_conditioning_chunk,
            "expanded_cross_attention": self.expanded_cross_attention_conditioning_chunk,
            "sum": self.sum_conditioning_chunk,
            "ignore": self.ignore_conditioning_chunk,
        }[self.conditioning_mode]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # self-attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first feed forward chunk
        layer_output = self.feed_forward_chunk(attention_output)

        # adapter begins
        residual = layer_output

        # conditioning with encoder output
        conditioning_output = self.conditioning_chunk(layer_output, encoder_hidden_states, encoder_attention_mask)

        # second feed forward chunk
        ffn_output = self.adapter_feed_forward_chunk(conditioning_output)
        ffn_output += residual

        outputs = (ffn_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def adapter_feed_forward_chunk(self, attention_output):
        attention_output_ln = self.adapter_LayerNorm(attention_output)
        intermediate_output = self.adapter_intermediate(attention_output_ln)
        layer_output = self.adapter_output(
            intermediate_output, attention_output
        )
        return layer_output
    
    """
    Cross-attention with the singular global embedding
    """
    def cross_attention_conditioning_chunk(self, layer_output, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        dtype = torch.float32
        extended_encoder_attention_mask = encoder_attention_mask[
            :, None, :, None,
        ]
        extended_encoder_attention_mask = extended_encoder_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_encoder_attention_mask = (
            1.0 - extended_encoder_attention_mask
        ) * torch.finfo(dtype).min
        cross_attention_outputs = self.adapter_crossattention(
            hidden_states=layer_output,
            encoder_hidden_states=encoder_hidden_states,  # encoder_hidden_states_proj,
            # encoder_attention_mask=attention_mask #if not attention_mask.any() else None,#encoder_attention_mask,
            encoder_attention_mask=extended_encoder_attention_mask,  # attention_mask, #
        )
        conditioning_output = cross_attention_outputs[0]
        return conditioning_output
    
    """
    Cross-attention with copied global embeddings
    """
    def expanded_cross_attention_conditioning_chunk(self, layer_output, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(-1, layer_output.size(1), -1)
        dtype = torch.float32
        extended_encoder_attention_mask = encoder_attention_mask[
            :, None, None, :
        ]
        extended_encoder_attention_mask = extended_encoder_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_encoder_attention_mask = (
            1.0 - extended_encoder_attention_mask
        ) * torch.finfo(dtype).min
        cross_attention_outputs = self.adapter_crossattention(
            hidden_states=layer_output,
            encoder_hidden_states=encoder_hidden_states,  # encoder_hidden_states_proj,
            # encoder_attention_mask=attention_mask #if not attention_mask.any() else None,#encoder_attention_mask,
            encoder_attention_mask=extended_encoder_attention_mask,  # attention_mask, #
        )
        conditioning_output = cross_attention_outputs[0]
        return conditioning_output
    
    """
    Trivial sum conditioning for debug.
    """
    def sum_conditioning_chunk(self, layer_output, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(-1, layer_output.size(1), -1)
        conditioning_output = layer_output + encoder_hidden_states
        return conditioning_output
    
    """
    Ignore conditioning for debug.
    """
    def ignore_conditioning_chunk(self, layer_output, encoder_hidden_states, encoder_attention_mask):
        return layer_output


class GlobalAdapterLayerMini(GlobalAdapterLayer):
    def __init__(self, cfg, config):
        adapter_config = deepcopy(config)
        adapter_config.intermediate_size = getattr(cfg, "adapter_intermediate_size")
        adapter_config.hidden_size = getattr(cfg, "adapter_hidden_size")
        super().__init__(cfg, config, adapter_config=adapter_config, qdim=config.hidden_size)
        # "adapter_" prefix is required so the freeze rule
        # ('adapter' not in pname -> requires_grad=False) keeps these trainable.
        self.adapter_downsize_layer = nn.Linear(config.hidden_size, adapter_config.hidden_size)
        self.adapter_upsize_layer = nn.Linear(adapter_config.hidden_size, config.hidden_size)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # self-attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first feed forward chunk
        layer_output = self.feed_forward_chunk(attention_output)

        # adapter begins
        residual = layer_output

        # conditioning with encoder output
        conditioning_output = self.conditioning_chunk(layer_output, encoder_hidden_states, encoder_attention_mask)

        # second feed forward chunk
        conditioning_output = self.adapter_downsize_layer(conditioning_output) # Added in the Mini version
        ffn_output = self.adapter_feed_forward_chunk(conditioning_output)
        ffn_output = self.adapter_upsize_layer(ffn_output) # Added in the Mini version
        ffn_output += residual

        outputs = (ffn_output,) + outputs

        return outputs


class GlobalAdapterEsmSelfAttention(EsmSelfAttention):
    def __init__(
        self, config, position_embedding_type=None, qdim=None, kdim=None, vdim=None
    ):
        super().__init__(config, position_embedding_type)
        if qdim is not None:
            self.query = nn.Linear(qdim, self.all_head_size)
        if kdim is not None:
            self.key = nn.Linear(kdim, self.all_head_size)
        if vdim is not None:
            self.value = nn.Linear(vdim, self.all_head_size)


class GlobalAdapterEsmAttention(EsmAttention):
    def __init__(self, config, qdim=None, kdim=None, vdim=None):
        super().__init__(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size if qdim is None else qdim, eps=config.layer_norm_eps)
        self.self = GlobalAdapterEsmSelfAttention(config, qdim=qdim, kdim=kdim, vdim=vdim)
        if qdim is not None:
            self.output.dense = nn.Linear(self.self.all_head_size, qdim)
