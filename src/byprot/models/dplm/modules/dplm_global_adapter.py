# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
from copy import deepcopy
from dataclasses import dataclass, field

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
from byprot.models.utils import NetConfig, get_net

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
    encoder_conditioning_mode: str = field(default="cross_attention")  # cross_attention, expanded_cross_attention, sum, ignore
    dplm_name: str = field(default="")
    from_huggingface: bool = field(default=True)
    net: NetConfig = field(default=NetConfig())
    lora: bool = field(default=True)


class DPLMWithConditionalGlobalAdapter(nn.Module):
    _default_cfg = DPLMWithGlobalAdapterConfig()

    @classmethod
    def from_pretrained(cls, cfg):
        net_override = {"conditioning_mode": cfg.encoder_conditioning_mode}
        net = DiffusionProteinLanguageModel.from_pretrained(cfg.dplm_name, from_huggingface=cfg.from_huggingface, net_override=net_override).net

        if cfg.encoder_conditioning_mode == "prepend":
            pass
        else:
            # change net.last_layer to GlobalAdapterLayer
            adapter = GlobalAdapterLayer(cfg, deepcopy(net.config))
            net_last_layer = net.esm.encoder.layer[-1]
            adapter.load_state_dict(net_last_layer.state_dict(), strict=False)
            net.esm.encoder.layer[-1] = adapter
            del net_last_layer

        dplm_adapter = cls(cfg, net)

        if cfg.encoder_conditioning_mode == "prepend":
            # TODO enable LoRA for prepend adapter
            pass
        else:
            for pname, param in dplm_adapter.named_parameters():
                if "adapter" not in pname:
                    param.requires_grad = False
            dplm_adapter.net.esm.encoder.emb_layer_norm_after.requires_grad_(True)
            dplm_adapter.net.esm.contact_head.requires_grad_(True)
            dplm_adapter.net.lm_head.requires_grad_(True)
        
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
        batch,
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
            else batch["prev_tokens"].ne(self.pad_id)
        )

        outputs = self.net(
            input_ids=batch["prev_tokens"],
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

        batch["prev_tokens"] = x_t
        logits = self.forward(
            batch,
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
    def __init__(self, cfg, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)

        kdim = vdim = getattr(cfg, "encoder_d_model", 512)
        config.hidden_dropout_prob = getattr(cfg, "adapter_dropout", 0.0)
        self.adapter_crossattention = GlobalAdapterEsmAttention(
            config, kdim=kdim, vdim=vdim
        )
        # config.intermediate_size = config.hidden_size // 2 # Notes: bottleneck ffn
        self.adapter_intermediate = EsmIntermediate(config)
        self.adapter_output = EsmOutput(config)

        if cfg.lora:
            def replace_with_lora(linear):
                linear = LoraLinear(
                    linear.weight,
                    bias=linear.bias,
                )
                return linear

            self.adapter_crossattention.self.query = replace_with_lora(self.adapter_crossattention.self.query)
            self.adapter_crossattention.self.key = replace_with_lora(self.adapter_crossattention.self.key)
            self.adapter_crossattention.self.value = replace_with_lora(self.adapter_crossattention.self.value)
            self.adapter_crossattention.output.dense = replace_with_lora(self.adapter_crossattention.output.dense)
            self.adapter_intermediate.dense = replace_with_lora(self.adapter_intermediate.dense)
            self.adapter_output.dense = replace_with_lora(self.adapter_output.dense)
            

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.adapter_LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.conditioning_mode = getattr(cfg, "encoder_conditioning_mode", "cross_attention")
        self.conditioning_chunk = {
            "cross_attention": self.cross_attention_conditioning_chunk,
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


class GlobalAdapterEsmSelfAttention(EsmSelfAttention):
    def __init__(
        self, config, position_embedding_type=None, kdim=None, vdim=None
    ):
        super().__init__(config, position_embedding_type)
        if kdim is not None:
            self.key = nn.Linear(kdim, self.all_head_size)
        if vdim is not None:
            self.value = nn.Linear(vdim, self.all_head_size)


class GlobalAdapterEsmAttention(EsmAttention):
    def __init__(self, config, kdim=None, vdim=None):
        super().__init__(config)
        self.self = GlobalAdapterEsmSelfAttention(config, kdim=kdim, vdim=vdim)


class LoraLinear(nn.Module):
    def __init__(self, weight, bias=None, r=1):
        super().__init__()
        self.in_features = weight.size(1)
        self.out_features = weight.size(0)
        self.r = r
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, requires_grad=True, device=weight.device))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r, requires_grad=True, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def __repr__(self):
        return f"LoraLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, r={self.r})"

    def forward(self, x):
        lora_update = self.lora_B @ self.lora_A
        updated_weights = self.weight + lora_update
        result = nn.functional.linear(x, updated_weights, self.bias)
        return result
