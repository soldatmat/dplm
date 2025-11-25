# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
import esm

from byprot.models import register_model


@dataclass
class ClassEncoderConfig:
    output_logits: bool = False
    embedding_dim: int = 512
    n_classes: int = 0 # has to be specified in the config
    weight_path: str = ""
    frozen: bool = False


@register_model("class_encoder")
class ClassEncoder(torch.nn.Module):
    def __init__(self, n_classes, embedding_dim, weight_path=None, frozen=False, output_logits=False):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = torch.nn.Embedding(n_classes, self.embedding_dim)

        if weight_path:
            weights = torch.load(weight_path, map_location="cpu")
            state_dict = {"weight": weights}
            self.encoder.load_state_dict(state_dict)
        
        self.frozen = frozen
        if self.frozen:
            self.encoder.weight.requires_grad = False

        # alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")
            # self.out_proj = torch.nn.Linear(self.embedding_dim, len(alphabet))
    
    # TODO add from_pretrained option

    def train(self, mode: bool = True):
        if self.frozen:
            self.encoder.train(False)
        else:
            self.encoder.train(mode)
        return self

    def forward(self, class_ids, output_logits=False, **kwargs):
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")

        encoder_out = self.encoder(class_ids)

        return encoder_out
        # if output_logits:
        #     logits = self.out_proj(encoder_out)
        #     return logits, encoder_out
        # else:
        #     return encoder_out
