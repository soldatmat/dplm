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
    n_classes: int = 0 # Has to be specified in the config


@register_model("class_encoder")
class ClassEncoder(torch.nn.Module):
    def __init__(self, n_classes, embedding_dim, freeze=True, output_logits=False,):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = torch.nn.Embedding(n_classes, self.embedding_dim)

        # if freeze:
            # TODO

        # alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")
            # self.out_proj = torch.nn.Linear(self.embedding_dim, len(alphabet))
    
    # TODO add from_pretrained option

    def forward(self, batch, output_logits=False, **kwargs):
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")

        encoder_out = self.encoder(batch["class_ids"])

        return encoder_out
        # if output_logits:
        #     logits = self.out_proj(encoder_out)
        #     return logits, encoder_out
        # else:
        #     return encoder_out
