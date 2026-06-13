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
    # A3 Option D (classifier-free guidance / condition dropout). When > 0,
    # during training each example's class embedding is replaced with a single
    # LEARNED null embedding with this probability (not zeros, not the mean).
    # 0.0 (default) -> behaviour identical to the original ClassEncoder.
    cfg_dropout: float = 0.0


@register_model("class_encoder")
class ClassEncoder(torch.nn.Module):
    def __init__(self, n_classes, embedding_dim, weight_path=None, frozen=False, output_logits=False, cfg_dropout=0.0):
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

        # A3 Option D: single LEARNED null embedding used for classifier-free
        # guidance. Always trainable (independent of `frozen`, which only
        # governs the per-class label embeddings). Initialised small so it
        # starts near the origin of the conditioning space without being zeros.
        self.cfg_dropout = cfg_dropout
        self.null_embedding = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim) * 0.02
        )

        # alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")
            # self.out_proj = torch.nn.Linear(self.embedding_dim, len(alphabet))

    # TODO add from_pretrained option

    def train(self, mode: bool = True):
        # The per-class label embedding follows `frozen`; the rest of the
        # module (the learned null embedding) follows `mode` as usual.
        super().train(mode)
        if self.frozen:
            self.encoder.train(False)
        else:
            self.encoder.train(mode)
        return self

    def forward(self, class_ids, output_logits=False, force_null=False, **kwargs):
        if output_logits:
            raise NotImplementedError("output_encoder_logits=True is not implemented.")

        encoder_out = self.encoder(class_ids)

        # force_null=True (inference, the unconditional CFG branch) replaces
        # every example with the learned null embedding.
        if force_null:
            return self.null_embedding.expand(encoder_out.size(0), -1)

        # Condition dropout (training only): with prob cfg_dropout swap an
        # example's class embedding for the learned null embedding.
        if self.training and self.cfg_dropout > 0.0:
            drop = torch.rand(
                encoder_out.size(0), device=encoder_out.device
            ) < self.cfg_dropout
            null = self.null_embedding.expand(encoder_out.size(0), -1)
            encoder_out = torch.where(drop.unsqueeze(1), null, encoder_out)

        return encoder_out
        # if output_logits:
        #     logits = self.out_proj(encoder_out)
        #     return logits, encoder_out
        # else:
        #     return encoder_out
