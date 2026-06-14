# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# A3 Option A: same-class-neighbor exemplar conditioning.
#
# Replaces the class-index Embedding lookup of ClassEncoder with a trainable
# PROJECTOR over a FROZEN, precomputed 640-d ESM mean embedding (run_41V):
#   - at TRAINING, the datamodule supplies `cond_emb` = the mean embedding of a
#     randomly-sampled DIFFERENT same-class enzyme (blocks trivial copying);
#   - at INFERENCE, only `class_ids` are available, so the encoder looks up the
#     per-class MEDOID embedding (a real on-manifold class member).
# The embedding SOURCE is frozen; only the projector (Linear / 2-layer MLP)
# trains.

from dataclasses import dataclass

import torch

from byprot import utils
from byprot.models import register_model

log = utils.get_logger(__name__)


@dataclass
class NeighborEncoderConfig:
    output_logits: bool = False
    embedding_dim: int = 640  # = encoder_d_model (model dim the projector maps to)
    source_dim: int = 640  # dim of the precomputed neighbor/medoid embeddings
    n_classes: int = 0  # has to be specified in the config
    # Path to the precompute_neighbor_conditioning.py artifact (.pt) holding
    # {class_medoid: [n_classes, source_dim]}. Required.
    neighbor_artifact_path: str = ""
    # Projector: "linear" (640->emb) or "mlp" (640->emb->emb with GELU).
    projector: str = "linear"
    projector_hidden_dim: int = 640
    # A4: classifier-free guidance over the NEIGHBOR conditioning path. When
    # cfg_dropout > 0, during training each example's projected neighbor
    # embedding is replaced with a single LEARNED null vector with this
    # probability; at inference force_null=True returns the same null. The null
    # lives in the OUTPUT (projected) space, so it composes with the existing
    # logit-space CFG generate path unchanged. 0.0 (default) -> behaviour
    # identical to the original neighbor-only NeighborEncoder (runs 3/4/5).
    cfg_dropout: float = 0.0


@register_model("class_encoder_neighbor")
class NeighborEncoder(torch.nn.Module):
    """Projects a (frozen) 640-d exemplar embedding into the conditioning space.

    forward() accepts a `cond_emb` kwarg (the per-example neighbor embedding,
    supplied by the datamodule during training). When `cond_emb` is None
    (inference), it falls back to the per-class medoid looked up by `class_ids`.
    """

    def __init__(
        self,
        n_classes,
        embedding_dim,
        source_dim=640,
        neighbor_artifact_path="",
        projector="linear",
        projector_hidden_dim=640,
        output_logits=False,
        cfg_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.source_dim = source_dim

        if output_logits:
            raise NotImplementedError("output_logits=True is not implemented.")

        if not neighbor_artifact_path:
            raise ValueError(
                "NeighborEncoder requires neighbor_artifact_path (the "
                "precompute_neighbor_conditioning.py .pt artifact)."
            )
        artifact = torch.load(neighbor_artifact_path, map_location="cpu")
        class_medoid = artifact["class_medoid"].float()  # [n_classes, source_dim]
        assert class_medoid.shape[1] == source_dim, (
            f"medoid source_dim {class_medoid.shape[1]} != cfg.source_dim {source_dim}"
        )
        if n_classes and class_medoid.shape[0] != n_classes:
            log.warning(
                f"NeighborEncoder: artifact has {class_medoid.shape[0]} classes "
                f"but cfg n_classes={n_classes}."
            )
        # Frozen inference exemplar table (per-class medoid). Buffer, not a
        # parameter -- never trained, travels with the checkpoint.
        self.register_buffer("class_medoid", class_medoid)

        # Trainable projector 640 -> embedding_dim.
        if projector == "linear":
            self.projector = torch.nn.Linear(source_dim, embedding_dim)
        elif projector == "mlp":
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(source_dim, projector_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(projector_hidden_dim, embedding_dim),
            )
        else:
            raise ValueError(f"Unknown projector type: {projector}")

        # A4: single LEARNED null embedding in the OUTPUT (projected) space, used
        # for classifier-free guidance over the neighbor path. Always trainable.
        # Initialised small so it starts near the origin without being zeros.
        # Mirrors ClassEncoder.null_embedding so dplm_class.generate's force_null
        # / logit-space CFG path works identically for both encoders.
        self.cfg_dropout = cfg_dropout
        self.null_embedding = torch.nn.Parameter(
            torch.randn(1, embedding_dim) * 0.02
        )

    def forward(
        self,
        class_ids,
        cond_emb=None,
        output_logits=False,
        force_null=False,
        **kwargs,
    ):
        if output_logits:
            raise NotImplementedError("output_logits=True is not implemented.")
        if cond_emb is None:
            # Inference path: look up the per-class medoid by class id.
            cond_emb = self.class_medoid[class_ids]
        cond_emb = cond_emb.to(self.projector[0].weight.dtype
                               if isinstance(self.projector, torch.nn.Sequential)
                               else self.projector.weight.dtype)
        encoder_out = self.projector(cond_emb)

        # force_null=True (inference, the unconditional CFG branch) replaces
        # every example with the learned null embedding.
        if force_null:
            return self.null_embedding.to(encoder_out.dtype).expand(
                encoder_out.size(0), -1
            )

        # Condition dropout (training only): with prob cfg_dropout swap an
        # example's projected neighbor embedding for the learned null embedding.
        if self.training and self.cfg_dropout > 0.0:
            drop = torch.rand(
                encoder_out.size(0), device=encoder_out.device
            ) < self.cfg_dropout
            null = self.null_embedding.to(encoder_out.dtype).expand(
                encoder_out.size(0), -1
            )
            encoder_out = torch.where(drop.unsqueeze(1), null, encoder_out)

        return encoder_out
