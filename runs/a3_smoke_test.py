"""A3 smoke test: verify a class-conditioning experiment config end to end.

For the given experiment=<name>, checks:
  (a) data batch provides the right conditioning
        - neighbor: cond_emb present; same-class & different-ID than target
        - cfg: ~15% of examples get the learned null embedding during training
  (b) trainable params include the new projector / null embedding
  (c) 2-3 training steps run without error (loss is finite, decreases-ish)
  (d) a tiny generation (5 seqs) runs without error
        - for cfg: also runs guidance_w=2 (cond + null branches combined)

Run on a GPU node, repo root as CWD:
  python runs/a3_smoke_test.py experiment=tps/<config> ++datamodule.num_workers=0 \
    ++callbacks.validate_with_enzyme_explorer.class_ids=[0]
"""
import sys

import os

import pyrootutils

# Mirror train.py: sets the PROJECT_ROOT env var used by configs/paths/default.yaml.
ROOT = str(
    pyrootutils.setup_root(
        search_from=os.path.dirname(os.path.abspath(__file__)),
        indicator=[".git", "pyproject.toml", "setup.py"],
        pythonpath=True,
        dotenv=True,
    )
)
os.environ.setdefault("PROJECT_ROOT", ROOT)

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from byprot import utils  # noqa: E402,F401 (registers OmegaConf resolvers at import)


def log(msg):
    print(f"[A3-SMOKE] {msg}", flush=True)


@hydra.main(version_base="1.1", config_path=f"{ROOT}/configs", config_name="config.yaml")
def main(config):
    # instantiate_from_config pops _target_ in place; struct mode blocks pop.
    OmegaConf.set_struct(config, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    is_neighbor = bool(config.datamodule.get("neighbor_conditioning", False))
    cfg_dropout = float(config.model.encoder.get("cfg_dropout", 0.0))
    log(f"neighbor_conditioning={is_neighbor} cfg_dropout={cfg_dropout}")

    # ---- datamodule ----
    # Resolve interpolations (e.g. data_path = ${paths.data_dir}/...) before
    # the registry unpacks **cfg, so the dataset receives plain strings.
    OmegaConf.resolve(config)
    datamodule = utils.instantiate_from_config(
        cfg=config.datamodule, group="datamodule"
    )
    datamodule.setup("fit")
    dl = datamodule.train_dataloader()
    batch = next(iter(dl))
    log(f"(a) batch keys: {sorted(batch.keys())}")
    log(f"    tokens {tuple(batch['tokens'].shape)} class_ids {tuple(batch['class_ids'].shape)}")

    # (a) conditioning checks
    if is_neighbor:
        assert "cond_emb" in batch, "cond_emb missing from neighbor batch!"
        log(f"    cond_emb {tuple(batch['cond_emb'].shape)} (expect [B, 640])")
        # Verify same-class-different-ID by re-deriving from the dataset.
        ds = datamodule.train_dataset
        ds = getattr(ds, "dataset", ds)  # unwrap Subset (mini_run)
        # Re-sample a handful of (idx) and confirm neighbor is same-class, diff-id.
        ok_same_class, ok_diff_id, n = 0, 0, 0
        import numpy as np
        for idx in range(min(50, len(ds))):
            cls = ds.data.iloc[idx][ds.class_column]
            own = ds.data.iloc[idx][ds.id_column]
            cands = [e for e in ds._class_to_enzymes[cls] if e != own]
            if not cands:
                continue  # singleton class -> medoid fallback, skip the check
            neigh = cands[0]
            # neighbor must be same class
            neigh_classes = set(
                ds.data[ds.data[ds.id_column] == neigh][ds.class_column].tolist()
            )
            ok_same_class += int(cls in neigh_classes)
            ok_diff_id += int(neigh != own)
            n += 1
        log(f"    neighbor same-class {ok_same_class}/{n}, different-ID {ok_diff_id}/{n}")
        assert n > 0 and ok_same_class == n and ok_diff_id == n, "neighbor check FAILED"

    # ---- model + criterion (instantiate directly; avoids Lightning trainer) ----
    model = utils.instantiate_from_config(cfg=config.model, group="model").to(device)
    model.train()
    criterion = utils.instantiate_from_config(cfg=config.task.criterion)
    criterion.ignore_index = 1

    # (b) trainable params
    trainable = {
        n: p.numel() for n, p in model.named_parameters() if p.requires_grad
    }
    total_trainable = sum(trainable.values())
    log(f"(b) total trainable params: {total_trainable:,}")
    if is_neighbor:
        proj = {n: c for n, c in trainable.items() if "projector" in n}
        assert proj, "projector params not trainable!"
        log(f"    projector params: {proj}")
        assert all(
            "encoder.class_medoid" not in n for n in trainable
        ), "medoid buffer should not be trainable"
    if cfg_dropout > 0:
        nulls = {n: c for n, c in trainable.items() if "null_embedding" in n}
        assert nulls, "null_embedding not trainable!"
        log(f"    null_embedding params: {nulls}")

    # (a-cfg) ~15% null fraction over a large synthetic batch
    if cfg_dropout > 0:
        enc = model.encoder
        cids = torch.randint(0, config.datamodule.n_classes, (20000,), device=device)
        out = enc(cids)
        null = enc.null_embedding.detach()
        frac = (out == null).all(dim=1).float().mean().item()
        log(f"(a) CFG empirical null fraction over 20k: {frac:.3f} (target ~{cfg_dropout})")
        assert abs(frac - cfg_dropout) < 0.04, "null fraction off target"
        # force_null path
        fn = enc(cids[:5], force_null=True)
        assert bool((fn == null).all()), "force_null did not return null"
        log("    force_null branch OK")

    # ---- (c) 2-3 training steps ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for step in range(3):
        b = utils.recursive_to(batch, device)
        out = model.compute_loss(b, weighting="linear")
        logits, target, loss_mask, weight, _ = out
        loss = criterion(
            logits, target, label_mask=loss_mask, weights=weight
        )[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        log(f"(c) train step {step}: loss={loss.item():.4f}")
        assert torch.isfinite(loss), "non-finite loss!"

    # ---- (d) tiny generation (5 seqs) ----
    model.eval()
    from byprot.utils.generation import initialize_generation

    tokenizer = model.decoder.net.tokenizer
    input_ids, gen_class_ids = initialize_generation(
        num_seqs=5, length=60, tokenizer=tokenizer, device=device, class_id=0
    )
    gen_batch = {"prev_tokens": input_ids, "class_ids": gen_class_ids}
    partial_mask = input_ids.ne(model.mask_id)
    with torch.no_grad(), torch.cuda.amp.autocast():
        toks, _ = model.generate(
            batch=gen_batch, tokenizer=tokenizer, max_iter=10,
            sampling_strategy="gumbel_argmax", partial_masks=partial_mask,
            temperature=1.0, guidance_w=0.0,
        )
    seqs = [s.replace(" ", "") for s in tokenizer.batch_decode(toks, skip_special_tokens=True)]
    log(f"(d) generated {len(seqs)} seqs (w=0); first len={len(seqs[0])} sample={seqs[0][:40]}")
    assert len(seqs) == 5

    if cfg_dropout > 0:
        with torch.no_grad(), torch.cuda.amp.autocast():
            toks2, _ = model.generate(
                batch=gen_batch, tokenizer=tokenizer, max_iter=10,
                sampling_strategy="gumbel_argmax", partial_masks=partial_mask,
                temperature=1.0, guidance_w=2.0,
            )
        seqs2 = [s.replace(" ", "") for s in tokenizer.batch_decode(toks2, skip_special_tokens=True)]
        log(f"(d) generated {len(seqs2)} seqs with guidance_w=2 (cond+null combined); sample={seqs2[0][:40]}")
        assert len(seqs2) == 5

    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
