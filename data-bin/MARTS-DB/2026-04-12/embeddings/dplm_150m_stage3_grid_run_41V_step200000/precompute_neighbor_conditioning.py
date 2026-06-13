"""
Precompute artifacts for A3 same-class-neighbor (Option A) conditioning.

Reads:
  - TPS_first_cyclization.csv            (data, keyed by Enzyme_marts_ID + First_cyclization_product)
  - TPS_first_cyclization_embeddings_mean.csv   (per-enzyme mean ESM emb, 640-d, keyed by id=Enzyme_marts_ID)

Writes (next to the embeddings csv):
  - neighbor_conditioning_emb.pt   dict:
        emb           FloatTensor [N_enz, 640]   per-enzyme mean embedding
        enzyme_ids    list[str]                  row -> Enzyme_marts_ID  (index into emb)
        class_names   list[str]                  sorted First_cyclization_product (id = position)
        class_medoid  FloatTensor [n_classes, 640]  per-class medoid (real member nearest centroid)
        medoid_enzyme list[str]                  enzyme id of each class medoid

The medoid for class c is the real class member whose embedding is closest
(L2) to the class centroid -- an on-manifold inference exemplar, per A3.

Run on the cluster (cwd this dir) with the dplm env python; pure pandas/torch.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "..", "TPS_first_cyclization.csv")
EMB_CSV = os.path.join(HERE, "TPS_first_cyclization_embeddings_mean.csv")
OUT = os.path.join(HERE, "neighbor_conditioning_emb.pt")

CLASS_COLUMN = "First_cyclization_product"
ID_COLUMN = "Enzyme_marts_ID"


def main():
    data = pd.read_csv(DATA_CSV)
    # The dataset's class2id is the sorted unique of the class column.
    class_names = sorted(data[CLASS_COLUMN].unique())
    class2id = {v: i for i, v in enumerate(class_names)}

    emb = pd.read_csv(EMB_CSV).drop_duplicates(subset="id")
    feat_cols = [c for c in emb.columns if c != "id"]
    assert len(feat_cols) == 640, f"expected 640 feature cols, got {len(feat_cols)}"

    enzyme_ids = emb["id"].tolist()
    emb_t = torch.tensor(emb[feat_cols].to_numpy(dtype=np.float32))
    id2row = {eid: i for i, eid in enumerate(enzyme_ids)}

    # Per-class set of enzyme ids (an enzyme can belong to several classes via
    # different substrate rows; treat each (enzyme, class) membership).
    members = {i: set() for i in range(len(class_names))}
    for _, r in data[[ID_COLUMN, CLASS_COLUMN]].drop_duplicates().iterrows():
        cid = class2id[r[CLASS_COLUMN]]
        if r[ID_COLUMN] in id2row:
            members[cid].add(r[ID_COLUMN])

    class_medoid = torch.zeros(len(class_names), 640, dtype=torch.float32)
    medoid_enzyme = []
    for cid in range(len(class_names)):
        ids = sorted(members[cid])
        rows = [id2row[e] for e in ids]
        sub = emb_t[rows]
        centroid = sub.mean(dim=0, keepdim=True)
        d = torch.cdist(sub, centroid).squeeze(1)
        j = int(torch.argmin(d).item())
        class_medoid[cid] = sub[j]
        medoid_enzyme.append(ids[j])

    out = {
        "emb": emb_t,
        "enzyme_ids": enzyme_ids,
        "class_names": class_names,
        "class_medoid": class_medoid,
        "medoid_enzyme": medoid_enzyme,
    }
    torch.save(out, OUT)
    print(f"Wrote {OUT}")
    print(f"  n_enzymes={emb_t.shape[0]} n_classes={len(class_names)}")
    print(f"  class sizes: {[len(members[c]) for c in range(len(class_names))]}")
    print(f"  medoid enzymes: {medoid_enzyme}")


if __name__ == "__main__":
    main()
