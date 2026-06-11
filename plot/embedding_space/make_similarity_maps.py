#!/usr/bin/env python3
"""Sequence- and structure-similarity 2D maps of the MARTS-DB training TPSs.

Mirrors the ESM-embedding map (make_pca_tsne_fig.py) but derives the 2D layout
from PAIRWISE SIMILARITY instead of ESM mean-embeddings:
  - sequence:  mmseqs all-vs-all  -> distance = 1 - fraction-identical (fident)
  - structure: foldseek all-vs-all -> distance = 1 - TM-score (alntmscore)

Each map has two panels: PCoA (classical MDS) + t-SNE(metric='precomputed'),
coloured by first-cyclization class with the SAME carbon-ordered, substrate-type
grouped palette as the ESM map. We embed UNIQUE enzymes (one structure/sequence
each) and then expand to the 1349 label-rows for colouring, so the three maps
(ESM / sequence / structure) are directly comparable.

Inputs come from slide306_eval/similarity_data/ (produced on Karolina). Outputs
go there too. Run locally with the base /opt/miniconda3 python3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

# palette + grouping reused from the ESM-map script (same dir)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import build_class_colors, make_palette, TYPE_ORDER  # noqa: E402

# class labels from the canonical kNN repo (same 1349 label-rows as the ESM map)
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference, N_CLASSES  # noqa: E402

DATA = Path("/Users/soldatmat/Documents/terpene_synthases/dplm/run/"
            "class_predictor/slide306_eval/similarity_data")
FOLDSEEK_TSV = DATA / "foldseek_marts_self_alignments.tsv"   # all-pairs
MMSEQS_TSV = DATA / "mmseqs_marts_allvall.tsv"               # all-pairs
OUT_SEQ = DATA / "seq_similarity_map.png"
OUT_STRUCT = DATA / "struct_similarity_map.png"

# documented column orders of the two all-pairs tables (no header in either)
FOLDSEEK_COLS = ["query", "target", "alntmscore", "qtmscore", "ttmscore", "lddt"]
MMSEQS_COLS = ["query", "target", "fident", "pident", "alnlen",
               "qcov", "tcov", "evalue", "bits"]


# --------------------------------------------------------------------------- #
def load_pairs(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read an all-pairs TSV, tolerating an optional header row."""
    first = path.read_text().splitlines()[0].split("\t")
    has_header = first[0].strip().lower() in ("query", "q") and not first[0].startswith("marts_")
    if has_header:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=cols)
    return df


def build_distance(df: pd.DataFrame, ids: list[str], sim_col: str) -> np.ndarray:
    """Symmetric distance matrix (1 - similarity) over `ids`; missing pairs -> 1."""
    pos = {i: k for k, i in enumerate(ids)}
    n = len(ids)
    S = np.zeros((n, n), dtype=float)
    q = df["query"].to_numpy()
    t = df["target"].to_numpy()
    s = df[sim_col].to_numpy(dtype=float)
    for qi, ti, si in zip(q, t, s):
        i = pos.get(qi)
        j = pos.get(ti)
        if i is None or j is None:
            continue
        if si > S[i, j]:
            S[i, j] = si
            S[j, i] = si
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 1.0)
    D = 1.0 - S
    np.clip(D, 0.0, 1.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def pcoa(D: np.ndarray):
    """Classical MDS (principal coordinates): first 2 axes + % variance."""
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0.0))
    pos_sum = w[w > 0].sum()
    pct = 100 * w[:2] / pos_sum if pos_sum > 0 else np.array([0.0, 0.0])
    return coords, pct


def tsne_precomputed(D: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, metric="precomputed", init="random",
                perplexity=30, random_state=0).fit_transform(D)


def grouped_legend(fig, cmap):
    """Legend grouped by substrate type (carbon order), matching the ESM map."""
    _, by_type = build_class_colors()
    handles, labels = [], []
    for ty in TYPE_ORDER:
        ids = by_type[ty]
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0))
        labels.append(f"$\\bf{{{ty}}}$  ({len(ids)} cls)")
        for c in ids:
            handles.append(Line2D([0], [0], marker="o", linestyle="",
                                  markerfacecolor=cmap(c), markeredgecolor="none",
                                  markersize=7))
            labels.append(f"   {c:>2d}")
    fig.legend(handles=handles, labels=labels,
               title="first-cyclization class\n(grouped by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, title_fontsize=9,
               ncol=1, frameon=True, framealpha=0.9, handletextpad=0.4,
               labelspacing=0.35)


def make_map(D, ids, ref_df, cmap, kind_title, metric_note, out_png):
    """Embed unique enzymes (PCoA + t-SNE), expand to label-rows, plot 2 panels."""
    coords_pcoa, pct = pcoa(D)
    coords_tsne = tsne_precomputed(D)
    pos = {i: k for k, i in enumerate(ids)}

    # expand to the label-rows present in this similarity set (mirror ESM figure)
    keep = ref_df[ref_df["id"].isin(pos)]
    row_idx = keep["id"].map(pos).to_numpy()
    y = keep["cls"].to_numpy()
    n_rows, n_uni = len(keep), len(ids)

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)
    for ax, coords, ttl in (
        (axes[0], coords_pcoa, f"PCoA  (axis1 {pct[0]:.1f}%, axis2 {pct[1]:.1f}%)"),
        (axes[1], coords_tsne, "t-SNE  (precomputed dist, perplexity 30)"),
    ):
        ax.scatter(coords[row_idx, 0], coords[row_idx, 1], c=y, cmap=cmap,
                   vmin=-0.5, vmax=N_CLASSES - 0.5, s=10, alpha=0.8, linewidths=0)
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PCo1"); axes[0].set_ylabel("PCo2")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    grouped_legend(fig, cmap)
    fig.suptitle(
        f"Training TPSs in {kind_title} space "
        "— coloured by first-cyclization class (hue = substrate type)",
        fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005,
             f"{metric_note}; {n_uni} unique enzymes, {n_rows} label-rows shown",
             fontsize=9, style="italic", ha="left", va="bottom")

    if not (str(out_png).startswith("/Volumes/") and not Path("/Volumes/data").exists()):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"saved {out_png}")
    plt.close(fig)
    return n_uni, n_rows


def main():
    _, y, groups = load_reference()
    ref_df = pd.DataFrame({"id": np.asarray(groups).astype(str), "cls": y})
    cmap = make_palette(N_CLASSES)
    print(f"label-rows: {len(ref_df)}  unique enzymes: {ref_df['id'].nunique()}")

    # ---- sequence-similarity map ----
    seq = load_pairs(MMSEQS_TSV, MMSEQS_COLS)
    seq_ids = sorted(set(seq["query"].astype(str)) | set(seq["target"].astype(str)))
    Dseq = build_distance(seq, seq_ids, "fident")
    print(f"[seq] pairs={len(seq)} ids={len(seq_ids)} "
          f"mean off-diag dist={Dseq[~np.eye(len(seq_ids),dtype=bool)].mean():.3f}")
    make_map(Dseq, seq_ids, ref_df, cmap,
             "pairwise SEQUENCE-identity (mmseqs)",
             "distance = 1 - mmseqs fraction-identical (fident)", OUT_SEQ)

    # ---- structure-similarity map ----
    st = load_pairs(FOLDSEEK_TSV, FOLDSEEK_COLS)
    st_ids = sorted(set(st["query"].astype(str)) | set(st["target"].astype(str)))
    Dst = build_distance(st, st_ids, "alntmscore")
    print(f"[struct] pairs={len(st)} ids={len(st_ids)} "
          f"mean off-diag dist={Dst[~np.eye(len(st_ids),dtype=bool)].mean():.3f}")
    make_map(Dst, st_ids, ref_df, cmap,
             "pairwise STRUCTURAL-similarity (foldseek TM-score)",
             "distance = 1 - foldseek alntmscore", OUT_STRUCT)


if __name__ == "__main__":
    main()
