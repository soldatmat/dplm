# embedding_space — TPS embedding / similarity landscape figures

Code for the "space-exploration" figures in the DPLM TPS deck: 2D maps of the
MARTS-DB training terpene synthases and of DPLM-generated sequences, in three
different similarity spaces.

## Data & output locations (read this first)

This dir holds **code only**. The input CSVs and the output PNGs live with the
rest of the generated-sequence eval data:

    ../../run/class_predictor/slide306_eval/                 # inputs + ESM/overlay PNGs
    ../../run/class_predictor/slide306_eval/similarity_data/ # seq/struct matrices + maps

The scripts reference those by absolute path, so they run from anywhere. The
class labels / reference embeddings come from the **canonical** standalone repo
`../../../tps-first-cyclization-knn` (its bundled `data/` is md5-identical to the
dplm data-bin copy). Run everything with the local base `/opt/miniconda3` python3
(has pandas, scikit-learn, matplotlib, python-pptx).

## Scripts → figures → slides (presentation/dplm.pptx)

| Script | Produces | Slide(s) |
|--------|----------|----------|
| `make_pca_tsne_fig.py` | ESM mean-embedding PCA + t-SNE map of the 1349 training TPSs, coloured by first-cyclization class (carbon-ordered, substrate-type hue families) | **#314** |
| `make_gen_overlay_figs.py` | 5 overlays of DPLM-generated seqs projected into the train ESM space: class-0 overview + per-arch CA/PRE/MINI, and class-1 MINI | **#316–#320** |
| `make_similarity_maps.py` | Sequence-identity map (mmseqs all-vs-all) and structural-similarity map (foldseek TM-score all-vs-all), each PCoA + t-SNE(precomputed) | (appended) |

### Deck writers (one open + one save; never run two concurrently)

| Script | Does |
|--------|------|
| `update_deck.py` | Swaps the slide-#314 image and appends the 5 overlay slides in a single pass |
| `replace_pca_tsne_image.py` | In-place image swap on slide #314 (superseded by `update_deck.py`) |
| `append_pca_tsne_slide.py` | Original single-slide appender for slide #314 (superseded by `update_deck.py`) |

**Deck discipline:** PowerPoint must be **closed** (a `presentation/~$dplm.pptx`
lock means it's open). All writers back up to `/tmp` first and title-match slides
(robust to index shifts).

## Methods notes

- **ESM map** (`make_pca_tsne_fig.py`): z-score per dim, PCA via SVD; t-SNE on the
  top-50 PCs (`init='pca'`, perplexity 30). Colour = first-cyclization class;
  hue family = substrate type, ordered by precursor carbon count
  (mono C10 → sesqui C15 → di C20 → sester C25 → sterol C30).
- **Overlays** (`make_gen_overlay_figs.py`): generated-seq embeddings (extracted
  with the run_41 V step-200000 encoder — same space as the reference) projected
  into the fixed train PCA (exact out-of-sample) + a per-class joint t-SNE. Train
  shown as light-grey context with the conditioned/target class in gold; baseline
  as black ✕; conditional models in distinct colours.
- **Similarity maps** (`make_similarity_maps.py`): distance = `1 − similarity`
  (sequence: mmseqs `fident`; structure: foldseek `alntmscore`), missing pairs →
  max distance. Embeds **unique** enzymes via PCoA (classical MDS) + t-SNE
  (`metric='precomputed'`), then expands to the label-rows for colouring so all
  three maps are comparable. NB: the foldseek table reports only ~top-100
  neighbours/query, so its distance matrix is saturated and **PCoA variance is
  low — read the t-SNE panel** for the structural map.

The all-vs-all inputs are produced on Karolina (foldseek from the 2026-06-08
campaign self-search; mmseqs in the `dplm` conda env). See the project `Runs.md`
for the cluster jobs.
