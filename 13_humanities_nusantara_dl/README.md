# Project 13: Humanities Classical Literature Stylometry & Nusantara Pantun Coherence DL

## Overview
This digital humanities deep learning benchmark models literary style, archaic lexical evolution, and Nusantara oral-poetic prosody across two real public literary corpora: classical English poetry (`merve/poetry`) and Indonesian *pantun* heritage (`maercaestro/pantun` + `LorthGyu/indonesian-pantun`).

## Real Literary Corpora (`src/dataset.py`)
1. **Track A -- World Classical Literature Stylometry (`merve/poetry`)**:
   - ~573 classical poems annotated with `author`, `content`, `poem name`, `age` (`Renaissance` vs `Modern`), and `type` (`Love`, `Nature`, `Mythology & Folklore`).
   - Deduplicated prior to stratified 75/10/15 splitting for two tasks:
     - **Track A1**: Literary Era Classification (`Renaissance` vs `Modern`).
     - **Track A2**: Thematic Genre Classification (`Love`, `Nature`, `Mythology & Folklore`).

2. **Track B -- Indonesian Nusantara Literary Heritage (`maercaestro/pantun` + `LorthGyu/indonesian-pantun`)**:
   - Streams up to 3,000 real 4-line Indonesian *pantun* stanzas from `maercaestro/pantun` (`structured_pantun2.csv`) combined with annotated stanzas from `LorthGyu/indonesian-pantun` (`theme`, `lines`, `rhyme_scheme`, `region`).
   - **Real vs Permuted Sampiran-Isi Coherence & Rhyme-Meter Stylometry Benchmark**:
     - Authentic *pantun* stanzas (`y=1`) are split 75/10/15 FIRST (strict zero stanza/line leakage across splits), then paired within each split against hard negatives (`y=0`):
       1. Cross-stanza *sampiran-isi* swaps (fluent lines whose `a-b-a-b` phonetic cadence and semantic pairing are broken).
       2. Intra-stanza rhyme-order perturbations (`[L1, L2, L4, L3]`, `a-b-b-a` order with identical word multisets -- exposing bag-of-words limitations).

## Architecture (`src/model.py`)
- **`DualScaleHumanitiesNet`**:
  1. **Char-Phonetic 1D Multi-Scale CNN** (kernels `3, 5, 7`): captures archaic English inflectional suffixes (`-eth`, `-est`, `thou`/`thy`), Nusantara rhyme endings (`-an`, `-ang`, `-i`), and computes pairwise cross-line suffix phonetic cosine similarities (`L1-L3`, `L2-L4`, `L1-L2`, `L3-L4`, `L1-L4`, `L2-L3`).
  2. **Word-Level BiGRU + Additive Self-Attention**: captures sequential poetic phrasing and returns per-token attention weights for close-reading interpretability.
- **Baseline Comparison**: Word + Character n-gram **TF-IDF + LogisticRegression** linear baseline evaluated on identical train/test splits for Track A1, Track A2, and Track B.

## Reproduction
```bash
# Offline CPU unit tests
pytest tests/test_humanities_dl.py -v

# Train and evaluate Track A (Era + Theme) and Track B (Nusantara Pantun)
python train.py --epochs 15 --batch-size 32 --seed 42
```

## Outputs
- `reports/metrics.json`: Accuracy, Macro-F1, ROC-AUC vs TF-IDF baseline, plus top salient attended tokens per literary class.
- `reports/humanities_stylometry.png`: 4-panel visual proof (Era/Theme/Pantun Macro-F1 comparison, Nusantara Pantun ROC curve, Renaissance vs Modern close-reading token salience, Nusantara Pantun token salience).
- `artifacts/dual_scale_era.pt`, `artifacts/dual_scale_theme.pt`, `artifacts/dual_scale_pantun.pt`: Saved model checkpoints.
