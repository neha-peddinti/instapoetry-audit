# Instapoetry aesthetic clustering proof of concept

This project aims to look at linguistic convergence while interacting with different types of Instagram poetry content. Current code computes a "linguistic feature representation" vector for each post, and performs PCA to examine whether poems cluster into distinct aesthetics as a proof-of-concept.

## Project structure

- `data/poems.json`: poem data with each item as an object containing an `id` and a list of lines.
- `src/linguistic_features.py`: feature extraction logic for lexical, syntactic, and semantic measures. Details in paper.
- `src/pca_analysis.py`: loads the poem data, computes the feature matrix, runs PCA, and saves a plot.
- `data/lexicons/`: word lists used by `linguistic_features.py` (see below).
- `src/lexicons/`: code that builds `data/lexicons/style_lexicon.csv`.

## Lexicons

| File | Features | Source | License |
|---|---|---|---|
| `style_lexicon.csv` | `lit_col_score`, `subj_obj_score` | Built by `src/lexicons/build_style_lexicon.py` (Brooke, Wang & Hirst 2010 seed-similarity method) | — |
| `concreteness.txt` | `concreteness` | Brysbaert et al. (2014) | Research use |
| `nrc_emolex.json` | `nrc_positive`, `nrc_negative` | NRC Emotion Lexicon, Mohammad & Turney (2013), via the NRCLex package | Free for research |
| `empath_categories.tsv` | `empath_health`, `empath_content` | Empath built-in categories, Fast et al. (2016) | MIT |
| `socal/` | style lexicon build only (subjectivity) | SO-CAL, Taboada et al. (2011) | CC BY-NC-SA 4.0 |

To rebuild the style lexicon (needs `python -m spacy download en_core_web_lg` for word vectors):

```bash
python3 src/lexicons/build_style_lexicon.py data/lexicons/style_lexicon.csv
```

The literary, colloquial, and objective seed lists in `src/lexicons/seeds.py` were drafted for this project and should be reviewed.

## Data format

First need to populate `data/poems.json` with entries like this:

```json
[
  {
    "id": "poem_0xx",
    "lines": [
      "line 1",
      "line 2",
      "line 3"
    ]
  }
]
```

## Running PCA analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/pca_analysis.py --data data/poems.json --output outputs/pca_poem_projection.png
```

