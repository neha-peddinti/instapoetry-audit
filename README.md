# Instapoetry aesthetic clustering proof of concept

This project is a lightweight pipeline for exploring linguistic similarity among Instagram poetry samples. The goal is to represent each poem as a numeric feature vector, reduce the dimensionality with PCA, and examine whether poems cluster into distinct aesthetics.

## Project structure

- `data/poems.json`: poem data with each item as an object containing an `id` and a list of lines.
- `src/linguistic_features.py`: feature extraction logic for lexical, syntactic, and semantic measures.
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
| `socal/` | style lexicon build only (subjective seeds) | SO-CAL, Taboada et al. (2011) | CC BY-NC-SA 4.0 |
| `warriner.csv` | not used yet | Valence/arousal/dominance norms, Warriner et al. (2013) | Research use |

To rebuild the style lexicon (needs `python -m spacy download en_core_web_lg` for word vectors):

```bash
python3 src/lexicons/build_style_lexicon.py data/lexicons/style_lexicon.csv
```

The literary, colloquial, and objective seed lists in `src/lexicons/seeds.py` were drafted for this project and should be reviewed.

## Data format

Populate `data/poems.json` with entries like this:

```json
[
  {
    "id": "poem_001",
    "lines": [
      "you are the softest thing",
      "i have ever known",
      "and still i am learning how to stay"
    ]
  }
]
```

Do not invent sample data for the proof of concept. Fill this in with your actual poems only.

## Feature design

The current implementation uses a compact feature set that is easy to expand and aligns with the kind of linguistic dimensions discussed in the project: lexical diversity, function-word frequency, line rhythm, punctuation, pronoun use, and theme keywords related to love, heartbreak, grief, and identity.

You can adjust the feature list in `src/linguistic_features.py` to match the exact coding scheme from your citation document.

## Run the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/pca_analysis.py --data data/poems.json --output outputs/pca_poem_projection.png
```

The script will print per-poem PCA coordinates and save a scatter plot of poems in the first two principal components.

## Notes

- The dataset starts empty intentionally so you can populate it with real poem texts without generating synthetic content.
- The current version is deliberately minimal and suitable for a proof of concept or initial exploratory analysis.
