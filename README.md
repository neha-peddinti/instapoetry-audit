# Instapoetry aesthetic clustering proof of concept

This project aims to look at linguistic convergence while interacting with different types of Instagram poetry content. Current code computes a "linguistic feature representation" vector for each post, and plots the feature distributions to examine whether poems cluster into distinct aesthetics as a proof-of-concept. (PCA was unhelpful with this.)

## Project structure

- `data/poems.json`: poem data with each item as an object containing an `id` and a list of lines.
- `src/linguistic_features.py`: feature extraction logic for lexical, syntactic, and semantic measures. Details in paper.
- `src/pca_analysis.py`: loads the poem data, computes the feature matrix, runs PCA, and saves a plot.
- `data/lexicons/`: word lists used by `linguistic_features.py` (see below).
- `src/lexicons/`: code that builds `data/lexicons/style_lexicon.csv`.

## Feature set, implementation method, and source

The feature vector used in `src/linguistic_features.py` combines lexical, syntactic, stylistic, and semantic features, and it currently uses only the positive/negative labels from NRc.

| Feature | How it is computed in code | Source | Citation / method |
|---|---|---|---|
| `concreteness` | Mean concreteness rating for content words (nouns, verbs, adjectives, adverbs) after lemma lookup | `data/lexicons/concreteness.txt` | Brysbaert et al. (2014) concreteness scores|
| `absolutist` | Ratio of tokens in the custom `ABSOLUTIST` list (`always`, `never`, `completely`, etc.) | custom list (from Claude) in `src/linguistic_features.py` | Kao & Jurafsky (2012) |
| `subj_obj_score` | Ratio of subjective tokens among subjective/objective tokens after lemmatization and stopword filtering | `data/lexicons/style_lexicon.csv` built via `src/lexicons/build_style_lexicon.py` | Subjective/objective score from the SO-CAL style lexicon; from Brooke, Wang & Hirst (2010) and Taboada et al. (2011) |
| `word_freq` | Mean Zipf frequency of all alphabetic tokens | `wordfreq` package | Zipf lexical frequency used as a familiarity/word prevalence estimate |
| `mattr` | Moving-average type-token ratio over rolling 10-token window | implemented in `mattr()` in `src/linguistic_features.py` | Standard MATTR lexical diversity calculation (Covington & McFall, 2010) |
| `simple_sentence_ratio` | Fraction of sentences classified as simple by spaCy parsed patterns | spaCy sentence parsing | sentences without subordinate clauses and with one main verb |
| `complex_sentence_ratio` | Fraction of sentences with subordinate-clause markers such as `advcl`, `relcl`, `ccomp`, `xcomp`, etc. | spaCy dependency parse | Heuristic using sentence structure, following common dependency-based complexity metrics |
| `compound_sentence_ratio` | Fraction of sentences with coordination or multiple finite verbs | spaCy parse + conjunction detection | Custom metric using spaCy labels|
| `nrc_positive` | Proportion of lemmatized tokens found in the NRC positive class | `data/lexicons/nrc_emolex.json` | NRC Emotion Lexicon; Mohammad & Turney (2013); positive polarity class |
| `nrc_negative` | Proportion of lemmatized tokens found in the NRC negative class | `data/lexicons/nrc_emolex.json` | NRC Emotion Lexicon; Mohammad & Turney (2013); negative polarity class |
| `empath_psych_theme` | Proportion of lemmatized tokens matching Empath mental-state / wellbeing-related categories | `data/lexicons/empath_categories.tsv` | Empath category-based semantic lexicon; Fast et al. (2016) |
| `empath_physical_theme` | Proportion of lemmatized tokens matching Empath body / health categories | `data/lexicons/empath_categories.tsv` | Empath category-based semantic lexicon; Fast et al. (2016) |
| `n_words` | Total number of alphabetic tokens after tokenization | computed directly from the poem text | Structural word count |
| `n_lines` | Number of non-empty lines in the poem | computed directly from the poem text | Structural line count |
| `words_per_line` | `n_words / n_lines` | computed directly | Basic line-density metric |
| `lowercase_ratio` | Fraction of lines that are entirely lowercase | computed directly from the poem text | Orthographic style metric |
| `enjambment` | `1 - endstop_rate`, where endstop is the proportion of lines ending in punctuation | computed directly from line endings | Run-on / line-break style metric |

Important note: the current project uses the NRC lexicon only for the binary polarity dimensions (`positive` and `negative`) in the main feature vector. The broader NRC emotion taxonomy is available in the source lexicon, but it is not currently being used as a separate feature set in `compute_features()`.

## Lexicons and their role

| File | Used for | Source | Notes |
|---|---|---|---|
| `data/lexicons/style_lexicon.csv` | `subj_obj_score` | Built by `src/lexicons/build_style_lexicon.py` using seed lists in `src/lexicons/seeds.py` and style expansion procedures | Subjective/objective lexical polarity for stylistic texture |
| `data/lexicons/concreteness.txt` | `concreteness` | Brysbaert et al. (2014) concreteness norms | Mean concreteness ratings for content words |
| `data/lexicons/nrc_emolex.json` | `nrc_positive`, `nrc_negative` | NRC Emotion Lexicon, Mohammad & Turney (2013) | Uses the positive/negative polarity tags in the current vector |
| `data/lexicons/empath_categories.tsv` | `empath_psych_theme`, `empath_physical_theme` | Empath categories, Fast et al. (2016) | Category-level semantic themes, not a full sentiment model by itself |
| `data/lexicons/socal/` | support for style lexicon building | SO-CAL lexicons, Taboada et al. (2011) | Used in the style lexicon generation pipeline |

To rebuild the style lexicon (needs `python -m spacy download en_core_web_lg` for word vectors):

```bash
python3 src/lexicons/build_style_lexicon.py data/lexicons/style_lexicon.csv
```

The seed lists in `src/lexicons/seeds.py` were drafted for this project and should be reviewed as needed.

## Current sock-puppet audit workflow

The current approach is to use a browser automation tool like Playwright to collect the sock puppet data.

### Workflow

1. Collect screenshots of feed pages or post cards in category folders under `data/instagram_corpus/`.
   - Example structure:
     - `data/instagram_corpus/happy-poetry/`
     - `data/instagram_corpus/langston-hughes/`
2. Run:

```bash
.venv/bin/python src/images_to_poems.py data/instagram_corpus --output data/poems.json
```

3. `images_to_poems.py` recursively finds all image files, OCRs them, splits multi-poem screenshots if needed, and writes JSON entries into `data/poems.json` with the poem text and the source category.
4. The feature scripy in `src/linguistic_features.py` computes the poem-level vector for each data (poem) entry.
5. Use the audit scripts to summarize or visualize the corpus:

```bash
.venv/bin/python src/feature_audit.py --data data/poems.json --output outputs/feature_audit.csv
.venv/bin/python src/feature_policy_plots.py --data data/poems.json --output outputs/policy_dashboard_full.png
```

6. Engagement policies for different sock puppet users are set by comparing distributions across categories.

## TBD

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

