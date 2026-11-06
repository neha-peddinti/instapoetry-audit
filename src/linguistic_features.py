import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None

# This module follows the actual feature table used in the project specification.
# The metrics correspond to Kao & Jurafsky (2012/2015), Verma & Srinivasan (2019),
# Brysbaert et al. (2014), Lu (2010), and the original PoemFeatures implementation.

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lexicons"

if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
else:
    nlp = None


def _need(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return None
    return path


def _load_resources():
    resources = {}

    style_path = _need("style_lexicon.csv")
    if style_path is not None:
        d = pd.read_csv(style_path, keep_default_na=False)
        resources["subj"] = set(d.loc[d["subj_obj_pole"] == "subjective", "word"])
        resources["obj"] = set(d.loc[d["subj_obj_pole"] == "objective", "word"])
        resources["subj_obj_fs"] = dict(zip(d["word"], d["subj_obj"]))

    concreteness_path = _need("concreteness.txt")
    if concreteness_path is not None:
        d = pd.read_csv(concreteness_path, sep="\t")
        resources["conc"] = dict(zip(d["Word"].astype(str).str.lower(), d["Conc.M"]))

    nrc_path = _need("nrc_emolex.json")
    if nrc_path is not None:
        d = json.loads(nrc_path.read_text())
        resources["nrc"] = set(d)
        resources["pos"] = {w for w, tags in d.items() if "positive" in tags}
        resources["neg"] = {w for w, tags in d.items() if "negative" in tags}

    empath_path = _need("empath_categories.tsv")
    if empath_path is not None:
        with open(empath_path, "r", encoding="utf-8") as f:
            raw_cats = {line.split("\t")[0]: set(line.strip().split("\t")[1:]) for line in f if line.strip()}
        cats = {k.strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in raw_cats.items()}
        resources["health"] = cats.get("health", set())
        resources["content"] = cats.get("contentment", set())

        def _merge_aliases(*names):
            merged = set()
            for name in names:
                normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
                merged |= cats.get(normalized, set())
            return merged

        resources["psychological_wellbeing"] = _merge_aliases(
            "wellbeing",
            "mental_state",
            "cheerfulness",
            "happiness",
            "contentment",
            "nervousness",
            "hate",
            "suffering",
            "weakness",
            "horror",
            "aggression",
            "envy",
            "fear",
        )
        resources["physical_theme"] = _merge_aliases("body", "health")

    return resources


RES = _load_resources()

# The PoemFeatures project uses a generalized-language dictionary for absolutist
# wording. We keep the list as a direct implementation of the metric in the table.
ABSOLUTIST = {
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
    "never", "nothing", "totally", "whole", "none"
}


def clean_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text).replace("\r", "").split("\n") if line.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def mattr(tokens: List[str], window: int = 10) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    values = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(values))


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _sentence_structure_stats(doc) -> dict:
    sentences = list(doc.sents) if doc is not None else []
    if not sentences:
        return {
            "simple": 0.0,
            "complex": 0.0,
            "compound": 0.0,
            "fragment": 0.0,
            "total": 0,
        }

    simple = 0
    complex = 0
    compound = 0
    fragment = 0
    total = 0

    for sent in sentences:
        sent_tokens = [t for t in sent if t.is_alpha]
        if not sent_tokens:
            continue
        total += 1
        finite_verbs = [t for t in sent if t.pos_ in {"VERB", "AUX"}]
        if not finite_verbs:
            fragment += 1
            continue
        subordinate = any(t.dep_ in {"advcl", "relcl", "ccomp", "xcomp", "acl", "acomp"} for t in sent)
        has_conj = any(t.dep_ == "conj" or t.text.lower() in {"and", "but", "or", "yet", "so"} for t in sent)

        if subordinate:
            complex += 1
        elif has_conj and len(finite_verbs) > 1:
            compound += 1
        else:
            simple += 1

    total = max(total, 1)
    return {
        "simple": simple / total,
        "complex": complex / total,
        "compound": compound / total,
        "fragment": fragment / total,
        "total": total,
    }


def feature_names() -> List[str]:
    return [
        "concreteness",
        "absolutist",
        "subj_obj_score",
        "word_freq",
        "mattr",
        "simple_sentence_ratio",
        "complex_sentence_ratio",
        "compound_sentence_ratio",
        "nrc_positive",
        "nrc_negative",
        "empath_psych_theme",
        "empath_physical_theme",
        "n_words",
        "n_lines",
        "words_per_line",
        "lowercase_ratio",
        "enjambment",
    ]


def collect_unmapped_nrc_words(lines: Iterable[str]) -> dict:
    """Return content words not found in the NRC lexicon.

    When spaCy POS tagging is available, nouns and verbs are split into separate
    lists. When the lightweight blank model is used, we fall back to a content-word
    bucket instead of pretending we have perfect noun/verb labels.
    """
    poem_lines = clean_lines("\n".join(str(line) for line in lines))
    text = "\n".join(poem_lines)
    if not text.strip():
        return {"nouns": [], "verbs": [], "other_content": []}

    doc = nlp(text) if nlp is not None and text.strip() else None
    nrc_words = RES.get("nrc", set())
    nouns = set()
    verbs = set()
    other_content = set()

    if doc is not None:
        for token in doc:
            if not token.is_alpha:
                continue
            word = token.text.lower()
            lemma = token.lemma_.lower() if getattr(token, "lemma_", None) else word
            lookup = lemma if lemma else word
            if not lookup or word in nrc_words or lookup in nrc_words:
                continue
            if token.pos_ == "NOUN":
                nouns.add(lookup)
            elif token.pos_ == "VERB":
                verbs.add(lookup)
            elif not token.is_stop:
                other_content.add(lookup)
        return {
            "nouns": sorted(nouns),
            "verbs": sorted(verbs),
            "other_content": sorted(other_content),
        }

    for word in re.findall(r"[A-Za-z']+", text.lower()):
        if not word or word in nrc_words:
            continue
        if word in {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her", "i", "you", "he", "she", "we", "they", "them", "me", "us"}:
            continue
        other_content.add(word)

    return {
        "nouns": [],
        "verbs": [],
        "other_content": sorted(other_content),
    }


def compute_features(lines: Iterable[str], include_unmapped_nrc_words: bool = False) -> dict:
    poem_lines = clean_lines("\n".join(str(line) for line in lines))
    text = "\n".join(poem_lines)
    doc = nlp(text) if nlp is not None and text.strip() else None
    words = [t for t in doc if t.is_alpha] if doc is not None else []
    word_tokens = [t.text.lower() for t in words]
    lemma_tokens = [t.lemma_.lower() if getattr(t, "lemma_", None) else t.text.lower() for t in words]
    non_stop_words = [t for t in words if not t.is_stop and t.pos_ != "PROPN"]
    content_words = [t for t in words if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]

    if not word_tokens:
        return {name: 0.0 for name in feature_names()}

    # Lexical features
    if "conc" in RES:
        conc_vals = []
        for t in content_words:
            val = RES["conc"].get(t.text.lower(), RES["conc"].get(t.lemma_.lower()))
            if val is not None:
                conc_vals.append(float(val))
        concreteness = float(np.mean(conc_vals)) if conc_vals else 0.0
    else:
        concreteness = 0.0

    absolutist_count = sum(1 for w in word_tokens if w in ABSOLUTIST)
    absolutist = _safe_ratio(absolutist_count, len(word_tokens))

    if "subj" in RES and "obj" in RES:
        subj_count = sum(1 for t in non_stop_words if t.text.lower() in RES["subj"] or t.lemma_.lower() in RES["subj"])
        obj_count = sum(1 for t in non_stop_words if t.text.lower() in RES["obj"] or t.lemma_.lower() in RES["obj"])
        subj_obj = _safe_ratio(subj_count, subj_count + obj_count)
    else:
        subj_obj = 0.0

    if zipf_frequency is not None:
        word_freq = float(np.mean([zipf_frequency(w, "en") for w in word_tokens]))
    else:
        word_freq = 0.0

    mtr = mattr(word_tokens)

    # Syntactic features
    sentence_stats = _sentence_structure_stats(doc)
    simple_ratio = sentence_stats["simple"]
    complex_ratio = sentence_stats["complex"]
    compound_ratio = sentence_stats["compound"]

    # Semantic features
    if "pos" in RES:
        pos_count = sum(1 for w in lemma_tokens if w in RES["pos"])
        nrc_positive = _safe_ratio(pos_count, len(word_tokens))
    else:
        nrc_positive = 0.0

    if "neg" in RES:
        neg_count = sum(1 for w in lemma_tokens if w in RES["neg"])
        nrc_negative = _safe_ratio(neg_count, len(word_tokens))
    else:
        nrc_negative = 0.0

    if "psychological_wellbeing" in RES:
        wellbeing_count = sum(1 for w in lemma_tokens if w in RES["psychological_wellbeing"])
        empath_psych_theme = _safe_ratio(wellbeing_count, len(word_tokens))
    else:
        empath_psych_theme = 0.0

    if "physical_theme" in RES:
        physical_count = sum(1 for w in lemma_tokens if w in RES["physical_theme"])
        empath_physical_theme = _safe_ratio(physical_count, len(word_tokens))
    else:
        empath_physical_theme = 0.0

    # Structural features
    n_words = len(word_tokens)
    n_lines = max(len(poem_lines), 1)
    words_per_line = _safe_ratio(n_words, n_lines)
    lowercase_ratio = sum(1 for line in poem_lines if line == line.lower()) / n_lines if poem_lines else 0.0

    endstop = 0
    for line in poem_lines:
        if line.endswith((".", "!", "?", ";", ":", ",")):
            endstop += 1
    enjambment = 1.0 - (endstop / n_lines) if n_lines else 0.0

    features = {
        "concreteness": float(concreteness),
        "absolutist": float(absolutist),
        "subj_obj_score": float(subj_obj),
        "word_freq": float(word_freq),
        "mattr": float(mtr),
        "simple_sentence_ratio": float(simple_ratio),
        "complex_sentence_ratio": float(complex_ratio),
        "compound_sentence_ratio": float(compound_ratio),
        "nrc_positive": float(nrc_positive),
        "nrc_negative": float(nrc_negative),
        "empath_psych_theme": float(empath_psych_theme),
        "empath_physical_theme": float(empath_physical_theme),
        "n_words": float(n_words),
        "n_lines": float(n_lines),
        "words_per_line": float(words_per_line),
        "lowercase_ratio": float(lowercase_ratio),
        "enjambment": float(enjambment),
    }

    if include_unmapped_nrc_words:
        missing = collect_unmapped_nrc_words(poem_lines)
        features["nrc_unmapped_nouns"] = missing["nouns"]
        features["nrc_unmapped_verbs"] = missing["verbs"]
        features["nrc_unmapped_other_content"] = missing["other_content"]

    return features


def poem_feature_vector(lines: Iterable[str]) -> np.ndarray:
    features = compute_features(lines)
    ordered_keys = feature_names()
    return np.array([features[key] for key in ordered_keys], dtype=float)


def main() -> None:
    """
    Allows you to see feature values for a list of sentences from command line and compare them.
    """

    parser = argparse.ArgumentParser(description="Compute poem feature values for one or more sentences.")
    parser.add_argument(
        "sentences",
        nargs="*",
        help="Sentence strings to score. If omitted, read from standard input line by line.",
    )
    args = parser.parse_args()

    if args.sentences:
        lines = args.sentences
    else:
        import sys
        lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    if not lines:
        print("No sentences supplied.")
        return

    values = {}
    for idx, sentence in enumerate(lines, start=1):
        sentence_features = compute_features([sentence])
        values[f"sentence_{idx}"] = {"sentence": sentence.strip(), **sentence_features}

    feature_order = feature_names()
    sentence_names = [values[f"sentence_{i}"]["sentence"] for i in range(1, len(lines) + 1)]
    feature_rows = {
        feature: [values[f"sentence_{i}"][feature] for i in range(1, len(lines) + 1)]
        for feature in feature_order
    }
    df = pd.DataFrame(feature_rows, index=sentence_names).T
    df.index.name = "feature"
    df.columns.name = "sentence"
    print(df.to_string())


if __name__ == "__main__":
    main()
