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
    except OSError:  # pragma: no cover
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
else:  # pragma: no cover
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
        resources["lit"] = set(d.loc[d["lit_col_pole"] == "literary", "word"])
        resources["col"] = set(d.loc[d["lit_col_pole"] == "colloquial", "word"])
        resources["subj"] = set(d.loc[d["subj_obj_pole"] == "subjective", "word"])
        resources["obj"] = set(d.loc[d["subj_obj_pole"] == "objective", "word"])
        resources["lit_col_fs"] = dict(zip(d["word"], d["lit_col"]))
        resources["subj_obj_fs"] = dict(zip(d["word"], d["subj_obj"]))

    concreteness_path = _need("concreteness.txt")
    if concreteness_path is not None:
        d = pd.read_csv(concreteness_path, sep="\t")
        resources["conc"] = dict(zip(d["Word"].astype(str).str.lower(), d["Conc.M"]))

    nrc_path = _need("nrc_emolex.json")
    if nrc_path is not None:
        d = json.loads(nrc_path.read_text())
        resources["pos"] = {w for w, tags in d.items() if "positive" in tags}
        resources["neg"] = {w for w, tags in d.items() if "negative" in tags}

    empath_path = _need("empath_categories.tsv")
    if empath_path is not None:
        with open(empath_path, "r", encoding="utf-8") as f:
            cats = {line.split("\t")[0]: set(line.strip().split("\t")[1:]) for line in f if line.strip()}
        resources["health"] = cats.get("health", set())
        resources["content"] = cats.get("contentment", set())

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
        "lit_col_score",
        "concreteness",
        "absolutist",
        "subj_obj_score",
        "word_freq",
        "mattr",
        "simple_sentence_ratio",
        "complex_sentence_ratio",
        "compound_sentence_ratio",
        "fragment_rate",
        "nrc_positive",
        "nrc_negative",
        "empath_health",
        "empath_content",
        "n_words",
        "n_lines",
        "words_per_line",
        "lowercase_ratio",
        "enjambment",
    ]


def compute_features(lines: Iterable[str]) -> dict:
    poem_lines = clean_lines("\n".join(str(line) for line in lines))
    text = "\n".join(poem_lines)
    doc = nlp(text) if nlp is not None and text.strip() else None
    words = [t for t in doc if t.is_alpha] if doc is not None else []
    word_tokens = [t.text.lower() for t in words]
    non_stop_words = [t for t in words if not t.is_stop and t.pos_ != "PROPN"]
    content_words = [t for t in words if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]

    if not word_tokens:
        return {name: 0.0 for name in feature_names()}

    # Lexical features
    if "lit" in RES and "col" in RES:
        lit_count = sum(1 for t in non_stop_words if t.text.lower() in RES["lit"] or t.lemma_.lower() in RES["lit"])
        col_count = sum(1 for t in non_stop_words if t.text.lower() in RES["col"] or t.lemma_.lower() in RES["col"])
        lit_col = _safe_ratio(lit_count, lit_count + col_count)
    else:
        lit_col = 0.0

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
    fragment_rate = sentence_stats["fragment"]

    # Semantic features
    if "pos" in RES:
        pos_count = sum(1 for w in word_tokens if w in RES["pos"])
        nrc_positive = _safe_ratio(pos_count, len(word_tokens))
    else:
        nrc_positive = 0.0

    if "neg" in RES:
        neg_count = sum(1 for w in word_tokens if w in RES["neg"])
        nrc_negative = _safe_ratio(neg_count, len(word_tokens))
    else:
        nrc_negative = 0.0

    if "health" in RES:
        health_count = sum(1 for w in word_tokens if w in RES["health"])
        empath_health = _safe_ratio(health_count, len(word_tokens))
    else:
        empath_health = 0.0

    if "content" in RES:
        content_count = sum(1 for w in word_tokens if w in RES["content"])
        empath_content = _safe_ratio(content_count, len(word_tokens))
    else:
        empath_content = 0.0

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

    return {
        "lit_col_score": float(lit_col),
        "concreteness": float(concreteness),
        "absolutist": float(absolutist),
        "subj_obj_score": float(subj_obj),
        "word_freq": float(word_freq),
        "mattr": float(mtr),
        "simple_sentence_ratio": float(simple_ratio),
        "complex_sentence_ratio": float(complex_ratio),
        "compound_sentence_ratio": float(compound_ratio),
        "fragment_rate": float(fragment_rate),
        "nrc_positive": float(nrc_positive),
        "nrc_negative": float(nrc_negative),
        "empath_health": float(empath_health),
        "empath_content": float(empath_content),
        "n_words": float(n_words),
        "n_lines": float(n_lines),
        "words_per_line": float(words_per_line),
        "lowercase_ratio": float(lowercase_ratio),
        "enjambment": float(enjambment),
    }


def poem_feature_vector(lines: Iterable[str]) -> np.ndarray:
    features = compute_features(lines)
    ordered_keys = feature_names()
    return np.array([features[key] for key in ordered_keys], dtype=float)
