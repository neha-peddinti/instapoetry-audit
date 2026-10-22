"""
Builds word-level style scores for two dimensions:
  lit_col   literary (+) vs. colloquial (-)
  subj_obj  subjective (+) vs. objective (-)

Method follows Brooke, Wang & Hirst (2010), the seed-similarity approach
that Brooke & Hirst (2013) extend and Verma & Srinivasan (2019) use:

  FS0(w) = sum over seeds s != w of  W_s * FS(s) * cos(w, s)
  FS(w)  = (FS0(w) - FS0("and")) / N

  FS(s)  = +1 for pole-A seeds, -1 for pole-B seeds
  W_s    = 1 / (size of s's seed set), so unequal set sizes cancel out
  "and"  is the stylistically neutral reference word
  N      scales scores into [-1, 1]
  s != w is leave-one-out for seed words

Substitution: the original used LSA vectors from a large web corpus; this
script uses the 300-d pretrained word vectors in spaCy's en_core_web_lg.

A word counts toward a pole (the n_a / n_b in the feature table) when its
FS(w) is at least as extreme as the Q-th percentile of that pole's seed
scores (default Q = 50: the word scores at least as extreme as the median seed).

Usage:
  python src/lexicons/build_style_lexicon.py data/lexicons/style_lexicon.csv [--q 50]
Writes the lexicon CSV and a validation report next to it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).parent))
from seeds import LITERARY, COLLOQUIAL, OBJECTIVE

DATA = Path(__file__).resolve().parents[2] / "data" / "lexicons"
VOCAB_SIZE = 50000
SUBJ_PER_POLE = 75        # 150 subjective seeds total, as in Brooke & Hirst


def load_socal_subjective():
    """Strongly positive and strongly negative SO-CAL words (|SO| >= 4),
    most frequent first, half positive and half negative."""
    pos, neg = [], []
    for f in sorted((DATA / "socal").glob("*_dictionary1.11.txt")):
        for line in open(f, encoding="latin-1"):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2 or not parts[0].strip().isalpha():
                continue
            w, so = parts[0].strip().lower(), float(parts[1])
            if so >= 4:
                pos.append(w)
            elif so <= -4:
                neg.append(w)
    rank = lambda ws: sorted(set(ws), key=lambda w: -zipf_frequency(w, "en"))
    return rank(pos)[:SUBJ_PER_POLE] + rank(neg)[:SUBJ_PER_POLE]


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def score_dimension(words, M, idx, seeds_a, seeds_b):
    a = [idx[w] for w in seeds_a if w in idx]
    b = [idx[w] for w in seeds_b if w in idx]
    sims_a = M @ M[a].T                       # cosine (rows are unit vectors)
    sims_b = M @ M[b].T
    # leave-one-out: a seed does not vote for itself
    for j, i in enumerate(a):
        sims_a[i, j] = 0
    for j, i in enumerate(b):
        sims_b[i, j] = 0
    fs0 = sims_a.sum(1) / len(a) - sims_b.sum(1) / len(b)
    fs = fs0 - fs0[idx["and"]]
    return fs / np.abs(fs).max(), len(a), len(b)


def seed_accuracy(fs, idx, seeds_a, seeds_b):
    ok = [fs[idx[w]] > 0 for w in seeds_a if w in idx] + \
         [fs[idx[w]] < 0 for w in seeds_b if w in idx]
    return float(np.mean(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--q", type=float, default=50,
                    help="seed percentile used as each pole's threshold")
    args = ap.parse_args()

    subjective = load_socal_subjective()
    lists = {"literary": LITERARY, "colloquial": COLLOQUIAL,
             "subjective": subjective, "objective": OBJECTIVE}
    # remove words that occur in more than one list (Brooke & Hirst 2013)
    counts = pd.Series([w for ws in lists.values() for w in set(ws)]).value_counts()
    dup = set(counts[counts > 1].index)
    lists = {k: [w for w in dict.fromkeys(v) if w not in dup] for k, v in lists.items()}

    nlp = spacy.load("en_core_web_lg", exclude=["parser", "ner", "tagger", "lemmatizer"])
    vocab = [w for w in top_n_list("en", VOCAB_SIZE) if w.isalpha()]
    vocab = list(dict.fromkeys(vocab + [w for ws in lists.values() for w in ws] + ["and"]))
    words = [w for w in vocab if nlp.vocab.has_vector(w)]
    idx = {w: i for i, w in enumerate(words)}
    M = unit(np.array([nlp.vocab.get_vector(w) for w in words], dtype=np.float32))

    report = [f"Vocabulary: {len(words)} words with vectors",
              f"Removed from seed lists (in more than one list): {sorted(dup) or 'none'}", ""]
    out = pd.DataFrame({"word": words})
    for name, a, b in [("lit_col", "literary", "colloquial"),
                       ("subj_obj", "subjective", "objective")]:
        fs, na, nb = score_dimension(words, M, idx, lists[a], lists[b])
        # Seed-calibrated thresholds: a word joins a pole if it scores at least as
        # far toward that pole as the least extreme (Q)% of the pole's own seeds
        # (leave-one-out scores). A single cutoff on |FS| does not work with word
        # vectors: the neutral reference "and" is not at the center of the
        # distribution, and everyday words (wet, said, faces) sit close to slang.
        sa = fs[[idx[w] for w in lists[a] if w in idx]]
        sb = fs[[idx[w] for w in lists[b] if w in idx]]
        hi = float(np.percentile(sa, args.q))
        lo = float(np.percentile(sb, 100 - args.q))
        out[name] = fs
        out[f"{name}_pole"] = np.where(fs >= hi, a, np.where(fs <= lo, b, ""))
        acc = seed_accuracy(fs, idx, lists[a], lists[b])
        top_a = out.sort_values(name, ascending=False)["word"].head(25).tolist()
        top_b = out.sort_values(name)["word"].head(25).tolist()
        report += [f"## {name}: {a} (+) vs {b} (-)",
                   f"seeds with vectors: {na} {a}, {nb} {b}",
                   f"leave-one-out seed accuracy (sign correct): {acc:.2f}",
                   f"thresholds: {a} if FS >= {hi:.3f}, {b} if FS <= {lo:.3f} "
                   f"({args.q:g}th percentile of each pole's seed scores)",
                   f"words assigned: {(out[f'{name}_pole'] == a).sum()} {a}, "
                   f"{(out[f'{name}_pole'] == b).sum()} {b}",
                   f"most {a}: {', '.join(top_a)}",
                   f"most {b}: {', '.join(top_b)}", ""]
    out.to_csv(args.out, index=False, float_format="%.4f")
    rep = Path(args.out).with_suffix(".report.txt")
    rep.write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
