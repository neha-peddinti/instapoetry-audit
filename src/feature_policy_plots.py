import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.linguistic_features import compute_features, feature_names
except ModuleNotFoundError:  # pragma: no cover
    from linguistic_features import compute_features, feature_names

FEATURES_FOR_POLICY = feature_names()


def load_poems(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Poem JSON must be a list of poem objects.")
    return data


def build_policy_summary(poems: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    poems = list(poems)
    categories = sorted({str(poem.get("category", "uncategorized")) for poem in poems})
    results = []
    for feature in FEATURES_FOR_POLICY:
        feature_values = []
        for poem in poems:
            feature_values.append(compute_features(poem.get("lines", [])).get(feature, 0.0))
        results.append({"feature": feature, "mean": float(np.mean(feature_values)), "max": float(np.max(feature_values))})

    return {
        "categories": categories,
        "features": FEATURES_FOR_POLICY,
        "rows": len(results),
        "data": results,
    }


def generate_policy_dashboard(poems: Iterable[Dict[str, Any]], output_path: str | Path) -> str:
    poems = list(poems)
    if not poems:
        raise ValueError("No poems were provided.")

    categories = sorted({str(poem.get("category", "uncategorized")) for poem in poems})
    values_by_feature = {feature: [] for feature in FEATURES_FOR_POLICY}
    for poem in poems:
        feats = compute_features(poem.get("lines", []))
        for feature in FEATURES_FOR_POLICY:
            values_by_feature[feature].append(float(feats.get(feature, 0.0)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 5
    n_rows = (len(FEATURES_FOR_POLICY) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = np.asarray(axes).flatten()

    for ax, feature in zip(axes, FEATURES_FOR_POLICY):
        vals = values_by_feature[feature]
        ax.boxplot(vals, vert=True, patch_artist=True)
        ax.set_title(feature)
        ax.set_xticks([])
        if max(vals) <= 1:
            upper = max(1.0, max(vals) * 1.2)
        else:
            upper = max(vals) * 1.1
        lower = min(0.0, min(vals) * 1.1)
        ax.set_ylim(lower, upper)

    for ax in axes[len(FEATURES_FOR_POLICY):]:
        ax.axis("off")

    fig.suptitle("Feature distribution across corpus", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=300)
    return str(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate policy-relevant feature plots for poem corpora.")
    parser.add_argument("--data", default="data/poems.json", help="Path to the poem JSON corpus.")
    parser.add_argument("--output", default="outputs/policy_dashboard.png", help="Path to the dashboard plot.")
    args = parser.parse_args()

    poems = load_poems(args.data)
    output = generate_policy_dashboard(poems, args.output)
    print(f"Saved feature policy dashboard to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
