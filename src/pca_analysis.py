import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from linguistic_features import poem_feature_vector


def load_poems(path: str):
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    return poems


def main():
    parser = argparse.ArgumentParser(description="Run PCA on poem feature vectors")
    parser.add_argument("--data", default="data/poems.json", help="Path to poem JSON file")
    parser.add_argument("--output", default="outputs/pca_poem_projection.png", help="Where to save the PCA scatter plot")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    poems = load_poems(str(data_path))

    if not poems:
        print("No poems found in the dataset. Add poem entries to data/poems.json before running the analysis.")
        return 0

    ids = []
    matrix = []

    for poem in poems:
        poem_id = poem.get("id")
        lines = poem.get("lines", [])

        if poem_id is None:
            raise ValueError("Each poem entry requires an 'id' field.")

        matrix.append(poem_feature_vector(lines))
        ids.append(str(poem_id))

    X = np.vstack(matrix)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    print("Poem ID | PC1 | PC2")
    for poem_id, coords in zip(ids, reduced):
        print(f"{poem_id} | {coords[0]:.4f} | {coords[1]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=60, alpha=0.8)

    for poem_id, (x, y) in zip(ids, reduced):
        plt.annotate(poem_id, (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points")

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Poem aesthetic clustering via PCA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved PCA plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
