import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from linguistic_features import poem_feature_vector

FEATURE_NAMES = [
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


def load_poems(path: str):
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    return poems


def main():
    parser = argparse.ArgumentParser(description="Run PCA on poem feature vectors")
    parser.add_argument("--data", default="data/poems.json", help="Path to poem JSON file")
    parser.add_argument("--output", default="outputs/pca_poem_projection.png", help="Where to save the PCA scatter plot")
    parser.add_argument("--components", type=int, default=2, choices=[2, 3], help="Number of principal components to visualize (2 or 3)")
    parser.add_argument(
        "--color-feature",
        default=None,
        choices=FEATURE_NAMES,
        help="Feature name to use for point coloring; if omitted, categories are not color-coded.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    poems = load_poems(str(data_path))

    if not poems:
        print("No poems found in the dataset. Add poem entries to data/poems.json before running the analysis.")
        return 0

    ids = []
    categories = []
    matrix = []

    for poem in poems:
        poem_id = poem.get("id")
        lines = poem.get("lines", [])
        category = poem.get("category", "uncategorized")

        if poem_id is None:
            raise ValueError("Each poem entry requires an 'id' field.")

        feature_vector = poem_feature_vector(lines)
        matrix.append(feature_vector)
        ids.append(str(poem_id))
        categories.append(str(category))

    X = np.vstack(matrix)
    max_components = min(X.shape[0], X.shape[1])
    n_components = min(args.components, max_components)
    if n_components < args.components:
        print(f"Requested {args.components}D PCA, but only {max_components} valid dimensions are available. Using {n_components}D PCA instead.")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)

    if args.color_feature is not None:
        feature_index = FEATURE_NAMES.index(args.color_feature)
        feature_values = np.array([vector[feature_index] for vector in matrix], dtype=float)
        vmin = float(np.min(feature_values))
        vmax = float(np.max(feature_values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
    else:
        feature_values = None

    if n_components == 2:
        print("Poem ID | Category | PC1 | PC2")
        for poem_id, category, coords in zip(ids, categories, reduced):
            print(f"{poem_id} | {category} | {coords[0]:.4f} | {coords[1]:.4f}")

        fig, ax = plt.subplots(figsize=(9, 7))
        if args.color_feature is not None:
            for idx, poem_id in enumerate(ids):
                x, y = reduced[idx, 0], reduced[idx, 1]
                color = cmap(norm(feature_values[idx]))
                ax.scatter(x, y, s=60, alpha=0.8, color=color)
                ax.annotate(poem_id, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=args.color_feature)
        else:
            unique_categories = sorted(set(categories))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
            color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
            for category in unique_categories:
                category_points = [i for i, c in enumerate(categories) if c == category]
                xs = reduced[category_points, 0]
                ys = reduced[category_points, 1]
                ax.scatter(xs, ys, s=60, alpha=0.8, color=color_map[category], label=category)
                for idx in category_points:
                    ax.annotate(ids[idx], (reduced[idx, 0], reduced[idx, 1]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"Poem aesthetic clustering via PCA ({args.color_feature or 'category'} view)")
        fig.tight_layout()
    else:
        print("Poem ID | Category | PC1 | PC2 | PC3")
        for poem_id, category, coords in zip(ids, categories, reduced):
            print(f"{poem_id} | {category} | {coords[0]:.4f} | {coords[1]:.4f} | {coords[2]:.4f}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        if args.color_feature is not None:
            for idx, poem_id in enumerate(ids):
                x, y, z = reduced[idx]
                color = cmap(norm(feature_values[idx]))
                ax.scatter(x, y, z, s=60, alpha=0.8, color=color)
                ax.text(x, y, z, poem_id, fontsize=7)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=args.color_feature)
        else:
            unique_categories = sorted(set(categories))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
            color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
            for category in unique_categories:
                category_points = [i for i, c in enumerate(categories) if c == category]
                xs = reduced[category_points, 0]
                ys = reduced[category_points, 1]
                zs = reduced[category_points, 2]
                ax.scatter(xs, ys, zs, s=60, alpha=0.8, color=color_map[category], label=category)
                for idx in category_points:
                    ax.text(reduced[idx, 0], reduced[idx, 1], reduced[idx, 2], ids[idx], fontsize=7)
            ax.legend(title="Category")

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title(f"Poem aesthetic clustering via 3D PCA ({args.color_feature or 'category'} view)")

    plt.savefig(output_path, dpi=300)
    print(f"Saved PCA plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
