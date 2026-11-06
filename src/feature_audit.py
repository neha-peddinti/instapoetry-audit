import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from src.linguistic_features import compute_features, feature_names
except ModuleNotFoundError:  # pragma: no cover
    from linguistic_features import compute_features, feature_names


def load_poems(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Poem JSON must be a list of poem objects.")
    return data


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize_features_by_category(poems: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for poem in poems:
        category = str(poem.get("category", "uncategorized"))
        lines = poem.get("lines", [])
        features = compute_features(lines)
        grouped.setdefault(category, {name: [] for name in feature_names()})
        for feature in feature_names():
            grouped[category][feature].append(_safe_float(features.get(feature, 0.0)))

    for category, feature_values in grouped.items():
        for feature in feature_names():
            stats = _feature_stats(feature_values[feature])
            rows.append(
                {
                    "category": category,
                    "feature": feature,
                    "count": len(feature_values[feature]),
                    **stats,
                }
            )

    return rows


def top_rule_hits(poems: Iterable[Dict[str, Any]], feature: str, top_k: int = 10) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for poem in poems:
        feature_value = _safe_float(compute_features(poem.get("lines", [])).get(feature, 0.0))
        scored.append(
            {
                "id": poem.get("id", "unknown"),
                "category": poem.get("category", "uncategorized"),
                "feature": feature,
                "value": feature_value,
            }
        )
    scored.sort(key=lambda item: item["value"], reverse=True)
    return scored[:top_k]


def write_summary_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "feature", "count", "mean", "std", "min", "max", "p90", "p95"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize feature distributions by category for engagement-policy audits.")
    parser.add_argument("--data", default="data/poems.json", help="Path to the poem JSON corpus.")
    parser.add_argument("--output", default="outputs/feature_audit.csv", help="Path to the CSV summary file.")
    parser.add_argument("--feature", default=None, help="Optional feature to print the strongest poems for; e.g. nrc_negative or absolutist.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of highest-scoring poems to print when --feature is set.")
    args = parser.parse_args()

    poems = load_poems(args.data)
    rows = summarize_features_by_category(poems)
    write_summary_csv(rows, args.output)

    if args.feature is not None:
        if args.feature not in feature_names():
            raise ValueError(f"Unsupported feature name: {args.feature}. Choose from: {', '.join(feature_names())}")
        hits = top_rule_hits(poems, args.feature, top_k=args.top_k)
        print(f"Top {args.top_k} poems by {args.feature}:")
        for hit in hits:
            print(f"  {hit['id']} | {hit['category']} | {hit['value']:.4f}")

    print(f"Saved category feature audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
