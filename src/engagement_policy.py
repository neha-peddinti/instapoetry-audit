import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from src.linguistic_features import compute_features, feature_names
except ModuleNotFoundError:  # pragma: no cover
    from linguistic_features import compute_features, feature_names

DEFAULT_POLICY = {
    "like_rate": {"threshold": 0.12, "high": 0.20, "weight": 0.40},
    "save_rate": {"threshold": 0.08, "high": 0.12, "weight": 0.30},
    "dwell_time_sec": {"threshold": 30, "high": 60, "weight": 0.30},
}


def _normalize(value: float, threshold: float, high: float) -> float:
    if high <= threshold:
        raise ValueError("Policy config requires high > threshold.")
    if value <= threshold:
        return max(0.0, min(1.0, value / max(threshold, 1e-9)))
    return min(1.0, 1.0 + ((value - threshold) / max(high - threshold, 1e-9)) * 0.5)


def classify_engagement(
    like_rate: float,
    save_rate: float,
    dwell_time_sec: float,
    comment_rate: float = 0.0,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = profile or DEFAULT_POLICY
    metrics = {
        "like_rate": float(like_rate),
        "save_rate": float(save_rate),
        "dwell_time_sec": float(dwell_time_sec),
    }

    weighted_score = 0.0
    score_components = {}
    for metric_name, cfg in profile.items():
        metric_value = metrics[metric_name]
        threshold = float(cfg["threshold"])
        high = float(cfg["high"])
        norm = _normalize(metric_value, threshold, high)
        weight = float(cfg.get("weight", 1.0))
        weighted_score += norm * weight
        score_components[metric_name] = {
            "value": metric_value,
            "threshold": threshold,
            "high": high,
            "normalized": norm,
            "weight": weight,
        }

    total_weight = sum(float(cfg.get("weight", 1.0)) for cfg in profile.values())
    score = weighted_score / total_weight if total_weight else 0.0

    if score >= 0.70:
        band = "high"
    elif score >= 0.40:
        band = "medium"
    else:
        band = "low"

    recommendation = {
        "low": "Keep the post in a conservative testing bucket; improve emotional clarity or line structure before scaling exposure.",
        "medium": "Use the post as a monitored variant; continue tracking dwell time and saves before promoting it.",
        "high": "This post is a strong candidate for a broader test or a higher-visibility placement, but monitor rate drift over time.",
    }[band]

    return {
        "band": band,
        "score": round(float(score), 4),
        "metrics": metrics,
        "components": score_components,
        "recommendation": recommendation,
    }


def _compute_feature_distribution(poems: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    values_by_feature: Dict[str, List[float]] = {feature: [] for feature in feature_names()}
    for poem in poems:
        feat = compute_features(poem.get("lines", []))
        for feature in feature_names():
            values_by_feature[feature].append(float(feat.get(feature, 0.0)))

    summary: Dict[str, Dict[str, float]] = {}
    for feature, values in values_by_feature.items():
        arr = sorted(float(v) for v in values)
        if not arr:
            summary[feature] = {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
            continue
        n = len(arr)
        summary[feature] = {
            "mean": float(sum(arr) / n),
            "p25": float(arr[max(0, int(n * 0.25) - 1)]),
            "p50": float(arr[max(0, int(n * 0.50) - 1)]),
            "p75": float(arr[min(n - 1, int(n * 0.75) - 1)]),
        }
    return summary


def _category_features(poems: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for poem in poems:
        category = str(poem.get("category", "uncategorized"))
        feature_vector = compute_features(poem.get("lines", []))
        grouped.setdefault(category, [])
        grouped[category].append(float(feature_vector.get("nrc_positive", 0.0) + feature_vector.get("nrc_negative", 0.0)))

    return {
        category: {
            "count": len(values),
            "avg_sentiment_load": sum(values) / len(values) if values else 0.0,
        }
        for category, values in grouped.items()
    }


def build_feature_driven_policies(poems: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    poem_list = list(poems)
    if not poem_list:
        return []

    dist = _compute_feature_distribution(poem_list)
    profile_specs = [
        {
            "id": "minimalist_readability",
            "name": "Minimalist readability",
            "feature_signals": {
                "simple_sentence_ratio": {"op": ">=", "value": dist.get("simple_sentence_ratio", {}).get("p75", 0.0)},
                "words_per_line": {"op": "<=", "value": dist.get("words_per_line", {}).get("p50", 0.0)},
                "nrc_positive": {"op": ">=", "value": dist.get("nrc_positive", {}).get("p50", 0.0)},
            },
            "interaction_targets": {"like_rate": 0.18, "save_rate": 0.10, "dwell_time_sec": 40},
            "rationale": "Short, readable lines with low complexity tend to attract quick likes and repeat viewing.",
        },
        {
            "id": "affective_optimism",
            "name": "Affective optimism",
            "feature_signals": {
                "nrc_positive": {"op": ">=", "value": dist.get("nrc_positive", {}).get("p75", 0.0)},
                "empath_psych_theme": {"op": ">=", "value": dist.get("empath_psych_theme", {}).get("p50", 0.0)},
                "words_per_line": {"op": "<=", "value": dist.get("words_per_line", {}).get("p75", 0.0)},
            },
            "interaction_targets": {"like_rate": 0.20, "save_rate": 0.12, "dwell_time_sec": 45},
            "rationale": "Warm, psychologically resonant poems often get fast likes and meaningful saves.",
        },
        {
            "id": "grief_retention",
            "name": "Grief retention",
            "feature_signals": {
                "nrc_negative": {"op": ">=", "value": dist.get("nrc_negative", {}).get("p75", 0.0)},
                "empath_psych_theme": {"op": ">=", "value": dist.get("empath_psych_theme", {}).get("p75", 0.0)},
                "dwell_time_sec": {"op": ">=", "value": dist.get("dwell_time_sec", {}).get("p50", 0.0)},
            },
            "interaction_targets": {"like_rate": 0.14, "save_rate": 0.13, "dwell_time_sec": 55},
            "rationale": "Emotionally heavy poems may not spike likes immediately, but they often drive longer dwell and stronger saves.",
        },
        {
            "id": "lexical_depth",
            "name": "Lexical depth / complexity",
            "feature_signals": {
                "mattr": {"op": ">=", "value": dist.get("mattr", {}).get("p75", 0.0)},
                "complex_sentence_ratio": {"op": ">=", "value": dist.get("complex_sentence_ratio", {}).get("p50", 0.0)},
                "concreteness": {"op": ">=", "value": dist.get("concreteness", {}).get("p50", 0.0)},
            },
            "interaction_targets": {"like_rate": 0.12, "save_rate": 0.11, "dwell_time_sec": 60},
            "rationale": "More complex, lexically rich poems often reward sustained reading and later revisits.",
        },
        {
            "id": "high_emotional_density",
            "name": "High emotional density",
            "feature_signals": {
                "nrc_negative": {"op": ">=", "value": dist.get("nrc_negative", {}).get("p50", 0.0)},
                "empath_physical_theme": {"op": ">=", "value": dist.get("empath_physical_theme", {}).get("p50", 0.0)},
                "subj_obj_score": {"op": ">=", "value": dist.get("subj_obj_score", {}).get("p75", 0.0)},
            },
            "interaction_targets": {"like_rate": 0.16, "save_rate": 0.13, "dwell_time_sec": 50},
            "rationale": "Emotionally dense poems often perform best when the language is vivid and body-oriented rather than purely abstract.",
        },
    ]

    policies = []
    for profile in profile_specs:
        profile_copy = {
            "id": profile["id"],
            "name": profile["name"],
            "feature_signals": profile["feature_signals"],
            "interaction_targets": profile["interaction_targets"],
            "rationale": profile["rationale"],
        }
        policies.append(profile_copy)
    return policies


def summarize_engagement_policy(poems: Iterable[Dict[str, Any]], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    poem_list = list(poems)
    categories = sorted({str(poem.get("category", "uncategorized")) for poem in poem_list})
    profile = profile or DEFAULT_POLICY
    rules = {
        metric: {
            "threshold": float(cfg["threshold"]),
            "high": float(cfg["high"]),
            "weight": float(cfg.get("weight", 1.0)),
        }
        for metric, cfg in profile.items()
    }
    return {
        "categories": categories,
        "category_summary": _category_features(poem_list),
        "rules": rules,
        "engagement_policies": build_feature_driven_policies(poem_list),
        "policy_notes": {
            "like_rate": "Use this as a view-through signal; optimize for sustained attention, not just initial approval.",
            "save_rate": "Saves indicate a post is useful, revisitable, or emotionally portable.",
            "dwell_time_sec": "Dwell time is the strongest proxy for whether a poem is actually being read rather than simply seen.",
        },
    }


def _load_poems(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Poem JSON must be a list of poem objects.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate engagement-rule summaries for a poem corpus.")
    parser.add_argument("--data", default="data/poems.json", help="Path to the poem JSON corpus.")
    parser.add_argument("--output", default="outputs/engagement_policy.json", help="Where to save the policy summary JSON.")
    parser.add_argument("--like-rate", type=float, default=None, help="Optional like rate to evaluate directly.")
    parser.add_argument("--save-rate", type=float, default=None, help="Optional save rate to evaluate directly.")
    parser.add_argument("--dwell-time", type=float, default=None, help="Optional dwell time in seconds to evaluate directly.")
    args = parser.parse_args()

    poems = _load_poems(args.data)
    summary = summarize_engagement_policy(poems)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if args.like_rate is not None or args.save_rate is not None or args.dwell_time is not None:
        classification = classify_engagement(
            like_rate=args.like_rate if args.like_rate is not None else 0.0,
            save_rate=args.save_rate if args.save_rate is not None else 0.0,
            dwell_time_sec=args.dwell_time if args.dwell_time is not None else 0.0,
        )
        print(json.dumps(classification, indent=2))
    else:
        print(f"Saved engagement policy summary to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
