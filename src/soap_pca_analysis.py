import json
import re
from pathlib import Path
from typing import Any, Dict, List


def clean_post_text(text: str) -> str:
    """Remove obvious Instagram title/author metadata while preserving poem text."""
    if text is None:
        return ""

    cleaned_lines: List[str] = []
    lines = str(text).splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("by "):
            continue
        if lower.startswith("@"):
            continue
        if idx == 0 and len(stripped) <= 45 and stripped.istitle() and not any(ch in stripped for ch in ".!?"):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def load_local_corpus(path: str | Path) -> List[Dict[str, Any]]:
    """Load a local folder of JSON exports into the same record format consumed downstream."""
    corpus_dir = Path(path)
    records: List[Dict[str, Any]] = []

    for json_path in sorted(corpus_dir.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            record = dict(payload)
            text = payload.get("text") or payload.get("caption") or payload.get("content") or ""
            record["pk"] = str(payload.get("pk") or payload.get("id") or json_path.stem)
            record["text"] = clean_post_text(str(text))
            records.append(record)

    return records
