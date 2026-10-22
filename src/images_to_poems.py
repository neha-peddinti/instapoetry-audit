"""
OCR a folder of poem images into poems.json, remove title/author.
Uses tesseract via pytesseract and Pillow to read images.

"""

import json
import sys
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def extract_lines(image_path: Path) -> List[str]:
    text = pytesseract.image_to_string(Image.open(image_path))
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):  # drop social media handles
            continue
        lines.append(line)
    return lines


def main(folder: str, out_file: str = "data/poems.json") -> None:
    images = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    poems = [
        {"id": f"poem_{i:03d}", "lines": extract_lines(p)}
        for i, p in enumerate(images, start=1)
    ]
    Path(out_file).write_text(
        json.dumps(poems, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(poems)} poems to {out_file}")
    for p, poem in zip(images, poems):
        print(f"  {poem['id']} <- {p.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python images_to_poems_json.py <image_folder>")
    main(sys.argv[1])