"""OCR poem images and split feed screenshots into individual poem crops."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def extract_lines(image_path: Path | str | Image.Image) -> List[str]:
    if isinstance(image_path, (str, Path)):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    text = pytesseract.image_to_string(image)
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            continue
        lines.append(line)
    return lines


def _ocr_word_boxes(image: Image.Image) -> List[Dict[str, Any]]:
    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=Output.DICT,
    )

    boxes: List[Dict[str, Any]] = []
    for idx, text in enumerate(data.get("text", [])):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        left = int(float(data["left"][idx]))
        top = int(float(data["top"][idx]))
        width = int(float(data["width"][idx]))
        height = int(float(data["height"][idx]))
        if width <= 8 or height <= 8:
            continue
        boxes.append({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "text": cleaned,
        })
    return boxes


def _cluster_text_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for box in sorted(boxes, key=lambda b: (b["top"], b["left"])):
        cx = box["left"] + box["width"] / 2
        cy = box["top"] + box["height"] / 2
        matched = None
        for cluster in clusters:
            if abs(cx - cluster["cx"]) <= max(180, cluster["width"] * 0.9) and abs(cy - cluster["cy"]) <= max(180, cluster["height"] * 0.9):
                matched = cluster
                break

        if matched is None:
            clusters.append({
                "left": box["left"],
                "top": box["top"],
                "right": box["left"] + box["width"],
                "bottom": box["top"] + box["height"],
                "cx": cx,
                "cy": cy,
                "width": box["width"],
                "height": box["height"],
                "count": 1,
            })
        else:
            matched["left"] = min(matched["left"], box["left"])
            matched["top"] = min(matched["top"], box["top"])
            matched["right"] = max(matched["right"], box["left"] + box["width"])
            matched["bottom"] = max(matched["bottom"], box["top"] + box["height"])
            matched["cx"] = (matched["left"] + matched["right"]) / 2
            matched["cy"] = (matched["top"] + matched["bottom"]) / 2
            matched["width"] = matched["right"] - matched["left"]
            matched["height"] = matched["bottom"] - matched["top"]
            matched["count"] += 1

    return clusters


def detect_poem_regions(image_path: str | Path) -> List[Tuple[int, int, int, int]]:
    image = Image.open(image_path).convert("RGB")
    boxes = _ocr_word_boxes(image)
    if not boxes:
        return [(0, 0, image.width, image.height)]

    clusters = _cluster_text_boxes(boxes)
    regions: List[Tuple[int, int, int, int]] = []
    for cluster in clusters:
        pad = max(20, min(cluster["width"], cluster["height"]) // 4)
        x0 = max(0, int(cluster["left"]) - pad)
        y0 = max(0, int(cluster["top"]) - pad)
        x1 = min(image.width, int(cluster["right"]) + pad)
        y1 = min(image.height, int(cluster["bottom"]) + pad)
        if x1 - x0 <= 20 or y1 - y0 <= 20:
            continue
        regions.append((x0, y0, x1, y1))

    if not regions:
        return [(0, 0, image.width, image.height)]
    return regions


def split_feed_screenshot(
    image_path: str | Path,
    output_dir: str | Path | None = None,
) -> List[Path]:
    image = Image.open(image_path).convert("RGB")
    out_dir = Path(output_dir) if output_dir is not None else Path(image_path).with_suffix("") / "poem_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = detect_poem_regions(image_path)
    crop_paths: List[Path] = []
    for idx, region in enumerate(regions, start=1):
        crop = image.crop(region)
        crop_path = out_dir / f"poem_{idx:03d}.png"
        crop.save(crop_path)
        crop_paths.append(crop_path)
    return crop_paths


def poem_dict_from_image(image_path: Path | str, poem_id: str) -> Dict[str, Any]:
    return {"id": poem_id, "lines": extract_lines(image_path)}


def process_feed_screenshot(
    image_path: str | Path,
    out_file: str | Path = "data/poems.json",
    output_dir: str | Path | None = None,
) -> List[Dict[str, Any]]:
    crop_paths = split_feed_screenshot(image_path, output_dir=output_dir)
    poems = []
    for idx, crop in enumerate(crop_paths, start=1):
        poem_id = f"poem_{idx:03d}"
        poems.append(poem_dict_from_image(crop, poem_id))
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(poems, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(poems)} poems from feed screenshot to {out}")
    return poems


def _iter_image_files(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def process_folder_images(folder: str | Path, out_file: str | Path = "data/poems.json") -> List[Dict[str, Any]]:
    folder = Path(folder)
    images = _iter_image_files(folder)
    poems = [
        poem_dict_from_image(p, f"poem_{i:03d}")
        for i, p in enumerate(images, start=1)
    ]
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(poems, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(poems)} poems to {out}")
    for p, poem in zip(images, poems):
        print(f"  {poem['id']} <- {p.name}")
    return poems


def process_feed_folder(
    folder: str | Path,
    out_file: str | Path = "data/poems.json",
    crop_root: str | Path | None = None,
) -> List[Dict[str, Any]]:
    folder = Path(folder)
    screenshot_files = _iter_image_files(folder)
    if not screenshot_files:
        raise FileNotFoundError(f"No image files found in {folder}")

    poems: List[Dict[str, Any]] = []
    crop_root_path = Path(crop_root) if crop_root is not None else folder / "poem_crops"
    crop_root_path.mkdir(parents=True, exist_ok=True)

    for screenshot in screenshot_files:
        screenshot_crops = split_feed_screenshot(screenshot, output_dir=crop_root_path / screenshot.stem)
        for idx, crop in enumerate(screenshot_crops, start=1):
            poem_id = f"{screenshot.stem}_{idx:03d}"
            poems.append(poem_dict_from_image(crop, poem_id))

    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(poems, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(poems)} poems from {len(screenshot_files)} screenshots to {out}")
    return poems


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR poem images or split a feed screenshot into individual poem crops.")
    parser.add_argument("path", help="Path to a folder of images or a single screenshot file.")
    parser.add_argument("--output", default="data/poems.json", help="Where to write poem JSON output.")
    parser.add_argument("--split-feed", action="store_true", help="Treat the input as one screenshot containing multiple poem cards.")
    parser.add_argument("--crop-dir", default=None, help="Directory to save poem crops when splitting a feed screenshot.")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        if args.split_feed:
            process_feed_folder(target, out_file=args.output, crop_root=args.crop_dir)
        else:
            process_folder_images(target, out_file=args.output)
        return

    if args.split_feed:
        crop_dir = args.crop_dir or target.parent / "poem_crops"
        process_feed_screenshot(target, out_file=args.output, output_dir=crop_dir)
        return

    poem = poem_dict_from_image(target, "poem_001")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([poem], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote 1 poem to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python images_to_poems.py <image_folder_or_feed_screenshot> [--split-feed] [--output data/poems.json]")
    main()