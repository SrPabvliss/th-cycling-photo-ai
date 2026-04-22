"""Generate synthetic bib number images for OCR pretraining (Phase 1).

Produces 200K images of random 1-4 digit numbers with:
- Sport/display fonts (Bebas Neue, Oswald, Impact, Anton, etc.)
- Varied backgrounds (solid colors, gradients, noise)
- Realistic distortions (blur, rotation, skew, noise)

Output: LMDB dataset ready for PARSeq training + txt list for PP-OCRv5.

Usage:
    uv run python scripts/generate_synthetic_bibs.py
    uv run python scripts/generate_synthetic_bibs.py --count 200000
    uv run python scripts/generate_synthetic_bibs.py --count 1000 --preview  # test run
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import lmdb
import os
import random
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "ocr" / "synthetic"
FONTS_DIR = PROJECT_ROOT / "data" / "ocr" / "fonts"

# Google Fonts that look like racing bibs (all OFL licensed)
# We'll use system fonts as fallback + download sport fonts
FONT_URLS = {
    "BebasNeue": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Oswald-Bold": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Anton": "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
    "BigShouldersDisplay": "https://github.com/google/fonts/raw/main/ofl/bigshouldersdisplay/BigShouldersDisplay%5Bwght%5D.ttf",
    "BarlowCondensed-Bold": "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
    "RobotoCondensed-Bold": "https://github.com/google/fonts/raw/main/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf",
}


def download_fonts():
    """Download sport fonts if not cached."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    import urllib.request

    fonts = []
    for name, url in FONT_URLS.items():
        font_path = FONTS_DIR / f"{name}.ttf"
        if not font_path.exists():
            print(f"  Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, font_path)
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        fonts.append(str(font_path))

    # Add system Impact if available
    system_fonts = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    ]
    for sf in system_fonts:
        if os.path.exists(sf):
            fonts.append(sf)
            break

    print(f"  {len(fonts)} fonts available")
    return fonts


def random_bib_number() -> str:
    """Generate random bib number 1-4 digits, weighted like real races."""
    r = random.random()
    if r < 0.03:  # 3% single digit
        return str(random.randint(1, 9))
    elif r < 0.35:  # 32% two digits
        return str(random.randint(10, 99))
    elif r < 1.0:  # 65% three digits
        return str(random.randint(100, 999))


def random_background(w: int, h: int) -> Image.Image:
    """Generate varied background."""
    choice = random.random()

    if choice < 0.3:
        # Solid color (white, light gray, cream, light blue — like bib paper)
        colors = [
            (255, 255, 255), (240, 240, 240), (255, 250, 240),
            (220, 230, 240), (245, 245, 220), (200, 200, 200),
            (180, 180, 180), (255, 255, 230),
        ]
        bg = Image.new("RGB", (w, h), random.choice(colors))

    elif choice < 0.6:
        # Gradient
        bg = Image.new("RGB", (w, h))
        c1 = tuple(random.randint(150, 255) for _ in range(3))
        c2 = tuple(random.randint(150, 255) for _ in range(3))
        for y in range(h):
            r = y / max(h - 1, 1)
            color = tuple(int(c1[i] * (1 - r) + c2[i] * r) for i in range(3))
            ImageDraw.Draw(bg).line([(0, y), (w, y)], fill=color)

    elif choice < 0.85:
        # Noisy solid (simulates fabric texture)
        base = tuple(random.randint(180, 255) for _ in range(3))
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)

    else:
        # Dark background (some bibs are dark with white text)
        colors = [
            (30, 30, 30), (0, 0, 80), (80, 0, 0), (0, 60, 0),
            (50, 50, 50), (20, 20, 60),
        ]
        bg = Image.new("RGB", (w, h), random.choice(colors))

    return bg


def random_text_color(bg_color: tuple) -> tuple:
    """Pick text color with good contrast against background."""
    brightness = sum(bg_color[:3]) / 3
    if brightness > 150:
        # Dark text on light bg
        colors = [
            (0, 0, 0), (20, 20, 20), (40, 40, 40),
            (0, 0, 80), (80, 0, 0), (0, 60, 0),
        ]
    else:
        # Light text on dark bg
        colors = [
            (255, 255, 255), (240, 240, 240), (255, 255, 0),
            (255, 200, 0),
        ]
    return random.choice(colors)


def generate_one(text: str, fonts: list[str], size: tuple = (128, 32)) -> Image.Image:
    """Generate one synthetic bib number image."""
    w, h = size

    # Background
    bg = random_background(w, h)

    # Sample bg color for contrast
    bg_sample = np.array(bg).mean(axis=(0, 1))

    # Font
    font_path = random.choice(fonts)
    font_size = random.randint(int(h * 0.5), int(h * 0.9))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Text color
    text_color = random_text_color(tuple(int(x) for x in bg_sample))

    # Draw text centered
    draw = ImageDraw.Draw(bg)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Random offset from center
    offset_x = random.randint(-int(w * 0.05), int(w * 0.05))
    offset_y = random.randint(-int(h * 0.05), int(h * 0.05))
    x = (w - tw) // 2 + offset_x
    y = (h - th) // 2 + offset_y - bbox[1]

    # Optional stroke/outline
    if random.random() < 0.2:
        stroke_width = random.randint(1, 2)
        stroke_color = (255, 255, 255) if sum(text_color) < 384 else (0, 0, 0)
        draw.text((x, y), text, font=font, fill=text_color,
                  stroke_width=stroke_width, stroke_fill=stroke_color)
    else:
        draw.text((x, y), text, font=font, fill=text_color)

    # Augmentations
    img = bg

    # Rotation (slight)
    if random.random() < 0.4:
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, expand=False, fillcolor=tuple(int(x) for x in bg_sample))

    # Perspective warp (subtle)
    if random.random() < 0.3:
        coeffs = [1 + random.uniform(-0.05, 0.05) for _ in range(8)]
        try:
            img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        except Exception:
            pass

    # Blur
    if random.random() < 0.3:
        radius = random.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Brightness
    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # JPEG compression artifacts
    if random.random() < 0.2:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 70))
        buf.seek(0)
        img = Image.open(buf).copy()

    # Noise
    if random.random() < 0.25:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, random.uniform(5, 20), arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    return img.convert("RGB")


def write_lmdb(output_dir: Path, images: list[tuple[Image.Image, str]]):
    """Write images + labels to LMDB format (PARSeq/deep-text-recognition standard)."""
    lmdb_dir = output_dir / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Estimate map size: ~50KB per image average, with headroom
    map_size = len(images) * 100 * 1024  # 100KB per entry

    env = lmdb.open(str(lmdb_dir), map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, (img, label) in enumerate(images):
            # Save image as bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Keys follow clovaai/deep-text-recognition-benchmark convention
            img_key = f"image-{idx+1:09d}".encode()
            label_key = f"label-{idx+1:09d}".encode()

            txn.put(img_key, img_bytes)
            txn.put(label_key, label.encode())

        # Store dataset size
        txn.put("num-samples".encode(), str(len(images)).encode())

    env.close()
    print(f"  LMDB written: {lmdb_dir} ({len(images)} samples)")


def write_txt_list(output_dir: Path, images: list[tuple[Image.Image, str]]):
    """Write images + txt label list (PP-OCRv5 format)."""
    imgs_dir = output_dir / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for idx, (img, label) in enumerate(images):
        fname = f"syn_{idx:07d}.jpg"
        img.save(imgs_dir / fname, quality=95)
        labels.append(f"{fname}\t{label}")

    label_file = output_dir / "labels.txt"
    label_file.write_text("\n".join(labels))
    print(f"  TXT list written: {label_file} ({len(labels)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bib numbers")
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="Save preview grid")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading sport fonts...")
    fonts = download_fonts()
    if not fonts:
        print("ERROR: No fonts available. Cannot generate.")
        return

    print(f"\nGenerating {args.count} synthetic bib images...")

    images: list[tuple[Image.Image, str]] = []
    for i in range(args.count):
        text = random_bib_number()
        img = generate_one(text, fonts, size=(128, 32))
        images.append((img, text))

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{args.count} ({(i+1)/args.count*100:.0f}%)")

    # Preview grid
    if args.preview:
        preview_dir = OUTPUT_DIR / "preview"
        preview_dir.mkdir(exist_ok=True)
        for i in range(min(100, len(images))):
            img, label = images[i]
            img.save(preview_dir / f"{i:04d}_{label}.jpg")
        print(f"  Preview: {preview_dir}")

    # Write LMDB (PARSeq)
    print("\nWriting LMDB format (PARSeq)...")
    write_lmdb(OUTPUT_DIR, images)

    # Write TXT list (PP-OCRv5)
    print("Writing TXT list format (PP-OCRv5)...")
    write_txt_list(OUTPUT_DIR, images)

    # Stats
    lengths = {}
    for _, label in images:
        l = len(label)
        lengths[l] = lengths.get(l, 0) + 1

    print(f"\n{'='*60}")
    print(f"Generated {len(images)} synthetic bib images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nDigit length distribution:")
    for l in sorted(lengths):
        print(f"  {l} digits: {lengths[l]} ({lengths[l]/len(images)*100:.0f}%)")
    print(f"\nFormats:")
    print(f"  LMDB: {OUTPUT_DIR / 'lmdb'}")
    print(f"  TXT:  {OUTPUT_DIR / 'labels.txt'}")


if __name__ == "__main__":
    main()
