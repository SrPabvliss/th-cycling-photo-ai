from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image


def extract_crop(
    image_path: Path, *, x: int, y: int, w: int, h: int,
    padding_ratio: float = 0.12,
) -> Image.Image:
    img = Image.open(image_path)
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(img.width, x + w + pad_x)
    bottom = min(img.height, y + h + pad_y)
    return img.crop((left, top, right, bottom))


def crop_sha256_of(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()
