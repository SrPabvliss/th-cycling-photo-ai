from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps


def open_image_oriented(image_path: Path) -> Image.Image:
    """Open an image and apply EXIF orientation. Required for Sony/iPhone JPEGs
    that store rotation in EXIF rather than rotating pixels — st.image and
    PIL.Image.open both ignore EXIF by default, producing sideways previews
    and (worse) bboxes drawn in rotated coordinate space.
    """
    return ImageOps.exif_transpose(Image.open(image_path))


_ORIENTED_CACHE_DIR = Path("data/exploratorio/_oriented_cache")


def oriented_image_path(image_path: Path) -> Path:
    """Return a cached EXIF-baked copy of `image_path`. If the source has no
    EXIF rotation (or rotation is identity), returns the source path unchanged.
    Otherwise writes a transposed JPEG to the cache dir keyed by source name
    and mtime, so detectors and the UI see the same oriented pixels.
    """
    src_img = Image.open(image_path)
    exif = src_img.getexif()
    orientation = exif.get(0x0112, 1) if exif else 1
    if orientation == 1:
        return image_path  # nothing to bake

    _ORIENTED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = str(image_path).replace("/", "__").replace(":", "_")
    cache_path = _ORIENTED_CACHE_DIR / f"{safe_name}__o{orientation}.jpg"
    if cache_path.exists() and cache_path.stat().st_mtime >= image_path.stat().st_mtime:
        return cache_path

    oriented = ImageOps.exif_transpose(src_img).convert("RGB")
    oriented.save(cache_path, format="JPEG", quality=95)
    return cache_path


def extract_crop(
    image_path: Path, *, x: int, y: int, w: int, h: int,
    padding_ratio: float = 0.12,
) -> Image.Image:
    img = open_image_oriented(image_path)
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
