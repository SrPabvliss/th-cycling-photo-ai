from PIL import Image
from apps.comparison_viewer.components.crop_utils import (
    extract_crop, crop_sha256_of,
)


def test_crop_returns_pil_image(tmp_path):
    img = Image.new("RGB", (200, 200), color=(0, 0, 0))
    src = tmp_path / "i.jpg"
    img.save(src)
    crop = extract_crop(src, x=10, y=20, w=50, h=60, padding_ratio=0.0)
    assert crop.size == (50, 60)


def test_crop_sha_stable():
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    assert crop_sha256_of(img1) == crop_sha256_of(img2)


def test_crop_sha_differs_for_different_content():
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 0, 255))
    assert crop_sha256_of(img1) != crop_sha256_of(img2)
