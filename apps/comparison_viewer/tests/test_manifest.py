import json
from pathlib import Path

from PIL import Image

from apps.comparison_viewer.storage.manifest import build_manifest


def _make_jpg(path: Path, w=100, h=80):
    img = Image.new("RGB", (w, h), color=(255, 0, 0))
    img.save(path, format="JPEG")


def test_build_manifest_assigns_groups(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_jpg(images_dir / "IMG_1.jpg")
    _make_jpg(images_dir / "IMG_2.jpg")
    _make_jpg(images_dir / "IMG_3.jpg")

    groups_yaml = tmp_path / "groups.yaml"
    groups_yaml.write_text(
        "dorsal_9:\n  - IMG_1.jpg\n  - IMG_2.jpg\n"
    )

    out = tmp_path / "manifest.json"
    build_manifest(images_dir, groups_yaml, out)
    data = json.loads(out.read_text())
    assert data["n_images"] == 3
    by_filename = {im["filename"]: im for im in data["images"]}
    assert by_filename["IMG_1.jpg"]["group_id"] == "dorsal_9"
    assert by_filename["IMG_1.jpg"]["photo_index_in_group"] == 1
    assert by_filename["IMG_3.jpg"]["group_id"] is None


def test_build_manifest_without_groups_yaml(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_jpg(images_dir / "IMG_X.jpg")
    out = tmp_path / "manifest.json"
    build_manifest(images_dir, None, out)
    data = json.loads(out.read_text())
    assert data["n_images"] == 1
    assert data["images"][0]["group_id"] is None


def test_manifest_sha_is_stable(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_jpg(images_dir / "IMG_A.jpg")
    out1 = tmp_path / "m1.json"
    out2 = tmp_path / "m2.json"
    build_manifest(images_dir, None, out1)
    build_manifest(images_dir, None, out2)
    sha1 = json.loads(out1.read_text())["images"][0]["sha256"]
    sha2 = json.loads(out2.read_text())["images"][0]["sha256"]
    assert sha1 == sha2
