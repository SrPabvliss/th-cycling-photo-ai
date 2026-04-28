"""Tests for validation_set loader (F3).

Uses tmp_path fixtures + monkeypatched paths so tests don't depend on the
real data/color/ contents.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cycling_photo_ai.color.dataset import validation_set
from cycling_photo_ai.color.dataset.validation_set import (
    ValidationCrop,
    label_distribution,
    load_validation_set,
)


@pytest.fixture
def fake_dataset(tmp_path: Path, monkeypatch) -> Path:
    crops_dir = tmp_path / "crops"
    labels_dir = tmp_path / "labels"
    crops_dir.mkdir()
    labels_dir.mkdir()
    (crops_dir / "helmet").mkdir()
    (crops_dir / "bicycle").mkdir()

    # metadata.csv
    metadata_rows = [
        {
            "crop_id": 0, "crop_file": "helmet/img_00000.jpg", "region": "helmet",
            "source_image": "a.jpg", "source_split": "test",
            "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 100, "bbox_y2": 200,
            "crop_w": 90, "crop_h": 180, "label_top1": "", "label_top2": "", "notes": "",
        },
        {
            "crop_id": 1, "crop_file": "bicycle/img_00001.jpg", "region": "bicycle",
            "source_image": "b.jpg", "source_split": "test",
            "bbox_x1": 5, "bbox_y1": 5, "bbox_x2": 50, "bbox_y2": 60,
            "crop_w": 45, "crop_h": 55, "label_top1": "", "label_top2": "", "notes": "",
        },
        {
            "crop_id": 2, "crop_file": "helmet/img_00002.jpg", "region": "helmet",
            "source_image": "c.jpg", "source_split": "valid",
            "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 80, "bbox_y2": 80,
            "crop_w": 80, "crop_h": 80, "label_top1": "", "label_top2": "", "notes": "",
        },
    ]
    with open(crops_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    # labels.jsonl: one labeled, one skipped, one labeled top1+top2
    labels = [
        {"crop_file": "helmet/img_00000.jpg", "region": "helmet",
         "top1": "rojo", "top2": None, "notes": ""},
        {"crop_file": "bicycle/img_00001.jpg", "region": "bicycle",
         "top1": None, "top2": None, "notes": "skipped"},
        {"crop_file": "helmet/img_00002.jpg", "region": "helmet",
         "top1": "azul", "top2": "blanco", "notes": "two-tone"},
    ]
    jsonl_path = labels_dir / "validation.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in labels:
            f.write(json.dumps(entry) + "\n")

    monkeypatch.setattr(validation_set, "COLOR_CROPS_DIR", crops_dir)
    monkeypatch.setattr(validation_set, "COLOR_LABELS_DIR", labels_dir)
    return tmp_path


class TestLoadValidationSet:
    def test_excludes_skipped_by_default(self, fake_dataset):
        crops = load_validation_set()
        assert len(crops) == 2
        files = {c.crop_file for c in crops}
        assert "bicycle/img_00001.jpg" not in files

    def test_includes_skipped_when_flag_set(self, fake_dataset):
        # Skipped crops have top1=None — still excluded because we require top1
        crops = load_validation_set(include_skipped=True)
        assert len(crops) == 2  # still drops top1=None

    def test_region_filter(self, fake_dataset):
        crops = load_validation_set(region="helmet")
        assert len(crops) == 2
        assert all(c.region == "helmet" for c in crops)

    def test_returns_validation_crop_instances(self, fake_dataset):
        crops = load_validation_set()
        assert all(isinstance(c, ValidationCrop) for c in crops)
        crop = next(c for c in crops if c.crop_file == "helmet/img_00000.jpg")
        assert crop.top1 == "rojo"
        assert crop.top2 is None
        assert crop.bbox == (10, 20, 100, 200)
        assert crop.source_split == "test"

    def test_top2_preserved(self, fake_dataset):
        crops = load_validation_set()
        crop = next(c for c in crops if c.top2 == "blanco")
        assert crop.top1 == "azul"
        assert crop.top2 == "blanco"
        assert crop.notes == "two-tone"

    def test_sorted_by_crop_file(self, fake_dataset):
        crops = load_validation_set()
        files = [c.crop_file for c in crops]
        assert files == sorted(files)


class TestLabelDistribution:
    def test_histogram(self, fake_dataset):
        crops = load_validation_set()
        hist = label_distribution(crops)
        assert hist == {"azul": 1, "rojo": 1}

    def test_sorted_descending(self):
        crops = [
            ValidationCrop("a", "helmet", "rojo", None, "", "x", "test", (0, 0, 1, 1)),
            ValidationCrop("b", "helmet", "rojo", None, "", "x", "test", (0, 0, 1, 1)),
            ValidationCrop("c", "helmet", "azul", None, "", "x", "test", (0, 0, 1, 1)),
        ]
        hist = label_distribution(crops)
        assert list(hist.keys()) == ["rojo", "azul"]
        assert hist["rojo"] == 2


class TestErrors:
    def test_missing_metadata_raises(self, tmp_path, monkeypatch):
        crops_dir = tmp_path / "crops"
        crops_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "validation.jsonl").write_text("")

        monkeypatch.setattr(validation_set, "COLOR_CROPS_DIR", crops_dir)
        monkeypatch.setattr(validation_set, "COLOR_LABELS_DIR", labels_dir)

        with pytest.raises(FileNotFoundError, match="Metadata"):
            load_validation_set()

    def test_missing_labels_raises(self, tmp_path, monkeypatch):
        crops_dir = tmp_path / "crops"
        crops_dir.mkdir()
        (crops_dir / "metadata.csv").write_text("crop_file\n")
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        monkeypatch.setattr(validation_set, "COLOR_CROPS_DIR", crops_dir)
        monkeypatch.setattr(validation_set, "COLOR_LABELS_DIR", labels_dir)

        with pytest.raises(FileNotFoundError, match="Labels"):
            load_validation_set()
