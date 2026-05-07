"""Sanity tests for the smoke test script.

Tests that the script imports correctly and runs with mocked pipeline calls.
No actual API calls are made during testing.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image
import pytest

from apps.comparison_viewer.scripts.smoke_test_all import smoke_one


@pytest.fixture
def tmp_image_with_manifest(tmp_path):
    """Create a temporary directory with 1 test image and manifest."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create a single test image.
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    img_path = images_dir / "TEST_IMG_001.jpg"
    img.save(img_path, format="JPEG")

    return img_path


def test_smoke_one_imports():
    """Verify the smoke_one function can be imported."""
    assert callable(smoke_one)


@pytest.mark.asyncio
async def test_smoke_one_runs_with_mocked_pipeline(tmp_image_with_manifest, monkeypatch):
    """Test smoke_one with fully mocked pipeline_runner.run_stage.

    Mocks run_stage to yield deterministic done events for all 16 systems.
    Verifies that the script processes events correctly and returns a
    properly formatted results dict.
    """
    image_path = tmp_image_with_manifest

    # Mock run_stage to yield done events for all requested systems.
    async def mock_run_stage(
        *,
        domain: str,
        image_sha256: str,
        system_ids: list[str],
        mode: str,
        experiments_root: Path,
        parent_crop_sha256: str | None = None,
        region: str | None = None,
        **adapter_kwargs,
    ):
        """Yields deterministic done events for all systems in system_ids."""
        for sid in system_ids:
            # Create a mock CallRecord with minimal required fields.
            rec = MagicMock()
            rec.error_category = None
            rec.cost_usd = 0.01  # Mock cost
            rec.normalized_output = {
                "bboxes": [
                    {
                        "x": 10,
                        "y": 10,
                        "w": 50,
                        "h": 40,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class_name": "competidor_number",
                        "label": "competidor_number",
                        "crop_id": "c1",
                        "crop_sha256": "a" * 64,
                    },
                    {
                        "x": 100,
                        "y": 100,
                        "w": 80,
                        "h": 60,
                        "confidence": 0.85,
                        "class_id": 1,
                        "class_name": "helmet",
                        "label": "helmet",
                        "crop_id": "c2",
                        "crop_sha256": "b" * 64,
                    },
                    {
                        "x": 150,
                        "y": 150,
                        "w": 100,
                        "h": 80,
                        "confidence": 0.8,
                        "class_id": 2,
                        "class_name": "cyclist_clothes",
                        "label": "cyclist_clothes",
                        "crop_id": "c3",
                        "crop_sha256": "c" * 64,
                    },
                    {
                        "x": 200,
                        "y": 200,
                        "w": 120,
                        "h": 100,
                        "confidence": 0.75,
                        "class_id": 3,
                        "class_name": "bicycle",
                        "label": "bicycle",
                        "crop_id": "c4",
                        "crop_sha256": "d" * 64,
                    },
                ]
            }
            # Differentiate output by domain and system.
            if domain == "ocr":
                rec.normalized_output = {
                    "predicted_text": f"OCR_{sid}",
                    "text": f"OCR_{sid}",
                    "confidence": 0.95,
                }
            elif domain == "color":
                rec.normalized_output = {
                    "primary_color": "blue",
                    "secondary_color": "red",
                    "confidence_primary": 0.92,
                }
            yield ("done", sid, rec)

    # Patch pipeline_runner.run_stage to use our mock.
    with patch(
        "apps.comparison_viewer.scripts.smoke_test_all.run_stage",
        side_effect=mock_run_stage,
    ):
        results = await smoke_one(image_path)

    # Verify we got results for all 16 systems.
    assert len(results) == 16

    # Verify all results are PASS or SKIP (no actual API calls fail in mock).
    for sid, status in results.items():
        assert status in ("PASS", "SKIP (no bib crop)", "SKIP (no detection)", "SKIP (no bbox)"), \
            f"{sid} has unexpected status: {status}"

    # Count PASS results (at least detection systems should pass).
    pass_count = sum(1 for v in results.values() if v == "PASS")
    assert pass_count >= 4, f"Expected at least 4 detection systems to pass, got {pass_count}"


@pytest.mark.asyncio
async def test_smoke_one_handles_detection_failure(tmp_image_with_manifest, monkeypatch):
    """Test that smoke_one correctly reports FAIL for a detection system.

    Verifies error_category is captured and formatted.
    """
    image_path = tmp_image_with_manifest

    async def mock_run_stage(
        *,
        domain: str,
        image_sha256: str,
        system_ids: list[str],
        mode: str,
        experiments_root: Path,
        parent_crop_sha256: str | None = None,
        region: str | None = None,
        **adapter_kwargs,
    ):
        """Yields mixed done/error events."""
        for idx, sid in enumerate(system_ids):
            if idx == 0:
                # First system fails with an error.
                rec = MagicMock()
                rec.error_category = "timeout"
                rec.cost_usd = 0.0
                yield ("error", sid, rec)
            else:
                # Rest succeed.
                rec = MagicMock()
                rec.error_category = None
                rec.cost_usd = 0.01
                rec.normalized_output = {"bboxes": []} if domain == "detection" else {}
                yield ("done", sid, rec)

    with patch(
        "apps.comparison_viewer.scripts.smoke_test_all.run_stage",
        side_effect=mock_run_stage,
    ):
        results = await smoke_one(image_path)

    # Verify detection systems (first batch).
    # At least one should show FAIL (timeout).
    fail_count = sum(1 for v in results.values() if v.startswith("FAIL"))
    assert fail_count > 0, "Expected at least 1 FAIL status"


def test_smoke_one_main_exit_code_zero(tmp_image_with_manifest, monkeypatch, capsys):
    """Test that main() exits with code 0 when all systems pass.

    Uses monkeypatch to mock sys.argv and asyncio.run.
    """
    import sys
    from apps.comparison_viewer.scripts.smoke_test_all import main

    # Mock asyncio.run to return all-pass results.
    mock_results = {
        f"system_{i}": "PASS" for i in range(16)
    }

    with patch("apps.comparison_viewer.scripts.smoke_test_all.asyncio.run", return_value=mock_results):
        with patch.object(sys, "argv", ["smoke_test_all.py", "--image", str(tmp_image_with_manifest)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, "Expected exit code 0 for all-pass results"


def test_smoke_one_main_exit_code_nonzero(tmp_image_with_manifest, monkeypatch, capsys):
    """Test that main() exits with N when N systems fail."""
    import sys
    from apps.comparison_viewer.scripts.smoke_test_all import main

    # Mock asyncio.run to return 3 failures.
    mock_results = {
        "system_0": "FAIL (timeout)",
        "system_1": "FAIL (api_error)",
        "system_2": "FAIL (schema)",
        **{f"system_{i}": "PASS" for i in range(3, 16)},
    }

    with patch("apps.comparison_viewer.scripts.smoke_test_all.asyncio.run", return_value=mock_results):
        with patch.object(sys, "argv", ["smoke_test_all.py", "--image", str(tmp_image_with_manifest)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 3, "Expected exit code 3 for 3 failures"


def test_smoke_one_main_missing_image():
    """Test that main() exits with error when image does not exist."""
    import sys
    from apps.comparison_viewer.scripts.smoke_test_all import main

    with patch.object(sys, "argv", ["smoke_test_all.py", "--image", "/nonexistent/image.jpg"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1, "Expected exit code 1 for missing image"
