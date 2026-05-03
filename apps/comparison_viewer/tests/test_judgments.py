from pathlib import Path
from apps.comparison_viewer.storage.judgments import (
    append_judgment, load_judgments_for_image,
)
from apps.comparison_viewer.storage.schemas import JudgmentRecord


def test_append_then_load(tmp_path):
    j = JudgmentRecord(
        session_id="s1", image_sha256="a" * 64, stage="detection",
        system_id="yolo11m", judgment_codes=["correct"],
        judged_at="2026-05-03T00:00:00Z",
    )
    append_judgment(tmp_path, j)
    loaded = load_judgments_for_image(tmp_path, "a" * 64)
    assert len(loaded) == 1
    assert loaded[0].judgment_codes == ["correct"]


def test_last_write_wins(tmp_path):
    j1 = JudgmentRecord(session_id="s1", image_sha256="a" * 64,
                         stage="ocr", system_id="parseq_base",
                         parent_crop_sha256="b" * 64,
                         judgment_codes=["correct"], judged_at="t1")
    j2 = JudgmentRecord(session_id="s1", image_sha256="a" * 64,
                         stage="ocr", system_id="parseq_base",
                         parent_crop_sha256="b" * 64,
                         judgment_codes=["wrong"],
                         correct_value="9", judged_at="t2")
    append_judgment(tmp_path, j1)
    append_judgment(tmp_path, j2)
    loaded = load_judgments_for_image(tmp_path, "a" * 64,
                                       last_write_wins=True)
    assert len(loaded) == 1
    assert loaded[0].judgment_codes == ["wrong"]
