from __future__ import annotations

from cycling_photo_ai.telemetry.report_buffer import ReportBuffer


def _report(name: str) -> dict:
    return {"report_id": name}


def test_take_returns_fifo_and_return_front_restores_order():
    buffer = ReportBuffer(max_size=10)
    for name in ("a", "b", "c"):
        buffer.add(_report(name))

    taken = buffer.take(2)
    assert [r["report_id"] for r in taken] == ["a", "b"]
    assert len(buffer) == 1

    buffer.return_front(taken)
    assert [r["report_id"] for r in buffer.take(10)] == ["a", "b", "c"]
    assert len(buffer) == 0


def test_full_buffer_drops_oldest_but_keeps_boot():
    buffer = ReportBuffer(max_size=3)
    buffer.add(_report("boot"), protected=True)
    for name in ("a", "b", "c"):
        buffer.add(_report(name))

    assert len(buffer) == 3
    assert buffer.dropped == 1
    assert [r["report_id"] for r in buffer.take(10)] == ["boot", "b", "c"]


def test_returned_reports_never_overflow_the_bound():
    buffer = ReportBuffer(max_size=2)
    buffer.add(_report("a"))
    taken = buffer.take(1)
    buffer.add(_report("b"))
    buffer.add(_report("c"))

    buffer.return_front(taken)

    assert len(buffer) == 2
    assert buffer.dropped == 1


def test_boot_stays_protected_after_a_failed_delivery():
    buffer = ReportBuffer(max_size=3)
    buffer.add(_report("boot"), protected=True)

    buffer.return_front(buffer.take(10))
    for name in ("a", "b", "c", "d"):
        buffer.add(_report(name))

    assert [r["report_id"] for r in buffer.take(10)][0] == "boot"
    assert buffer.dropped == 2
