import pytest

from tapes_eject.autolabel import Autolabel, AutolabelError

BASE = "http://h:9996/v1/cassettes/autolabel"


def scripted(*answers):
    calls = []
    queue = list(answers)

    def transport(method, url, body):
        calls.append((method, url, body))
        return queue.pop(0)

    return transport, calls


def result(sessions, reason=None):
    return {"label": "x", "sessions": sessions, "reason": reason, "matched": 0}


def test_run_posts_then_polls_until_done():
    t, calls = scripted(
        {"id": "r1", "state": "running", "progress": "reading sessions"},
        {"id": "r1", "state": "running", "progress": "scanning 2 sessions"},
        {"id": "r1", "state": "done", "result": result([])},
    )
    seen = []
    got = Autolabel(BASE, t, sleep=lambda s: None).run("apology", ["s1"], on_progress=seen.append)
    assert got["sessions"] == []
    assert calls[0] == (
        "POST",
        f"{BASE}/run",
        {"label": "apology", "session_ids": ["s1"], "apply": False},
    )
    assert calls[1][:2] == ("GET", f"{BASE}/runs/r1")
    assert seen == ["scanning 2 sessions"]


def test_run_raises_when_the_job_failed():
    t, _ = scripted(
        {"id": "r1", "state": "running"}, {"id": "r1", "state": "failed", "error": "boom"}
    )
    with pytest.raises(AutolabelError, match="failed: boom"):
        Autolabel(BASE, t, sleep=lambda s: None).run("apology", ["s1"])


def test_run_gives_up_after_max_wait():
    t, _ = scripted(*[{"id": "r1", "state": "running"}] * 5)
    client = Autolabel(BASE, t, poll_seconds=1, max_wait_seconds=2, sleep=lambda s: None)
    with pytest.raises(AutolabelError, match="still running"):
        client.run("apology", ["s1"])


def test_matched_sessions_chunks_and_unions():
    t, calls = scripted(
        {
            "id": "a",
            "state": "done",
            "result": result(
                [
                    {"session_id": "s1", "matched": True, "turns": []},
                    {"session_id": "s2", "matched": False, "turns": []},
                ]
            ),
        },
        {
            "id": "b",
            "state": "done",
            "result": result(
                [
                    {"session_id": "s3", "matched": True, "turns": []},
                ]
            ),
        },
    )
    client = Autolabel(BASE, t, sleep=lambda s: None)
    matched, unknown = client.matched_sessions("no-outcome", ["s1", "s2", "s3"], chunk=2)
    assert matched == {"s1", "s3"} and unknown == set()
    assert [c[2]["session_ids"] for c in calls] == [["s1", "s2"], ["s3"]]


def test_a_failed_chunk_or_missing_session_is_unknown_not_fatal():
    t, _ = scripted(
        {"id": "a", "state": "failed", "error": "paperctl export failed"},
        {
            "id": "b",
            "state": "done",
            "result": {
                **result(
                    [
                        {"session_id": "s3", "matched": True, "turns": []},
                    ]
                ),
                "missing": ["s4"],
            },
        },
    )
    client = Autolabel(BASE, t, sleep=lambda s: None)
    matched, unknown = client.matched_sessions("no-outcome", ["s1", "s2", "s3", "s4"], chunk=2)
    assert matched == {"s3"}
    assert unknown == {"s1", "s2", "s4"}


def test_matched_sessions_raises_on_a_reason():
    t, _ = scripted({"id": "a", "state": "done", "result": result([], reason="needs_judge")})
    with pytest.raises(AutolabelError, match="needs_judge"):
        Autolabel(BASE, t, sleep=lambda s: None).matched_sessions("pushback", ["s1"])


def test_turn_evidence_maps_trace_to_evidence_and_passes_reason():
    turns = [
        {"turn_id": "trc_1", "evidence": "no, use the other flag"},
        {"turn_id": "trc_1", "evidence": "dup"},
    ]
    t, _ = scripted(
        {
            "id": "a",
            "state": "done",
            "result": result(
                [{"session_id": "s1", "matched": True, "turns": turns}], reason="needs_judge"
            ),
        }
    )
    ev, reason = Autolabel(BASE, t, sleep=lambda s: None).turn_evidence("pushback", ["s1"])
    assert ev == {"trc_1": "no, use the other flag"}
    assert reason == "needs_judge"
