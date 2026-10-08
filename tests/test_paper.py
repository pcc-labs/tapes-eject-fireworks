import json
import subprocess

import pytest

from tapes_eject import paper
from tapes_eject.paper import Paper, PaperError
from tests.helpers import FakeRunner, record


def test_labels_parses_the_list():
    run = FakeRunner([(("list-labels",), {"labels": [{"id": "L1", "name": "pushback"}]})])
    assert Paper(run).labels() == [{"id": "L1", "name": "pushback"}]


def test_attachments_follow_the_cursor_to_the_end():
    run = FakeRunner(
        [
            (("--cursor", "c1"), {"attachments": [{"primitive_id": "b"}], "next_cursor": None}),
            (
                ("list-label-attachments",),
                {"attachments": [{"primitive_id": "a"}], "next_cursor": "c1"},
            ),
        ]
    )
    got = Paper(run).attachments("L1", "trace")
    assert [a["primitive_id"] for a in got] == ["a", "b"]
    assert run.calls[0][-4:] == ["--primitive-type", "trace", "--limit", "200"]


def test_sessions_passes_label_json_and_pages_until_limit():
    run = FakeRunner(
        [
            (("--cursor", "n1"), {"items": [{"id": "s3"}], "next_cursor": None}),
            (("sessions", "list"), {"items": [{"id": "s1"}, {"id": "s2"}], "next_cursor": "n1"}),
        ]
    )
    got = Paper(run).sessions(label="golden", limit=3)
    assert [s["id"] for s in got] == ["s1", "s2", "s3"]
    assert "--json" in run.calls[0] and ["--label", "golden"] == run.calls[0][-2:]


def test_export_never_passes_detail_and_returns_the_record():
    rec = record("s1", [("trc_1", "hi", "hello")])
    run = FakeRunner(
        [(("status",), "paperd: stopped\n"), (("sessions", "export", "s1"), json.dumps(rec) + "\n")]
    )
    assert Paper(run).export_session("s1") == rec
    assert "--detail" not in run.calls[0]


def test_org_slug_goes_first():
    run = FakeRunner([(("list-labels",), {"labels": []})])
    Paper(run, org_slug="acme").labels()
    assert run.calls[0][:2] == ["--org-slug", "acme"]


def test_paperctl_timeout_becomes_paper_error(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="paperctl", timeout=1)

    monkeypatch.setattr(paper.subprocess, "run", slow)
    with pytest.raises(PaperError, match="timed out"):
        paper.paperctl(["sessions", "export", "s1"], 1)


def test_export_session_reports_a_truncated_record_as_an_error():
    cut = '{"schema": "2026-06-15", "session": {"id": "s1", "title": "half a rec'
    paper = Paper(
        run=FakeRunner([(("status",), "paperd: stopped\n"), (("sessions", "export"), cut)])
    )
    with pytest.raises(PaperError, match="truncated at"):
        paper.export_session("s1")


STATUS = "paperd:    running (pid 1)\nproxy:     127.0.0.1:51539\nauth:      healthy\n"


def test_export_session_reads_core_traces_through_the_proxy_found_in_status():
    run = FakeRunner([(("status",), STATUS)])
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        return json.dumps(record("s1", [("trc_1", "hi", "ok")])).encode()

    paper = Paper(run, fetch=fetch)
    rec = paper.export_session("s1")
    assert rec["session"]["id"] == "s1" and rec["traces"]
    assert urls == ["http://127.0.0.1:51539/v1/sessions/s1/traces"]
    assert paper.export_session("s2") and len(run.calls) == 1  # status asked once


def test_export_session_maps_proxy_outcomes_to_paper_errors():
    paper = Paper(FakeRunner([]), proxy="http://127.0.0.1:1/", fetch=lambda u, t: b"")
    assert paper.export_session("gone") is None
    cut = Paper(FakeRunner([]), proxy="http://p", fetch=lambda u, t: b'{"schema": "x", "sess')
    with pytest.raises(PaperError, match="truncated at"):
        cut.export_session("s1")

    def down(u, t):
        raise PaperError(f"GET {u} could not reach paperd's proxy: refused")

    with pytest.raises(PaperError, match="could not reach"):
        Paper(FakeRunner([]), proxy="http://p", fetch=down).export_session("s1")


def test_export_session_falls_back_to_paperctl_without_a_proxy():
    rec = record("s1", [("trc_1", "hi", "ok")])
    run = FakeRunner(
        [(("status",), "paperd: stopped\n"), (("sessions", "export"), json.dumps(rec))]
    )
    paper = Paper(run, fetch=lambda u, t: (_ for _ in ()).throw(AssertionError("no proxy call")))
    assert paper.export_session("s1")["session"]["id"] == "s1"
    assert run.calls[-1][-3:] == ["sessions", "export", "s1"]


def test_proxy_from_status():
    from tapes_eject.paper import proxy_from_status

    assert proxy_from_status(STATUS) == "http://127.0.0.1:51539"
    assert proxy_from_status("paperd: stopped") is None
