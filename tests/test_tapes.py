import json

from tapes_eject.config import load
from tapes_eject.export import run_export
from tapes_eject.session import parse_session
from tapes_eject.tapes import (
    Tapes,
    find_labels,
    is_pushback,
    mark,
    merge_auto,
    read_labels,
    write_labels,
)
from tests.helpers import record


def _item(sid, turns=3, seen="2026-10-01T00:00:00Z"):
    return {"id": sid, "last_seen_at": seen, "rollup": {"turn_count": turns, "usage": {}}}


class FakeAPI:
    """Answers GET by path: /v1/sessions pages and /v1/sessions/<id>/traces records."""

    def __init__(self, pages, records):
        self.pages, self.records, self.urls = pages, records, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        path = url.split("127.0.0.1:18081", 1)[1]
        if path.startswith("/v1/sessions?"):
            cursor = path.split("cursor=")[1] if "cursor=" in path else ""
            return json.dumps(self.pages[cursor]).encode()
        sid = path.split("/")[3]
        return json.dumps(self.records[sid]).encode() if sid in self.records else b""


def test_pushback_catches_corrections_not_ordinary_requests():
    assert is_pushback("don't delete the file. just don't write an uneeded test.")
    assert is_pushback("you changed ut")
    assert is_pushback("No, use uv instead")
    assert is_pushback("that is not what I asked for")
    assert not is_pushback("add a --dry-run flag to export")
    assert not is_pushback("now open a PR")


def test_find_labels_marks_turns_and_never_the_first_prompt():
    sess = parse_session(
        record(
            "s1",
            [
                ("t1", "no tests please, add the flag", "Added it."),
                ("t2", "don't touch config.py", "You're right, I'll revert config.py."),
            ],
        )
    )
    rows = find_labels(sess)
    assert [(r["label"], r["primitive_id"]) for r in rows] == [
        ("pushback", "t2"),
        ("apology", "t2"),
    ]
    assert rows[0]["evidence"] == "don't touch config.py" and rows[0]["source"] == "auto"


def test_rescans_replace_auto_labels_and_keep_hand_set_ones(tmp_path):
    old = [
        {
            "label": "pushback",
            "primitive_type": "trace",
            "primitive_id": "t9",
            "session_id": "s1",
            "source": "auto",
        },
        {
            "label": "pushback",
            "primitive_type": "trace",
            "primitive_id": "t8",
            "session_id": "s2",
            "source": "auto",
        },
    ]
    rows = mark(old, "golden", ["s1"], remove=False)
    rows = merge_auto(rows, [], scanned={"s1"})
    assert {(r["label"], r["session_id"]) for r in rows} == {("golden", "s1"), ("pushback", "s2")}
    assert [r["label"] for r in mark(rows, "pushback", ["s2"], remove=True)] == ["golden"]
    path = tmp_path / "local_labels.jsonl"
    write_labels(path, rows)
    assert [r["label"] for r in read_labels(path)] == ["golden", "pushback"]


def test_tapes_source_feeds_export_with_local_labels(tmp_path):
    labels = tmp_path / "local_labels.jsonl"
    write_labels(
        labels,
        [
            {
                "label": "pushback",
                "primitive_type": "trace",
                "primitive_id": "tb",
                "session_id": "s1",
                "evidence": "no",
                "source": "auto",
            },
            {
                "label": "golden",
                "primitive_type": "session",
                "primitive_id": "s2",
                "session_id": "s2",
                "evidence": None,
                "source": "manual",
            },
        ],
    )
    api = FakeAPI(
        pages={
            "": {"items": [_item("s1"), _item("s2")], "next_cursor": "c2"},
            "c2": {"items": [_item("s3", turns=0)]},
        },
        records={
            "s1": record("s1", [("ta", "add a flag", "done"), ("tb", "no, not there", "ok")]),
            "s2": record("s2", [("tc", "fix the test", "fixed")]),
        },
    )
    tapes = Tapes("http://127.0.0.1:18081", labels, fetch=api)
    assert {lab["name"]: lab["usage"] for lab in tapes.labels()} == {
        "pushback": {"trace": 1},
        "golden": {"session": 1},
    }
    assert [it["id"] for it in tapes.sessions(label="pushback")] == ["s1"]
    assert [it["id"] for it in tapes.sessions(limit=2)] == ["s1", "s2"]

    cfg = load({"TAPES_EJECT_EXPORT_PAUSE": "0"})
    ex = run_export(tapes, cfg, log=lambda _: None)
    assert sorted(s["session_id"] for s in ex.sessions) == ["s1", "s2"]
    assert {(r["label"], r["session_id"], r["turn_id"]) for r in ex.labels} == {
        ("pushback", "s1", "tb"),
        ("golden", "s2", None),
    }
    assert not ex.failed and not ex.unmapped
