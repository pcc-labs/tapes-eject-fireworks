import json

import pytest

from tapes_eject.cli import export_status, label_candidates, main
from tapes_eject.config import load

CFG = load({})


def _item(sid, turns, output_tokens):
    return {"id": sid, "rollup": {"turn_count": turns, "usage": {"output_tokens": output_tokens}}}


def test_label_leaves_out_oversized_sessions_before_the_cassette_exports_them():
    items = [
        _item("small", 3, 3_900),
        _item("empty", 0, 0),
        _item("whale", 3, 796_463),
        _item("long", 351, 2_744_041),
        _item("ok", 28, 294_931),
    ]
    ids, skipped = label_candidates(items, CFG)
    assert ids == ["small", "ok"]  # the empty session is neither sent nor reported
    assert skipped == [
        ("whale", "796463 output tokens > max 400000"),
        ("long", "351 turns > max 150"),
    ]


def test_export_fails_on_problems_unless_partial_is_allowed():
    bad = {"sessions": 2, "failed": [["s3", "timed out"]], "outcome_unknown": 0}
    ok = {"sessions": 2, "failed": [], "outcome_unknown": 0}
    assert export_status(ok, allow_partial=False) == 0
    assert export_status(bad, allow_partial=False) == 1
    assert export_status(bad, allow_partial=True) == 0


def _write(d, name, rows):
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_prepare_refuses_a_partial_export_then_writes_the_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    _write(d, "sessions.jsonl", [{"session_id": "s1", "has_outcome": True, "model": "m"}])
    _write(
        d,
        "turns.jsonl",
        [
            {
                "session_id": "s1",
                "turn_id": "t1",
                "ordinal": 0,
                "user_prompt": "hi",
                "agent_text": "hello",
            }
        ],
    )
    _write(d, "labels.jsonl", [])
    (d / "report.json").write_text(json.dumps({"failed": [["s2", "x"]], "outcome_unknown": 0}))
    with pytest.raises(SystemExit, match="refusing a partial export"):
        main(["prepare"])
    assert main(["prepare", "--force"]) == 0
    training = [json.loads(line) for line in (d / "training.jsonl").read_text().splitlines()]
    assert training[0]["session_id"] == "s1"
    assert (d / "eval_cases.jsonl").exists() and (d / "corrections.jsonl").exists()
